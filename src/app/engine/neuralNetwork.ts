import {
  LayerConfig,
  NetworkConfig,
  ForwardPassResult,
  BackwardPassResult,
  LayerGradients,
  LatexEquation,
  WeightInit,
} from './types';
import { applyActivation, applyActivationDerivative } from './activations';
import { computeLoss, computeLossDerivative } from './losses';

export class NeuralNetwork {
  layers: LayerConfig[];
  weights: number[][][]; // weights[wl][neuron_out][neuron_in]  (wl = weight-layer index)
  biases: number[][];    // biases[wl][neuron]
  config: NetworkConfig;

  constructor(config: NetworkConfig) {
    this.config = config;
    this.layers = config.layers;
    this.weights = [];
    this.biases = [];
    this.initWeights(config.weightInit);
  }

  initWeights(init: WeightInit): void {
    this.weights = [];
    this.biases = [];
    for (let l = 1; l < this.layers.length; l++) {
      const fanIn  = this.layers[l - 1].neurons;
      const fanOut = this.layers[l].neurons;
      const W: number[][] = [];
      const b: number[]   = [];
      for (let j = 0; j < fanOut; j++) {
        const row: number[] = [];
        for (let k = 0; k < fanIn; k++) row.push(this.initValue(init, fanIn, fanOut));
        W.push(row);
        b.push(0);
      }
      this.weights.push(W);
      this.biases.push(b);
    }
  }

  private initValue(init: WeightInit, fanIn: number, fanOut: number): number {
    const r = Math.random() * 2 - 1;
    switch (init) {
      case 'random':  return r * 0.5;
      case 'xavier':  return r * Math.sqrt(6 / (fanIn + fanOut));
      case 'he':      return (Math.random() * 2 - 1) * Math.sqrt(2 / fanIn);
      default:        return r * 0.5;
    }
  }

  // ── Forward pass ──────────────────────────────────────────────────────────
  forward(input: number[]): ForwardPassResult {
    const zValues: number[][] = [];
    const aValues: number[][] = [input.slice()];

    for (let l = 0; l < this.weights.length; l++) {
      const W = this.weights[l];
      const b = this.biases[l];
      const prevA = aValues[l];
      const z: number[] = [];
      for (let j = 0; j < W.length; j++) {
        let s = b[j];
        for (let k = 0; k < W[j].length; k++) s += W[j][k] * prevA[k];
        z.push(s);
      }
      zValues.push(z);
      aValues.push(applyActivation(this.layers[l + 1].activation, z));
    }

    return { zValues, aValues, input: input.slice(), output: aValues[aValues.length - 1].slice() };
  }

  // ── Backward pass ─────────────────────────────────────────────────────────
  backward(target: number[], forwardResult: ForwardPassResult): BackwardPassResult {
    const { zValues, aValues } = forwardResult;
    const L = this.weights.length;
    const layerGradients: LayerGradients[] = new Array(L);
    const predicted = aValues[aValues.length - 1];
    const loss = computeLoss(this.config.lossFunction, predicted, target);

    let dA = computeLossDerivative(this.config.lossFunction, predicted, target);

    for (let l = L - 1; l >= 0; l--) {
      const z          = zValues[l];
      const activation = this.layers[l + 1].activation;
      const a          = aValues[l + 1];
      const prevA      = aValues[l];
      const W          = this.weights[l];

      let delta: number[];
      if (activation === 'softmax' && this.config.lossFunction === 'cce') {
        delta = dA.slice(); // combined gradient
      } else {
        const dAct = applyActivationDerivative(activation, z, a);
        delta = dA.map((da, j) => da * dAct[j]);
      }

      const dW     = delta.map(d => prevA.map(pa => d * pa));
      const db     = delta.slice();
      const dA_prev: number[] = new Array(prevA.length).fill(0);
      for (let j = 0; j < delta.length; j++)
        for (let k = 0; k < prevA.length; k++)
          dA_prev[k] += W[j][k] * delta[j];

      layerGradients[l] = { dW, db, dA_prev, delta, dA: dA.slice() };
      dA = dA_prev;
    }

    return { layerGradients, loss, target: target.slice(), predicted: predicted.slice() };
  }

  updateWeights(backwardResult: BackwardPassResult, lr: number): void {
    for (let l = 0; l < this.weights.length; l++) {
      const { dW, db } = backwardResult.layerGradients[l];
      for (let j = 0; j < this.weights[l].length; j++) {
        for (let k = 0; k < this.weights[l][j].length; k++)
          this.weights[l][j][k] -= lr * dW[j][k];
        this.biases[l][j] -= lr * db[j];
      }
    }
  }

  cloneWeights(): { weights: number[][][]; biases: number[][] } {
    return {
      weights: this.weights.map(W => W.map(row => row.slice())),
      biases:  this.biases.map(b => b.slice()),
    };
  }

  // ── Forward equations (full algebra per neuron) ───────────────────────────
  getForwardEquations(layerIdx: number, fr: ForwardPassResult): LatexEquation[] {
    const eqs: LatexEquation[] = [];
    const prevA     = fr.aValues[layerIdx];
    const z         = fr.zValues[layerIdx];
    const a         = fr.aValues[layerIdx + 1];
    const W         = this.weights[layerIdx];
    const b         = this.biases[layerIdx];
    const activation = this.layers[layerIdx + 1].activation;
    const l          = layerIdx + 1;
    const maxN       = Math.min(3, W.length); // neurons to show

    // ── Step 1: z = W·a_prev + b ──────────────────────────────────────
    const zNumLines: string[] = [];
    for (let j = 0; j < maxN; j++) {
      const row = W[j];
      // Terms: w_{j,k} * a_k
      const terms = row.map((w, k) => `(${f(w)})(${f(prevA[k])})`).join(' + ');
      // Evaluated terms
      const vals  = row.map((w, k) => f(w * prevA[k])).join(' + ');
      zNumLines.push(
        `z_{${j+1}}^{[${l}]} = ${terms} + ${f(b[j])}`,
        `\\phantom{z_{${j+1}}^{[${l}]}} = ${vals} + ${f(b[j])} = ${f(z[j])}`
      );
      if (j < maxN - 1) zNumLines.push('');
    }
    if (W.length > maxN) zNumLines.push('\\vdots');

    eqs.push({
      label: `\\text{Step 1 — Pre-activation}\\quad z^{[${l}]} = W^{[${l}]} \\cdot a^{[${l-1}]} + b^{[${l}]}`,
      symbolic: `z_j^{[${l}]} = \\sum_{k} w_{jk}^{[${l}]} \\cdot a_k^{[${l-1}]} + b_j^{[${l}]}`,
      numeric: zNumLines.join('\\\\'),
    });

    // ── Step 2: a = activation(z) ─────────────────────────────────────
    const actFormula = getActivationFormula(activation);
    const aNumLines: string[] = [];
    for (let j = 0; j < maxN; j++) {
      aNumLines.push(
        `a_{${j+1}}^{[${l}]} = ${activationSubstituted(activation, z[j], a[j])}`
      );
    }
    if (a.length > maxN) aNumLines.push('\\vdots');

    eqs.push({
      label: `\\text{Step 2 — Activation}\\quad a^{[${l}]} = \\text{${activation}}\\!\\left(z^{[${l}]}\\right)`,
      symbolic: actFormula,
      numeric: aNumLines.join('\\\\'),
    });

    return eqs;
  }

  // ── Backward equations (full chain-rule derivation per layer) ─────────────
  getBackwardEquations(layerIdx: number, br: BackwardPassResult, fr: ForwardPassResult): LatexEquation[] {
    const eqs: LatexEquation[] = [];
    const grad       = br.layerGradients[layerIdx];
    const l          = layerIdx + 1;
    const isOutput   = layerIdx === this.weights.length - 1;
    const activation = this.layers[l].activation;
    const a          = fr.aValues[l];
    const z          = fr.zValues[layerIdx];
    const prevA      = fr.aValues[layerIdx];
    const W          = this.weights[layerIdx];
    const isSoftCCE  = activation === 'softmax' && this.config.lossFunction === 'cce';
    const maxN       = Math.min(3, grad.delta.length);
    const lr         = this.config.learningRate;

    // ── Step 1: Incoming gradient ∂L/∂a^{[l]} ────────────────────────
    if (isSoftCCE) {
      // Softmax + CCE: combined gradient δ = ŷ − y, skip steps 1-2-3 into one
      eqs.push({
        label: `\\text{Step 1 — Softmax + CCE combined gradient}\\quad \\delta^{[${l}]} = \\hat{y} - y`,
        symbolic: `\\frac{\\partial L}{\\partial z^{[${l}]}} = \\hat{y}^{[${l}]} - y \\quad \\text{(softmax \u2295 CCE identity)}`,
        numeric: grad.delta.slice(0, maxN).map((d, j) =>
          `\\delta_{${j+1}}^{[${l}]} = \\hat{y}_{${j+1}} - y_{${j+1}} = ${f(a[j])} - ${f(br.target[j])} = ${f(d)}`
        ).join('\\\\'),
      });
    } else {
      if (isOutput) {
        // Loss derivative formula + numeric substitution
        eqs.push({
          label: `\\text{Step 1 — Loss gradient}\\quad \\dfrac{\\partial L}{\\partial a^{[${l}]}}`,
          symbolic: getLossDerivativeFormula(this.config.lossFunction, l),
          numeric: grad.dA.slice(0, maxN).map((da, j) =>
            `\\frac{\\partial L}{\\partial a_{${j+1}}^{[${l}]}} = ${lossDerivativeSubstituted(this.config.lossFunction, a[j], br.target[j])} = ${f(da)}`
          ).join('\\\\'),
        });
      } else {
        eqs.push({
          label: `\\text{Step 1 — Backpropagated gradient}\\quad \\dfrac{\\partial L}{\\partial a^{[${l}]}}\\;\\text{(from layer }${l+1}\\text{)}`,
          symbolic: `\\frac{\\partial L}{\\partial a_k^{[${l}]}} = \\sum_j W_{jk}^{[${l+1}]} \\cdot \\delta_j^{[${l+1}]}`,
          numeric: grad.dA.slice(0, maxN).map((da, k) =>
            `\\frac{\\partial L}{\\partial a_{${k+1}}^{[${l}]}} = ${f(da)}`
          ).join('\\\\'),
        });
      }

      // ── Step 2: Activation derivative f′(z^{[l]}) ──────────────────
      eqs.push({
        label: `\\text{Step 2 — Activation derivative}\\quad f'\\!\\left(z^{[${l}]}\\right)`,
        symbolic: getActivationDerivativeFormula(activation),
        numeric: z.slice(0, maxN).map((zj, j) => {
          const dact = activationDerivativeValue(activation, zj, a[j]);
          return `f'(z_{${j+1}}^{[${l}]}) = ${activationDerivativeSubstituted(activation, zj, a[j])} = ${f(dact)}`;
        }).join('\\\\'),
      });

      // ── Step 3: Delta δ = ∂L/∂a ⊙ f′(z) ───────────────────────────
      eqs.push({
        label: `\\text{Step 3 — Chain rule}\\quad \\delta^{[${l}]} = \\dfrac{\\partial L}{\\partial a^{[${l}]}} \\odot f'\\!\\left(z^{[${l}]}\\right)`,
        symbolic: `\\delta^{[${l}]} = \\frac{\\partial L}{\\partial a^{[${l}]}} \\odot f'\\!\\left(z^{[${l}]}\\right)`,
        numeric: grad.delta.slice(0, maxN).map((d, j) => {
          const dact = activationDerivativeValue(activation, z[j], a[j]);
          return `\\delta_{${j+1}}^{[${l}]} = ${f(grad.dA[j])} \\times ${f(dact)} = ${f(d)}`;
        }).join('\\\\'),
      });
    }

    // ── Step 4: Weight gradient ∂L/∂W^{[l]} = δ · a_prev^T ───────────
    const maxJ = Math.min(2, grad.delta.length);
    const maxK = Math.min(2, prevA.length);
    const dWLines: string[] = [];
    for (let j = 0; j < maxJ; j++) {
      for (let k = 0; k < maxK; k++) {
        dWLines.push(
          `\\frac{\\partial L}{\\partial w_{${j+1},${k+1}}^{[${l}]}} = \\delta_{${j+1}}^{[${l}]} \\cdot a_{${k+1}}^{[${l-1}]} = ${f(grad.delta[j])} \\times ${f(prevA[k])} = ${f(grad.dW[j][k])}`
        );
      }
    }
    if (W.length * prevA.length > maxJ * maxK) dWLines.push('\\vdots');

    eqs.push({
      label: `\\text{Step 4 — Weight gradient}\\quad \\dfrac{\\partial L}{\\partial W^{[${l}]}} = \\delta^{[${l}]} \\cdot \\bigl(a^{[${l-1}]}\\bigr)^{\\!\\top}`,
      symbolic: `\\frac{\\partial L}{\\partial W^{[${l}]}} = \\delta^{[${l}]} \\cdot \\left(a^{[${l-1}]}\\right)^{\\!\\top}`,
      numeric: dWLines.join('\\\\'),
    });

    // ── Step 5: Bias gradient ∂L/∂b^{[l]} = δ ────────────────────────
    eqs.push({
      label: `\\text{Step 5 — Bias gradient}\\quad \\dfrac{\\partial L}{\\partial b^{[${l}]}} = \\delta^{[${l}]}`,
      symbolic: `\\frac{\\partial L}{\\partial b_j^{[${l}]}} = \\delta_j^{[${l}]}`,
      numeric: grad.db.slice(0, maxN).map((d, j) =>
        `\\frac{\\partial L}{\\partial b_{${j+1}}^{[${l}]}} = ${f(d)}`
      ).join('\\\\'),
    });

    // ── Step 6: Propagate gradient back to a^{[l-1]} ─────────────────
    if (layerIdx > 0) {
      const maxKprev = Math.min(2, prevA.length);
      const maxJprev = Math.min(3, grad.delta.length);
      const bpLines: string[] = [];
      for (let k = 0; k < maxKprev; k++) {
        const terms = grad.delta.slice(0, maxJprev)
          .map((dj, j) => `w_{${j+1},${k+1}}^{[${l}]}(${f(dj)})`).join(' + ');
        const ellipsis = grad.delta.length > maxJprev ? ' + \\cdots' : '';
        bpLines.push(
          `\\frac{\\partial L}{\\partial a_{${k+1}}^{[${l-1}]}} = ${terms}${ellipsis} = ${f(grad.dA_prev[k])}`
        );
      }
      if (prevA.length > maxKprev) bpLines.push('\\vdots');

      eqs.push({
        label: `\\text{Step 6 — Propagate to}\\; a^{[${l-1}]}\\;\\text{(feeds next backward step)}`,
        symbolic: `\\frac{\\partial L}{\\partial a_k^{[${l-1}]}} = \\sum_j W_{jk}^{[${l}]} \\cdot \\delta_j^{[${l}]}`,
        numeric: bpLines.join('\\\\'),
      });
    }

    // ── Step 7: Apply gradient descent update ─────────────────────────
    const maxJu = Math.min(2, W.length);
    const maxKu = Math.min(2, prevA.length);
    const updLines: string[] = [];
    for (let j = 0; j < maxJu; j++) {
      for (let k = 0; k < maxKu; k++) {
        const oldW = W[j][k];
        const dw   = grad.dW[j][k];
        const newW = oldW - lr * dw;
        updLines.push(
          `w_{${j+1},${k+1}}^{[${l}]} \\leftarrow ${f(oldW)} - ${f(lr)} \\times ${f(dw)} = ${f(newW)}`
        );
      }
    }
    for (let j = 0; j < Math.min(2, W.length); j++) {
      const oldb = this.biases[layerIdx][j];
      const db   = grad.db[j];
      updLines.push(
        `b_{${j+1}}^{[${l}]} \\leftarrow ${f(oldb)} - ${f(lr)} \\times ${f(db)} = ${f(oldb - lr * db)}`
      );
    }
    if (W.length * prevA.length > maxJu * maxKu) updLines.push('\\vdots');

    eqs.push({
      label: `\\text{Step 7 — Gradient descent update}\\quad W^{[${l}]} \\leftarrow W^{[${l}]} - \\alpha \\cdot \\nabla W^{[${l}]}`,
      symbolic: `W^{[${l}]} \\leftarrow W^{[${l}]} - \\alpha \\cdot \\frac{\\partial L}{\\partial W^{[${l}]}}`,
      numeric: updLines.join('\\\\'),
    });

    return eqs;
  }
}

// ── Number formatting ─────────────────────────────────────────────────────────
function f(n: number): string {
  if (!isFinite(n) || isNaN(n)) return '0';
  const s = parseFloat(n.toFixed(4)).toString();
  return s;
}

// ── Activation helpers ────────────────────────────────────────────────────────
function getActivationFormula(act: string): string {
  switch (act) {
    case 'sigmoid': return '\\sigma(z) = \\dfrac{1}{1 + e^{-z}}';
    case 'relu':    return '\\text{ReLU}(z) = \\max(0,\\, z)';
    case 'tanh':    return '\\tanh(z) = \\dfrac{e^z - e^{-z}}{e^z + e^{-z}}';
    case 'softmax': return '\\text{softmax}(z_j) = \\dfrac{e^{z_j}}{\\displaystyle\\sum_k e^{z_k}}';
    case 'linear':  return 'f(z) = z';
    default:        return 'f(z)';
  }
}

function activationSubstituted(act: string, z: number, a: number): string {
  const zs = f(z);
  const as = f(a);
  switch (act) {
    case 'sigmoid': {
      const expNeg = f(Math.exp(-z));
      const denom  = f(1 + Math.exp(-z));
      return `\\sigma(${zs}) = \\dfrac{1}{1 + e^{${f(-z)}}} = \\dfrac{1}{1 + ${expNeg}} = \\dfrac{1}{${denom}} = ${as}`;
    }
    case 'relu':
      return `\\text{ReLU}(${zs}) = \\max(0,\\, ${zs}) = ${as}`;
    case 'tanh': {
      const ep = f(Math.exp(z));
      const en = f(Math.exp(-z));
      return `\\tanh(${zs}) = \\dfrac{${ep} - ${en}}{${ep} + ${en}} = ${as}`;
    }
    case 'softmax':
      return `\\text{softmax}(${zs}) = ${as}`;
    case 'linear':
      return `f(${zs}) = ${zs} = ${as}`;
    default:
      return `f(${zs}) = ${as}`;
  }
}

function getActivationDerivativeFormula(act: string): string {
  switch (act) {
    case 'sigmoid': return `\\sigma'(z) = \\sigma(z)\\,(1 - \\sigma(z))`;
    case 'relu':    return `\\text{ReLU}'(z) = \\begin{cases} 1 & z > 0 \\\\ 0 & z \\le 0 \\end{cases}`;
    case 'tanh':    return `\\tanh'(z) = 1 - \\tanh^2(z)`;
    case 'linear':  return `f'(z) = 1`;
    case 'softmax': return `\\text{softmax}'(z_j) = a_j(1 - a_j) \\quad \\text{(diagonal of Jacobian)}`;
    default:        return `f'(z)`;
  }
}

function activationDerivativeValue(act: string, z: number, a: number): number {
  switch (act) {
    case 'sigmoid': return a * (1 - a);
    case 'relu':    return z > 0 ? 1 : 0;
    case 'tanh':    return 1 - a * a;
    case 'linear':  return 1;
    case 'softmax': return a * (1 - a);
    default:        return 1;
  }
}

function activationDerivativeSubstituted(act: string, z: number, a: number): string {
  const zs   = f(z);
  const as   = f(a);
  const dact = activationDerivativeValue(act, z, a);
  switch (act) {
    case 'sigmoid':
      return `\\sigma(${zs})(1 - \\sigma(${zs})) = ${as} \\times (1 - ${as}) = ${as} \\times ${f(1 - a)}`;
    case 'relu':
      return z > 0 ? `1 \\quad (z = ${zs} > 0)` : `0 \\quad (z = ${zs} \\le 0)`;
    case 'tanh':
      return `1 - \\tanh^2(${zs}) = 1 - (${as})^2 = 1 - ${f(a * a)}`;
    case 'linear':
      return `1`;
    case 'softmax':
      return `${as}(1 - ${as})`;
    default:
      return f(dact);
  }
}

// ── Loss derivative helpers ───────────────────────────────────────────────────
function getLossDerivativeFormula(loss: string, l: number): string {
  switch (loss) {
    case 'mse':
      return `\\frac{\\partial L}{\\partial a_j^{[${l}]}} = \\frac{2}{n}\\left(\\hat{y}_j - y_j\\right)`;
    case 'bce':
      return `\\frac{\\partial L}{\\partial a_j^{[${l}]}} = -\\frac{y_j}{a_j} + \\frac{1 - y_j}{1 - a_j}`;
    case 'cce':
      return `\\frac{\\partial L}{\\partial a_j^{[${l}]}} = -\\frac{y_j}{a_j}`;
    default:
      return `\\frac{\\partial L}{\\partial a^{[${l}]}}`;
  }
}

function lossDerivativeSubstituted(loss: string, a: number, y: number): string {
  const as = f(a);
  const ys = f(y);
  switch (loss) {
    case 'mse': {
      const diff = f(2 * (a - y));
      return `\\tfrac{2}{1}(${as} - ${ys}) = 2 \\times ${f(a - y)} = ${diff}`;
    }
    case 'bce': {
      const t1 = y === 0 ? '0' : f(-y / Math.max(a, 1e-15));
      const t2 = y === 1 ? '0' : f((1 - y) / Math.max(1 - a, 1e-15));
      return `-\\frac{${ys}}{${as}} + \\frac{1-${ys}}{1-${as}} = ${t1} + ${t2}`;
    }
    case 'cce': {
      const t = y === 0 ? '0' : f(-y / Math.max(a, 1e-15));
      return `-\\frac{${ys}}{${as}} = ${t}`;
    }
    default:
      return f(a - y);
  }
}
