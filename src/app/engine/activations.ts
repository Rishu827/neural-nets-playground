import { ActivationFunction } from './types';

export function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

export function sigmoidDerivative(z: number): number {
  const s = sigmoid(z);
  return s * (1 - s);
}

export function relu(z: number): number {
  return Math.max(0, z);
}

export function reluDerivative(z: number): number {
  return z > 0 ? 1 : 0;
}

export function tanh(z: number): number {
  return Math.tanh(z);
}

export function tanhDerivative(z: number): number {
  const t = Math.tanh(z);
  return 1 - t * t;
}

export function softmax(zArr: number[]): number[] {
  const maxZ = Math.max(...zArr);
  const exps = zArr.map(z => Math.exp(z - maxZ));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sumExp);
}

// Jacobian of softmax (for a single element: s_i * (delta_ij - s_j))
// For backprop we handle this at the loss level, returning element-wise derivative
export function softmaxDerivativeElementWise(aArr: number[], idx: number): number {
  return aArr[idx] * (1 - aArr[idx]);
}

export function linear(z: number): number {
  return z;
}

export function linearDerivative(_z: number): number {
  return 1;
}

export function applyActivation(activation: ActivationFunction, z: number[], prevA?: number[]): number[] {
  switch (activation) {
    case 'sigmoid':
      return z.map(sigmoid);
    case 'relu':
      return z.map(relu);
    case 'tanh':
      return z.map(tanh);
    case 'softmax':
      return softmax(z);
    case 'linear':
      return z.map(linear);
    default:
      return z.map(sigmoid);
  }
}

// Returns element-wise derivative of activation w.r.t. z
// For softmax this is simplified; full softmax backprop is handled via loss
export function applyActivationDerivative(activation: ActivationFunction, z: number[], a: number[]): number[] {
  switch (activation) {
    case 'sigmoid':
      return z.map(sigmoidDerivative);
    case 'relu':
      return z.map(reluDerivative);
    case 'tanh':
      return z.map(tanhDerivative);
    case 'softmax':
      // For softmax with CCE, caller handles combined gradient
      return a.map((ai, i) => ai * (1 - ai));
    case 'linear':
      return z.map(linearDerivative);
    default:
      return z.map(sigmoidDerivative);
  }
}

export function getActivationLatex(activation: ActivationFunction): string {
  switch (activation) {
    case 'sigmoid':
      return '\\sigma(z) = \\frac{1}{1+e^{-z}}';
    case 'relu':
      return '\\text{ReLU}(z) = \\max(0, z)';
    case 'tanh':
      return '\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}';
    case 'softmax':
      return '\\text{softmax}(z_i) = \\frac{e^{z_i}}{\\sum_j e^{z_j}}';
    case 'linear':
      return 'f(z) = z';
    default:
      return '\\sigma(z)';
  }
}
