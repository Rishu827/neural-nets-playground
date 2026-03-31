import { LossFunction } from './types';

const EPS = 1e-15;

export function computeLoss(lossFunction: LossFunction, predicted: number[], target: number[]): number {
  switch (lossFunction) {
    case 'mse':
      return mse(predicted, target);
    case 'bce':
      return bce(predicted, target);
    case 'cce':
      return cce(predicted, target);
    default:
      return mse(predicted, target);
  }
}

export function computeLossDerivative(
  lossFunction: LossFunction,
  predicted: number[],
  target: number[]
): number[] {
  switch (lossFunction) {
    case 'mse':
      return mseDerivative(predicted, target);
    case 'bce':
      return bceDerivative(predicted, target);
    case 'cce':
      return cceDerivative(predicted, target);
    default:
      return mseDerivative(predicted, target);
  }
}

// Mean Squared Error
export function mse(predicted: number[], target: number[]): number {
  const n = predicted.length;
  return predicted.reduce((sum, p, i) => sum + (p - target[i]) ** 2, 0) / n;
}

export function mseDerivative(predicted: number[], target: number[]): number[] {
  const n = predicted.length;
  return predicted.map((p, i) => (2 / n) * (p - target[i]));
}

// Binary Cross-Entropy
export function bce(predicted: number[], target: number[]): number {
  const n = predicted.length;
  return (
    -predicted.reduce(
      (sum, p, i) =>
        sum + target[i] * Math.log(Math.max(p, EPS)) + (1 - target[i]) * Math.log(Math.max(1 - p, EPS)),
      0
    ) / n
  );
}

export function bceDerivative(predicted: number[], target: number[]): number[] {
  return predicted.map((p, i) => {
    const pClipped = Math.max(Math.min(p, 1 - EPS), EPS);
    return (-target[i] / pClipped + (1 - target[i]) / (1 - pClipped));
  });
}

// Categorical Cross-Entropy
export function cce(predicted: number[], target: number[]): number {
  return -predicted.reduce((sum, p, i) => sum + target[i] * Math.log(Math.max(p, EPS)), 0);
}

export function cceDerivative(predicted: number[], target: number[]): number[] {
  // Combined with softmax: gradient = predicted - target
  return predicted.map((p, i) => p - target[i]);
}

export function getLossLatex(lossFunction: LossFunction): string {
  switch (lossFunction) {
    case 'mse':
      return 'L = \\frac{1}{n}\\sum_{i=1}^{n}(\\hat{y}_i - y_i)^2';
    case 'bce':
      return 'L = -\\frac{1}{n}\\sum_{i=1}^{n}[y_i\\log(\\hat{y}_i) + (1-y_i)\\log(1-\\hat{y}_i)]';
    case 'cce':
      return 'L = -\\sum_{i=1}^{n} y_i \\log(\\hat{y}_i)';
    default:
      return '';
  }
}
