export type ActivationFunction = 'sigmoid' | 'relu' | 'tanh' | 'softmax' | 'linear';
export type LossFunction = 'mse' | 'bce' | 'cce';
export type WeightInit = 'random' | 'xavier' | 'he';
export type DatasetName = 'xor' | 'iris' | 'mnist_like' | 'sine' | 'circles' | 'moons';

export interface LayerConfig {
  neurons: number;
  activation: ActivationFunction;
}

export interface NetworkConfig {
  layers: LayerConfig[];
  lossFunction: LossFunction;
  learningRate: number;
  weightInit: WeightInit;
}

export interface ForwardPassResult {
  zValues: number[][];   // zValues[layer][neuron] — pre-activation
  aValues: number[][];   // aValues[layer][neuron] — post-activation (layer 0 = input)
  input: number[];
  output: number[];
}

export interface LayerGradients {
  dW: number[][];  // dW[neuron_out][neuron_in]
  db: number[];
  dA_prev: number[];
  delta: number[];
  dA: number[];    // ∂L/∂a for this layer's activations (incoming gradient before activation derivative)
}

export interface BackwardPassResult {
  layerGradients: LayerGradients[];   // index 0 = first weight layer (between input and hidden1)
  loss: number;
  target: number[];    // ground-truth labels used in this pass
  predicted: number[]; // output activations from forward pass
}

export interface LatexEquation {
  label: string;
  symbolic: string;
  numeric: string;
}

export interface TrainingStep {
  epoch: number;
  loss: number;
  weights: number[][][];
  biases: number[][];
}

export interface Dataset {
  name: string;
  inputs: number[][];
  targets: number[][];
  featureNames: string[];
  classNames: string[];
  description: string;
}
