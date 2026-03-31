# Neural Network Visualizer

An interactive neural network visualization tool built for learning — watch forward and backward propagation unfold step by step, with full algebra, chain-rule derivations, and a live animated diagram.

## Features

### Math Mode
Step through a complete forward + backward pass one equation at a time:
- **Forward pass**: see every weighted sum `z = W·a + b` expanded per neuron, then the activation applied
- **Loss computation**: cross-entropy or MSE shown with numeric substitution
- **Backward pass**: full chain-rule derivation — loss gradient → activation derivative → delta → weight/bias gradients → gradient descent update with before/after values
- Equation cards render with KaTeX (proper math typesetting)
- The network diagram stays in sync — the active weight layer is highlighted and signal dots travel the correct edges (left-to-right for forward, right-to-left for backward)
- Keyboard navigation: `←` / `→` to step, `Space` to play/pause

### Playground Mode
Watch the network train in real time:
- Animated signal dots on edges during forward and backward passes
- Live loss curve with a red→green gradient line
- Convergence flash (green glow) when loss drops sharply
- Signal snapshot panel: activation bar chart per layer at the current epoch
- Loss trend indicator ▼ / ▲

### Network Configuration
| Setting | Options |
|---|---|
| Layers | Add / remove hidden layers |
| Neurons per layer | Editable per layer |
| Activation | Sigmoid, ReLU, Tanh, Softmax |
| Loss function | MSE, Cross-Entropy |
| Learning rate | Adjustable slider |
| Weight initialization | Random, Xavier, He |

### Datasets
| Dataset | Inputs | Outputs | Task |
|---|---|---|---|
| XOR | 2 | 1 | Binary classification |
| Circles | 2 | 1 | Non-linear binary classification |
| Moons | 2 | 1 | Non-linear binary classification |
| Sine | 1 | 1 | Regression |
| Iris (3-class) | 4 | 3 | Multi-class classification |
| MNIST-like | 64 | 10 | Digit classification |

Switching datasets automatically resizes the input and output layers to match.

### Diagram
- Nodes colored by activation value (heatmap)
- Dashed orange rings on nodes sized by gradient magnitude during backward pass
- Activation function symbol rendered inside each node
- Bias indicators (orange = positive, blue = negative)
- Hoverable nodes and edges with tooltips showing weights, activations, and gradients
- Resizable diagram panel via drag handle

## Getting Started

```bash
npm install
npm start
```

Open [http://localhost:4200](http://localhost:4200).

## Build

```bash
npm run build
```

Output goes to `dist/neural-nets/`.

## Tech Stack

- [Angular 17+](https://angular.dev) — standalone components, `OnPush` change detection
- [Tailwind CSS v4](https://tailwindcss.com) — utility classes via `@use 'tailwindcss'`
- [KaTeX](https://katex.org) — math typesetting for equations
- SVG + `<animateMotion>` — animated network diagram
- Pure TypeScript neural network engine (no ML libraries)

## Project Structure

```
src/
  app/
    engine/
      neuralNetwork.ts   — forward pass, backward pass, equation generation
      activations.ts     — sigmoid, relu, tanh, softmax + derivatives
      losses.ts          — MSE, cross-entropy + derivatives
      datasets.ts        — built-in training datasets
      types.ts           — shared interfaces
    components/
      network-diagram.component.ts   — SVG diagram with animations
      math-mode.component.ts         — step-by-step equation walkthrough
      playground-mode.component.ts   — live training view
      config-panel.component.ts      — network configuration UI
      dataset-panel.component.ts     — dataset selector
      graphs-panel.component.ts      — loss curve and other charts
      equation-display.component.ts  — KaTeX renderer wrapper
    app.ts               — root component, wires everything together
  styles.scss            — global styles, animations
```
