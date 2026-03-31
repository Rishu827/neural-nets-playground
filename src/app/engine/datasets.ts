import { Dataset, DatasetName } from './types';

export function getDataset(name: DatasetName): Dataset {
  switch (name) {
    case 'xor':
      return getXOR();
    case 'iris':
      return getIris();
    case 'mnist_like':
      return getMNISTLike();
    case 'sine':
      return getSine();
    case 'circles':
      return getCircles();
    case 'moons':
      return getMoons();
    default:
      return getXOR();
  }
}

function getXOR(): Dataset {
  return {
    name: 'XOR',
    description: 'Classic XOR problem: output is 1 when inputs differ',
    featureNames: ['x1', 'x2'],
    classNames: ['0', '1'],
    inputs: [[0, 0], [0, 1], [1, 0], [1, 1]],
    targets: [[0], [1], [1], [0]],
  };
}

function getIris(): Dataset {
  // Iris dataset: 3 classes, 4 features, 150 samples (hardcoded subset of 30)
  const data: { input: number[]; target: number[] }[] = [
    { input: [5.1, 3.5, 1.4, 0.2], target: [1, 0, 0] },
    { input: [4.9, 3.0, 1.4, 0.2], target: [1, 0, 0] },
    { input: [4.7, 3.2, 1.3, 0.2], target: [1, 0, 0] },
    { input: [4.6, 3.1, 1.5, 0.2], target: [1, 0, 0] },
    { input: [5.0, 3.6, 1.4, 0.2], target: [1, 0, 0] },
    { input: [5.4, 3.9, 1.7, 0.4], target: [1, 0, 0] },
    { input: [4.6, 3.4, 1.4, 0.3], target: [1, 0, 0] },
    { input: [5.0, 3.4, 1.5, 0.2], target: [1, 0, 0] },
    { input: [4.4, 2.9, 1.4, 0.2], target: [1, 0, 0] },
    { input: [4.9, 3.1, 1.5, 0.1], target: [1, 0, 0] },
    { input: [7.0, 3.2, 4.7, 1.4], target: [0, 1, 0] },
    { input: [6.4, 3.2, 4.5, 1.5], target: [0, 1, 0] },
    { input: [6.9, 3.1, 4.9, 1.5], target: [0, 1, 0] },
    { input: [5.5, 2.3, 4.0, 1.3], target: [0, 1, 0] },
    { input: [6.5, 2.8, 4.6, 1.5], target: [0, 1, 0] },
    { input: [5.7, 2.8, 4.5, 1.3], target: [0, 1, 0] },
    { input: [6.3, 3.3, 4.7, 1.6], target: [0, 1, 0] },
    { input: [4.9, 2.4, 3.3, 1.0], target: [0, 1, 0] },
    { input: [6.6, 2.9, 4.6, 1.3], target: [0, 1, 0] },
    { input: [5.2, 2.7, 3.9, 1.4], target: [0, 1, 0] },
    { input: [6.3, 3.3, 6.0, 2.5], target: [0, 0, 1] },
    { input: [5.8, 2.7, 5.1, 1.9], target: [0, 0, 1] },
    { input: [7.1, 3.0, 5.9, 2.1], target: [0, 0, 1] },
    { input: [6.3, 2.9, 5.6, 1.8], target: [0, 0, 1] },
    { input: [6.5, 3.0, 5.8, 2.2], target: [0, 0, 1] },
    { input: [7.6, 3.0, 6.6, 2.1], target: [0, 0, 1] },
    { input: [4.9, 2.5, 4.5, 1.7], target: [0, 0, 1] },
    { input: [7.3, 2.9, 6.3, 1.8], target: [0, 0, 1] },
    { input: [6.7, 2.5, 5.8, 1.8], target: [0, 0, 1] },
    { input: [7.2, 3.6, 6.1, 2.5], target: [0, 0, 1] },
  ];
  return {
    name: 'Iris',
    description: 'Classic iris flower classification (3 classes, 4 features)',
    featureNames: ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid'],
    classNames: ['setosa', 'versicolor', 'virginica'],
    inputs: data.map(d => d.input),
    targets: data.map(d => d.target),
  };
}

function getMNISTLike(): Dataset {
  // Synthetic 2D classification mimicking MNIST structure
  const rng = seededRandom(42);
  const inputs: number[][] = [];
  const targets: number[][] = [];
  const numClasses = 3;
  const samplesPerClass = 20;

  const centers = [
    [0.2, 0.8],
    [0.8, 0.2],
    [0.5, 0.5],
  ];

  for (let c = 0; c < numClasses; c++) {
    for (let i = 0; i < samplesPerClass; i++) {
      const x = clamp(centers[c][0] + rng() * 0.3 - 0.15, 0, 1);
      const y = clamp(centers[c][1] + rng() * 0.3 - 0.15, 0, 1);
      inputs.push([x, y]);
      const t = [0, 0, 0];
      t[c] = 1;
      targets.push(t);
    }
  }
  return {
    name: 'MNIST-like',
    description: 'Synthetic 2D multi-class classification',
    featureNames: ['x1', 'x2'],
    classNames: ['class_0', 'class_1', 'class_2'],
    inputs,
    targets,
  };
}

function getSine(): Dataset {
  // Sine wave regression
  const inputs: number[][] = [];
  const targets: number[][] = [];
  for (let i = 0; i < 40; i++) {
    const x = (i / 39) * 2 * Math.PI;
    inputs.push([x / (2 * Math.PI)]); // normalized 0..1
    targets.push([(Math.sin(x) + 1) / 2]); // normalized 0..1
  }
  return {
    name: 'Sine Wave',
    description: 'Sine wave regression (normalize input to [0,1])',
    featureNames: ['x'],
    classNames: ['sin(x)'],
    inputs,
    targets,
  };
}

function getCircles(): Dataset {
  const rng = seededRandom(7);
  const inputs: number[][] = [];
  const targets: number[][] = [];
  const n = 50;
  for (let i = 0; i < n; i++) {
    const angle = rng() * 2 * Math.PI;
    const inner = i < n / 2;
    const radius = inner ? 0.3 + rng() * 0.1 : 0.7 + rng() * 0.1;
    const x = 0.5 + radius * Math.cos(angle);
    const y = 0.5 + radius * Math.sin(angle);
    inputs.push([x, y]);
    targets.push(inner ? [1, 0] : [0, 1]);
  }
  return {
    name: 'Circles',
    description: 'Concentric circles binary classification',
    featureNames: ['x1', 'x2'],
    classNames: ['inner', 'outer'],
    inputs,
    targets,
  };
}

function getMoons(): Dataset {
  const rng = seededRandom(13);
  const inputs: number[][] = [];
  const targets: number[][] = [];
  const n = 50;
  for (let i = 0; i < n; i++) {
    const t = (i / (n / 2 - 1)) * Math.PI;
    if (i < n / 2) {
      const x = Math.cos(t) * 0.5 + 0.25 + rng() * 0.1;
      const y = Math.sin(t) * 0.3 + 0.5 + rng() * 0.1;
      inputs.push([x, y]);
      targets.push([1, 0]);
    } else {
      const ti = ((i - n / 2) / (n / 2 - 1)) * Math.PI;
      const x = 1 - Math.cos(ti) * 0.5 + 0.25 + rng() * 0.1;
      const y = 1 - Math.sin(ti) * 0.3 - 0.1 + rng() * 0.1;
      inputs.push([x, y]);
      targets.push([0, 1]);
    }
  }
  return {
    name: 'Moons',
    description: 'Two interleaving moons binary classification',
    featureNames: ['x1', 'x2'],
    classNames: ['moon_0', 'moon_1'],
    inputs,
    targets,
  };
}

// Seeded LCG random for reproducibility
function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}
