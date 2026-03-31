import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { NeuralNetwork } from './engine/neuralNetwork';
import { NetworkConfig, Dataset, DatasetName, ForwardPassResult } from './engine/types';
import { getDataset } from './engine/datasets';

import { ConfigPanelComponent } from './components/config-panel.component';
import { NetworkDiagramComponent } from './components/network-diagram.component';
import { MathModeComponent } from './components/math-mode.component';
import { PlaygroundModeComponent } from './components/playground-mode.component';
import { DatasetPanelComponent } from './components/dataset-panel.component';
import { GraphsPanelComponent } from './components/graphs-panel.component';

type AppMode = 'math' | 'playground';
type PanelTab = 'diagram' | 'graphs' | 'data';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ConfigPanelComponent,
    NetworkDiagramComponent,
    MathModeComponent,
    PlaygroundModeComponent,
    DatasetPanelComponent,
    GraphsPanelComponent,
  ],
  template: `
    <div class="app-root flex flex-col h-screen overflow-hidden bg-slate-900 text-slate-100">

      <!-- Header -->
      <header class="flex items-center gap-4 px-6 py-3 bg-slate-800 border-b border-slate-700 flex-shrink-0">
        <div class="flex items-center gap-2">
          <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
            <circle cx="5" cy="14" r="3" fill="#3b82f6"/>
            <circle cx="14" cy="7" r="3" fill="#8b5cf6"/>
            <circle cx="14" cy="21" r="3" fill="#8b5cf6"/>
            <circle cx="23" cy="14" r="3" fill="#22c55e"/>
            <line x1="8" y1="14" x2="11" y2="7" stroke="#475569" stroke-width="1.5"/>
            <line x1="8" y1="14" x2="11" y2="21" stroke="#475569" stroke-width="1.5"/>
            <line x1="17" y1="7" x2="20" y2="14" stroke="#475569" stroke-width="1.5"/>
            <line x1="17" y1="21" x2="20" y2="14" stroke="#475569" stroke-width="1.5"/>
          </svg>
          <span class="text-lg font-bold text-white">Neural Net Visualizer</span>
        </div>

        <!-- Mode tabs -->
        <div class="flex gap-1 ml-4 bg-slate-900 rounded-xl p-1">
          <button class="mode-tab" [class.active]="mode === 'math'" (click)="setMode('math')">
            ∑ Math Mode
          </button>
          <button class="mode-tab" [class.active]="mode === 'playground'" (click)="setMode('playground')">
            ▶ Playground
          </button>
        </div>

        <div class="flex-1"></div>

        <!-- Dataset quick pick -->
        <div class="flex items-center gap-2 text-xs text-slate-400">
          <span>Dataset:</span>
          <select class="header-select" [(ngModel)]="currentDatasetName" (change)="onDatasetChange()">
            <option value="xor">XOR</option>
            <option value="iris">Iris</option>
            <option value="mnist_like">MNIST-like</option>
            <option value="sine">Sine Wave</option>
            <option value="circles">Circles</option>
            <option value="moons">Moons</option>
          </select>
        </div>

        <!-- Status indicator -->
        <div class="flex items-center gap-2 text-xs">
          <div class="w-2 h-2 rounded-full" [class.bg-green-400]="isReady" [class.bg-yellow-400]="!isReady"></div>
          <span class="text-slate-400">{{ isReady ? 'Ready' : 'Loading...' }}</span>
        </div>
      </header>

      <!-- Main layout -->
      <div class="flex flex-1 overflow-hidden">

        <!-- Left: Config Panel -->
        <aside class="w-60 flex-shrink-0 overflow-y-auto border-r border-slate-700">
          <app-config-panel
            [config]="networkConfig"
            (configChange)="onConfigChange($event)"
            (reinitialize)="onReinitialize()">
          </app-config-panel>
        </aside>

        <!-- Center: Main content -->
        <main class="flex-1 flex flex-col overflow-hidden"
          style="background-image: radial-gradient(circle, #1e293b 1px, transparent 1px); background-size: 24px 24px;">

          <!-- Network Diagram with resizable height -->
          <div class="border-b border-slate-700 bg-slate-850 flex-shrink-0 relative"
            [style.height.px]="diagramHeight">
            <app-network-diagram
              [config]="networkConfig"
              [weights]="network?.weights ?? []"
              [biases]="network?.biases ?? []"
              [forwardResult]="currentForwardResult"
              [backwardResult]="currentBackwardResult"
              [activeLayer]="activeLayer"
              [mode]="diagramMode"
              (nodeClicked)="onDiagramNodeClick($event)">
            </app-network-diagram>
          </div>

          <!-- Drag resize handle -->
          <div class="resize-handle"
            (mousedown)="onDragStart($event)"
            title="Drag to resize">
          </div>

          <!-- Mode content area -->
          <div class="flex-1 overflow-y-auto p-4" [class.mode-switching]="modeSwitching">
            @if (mode === 'math') {
              <app-math-mode
                [network]="network"
                [dataset]="currentDataset"
                (activeLayerChange)="onActiveLayerChange($event)"
                (stepCompleted)="onStepCompleted($event)">
              </app-math-mode>
            }
            @if (mode === 'playground') {
              <app-playground-mode
                [network]="network"
                [dataset]="currentDataset"
                (epochCompleted)="onEpochCompleted($event)"
                (activeLayerChange)="onActiveLayerChange($event)">
              </app-playground-mode>
            }
          </div>
        </main>

        <!-- Right: Data + Graphs panel -->
        <aside class="w-72 flex-shrink-0 border-l border-slate-700 flex flex-col overflow-hidden">
          <!-- Right panel tabs -->
          <div class="flex bg-slate-800 border-b border-slate-700">
            @for (tab of rightTabs; track tab.id) {
              <button class="right-tab flex-1"
                [class.active]="rightPanel === tab.id"
                (click)="rightPanel = tab.id">
                {{ tab.label }}
              </button>
            }
          </div>

          <!-- Tab content -->
          <div class="flex-1 overflow-y-auto p-3">
            @if (rightPanel === 'data') {
              <app-dataset-panel
                [selectedDataset]="currentDatasetName"
                (datasetChange)="onDatasetPanelChange($event)">
              </app-dataset-panel>
            }

            @if (rightPanel === 'graphs') {
              <app-graphs-panel
                [network]="network"
                [dataset]="currentDataset"
                [lossHistory]="lossHistory"
                [epochCount]="epochCount">
              </app-graphs-panel>
            }

            @if (rightPanel === 'diagram') {
              <div class="flex flex-col gap-3">
                <div class="text-xs text-slate-400 font-semibold uppercase tracking-wide">Architecture Info</div>
                @if (network) {
                  <div class="flex flex-col gap-2">
                    @for (layer of networkConfig.layers; track $index) {
                      <div class="bg-slate-800 rounded-lg p-3 border border-slate-700">
                        <div class="text-xs font-semibold mb-1"
                          [class.text-blue-300]="$index === 0"
                          [class.text-purple-300]="$index > 0 && $index < networkConfig.layers.length - 1"
                          [class.text-green-300]="$index === networkConfig.layers.length - 1">
                          {{ $index === 0 ? 'Input Layer' : $index === networkConfig.layers.length - 1 ? 'Output Layer' : 'Hidden Layer ' + $index }}
                        </div>
                        <div class="text-xs text-slate-400">Neurons: <span class="text-white">{{ layer.neurons }}</span></div>
                        <div class="text-xs text-slate-400">Activation: <span class="text-yellow-300">{{ layer.activation }}</span></div>
                        @if ($index > 0) {
                          <div class="text-xs text-slate-400">
                            Params: <span class="text-white">{{ layer.neurons * networkConfig.layers[$index - 1].neurons + layer.neurons }}</span>
                          </div>
                        }
                      </div>
                    }

                    <div class="bg-slate-800 rounded-lg p-3 border border-slate-700">
                      <div class="text-xs font-semibold text-slate-300 mb-2">Training Config</div>
                      <div class="text-xs text-slate-400">Loss: <span class="text-yellow-300">{{ networkConfig.lossFunction }}</span></div>
                      <div class="text-xs text-slate-400">Learning rate: <span class="text-blue-300 font-mono">{{ networkConfig.learningRate }}</span></div>
                      <div class="text-xs text-slate-400">Init: <span class="text-purple-300">{{ networkConfig.weightInit }}</span></div>
                    </div>
                  </div>
                }
              </div>
            }
          </div>
        </aside>
      </div>

      <!-- Footer -->
      <footer class="text-xs text-slate-600 px-6 py-2 border-t border-slate-800 flex justify-between flex-shrink-0">
        <span>Neural Network Visualizer — Angular + TypeScript</span>
        @if (epochCount > 0) {
          <span>Trained {{ epochCount }} epochs · Last loss: {{ lossHistory[lossHistory.length - 1]?.toFixed(6) }}</span>
        }
      </footer>
    </div>
  `,
  styles: [`
    :host { display: block; height: 100vh; overflow: hidden; }
    .app-root { font-size: 14px; }
    .mode-tab {
      padding: 6px 18px; border-radius: 8px; border: none; background: transparent;
      color: #64748b; font-size: 13px; font-weight: 500; cursor: pointer; transition: all 0.15s;
    }
    .mode-tab.active { background: #1e3a5f; color: #93c5fd; }
    .mode-tab:hover:not(.active) { color: #94a3b8; }
    .header-select {
      background: #1e293b; border: 1px solid #334155; border-radius: 6px;
      color: #f1f5f9; padding: 4px 8px; font-size: 12px; outline: none;
    }
    .right-tab {
      padding: 8px 4px; border: none; background: transparent;
      color: #64748b; font-size: 12px; cursor: pointer; transition: all 0.15s;
      border-bottom: 2px solid transparent;
    }
    .right-tab.active { color: #93c5fd; border-bottom-color: #3b82f6; background: #0f172a; }
    .right-tab:hover:not(.active) { color: #94a3b8; }
    .bg-slate-850 { background: #0f1929; }

    /* Resize handle */
    .resize-handle {
      height: 6px; background: transparent; cursor: row-resize; flex-shrink: 0;
      border-top: 1px solid #1e293b; border-bottom: 1px solid #1e293b;
      transition: background 0.15s;
    }
    .resize-handle:hover { background: #334155; }

    /* Mode transition */
    @keyframes mode-fade-in {
      from { opacity: 0; }
      to   { opacity: 1; }
    }
    .mode-switching { animation: mode-fade-in 0.2s ease-out; }
  `]
})
export class App implements OnInit {
  mode: AppMode = 'math';
  modeSwitching = false;
  rightPanel: PanelTab = 'data';
  rightTabs = [
    { id: 'data' as PanelTab, label: 'Dataset' },
    { id: 'graphs' as PanelTab, label: 'Graphs' },
    { id: 'diagram' as PanelTab, label: 'Info' },
  ];

  network: NeuralNetwork | null = null;
  networkConfig: NetworkConfig = {
    layers: [
      { neurons: 2, activation: 'linear' },
      { neurons: 2, activation: 'sigmoid' },
      { neurons: 1, activation: 'sigmoid' },
    ],
    lossFunction: 'bce',
    learningRate: 0.1,
    weightInit: 'xavier',
  };

  currentDatasetName: DatasetName = 'xor';
  currentDataset: Dataset | null = null;
  currentForwardResult: ForwardPassResult | null = null;
  currentBackwardResult: import('./engine/types').BackwardPassResult | null = null;
  activeLayer = -1;
  diagramMode: 'forward' | 'backward' | 'idle' = 'idle';

  lossHistory: number[] = [];
  epochCount = 0;
  isReady = false;

  // Diagram resize state
  diagramHeight = 340;
  private isDragging = false;
  private dragStartY = 0;
  private dragStartH = 340;
  private boundDrag: ((e: MouseEvent) => void) | null = null;
  private boundDragEnd: ((e: MouseEvent) => void) | null = null;

  private modeSwitchTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(private cdr: ChangeDetectorRef) {}

  ngOnInit(): void {
    this.currentDataset = getDataset(this.currentDatasetName);
    this.network = new NeuralNetwork(this.networkConfig);
    if (this.currentDataset) {
      this.currentForwardResult = this.network.forward(this.currentDataset.inputs[0]);
    }
    this.isReady = true;
    this.cdr.markForCheck();
  }

  setMode(m: AppMode): void {
    if (this.mode === m) return;
    this.mode = m;
    this.diagramMode = 'idle';
    this.activeLayer = -1;
    // Trigger opacity transition
    this.modeSwitching = true;
    if (this.modeSwitchTimer) clearTimeout(this.modeSwitchTimer);
    this.modeSwitchTimer = setTimeout(() => {
      this.modeSwitching = false;
      this.cdr.markForCheck();
    }, 200);
  }

  onConfigChange(config: NetworkConfig): void {
    this.networkConfig = config;
    this.network = new NeuralNetwork(config);
    if (this.currentDataset) {
      this.currentForwardResult = this.network.forward(this.currentDataset.inputs[0]);
    }
    this.lossHistory = [];
    this.epochCount = 0;
    this.cdr.markForCheck();
  }

  onReinitialize(): void {
    if (this.network) {
      this.network.initWeights(this.networkConfig.weightInit);
      if (this.currentDataset) {
        this.currentForwardResult = this.network.forward(this.currentDataset.inputs[0]);
      }
      this.lossHistory = [];
      this.epochCount = 0;
      this.cdr.markForCheck();
    }
  }

  onDatasetChange(): void {
    this.currentDataset = getDataset(this.currentDatasetName);
    if (this.currentDataset) {
      this.adjustNetworkForDataset(this.currentDataset);
    }
    this.lossHistory = [];
    this.epochCount = 0;
    this.cdr.markForCheck();
  }

  onDatasetPanelChange(event: { name: DatasetName; dataset: Dataset }): void {
    this.currentDatasetName = event.name;
    this.currentDataset = event.dataset;
    if (this.currentDataset) {
      this.adjustNetworkForDataset(this.currentDataset);
    }
    this.lossHistory = [];
    this.epochCount = 0;
    this.cdr.markForCheck();
  }

  private adjustNetworkForDataset(dataset: Dataset): void {
    const inputSize = dataset.inputs[0].length;
    const outputSize = dataset.targets[0].length;
    const layers = this.networkConfig.layers.map((l, _i) => ({ ...l }));
    layers[0] = { ...layers[0], neurons: inputSize };
    layers[layers.length - 1] = { ...layers[layers.length - 1], neurons: outputSize };
    this.networkConfig = { ...this.networkConfig, layers };
    this.network = new NeuralNetwork(this.networkConfig);
    this.currentForwardResult = this.network.forward(dataset.inputs[0]);
  }

  onActiveLayerChange(layer: number): void {
    this.activeLayer = layer;
  }

  onDiagramNodeClick(_event: { layer: number; neuron: number }): void {
    // Reserved for future: jump math-mode to the step for this layer
  }

  onStepCompleted(event: { forwardResult: ForwardPassResult | null; activeLayer: number; mode: 'forward' | 'backward' | 'idle' }): void {
    this.currentForwardResult = event.forwardResult;
    this.activeLayer = event.activeLayer;
    this.diagramMode = event.mode ?? (event.activeLayer >= 0 ? 'forward' : 'idle');
    this.cdr.markForCheck();
  }

  onEpochCompleted(event: { loss: number; epoch: number; forwardResult: ForwardPassResult }): void {
    this.lossHistory = [...this.lossHistory, event.loss].slice(-500);
    this.epochCount = event.epoch;
    this.currentForwardResult = event.forwardResult;
    this.diagramMode = 'forward';
    this.cdr.markForCheck();
  }

  // ── Diagram resize handlers ──────────────────────────────────────────────
  onDragStart(evt: MouseEvent): void {
    this.isDragging = true;
    this.dragStartY = evt.clientY;
    this.dragStartH = this.diagramHeight;

    this.boundDrag = (e: MouseEvent) => this.onDrag(e);
    this.boundDragEnd = (e: MouseEvent) => this.onDragEnd(e);
    document.addEventListener('mousemove', this.boundDrag);
    document.addEventListener('mouseup', this.boundDragEnd);
    evt.preventDefault();
  }

  private onDrag(evt: MouseEvent): void {
    if (!this.isDragging) return;
    const delta = evt.clientY - this.dragStartY;
    this.diagramHeight = Math.min(500, Math.max(220, this.dragStartH + delta));
    this.cdr.markForCheck();
  }

  private onDragEnd(_evt: MouseEvent): void {
    this.isDragging = false;
    if (this.boundDrag) document.removeEventListener('mousemove', this.boundDrag);
    if (this.boundDragEnd) document.removeEventListener('mouseup', this.boundDragEnd);
    this.boundDrag = null;
    this.boundDragEnd = null;
  }
}
