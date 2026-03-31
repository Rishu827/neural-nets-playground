import { Component, Input, Output, EventEmitter, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { NetworkConfig, LayerConfig, ActivationFunction, LossFunction, WeightInit } from '../engine/types';

@Component({
  selector: 'app-config-panel',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="config-panel h-full overflow-y-auto bg-slate-800 border-r border-slate-700 p-4 flex flex-col gap-4">
      <div class="text-lg font-bold text-blue-400 flex items-center gap-2">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="3"/><path d="M19.07 4.93A10 10 0 0 0 4.93 19.07M19.07 4.93A10 10 0 1 1 4.93 19.07"/>
        </svg>
        Network Config
      </div>

      <!-- Layer count -->
      <div class="config-section">
        <label class="config-label">Hidden Layers</label>
        <div class="flex items-center gap-2">
          <button class="icon-btn" (click)="removeHiddenLayer()" [disabled]="hiddenLayers.length <= 1">−</button>
          <span class="flex-1 text-center text-white font-mono">{{ hiddenLayers.length }}</span>
          <button class="icon-btn" (click)="addHiddenLayer()" [disabled]="hiddenLayers.length >= 5">+</button>
        </div>
      </div>

      <!-- Input neurons -->
      <div class="config-section">
        <label class="config-label">Input Features</label>
        <input type="number" class="config-input" [(ngModel)]="inputNeurons" [min]="1" [max]="8" (change)="onConfigChange()">
      </div>

      <!-- Hidden layers configuration -->
      @for (layer of hiddenLayers; track $index) {
        <div class="config-section">
          <label class="config-label">Hidden Layer {{ $index + 1 }}</label>
          <div class="flex gap-2">
            <input type="number" class="config-input w-20" [(ngModel)]="layer.neurons" [min]="1" [max]="16"
              (change)="onConfigChange()" placeholder="neurons">
            <select class="config-select flex-1" [(ngModel)]="layer.activation" (change)="onConfigChange()">
              @for (act of activations; track act) {
                <option [value]="act">{{ act }}</option>
              }
            </select>
          </div>
        </div>
      }

      <!-- Output layer -->
      <div class="config-section">
        <label class="config-label">Output Layer</label>
        <div class="flex gap-2">
          <input type="number" class="config-input w-20" [(ngModel)]="outputNeurons" [min]="1" [max]="10"
            (change)="onConfigChange()" placeholder="neurons">
          <select class="config-select flex-1" [(ngModel)]="outputActivation" (change)="onConfigChange()">
            @for (act of activations; track act) {
              <option [value]="act">{{ act }}</option>
            }
          </select>
        </div>
      </div>

      <div class="border-t border-slate-600 my-1"></div>

      <!-- Loss function -->
      <div class="config-section">
        <label class="config-label">Loss Function</label>
        <select class="config-select" [(ngModel)]="lossFunction" (change)="onConfigChange()">
          <option value="mse">MSE</option>
          <option value="bce">Binary Cross-Entropy</option>
          <option value="cce">Categorical Cross-Entropy</option>
        </select>
      </div>

      <!-- Learning rate -->
      <div class="config-section">
        <label class="config-label">Learning Rate: <span class="text-blue-300 font-mono">{{ learningRate }}</span></label>
        <input type="range" class="w-full accent-blue-500" [(ngModel)]="learningRate"
          [min]="0.001" [max]="1" [step]="0.001" (change)="onConfigChange()">
      </div>

      <!-- Weight initialization -->
      <div class="config-section">
        <label class="config-label">Weight Initialization</label>
        <select class="config-select" [(ngModel)]="weightInit" (change)="onConfigChange()">
          <option value="random">Random</option>
          <option value="xavier">Xavier/Glorot</option>
          <option value="he">He</option>
        </select>
      </div>

      <!-- Reinitialize button -->
      <button class="btn-primary mt-2" (click)="onReinit()">
        Re-initialize Weights
      </button>

      <!-- Network summary -->
      <div class="border-t border-slate-600 mt-2 pt-3">
        <div class="text-xs text-slate-400 mb-2">Architecture Summary</div>
        <div class="text-xs text-slate-300 font-mono flex flex-wrap gap-1">
          <span class="px-2 py-1 bg-blue-900 rounded">{{ inputNeurons }}</span>
          <span class="text-slate-500">→</span>
          @for (l of hiddenLayers; track $index) {
            <span class="px-2 py-1 bg-slate-700 rounded">{{ l.neurons }}</span>
            <span class="text-slate-500">→</span>
          }
          <span class="px-2 py-1 bg-green-900 rounded">{{ outputNeurons }}</span>
        </div>
        <div class="text-xs text-slate-400 mt-2">
          Total params: <span class="text-white font-mono">{{ totalParams }}</span>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .config-panel { min-width: 240px; }
    .config-label { display: block; font-size: 12px; color: #94a3b8; margin-bottom: 4px; }
    .config-section { display: flex; flex-direction: column; gap: 4px; }
    .config-input {
      background: #1e293b; border: 1px solid #334155; border-radius: 6px;
      color: #f1f5f9; padding: 6px 10px; font-size: 13px; width: 100%;
      outline: none;
    }
    .config-input:focus { border-color: #3b82f6; }
    .config-select {
      background: #1e293b; border: 1px solid #334155; border-radius: 6px;
      color: #f1f5f9; padding: 6px 10px; font-size: 13px;
      outline: none; cursor: pointer;
    }
    .config-select:focus { border-color: #3b82f6; }
    .icon-btn {
      width: 32px; height: 32px; border-radius: 6px; border: 1px solid #334155;
      background: #1e293b; color: #f1f5f9; font-size: 18px; cursor: pointer;
      display: flex; align-items: center; justify-content: center;
    }
    .icon-btn:disabled { opacity: 0.4; cursor: not-allowed; }
    .icon-btn:hover:not(:disabled) { background: #334155; }
    .btn-primary {
      background: #2563eb; color: white; border: none; border-radius: 8px;
      padding: 8px 16px; font-size: 14px; cursor: pointer; width: 100%;
      transition: background 0.2s;
    }
    .btn-primary:hover { background: #1d4ed8; }
  `]
})
export class ConfigPanelComponent implements OnInit {
  @Input() config!: NetworkConfig;
  @Output() configChange = new EventEmitter<NetworkConfig>();
  @Output() reinitialize = new EventEmitter<void>();

  activations: ActivationFunction[] = ['sigmoid', 'relu', 'tanh', 'softmax', 'linear'];
  inputNeurons = 2;
  hiddenLayers: LayerConfig[] = [{ neurons: 2, activation: 'sigmoid' }];
  outputNeurons = 1;
  outputActivation: ActivationFunction = 'sigmoid';
  lossFunction: LossFunction = 'bce';
  learningRate = 0.1;
  weightInit: WeightInit = 'xavier';

  get totalParams(): number {
    const layerSizes = [this.inputNeurons, ...this.hiddenLayers.map(l => l.neurons), this.outputNeurons];
    let total = 0;
    for (let i = 1; i < layerSizes.length; i++) {
      total += layerSizes[i] * layerSizes[i - 1] + layerSizes[i];
    }
    return total;
  }

  ngOnInit(): void {
    if (this.config) {
      this.syncFromConfig();
    }
  }

  syncFromConfig(): void {
    const layers = this.config.layers;
    this.inputNeurons = layers[0].neurons;
    this.hiddenLayers = layers.slice(1, -1).map(l => ({ ...l }));
    this.outputNeurons = layers[layers.length - 1].neurons;
    this.outputActivation = layers[layers.length - 1].activation;
    this.lossFunction = this.config.lossFunction;
    this.learningRate = this.config.learningRate;
    this.weightInit = this.config.weightInit;
  }

  addHiddenLayer(): void {
    this.hiddenLayers = [...this.hiddenLayers, { neurons: 4, activation: 'relu' }];
    this.onConfigChange();
  }

  removeHiddenLayer(): void {
    if (this.hiddenLayers.length > 1) {
      this.hiddenLayers = this.hiddenLayers.slice(0, -1);
      this.onConfigChange();
    }
  }

  onConfigChange(): void {
    const newConfig: NetworkConfig = {
      layers: [
        { neurons: this.inputNeurons, activation: 'linear' },
        ...this.hiddenLayers.map(l => ({ neurons: Math.max(1, l.neurons), activation: l.activation })),
        { neurons: this.outputNeurons, activation: this.outputActivation },
      ],
      lossFunction: this.lossFunction,
      learningRate: this.learningRate,
      weightInit: this.weightInit,
    };
    this.configChange.emit(newConfig);
  }

  onReinit(): void {
    this.reinitialize.emit();
  }
}
