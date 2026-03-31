import {
  Component, Input, Output, EventEmitter, OnDestroy, OnChanges, SimpleChanges,
  ChangeDetectionStrategy, ChangeDetectorRef
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { NeuralNetwork } from '../engine/neuralNetwork';
import { Dataset, ForwardPassResult } from '../engine/types';

interface WeightChange {
  layer: number;
  neuron: number;
  oldW: number;
  dW: number;
  newW: number;
  delta: number;
}

interface SignalLayer {
  label: string;
  values: number[];
  color: string;
  maxAbs: number;
}

@Component({
  selector: 'app-playground-mode',
  standalone: true,
  imports: [CommonModule, FormsModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="playground flex flex-col gap-4 h-full">

      <!-- Controls -->
      <div class="flex items-center gap-3 bg-slate-800 rounded-xl px-4 py-3 flex-wrap">
        <div class="text-sm font-semibold text-green-300">Playground — Live Training</div>
        <div class="flex-1"></div>

        <div class="flex items-center gap-2 text-xs text-slate-400">
          Speed:
          <input type="range" class="w-20 accent-green-500" [(ngModel)]="speedMs"
            [min]="10" [max]="1000" [step]="10" (change)="onSpeedChange()">
          <span class="text-green-300 font-mono w-12">{{ speedLabel }}</span>
        </div>

        <div class="flex items-center gap-2 text-xs text-slate-400">
          Epochs/tick:
          <select class="mini-select" [(ngModel)]="epochsPerTick">
            <option [value]="1">1</option>
            <option [value]="5">5</option>
            <option [value]="10">10</option>
            <option [value]="50">50</option>
            <option [value]="100">100</option>
          </select>
        </div>

        <button class="play-btn" [class.pause]="isRunning" (click)="toggleTraining()">
          {{ isRunning ? '⏸ Pause' : '▶ Train' }}
        </button>

        <button class="ctrl-btn" (click)="resetTraining()">&#x27F3; Reset</button>
        <button class="ctrl-btn" (click)="stepOnce()" [disabled]="isRunning">Step</button>
      </div>

      <!-- Live stats bar -->
      <div class="grid grid-cols-4 gap-3">
        <div class="stat-chip">
          <div class="stat-chip-label">Epoch</div>
          <div class="stat-chip-value text-blue-300">{{ epochCount }}</div>
        </div>
        <div class="stat-chip" [class.converge-flash]="converging">
          <div class="stat-chip-label">Loss</div>
          <div class="stat-chip-value stat-value text-yellow-300">
            {{ currentLoss.toFixed(5) }}
            <span class="trend-indicator" [class.trend-down]="lossTrend < 0" [class.trend-up]="lossTrend > 0">
              {{ lossTrend < 0 ? '▼' : lossTrend > 0 ? '▲' : '' }}
            </span>
          </div>
        </div>
        <div class="stat-chip">
          <div class="stat-chip-label">Accuracy</div>
          <div class="stat-chip-value text-green-300">{{ (accuracy * 100).toFixed(1) }}%</div>
        </div>
        <div class="stat-chip">
          <div class="stat-chip-label">Avg |grad|</div>
          <div class="stat-chip-value text-purple-300">{{ avgGrad.toFixed(5) }}</div>
        </div>
      </div>

      <!-- Mini loss chart -->
      <div class="bg-slate-800 border border-slate-700 rounded-xl p-3">
        <div class="text-xs text-slate-500 mb-2 uppercase tracking-wide">Loss Curve (live)</div>
        <svg width="100%" height="70" viewBox="0 0 400 70" preserveAspectRatio="none">
          <defs>
            <linearGradient id="liveGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#22c55e" stop-opacity="0.4"/>
              <stop offset="100%" stop-color="#22c55e" stop-opacity="0"/>
            </linearGradient>
            <linearGradient id="lossLineGrad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stop-color="#ef4444"/>
              <stop offset="100%" stop-color="#22c55e"/>
            </linearGradient>
          </defs>
          @if (lossAreaPath) {
            <path [attr.d]="lossAreaPath" fill="url(#liveGrad)"/>
          }
          @if (lossPolyline) {
            <polyline [attr.points]="lossPolyline" fill="none" stroke="url(#lossLineGrad)" stroke-width="1.5"/>
          }
          <!-- Current epoch vertical line -->
          @if (lossPolyline && currentEpochX > 0) {
            <line [attr.x1]="currentEpochX" y1="0" [attr.x2]="currentEpochX" y2="70"
              stroke="#94a3b8" stroke-width="1" stroke-dasharray="3 2" opacity="0.5"/>
          }
          @if (!lossPolyline) {
            <text x="200" y="40" text-anchor="middle" font-size="11" fill="#334155">Start training...</text>
          }
        </svg>
      </div>

      <!-- Signal Snapshot panel -->
      <div class="bg-slate-800 border border-slate-700 rounded-xl p-4">
        <div class="text-xs text-slate-500 mb-3 uppercase tracking-wide">Signal Snapshot — Last Forward Pass</div>
        @if (signalLayers.length > 0) {
          <div class="flex flex-col gap-2">
            @for (sl of signalLayers; track sl.label) {
              <div class="signal-layer-row">
                <div class="signal-layer-label" [style.color]="sl.color">{{ sl.label }}</div>
                <div class="signal-bars">
                  @for (v of sl.values; track $index) {
                    <div class="signal-bar-wrap">
                      <div class="signal-bar"
                        [style.width]="barWidth(v, sl.maxAbs)"
                        [style.background]="sl.color"
                        [style.opacity]="barOpacity(v, sl.maxAbs)">
                      </div>
                      <span class="signal-val">{{ v.toFixed(2) }}</span>
                    </div>
                  }
                </div>
              </div>
            }
          </div>
        } @else {
          <div class="text-xs text-slate-600 text-center py-2">Run a forward pass to see activations</div>
        }
      </div>

      <!-- Weight changes table -->
      @if (weightChanges.length > 0) {
        <div class="bg-slate-800 border border-slate-700 rounded-xl p-3">
          <div class="text-xs text-slate-500 mb-2 uppercase tracking-wide">Recent Weight Updates</div>
          <div class="overflow-x-auto">
            <table class="text-xs w-full">
              <thead>
                <tr>
                  <th class="th">Layer</th>
                  <th class="th">Neuron</th>
                  <th class="th">Old W</th>
                  <th class="th">dW</th>
                  <th class="th">New W</th>
                  <th class="th">Change</th>
                </tr>
              </thead>
              <tbody>
                @for (wc of weightChanges.slice(0, 10); track $index) {
                  <tr>
                    <td class="td">{{ wc.layer }}</td>
                    <td class="td">{{ wc.neuron }}</td>
                    <td class="td font-mono text-blue-300">{{ wc.oldW.toFixed(4) }}</td>
                    <td class="td font-mono text-orange-300">{{ wc.dW.toFixed(4) }}</td>
                    <td class="td font-mono text-green-300">{{ wc.newW.toFixed(4) }}</td>
                    <td class="td">
                      <span [class.text-red-400]="wc.delta < 0" [class.text-green-400]="wc.delta >= 0">
                        {{ wc.delta > 0 ? '+' : '' }}{{ wc.delta.toFixed(4) }}
                      </span>
                    </td>
                  </tr>
                }
              </tbody>
            </table>
          </div>
        </div>
      }
    </div>
  `,
  styles: [`
    .mini-select {
      background: #1e293b; border: 1px solid #334155; border-radius: 4px;
      color: #f1f5f9; padding: 2px 6px; font-size: 12px; outline: none;
    }
    .ctrl-btn {
      padding: 5px 12px; border-radius: 6px; border: 1px solid #334155;
      background: #1e293b; color: #94a3b8; font-size: 12px; cursor: pointer; transition: all 0.15s;
    }
    .ctrl-btn:hover:not(:disabled) { border-color: #22c55e; color: #86efac; }
    .ctrl-btn:disabled { opacity: 0.4; cursor: not-allowed; }
    .play-btn {
      padding: 6px 16px; border-radius: 8px; border: none;
      background: #15803d; color: white; font-size: 13px; font-weight: 600;
      cursor: pointer; transition: background 0.15s;
    }
    .play-btn:hover { background: #166534; }
    .play-btn.pause { background: #b45309; }
    .play-btn.pause:hover { background: #92400e; }

    /* Stat chips */
    .stat-chip {
      background: #1e293b; border: 1px solid #334155; border-radius: 10px;
      padding: 12px 14px; text-align: center; position: relative;
    }
    .stat-chip-label { font-size: 10px; color: #64748b; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
    .stat-chip-value { font-size: 18px; font-weight: 700; font-family: monospace; display: flex; align-items: center; justify-content: center; gap: 4px; }
    .stat-value { transition: color 0.3s ease; }

    /* Trend indicator */
    .trend-indicator { font-size: 12px; }
    .trend-down { color: #22c55e; }
    .trend-up   { color: #ef4444; }

    /* Convergence flash */
    @keyframes converge-flash-pg {
      0%   { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.6); }
      50%  { box-shadow: 0 0 0 12px rgba(34, 197, 94, 0); }
      100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }
    .converge-flash { animation: converge-flash-pg 0.6s ease-out; }

    /* Signal Snapshot */
    .signal-layer-row {
      display: flex; align-items: flex-start; gap: 8px;
    }
    .signal-layer-label {
      font-size: 10px; font-weight: 700; font-family: ui-monospace, monospace;
      min-width: 36px; padding-top: 2px; flex-shrink: 0;
    }
    .signal-bars { display: flex; flex-direction: column; gap: 3px; flex: 1; }
    .signal-bar-wrap { display: flex; align-items: center; gap: 6px; }
    .signal-bar {
      height: 8px; border-radius: 3px; min-width: 2px;
      transition: width 0.3s ease;
    }
    .signal-val { font-size: 9px; color: #64748b; font-family: ui-monospace, monospace; min-width: 30px; }

    .th { background: #1e293b; color: #64748b; padding: 4px 8px; text-align: left; }
    .td { color: #cbd5e1; padding: 3px 8px; border-top: 1px solid #1e293b; }
  `]
})
export class PlaygroundModeComponent implements OnDestroy, OnChanges {
  @Input() network: NeuralNetwork | null = null;
  @Input() dataset: Dataset | null = null;
  @Output() epochCompleted = new EventEmitter<{ loss: number; epoch: number; forwardResult: ForwardPassResult }>();
  @Output() activeLayerChange = new EventEmitter<number>();

  isRunning = false;
  epochCount = 0;
  currentLoss = 0;
  prevLoss = 0;
  lossTrend = 0;
  converging = false;
  accuracy = 0;
  avgGrad = 0;
  speedMs = 100;
  epochsPerTick = 5;
  lossHistory: number[] = [];
  lossPolyline = '';
  lossAreaPath = '';
  currentEpochX = 0;
  weightChanges: WeightChange[] = [];
  signalLayers: SignalLayer[] = [];

  private intervalId: ReturnType<typeof setInterval> | null = null;
  private convergingTimer: ReturnType<typeof setTimeout> | null = null;

  get speedLabel(): string {
    if (this.speedMs < 100) return `${this.speedMs}ms`;
    return `${(this.speedMs / 1000).toFixed(1)}s`;
  }

  constructor(private cdr: ChangeDetectorRef) {}

  ngOnChanges(_changes: SimpleChanges): void {
    this.resetTraining();
  }

  ngOnDestroy(): void {
    this.stopTraining();
    if (this.convergingTimer) clearTimeout(this.convergingTimer);
  }

  toggleTraining(): void {
    if (this.isRunning) {
      this.stopTraining();
    } else {
      this.startTraining();
    }
  }

  startTraining(): void {
    if (!this.network || !this.dataset) return;
    this.isRunning = true;
    this.intervalId = setInterval(() => this.tick(), this.speedMs);
  }

  stopTraining(): void {
    this.isRunning = false;
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    this.cdr.markForCheck();
  }

  onSpeedChange(): void {
    if (this.isRunning) {
      this.stopTraining();
      this.startTraining();
    }
  }

  stepOnce(): void {
    if (!this.network || !this.dataset) return;
    this.runEpochs(1);
  }

  resetTraining(): void {
    this.stopTraining();
    this.epochCount = 0;
    this.currentLoss = 0;
    this.prevLoss = 0;
    this.lossTrend = 0;
    this.converging = false;
    this.accuracy = 0;
    this.avgGrad = 0;
    this.lossHistory = [];
    this.lossPolyline = '';
    this.lossAreaPath = '';
    this.currentEpochX = 0;
    this.weightChanges = [];
    this.signalLayers = [];
    this.cdr.markForCheck();
  }

  tick(): void {
    this.runEpochs(this.epochsPerTick);
  }

  runEpochs(n: number): void {
    if (!this.network || !this.dataset) return;
    const { inputs, targets } = this.dataset;
    let totalLoss = 0;
    let correct = 0;
    let totalGrad = 0;
    let gradCount = 0;
    const newChanges: WeightChange[] = [];

    for (let e = 0; e < n; e++) {
      for (let s = 0; s < inputs.length; s++) {
        const oldWeights = this.network.cloneWeights();
        const fwd = this.network.forward(inputs[s]);
        const bwd = this.network.backward(targets[s], fwd);
        this.network.updateWeights(bwd, this.network.config.learningRate);
        totalLoss += bwd.loss;

        const pred = fwd.output;
        const predIdx = pred.indexOf(Math.max(...pred));
        const targIdx = targets[s].indexOf(Math.max(...targets[s]));
        if (predIdx === targIdx) correct++;

        for (const lg of bwd.layerGradients) {
          for (const row of lg.dW) {
            for (const g of row) { totalGrad += Math.abs(g); gradCount++; }
          }
        }

        if (e === n - 1 && s === inputs.length - 1) {
          for (let l = 0; l < this.network.weights.length && newChanges.length < 15; l++) {
            const W = this.network.weights[l];
            const oldW = oldWeights.weights[l];
            const dW = bwd.layerGradients[l].dW;
            for (let j = 0; j < Math.min(3, W.length); j++) {
              for (let k = 0; k < Math.min(3, W[j].length); k++) {
                newChanges.push({
                  layer: l + 1,
                  neuron: j + 1,
                  oldW: oldW[j][k],
                  dW: dW[j][k],
                  newW: W[j][k],
                  delta: W[j][k] - oldW[j][k],
                });
              }
            }
          }
        }
      }
      this.epochCount++;
    }

    const totalSamples = n * inputs.length;
    this.prevLoss = this.currentLoss;
    this.currentLoss = totalLoss / totalSamples;
    this.lossTrend = this.prevLoss > 0 ? Math.sign(this.currentLoss - this.prevLoss) : 0;
    this.accuracy = correct / totalSamples;
    this.avgGrad = gradCount > 0 ? totalGrad / gradCount : 0;
    this.weightChanges = newChanges;

    // Convergence detection: 5% improvement
    if (this.prevLoss > 0 && this.currentLoss < this.prevLoss * 0.95) {
      this.converging = true;
      if (this.convergingTimer) clearTimeout(this.convergingTimer);
      this.convergingTimer = setTimeout(() => {
        this.converging = false;
        this.cdr.markForCheck();
      }, 600);
    }

    this.lossHistory = [...this.lossHistory, this.currentLoss].slice(-200);
    this.updateLossChart();

    const lastFwd = this.network.forward(inputs[0]);
    this.updateSignalSnapshot(lastFwd);
    this.epochCompleted.emit({ loss: this.currentLoss, epoch: this.epochCount, forwardResult: lastFwd });
    this.cdr.markForCheck();
  }

  updateLossChart(): void {
    const data = this.lossHistory;
    if (data.length < 2) { this.lossPolyline = ''; this.lossAreaPath = ''; this.currentEpochX = 0; return; }

    const maxL = Math.max(...data, 0.001);
    const W = 400; const H = 65;
    const pts = data.map((l, i) => {
      const x = (i / (data.length - 1)) * W;
      const y = H - (l / maxL) * (H - 5) - 2;
      return { x, y };
    });

    this.lossPolyline = pts.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
    const f = pts[0], la = pts[pts.length - 1];
    this.lossAreaPath = `M ${f.x},${H} L ${pts.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' L ')} L ${la.x},${H} Z`;
    this.currentEpochX = la.x;
  }

  updateSignalSnapshot(fwd: ForwardPassResult): void {
    if (!this.network) return;
    const layerColors = ['#3b82f6', '#8b5cf6', '#8b5cf6', '#22c55e'];
    const L = this.network.config.layers.length;
    this.signalLayers = fwd.aValues.map((vals, i) => {
      const color = i === 0 ? '#3b82f6' : i === L - 1 ? '#22c55e' : '#8b5cf6';
      const maxAbs = Math.max(...vals.map(Math.abs), 0.0001);
      return {
        label: `L${i}:`,
        values: vals.slice(0, 6), // show at most 6 neurons
        color,
        maxAbs,
      };
    });
    // suppress unused variable warning
    void layerColors;
  }

  barWidth(v: number, maxAbs: number): string {
    const pct = Math.min(100, (Math.abs(v) / maxAbs) * 100);
    return `${Math.max(2, pct)}%`;
  }

  barOpacity(v: number, maxAbs: number): number {
    return Math.max(0.3, Math.abs(v) / maxAbs);
  }
}
