import {
  Component, Input, OnChanges, SimpleChanges, ChangeDetectionStrategy
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { NeuralNetwork } from '../engine/neuralNetwork';
import { Dataset } from '../engine/types';

@Component({
  selector: 'app-graphs-panel',
  standalone: true,
  imports: [CommonModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="graphs-panel flex flex-col gap-4">

      <!-- Loss Curve -->
      <div class="graph-card">
        <div class="graph-title">Training Loss</div>
        <svg width="100%" [attr.height]="chartH + 30" viewBox="0 0 400 130" preserveAspectRatio="none">
          <defs>
            <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#3b82f6" stop-opacity="0.3"/>
              <stop offset="100%" stop-color="#3b82f6" stop-opacity="0"/>
            </linearGradient>
          </defs>
          @for (y of gridLines; track $index) {
            <line [attr.x1]="40" [attr.y1]="y" [attr.x2]="390" [attr.y2]="y"
              stroke="#1e293b" stroke-width="1"/>
          }
          @for (g of gridLabelData; track $index) {
            <text [attr.x]="35" [attr.y]="g.y + 4"
              text-anchor="end" font-size="8" fill="#475569" font-family="monospace">{{ g.label }}</text>
          }
          @if (lossAreaPath) {
            <path [attr.d]="lossAreaPath" fill="url(#lossGrad)"/>
          }
          @if (lossPolyline) {
            <polyline
              [attr.points]="lossPolyline"
              fill="none" stroke="#3b82f6" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
          }
          @for (pt of sparsePoints; track $index) {
            <circle [attr.cx]="pt.cx" [attr.cy]="pt.cy" r="2" fill="#3b82f6"/>
          }
          @if (lossHistory.length > 0) {
            <text x="390" [attr.y]="lastLossY - 4"
              text-anchor="end" font-size="9" fill="#93c5fd" font-family="monospace">
              {{ lastLoss.toFixed(4) }}
            </text>
          }
          @if (lossHistory.length === 0) {
            <text x="200" y="65"
              text-anchor="middle" font-size="11" fill="#334155">
              Start training to see loss curve
            </text>
          }
          <text x="215" y="125" text-anchor="middle" font-size="9" fill="#475569">Epoch</text>
        </svg>
      </div>

      <!-- Decision Boundary (2D only) -->
      @if (dataset && dataset.featureNames.length === 2) {
        <div class="graph-card">
          <div class="graph-title">Decision Boundary</div>
          <svg width="220" height="180" style="display:block; margin:0 auto; background:#0f172a; border-radius:8px; overflow:visible">
            @for (cell of boundaryCells; track $index) {
              <rect
                [attr.x]="cell.x" [attr.y]="cell.y" width="7" height="7"
                [attr.fill]="cell.color" fill-opacity="0.35"/>
            }
            @for (pt of dataPoints; track $index) {
              <circle
                [attr.cx]="pt.x" [attr.cy]="pt.y" r="4"
                [attr.fill]="pt.color" [attr.stroke]="pt.stroke" stroke-width="1"
                fill-opacity="0.9"/>
            }
            <text x="110" y="176" text-anchor="middle" font-size="9" fill="#475569">{{ dataset.featureNames[0] }}</text>
          </svg>
        </div>
      }

      <!-- Weight Distribution -->
      @if (network) {
        <div class="graph-card">
          <div class="graph-title">Weight Distribution</div>
          <svg width="100%" height="80" viewBox="0 0 400 80" preserveAspectRatio="none">
            @for (bar of weightHistBars; track $index) {
              <rect
                [attr.x]="bar.x" [attr.y]="bar.y" [attr.width]="bar.w" [attr.height]="bar.h"
                fill="#6366f1" fill-opacity="0.7"/>
            }
            <text x="200" y="75" text-anchor="middle" font-size="9" fill="#475569">Weight Value</text>
            @if (weightHistBars.length === 0) {
              <text x="200" y="40" text-anchor="middle" font-size="11" fill="#334155">No weights yet</text>
            }
          </svg>
        </div>
      }

      <!-- Stats row -->
      @if (network) {
        <div class="flex gap-3 flex-wrap">
          <div class="stat-card">
            <div class="stat-label">Epochs</div>
            <div class="stat-value text-blue-300">{{ epochCount }}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Current Loss</div>
            <div class="stat-value text-yellow-300">{{ lastLoss.toFixed(6) }}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Best Loss</div>
            <div class="stat-value text-green-300">{{ bestLoss === Infinity ? '—' : bestLoss.toFixed(6) }}</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Parameters</div>
            <div class="stat-value text-purple-300">{{ totalParams }}</div>
          </div>
        </div>
      }
    </div>
  `,
  styles: [`
    .graph-card {
      background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 14px;
    }
    .graph-title { font-size: 12px; font-weight: 600; color: #64748b; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em; }
    .stat-card {
      flex: 1; min-width: 80px; background: #1e293b; border: 1px solid #334155;
      border-radius: 10px; padding: 10px 14px;
    }
    .stat-label { font-size: 10px; color: #64748b; margin-bottom: 3px; }
    .stat-value { font-size: 18px; font-weight: 700; font-family: monospace; }
  `]
})
export class GraphsPanelComponent implements OnChanges {
  @Input() network: NeuralNetwork | null = null;
  @Input() dataset: Dataset | null = null;
  @Input() lossHistory: number[] = [];
  @Input() epochCount = 0;

  readonly Infinity = Infinity;

  chartH = 100;
  gridLines: number[] = [10, 30, 50, 70, 90, 110];
  gridLabelData: { y: number; label: string }[] = [];
  lossPolyline = '';
  lossAreaPath = '';
  sparsePoints: { cx: number; cy: number }[] = [];
  lastLoss = 0;
  lastLossY = 50;
  bestLoss = Infinity;
  boundaryCells: { x: number; y: number; color: string }[] = [];
  dataPoints: { x: number; y: number; color: string; stroke: string }[] = [];
  weightHistBars: { x: number; y: number; w: number; h: number }[] = [];
  totalParams = 0;
  classColors = ['#3b82f6', '#f97316', '#22c55e', '#a855f7', '#ef4444'];

  ngOnChanges(changes: SimpleChanges): void {
    this.updateLossCurve();
    if (changes['network'] || changes['dataset']) {
      this.updateDecisionBoundary();
      this.updateWeightHistogram();
      this.computeStats();
    }
    if (changes['epochCount']) {
      this.updateDecisionBoundary();
      this.updateWeightHistogram();
    }
  }

  updateLossCurve(): void {
    const data = this.lossHistory;
    if (data.length === 0) {
      this.lossPolyline = '';
      this.lossAreaPath = '';
      this.lastLoss = 0;
      return;
    }

    this.lastLoss = data[data.length - 1];
    this.bestLoss = Math.min(...data);

    const maxLoss = Math.max(...data, 0.001);
    const W = 350; const H = 110;
    const padL = 40;

    const pts = data.map((l, i) => {
      const x = padL + (i / Math.max(data.length - 1, 1)) * W;
      const y = H - ((l - 0) / (maxLoss - 0 || 1)) * (H - 10) - 5;
      return { x, y };
    });

    this.lossPolyline = pts.map(p => `${p.x},${p.y}`).join(' ');

    const first = pts[0];
    const last = pts[pts.length - 1];
    this.lossAreaPath = `M ${first.x},${H} L ${pts.map(p => `${p.x},${p.y}`).join(' L ')} L ${last.x},${H} Z`;
    this.lastLossY = pts[pts.length - 1].y;

    const step = Math.max(1, Math.floor(data.length / 20));
    this.sparsePoints = pts
      .filter((_, i) => i % step === 0 || i === pts.length - 1)
      .map(p => ({ cx: p.x, cy: p.y }));

    this.gridLines = [10, 30, 50, 70, 90, 110];
    this.gridLabelData = this.gridLines.map(y => ({
      y,
      label: (maxLoss - ((y - 5) / (H - 10)) * maxLoss).toFixed(2),
    }));
  }

  updateDecisionBoundary(): void {
    if (!this.network || !this.dataset || this.dataset.featureNames.length < 2) {
      this.boundaryCells = [];
      this.dataPoints = [];
      return;
    }

    const inputs = this.dataset.inputs;
    const xs = inputs.map(r => r[0]);
    const ys = inputs.map(r => r[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const rangeX = (maxX - minX) || 1;
    const rangeY = (maxY - minY) || 1;
    const pad = 0.1;
    const rX = rangeX * (1 + 2 * pad), rY = rangeY * (1 + 2 * pad);
    const oX = minX - rangeX * pad, oY = minY - rangeY * pad;

    const W = 220; const H = 170;
    const res = 7;
    const cols = Math.ceil(W / res);
    const rows = Math.ceil(H / res);

    const cells: { x: number; y: number; color: string }[] = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const fx = oX + (c / cols) * rX;
        const fy = oY + ((rows - r) / rows) * rY;
        const result = this.network.forward([fx, fy]);
        const out = result.output;
        const classIdx = out.indexOf(Math.max(...out));
        cells.push({ x: c * res, y: r * res, color: this.classColors[classIdx % this.classColors.length] });
      }
    }
    this.boundaryCells = cells;

    this.dataPoints = inputs.map((row, i) => {
      const px = ((row[0] - oX) / rX) * W;
      const py = H - ((row[1] - oY) / rY) * H;
      const tgt = this.dataset!.targets[i];
      const classIdx = tgt.indexOf(Math.max(...tgt));
      const result = this.network!.forward(row);
      const predClass = result.output.indexOf(Math.max(...result.output));
      const correct = predClass === classIdx;
      return {
        x: px, y: py,
        color: this.classColors[classIdx % this.classColors.length],
        stroke: correct ? '#ffffff' : '#ff0000',
      };
    });
  }

  updateWeightHistogram(): void {
    if (!this.network) return;
    const allWeights: number[] = [];
    for (const W of this.network.weights) {
      for (const row of W) {
        for (const w of row) allWeights.push(w);
      }
    }
    if (allWeights.length === 0) return;

    const minW = Math.min(...allWeights);
    const maxW = Math.max(...allWeights);
    const bins = 30;
    const counts = new Array(bins).fill(0);
    for (const w of allWeights) {
      const idx = Math.min(bins - 1, Math.floor(((w - minW) / (maxW - minW || 1)) * bins));
      counts[idx]++;
    }
    const maxCount = Math.max(...counts, 1);
    const barW = 400 / bins;
    const H = 65;
    this.weightHistBars = counts.map((c, i) => {
      const h = (c / maxCount) * H;
      return { x: i * barW, y: H - h, w: barW - 1, h };
    });
  }

  computeStats(): void {
    if (!this.network) return;
    let total = 0;
    const layers = this.network.config.layers;
    for (let i = 1; i < layers.length; i++) {
      total += layers[i].neurons * layers[i - 1].neurons + layers[i].neurons;
    }
    this.totalParams = total;
  }
}
