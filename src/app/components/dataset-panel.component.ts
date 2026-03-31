import { Component, Input, Output, EventEmitter, OnInit, OnChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Dataset, DatasetName } from '../engine/types';
import { getDataset } from '../engine/datasets';

@Component({
  selector: 'app-dataset-panel',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="bg-slate-800 rounded-xl p-4 flex flex-col gap-4">
      <div class="text-sm font-bold text-purple-400">Dataset Selection</div>

      <!-- Dataset buttons -->
      <div class="flex flex-wrap gap-2">
        @for (ds of datasetOptions; track ds.value) {
          <button class="dataset-btn"
            [class.active]="selectedDataset === ds.value"
            (click)="selectDataset(ds.value)">
            {{ ds.label }}
          </button>
        }
      </div>

      <!-- Dataset info -->
      @if (dataset) {
        <div class="bg-slate-900 rounded-lg p-3 text-xs text-slate-400">
          <div class="text-slate-200 font-semibold mb-1">{{ dataset.name }}</div>
          <div>{{ dataset.description }}</div>
          <div class="mt-2 flex gap-4">
            <span>Samples: <b class="text-white">{{ dataset.inputs.length }}</b></span>
            <span>Features: <b class="text-white">{{ dataset.featureNames.length }}</b></span>
            <span>Outputs: <b class="text-white">{{ dataset.targets[0]?.length }}</b></span>
          </div>
          <div class="mt-1">Features: <span class="text-blue-300">{{ dataset.featureNames.join(', ') }}</span></div>
          <div>Classes: <span class="text-green-300">{{ dataset.classNames.join(', ') }}</span></div>
        </div>
      }

      <!-- 2D Scatter plot (if 2 features) -->
      @if (dataset && dataset.featureNames.length === 2) {
        <div class="scatter-container">
          <div class="text-xs text-slate-400 mb-2">Data Preview (2D)</div>
          <svg width="220" height="160" style="background:#0f172a; border-radius:8px; display:block; margin:0 auto;">
            <text x="110" y="150" text-anchor="middle" font-size="9" fill="#475569">{{ dataset.featureNames[0] }}</text>
            <text x="8" y="80" text-anchor="middle" font-size="9" fill="#475569" transform="rotate(-90,8,80)">{{ dataset.featureNames[1] }}</text>
            @for (pt of scatterPoints; track $index) {
              <circle
                [attr.cx]="pt.x" [attr.cy]="pt.y" r="4"
                [attr.fill]="pt.color" fill-opacity="0.8"
              />
            }
          </svg>
          <div class="flex gap-3 justify-center mt-1 flex-wrap">
            @for (cls of classColors; track $index) {
              <span class="flex items-center gap-1 text-xs text-slate-400">
                <span class="inline-block w-3 h-3 rounded-full" [style.background]="cls"></span>
                {{ dataset.classNames[$index] || 'class ' + $index }}
              </span>
            }
          </div>
        </div>
      }

      <!-- Data table preview -->
      <div class="overflow-x-auto">
        <div class="text-xs text-slate-400 mb-1">Sample Data (first 5 rows)</div>
        <table class="data-table text-xs w-full">
          <thead>
            <tr>
              @for (f of dataset?.featureNames; track f) {
                <th>{{ f }}</th>
              }
              <th>target</th>
            </tr>
          </thead>
          <tbody>
            @for (row of previewRows; track $index) {
              <tr>
                @for (v of row.inputs; track $index) {
                  <td>{{ v.toFixed(3) }}</td>
                }
                <td>{{ row.target }}</td>
              </tr>
            }
          </tbody>
        </table>
      </div>
    </div>
  `,
  styles: [`
    .dataset-btn {
      padding: 4px 12px; border-radius: 20px; border: 1px solid #334155;
      background: #1e293b; color: #94a3b8; font-size: 12px; cursor: pointer;
      transition: all 0.15s;
    }
    .dataset-btn:hover { border-color: #8b5cf6; color: #a78bfa; }
    .dataset-btn.active { background: #4c1d95; border-color: #8b5cf6; color: #c4b5fd; }
    .data-table { border-collapse: collapse; }
    .data-table th { background: #1e293b; color: #64748b; padding: 4px 8px; text-align: left; }
    .data-table td { color: #cbd5e1; padding: 3px 8px; border-top: 1px solid #1e293b; }
    .data-table tr:hover td { background: #1e293b; }
  `]
})
export class DatasetPanelComponent implements OnInit, OnChanges {
  @Input() selectedDataset: DatasetName = 'xor';
  @Output() datasetChange = new EventEmitter<{ name: DatasetName; dataset: Dataset }>();

  dataset: Dataset | null = null;
  scatterPoints: { x: number; y: number; color: string }[] = [];
  classColors = ['#3b82f6', '#f97316', '#22c55e', '#a855f7', '#ef4444'];

  datasetOptions: { label: string; value: DatasetName }[] = [
    { label: 'XOR', value: 'xor' },
    { label: 'Iris', value: 'iris' },
    { label: 'MNIST-like', value: 'mnist_like' },
    { label: 'Sine Wave', value: 'sine' },
    { label: 'Circles', value: 'circles' },
    { label: 'Moons', value: 'moons' },
  ];

  get previewRows(): { inputs: number[]; target: string }[] {
    if (!this.dataset) return [];
    return this.dataset.inputs.slice(0, 5).map((inp, i) => ({
      inputs: inp,
      target: this.dataset!.targets[i].join(', '),
    }));
  }

  ngOnInit(): void {
    this.loadDataset(this.selectedDataset);
  }

  ngOnChanges(): void {
    this.loadDataset(this.selectedDataset);
  }

  selectDataset(name: DatasetName): void {
    this.selectedDataset = name;
    this.loadDataset(name);
    this.datasetChange.emit({ name, dataset: this.dataset! });
  }

  loadDataset(name: DatasetName): void {
    this.dataset = getDataset(name);
    this.buildScatterPoints();
  }

  buildScatterPoints(): void {
    if (!this.dataset || this.dataset.featureNames.length < 2) {
      this.scatterPoints = [];
      return;
    }

    const inputs = this.dataset.inputs;
    const xs = inputs.map(r => r[0]);
    const ys = inputs.map(r => r[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;

    this.scatterPoints = inputs.map((row, i) => {
      const px = 20 + ((row[0] - minX) / rangeX) * 180;
      const py = 140 - ((row[1] - minY) / rangeY) * 120;
      const tgt = this.dataset!.targets[i];
      const classIdx = tgt.indexOf(Math.max(...tgt));
      return { x: px, y: py, color: this.classColors[classIdx % this.classColors.length] };
    });
  }
}
