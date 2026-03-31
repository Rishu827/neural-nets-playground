import {
  Component, Input, Output, EventEmitter, OnChanges, SimpleChanges,
  ChangeDetectionStrategy, ChangeDetectorRef, OnDestroy, OnInit, HostListener
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { NeuralNetwork } from '../engine/neuralNetwork';
import { ForwardPassResult, BackwardPassResult, LatexEquation, Dataset } from '../engine/types';
import { getLossLatex } from '../engine/losses';
import { getActivationLatex } from '../engine/activations';
import { EquationDisplayComponent } from './equation-display.component';

type StepKind =
  | { type: 'intro';    title: string; equations: LatexEquation[] }
  | { type: 'forward';  layerIdx: number; equations: LatexEquation[] }
  | { type: 'loss';     equations: LatexEquation[] }
  | { type: 'backward'; layerIdx: number; equations: LatexEquation[] }
  | { type: 'update';   equations: LatexEquation[] };

interface PipelineStage {
  label: string;
  type: 'input' | 'forward' | 'loss' | 'backward' | 'update';
  layerIdx?: number;
  stepIdx: number;
}

@Component({
  selector: 'app-math-mode',
  standalone: true,
  imports: [CommonModule, FormsModule, EquationDisplayComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
<div class="mm-root">

  <!-- ── Controls bar ── -->
  <div class="ctrl-bar">
    <span class="ctrl-label">∑ Math Mode</span>

    <div class="flex items-center gap-1 text-xs text-slate-400">
      Sample:
      <select class="mini-sel" [(ngModel)]="sampleIdx" (change)="runPass()">
        @for (inp of (dataset?.inputs ?? []); track $index) {
          <option [value]="$index">#{{ $index }}</option>
        }
      </select>
    </div>

    <!-- auto-play -->
    <button class="ctrl-btn" [class.playing]="playing" (click)="togglePlay()" title="Auto-play steps">
      {{ playing ? '⏸' : '▶' }}
    </button>

    <button class="ctrl-btn" (click)="prevStep()" [disabled]="currentStep === 0">◀ Prev</button>
    <span class="step-counter">{{ currentStep + 1 }} / {{ steps.length }}</span>
    <button class="ctrl-btn" (click)="nextStep()" [disabled]="currentStep >= steps.length - 1">Next ▶</button>
    <button class="ctrl-btn reset-btn" (click)="runPass()">↺ Reset</button>
    <span class="kbd-hint">← → keys</span>

    <!-- speed -->
    @if (playing) {
      <div class="flex items-center gap-1 text-xs text-slate-400">
        Speed:
        <select class="mini-sel" [(ngModel)]="playSpeed">
          <option value="2000">Slow</option>
          <option value="1200">Normal</option>
          <option value="600">Fast</option>
        </select>
      </div>
    }
  </div>

  <!-- ── Phase pipeline timeline ── -->
  @if (pipelineStages.length > 0) {
    <div class="pipeline-row">
      @for (stage of pipelineStages; track stage.stepIdx) {
        <div class="pipeline-item"
          [class.pipeline-active]="stage.stepIdx === currentStep"
          [class.pipeline-done]="stage.stepIdx < currentStep"
          [class.pipeline-input]="stage.type === 'input'"
          [class.pipeline-fwd]="stage.type === 'forward'"
          [class.pipeline-loss]="stage.type === 'loss'"
          [class.pipeline-bwd]="stage.type === 'backward'"
          [class.pipeline-upd]="stage.type === 'update'"
          (click)="goToStep(stage.stepIdx)"
          [title]="stage.label">
          {{ stage.label }}
        </div>
        @if (stage.stepIdx < pipelineStages[pipelineStages.length - 1].stepIdx) {
          <div class="pipeline-arrow">→</div>
        }
      }
    </div>
  }

  <!-- ── Overall progress bar ── -->
  <div class="progress-track">
    <div class="progress-fill"
      [style.width]="progressPct + '%'"
      [class.progress-fwd]="progressPhase === 'forward'"
      [class.progress-loss]="progressPhase === 'loss'"
      [class.progress-bwd]="progressPhase === 'backward'"
      [class.progress-upd]="progressPhase === 'update'">
    </div>
  </div>

  <!-- ── Step progress dots ── -->
  <div class="dot-row">
    @for (s of steps; track $index) {
      <div class="dot"
        [class.dot-fwd]="s.type === 'forward' || s.type === 'intro'"
        [class.dot-loss]="s.type === 'loss'"
        [class.dot-bwd]="s.type === 'backward' || s.type === 'update'"
        [class.dot-active]="$index === currentStep"
        [class.dot-done]="$index < currentStep"
        (click)="goToStep($index)"
        [title]="stepTitle(s)">
      </div>
    }
  </div>

  <!-- ── Main step content ── -->
  @if (currentStepData) {
    <div class="step-body" [class.anim-enter]="animKey">

      <!-- Step header banner -->
      <div class="step-banner"
        [class.banner-fwd]="currentStepData.type === 'forward' || currentStepData.type === 'intro'"
        [class.banner-loss]="currentStepData.type === 'loss'"
        [class.banner-bwd]="currentStepData.type === 'backward' || currentStepData.type === 'update'">
        <div class="banner-title">
          <span class="step-icon">{{ stepIcon(currentStepData) }}</span>
          {{ stepTitle(currentStepData) }}
        </div>
        <div class="banner-sub">{{ stepSubtitle(currentStepData) }}</div>
      </div>

      <!-- Diagram cross-reference pill -->
      <div class="diag-ref">
        <div class="diag-ref-icon">&#x25B2; Diagram</div>
        <div class="diag-ref-text">{{ diagramRef(currentStepData) }}</div>
        <!-- Highlight indicator badges -->
        <div class="diag-badges">
          @for (b of diagramBadges(currentStepData); track $index) {
            <span class="diag-badge" [style.background]="b.bg" [style.color]="b.fg">{{ b.label }}</span>
          }
        </div>
      </div>

      <!-- Activation / loss formula card -->
      @if (currentStepData.type === 'forward' || currentStepData.type === 'intro') {
        <div class="eq-card card-fwd-hdr">
          <div class="eq-card-title text-blue-400">Activation: {{ layerActivation(currentStepData) }}</div>
          <app-equation-display [latex]="activationLatex(currentStepData)" [displayMode]="true"/>
        </div>
      }
      @if (currentStepData.type === 'loss') {
        <div class="eq-card card-loss-hdr">
          <div class="eq-card-title text-yellow-400">Loss Function</div>
          <app-equation-display [latex]="lossLatex" [displayMode]="true"/>
        </div>
      }

      <!-- Equation cards -->
      @for (eq of currentStepData.equations; track $index) {
        <div class="eq-card eq-reveal-card"
          [class.card-fwd]="currentStepData.type === 'forward' || currentStepData.type === 'intro'"
          [class.card-loss]="currentStepData.type === 'loss'"
          [class.card-bwd]="currentStepData.type === 'backward' || currentStepData.type === 'update'"
          [style.animation-delay]="($index * 90) + 'ms'">

          <div class="eq-card-title"
            [class.title-fwd]="currentStepData.type === 'forward' || currentStepData.type === 'intro'"
            [class.title-loss]="currentStepData.type === 'loss'"
            [class.title-bwd]="currentStepData.type === 'backward' || currentStepData.type === 'update'">
            <app-equation-display [latex]="eq.label" [displayMode]="false" containerClass="eq-title-katex"/>
          </div>

          <!-- Symbolic row -->
          <div class="eq-row">
            <div class="eq-row-label">Formula</div>
            <app-equation-display [latex]="eq.symbolic" [displayMode]="true"/>
          </div>

          <!-- Numeric row -->
          @if (eq.numeric) {
            <div class="eq-row mt-2">
              <div class="eq-row-label">Values</div>
              <app-equation-display [latex]="eq.numeric" [displayMode]="false"/>
            </div>
          }
        </div>
      }

      <!-- Loss summary strip -->
      @if (forwardResult && (currentStepData.type === 'loss' || currentStepData.type === 'backward' || currentStepData.type === 'update')) {
        <div class="loss-strip" [class.value-flash]="dataFlash">
          <div class="loss-kv">
            <div class="loss-key">Loss L</div>
            <div class="loss-val text-yellow-300">{{ backwardResult?.loss?.toFixed(6) }}</div>
          </div>
          <div class="loss-kv">
            <div class="loss-key">ŷ (predicted)</div>
            <div class="loss-val text-blue-300 font-mono text-xs">{{ forwardResult.output.map(v => v.toFixed(4)).join(', ') }}</div>
          </div>
          <div class="loss-kv">
            <div class="loss-key">y (target)</div>
            <div class="loss-val text-green-300 font-mono text-xs">{{ currentTarget.join(', ') }}</div>
          </div>
        </div>
      }

      <!-- Weight update summary -->
      @if (currentStepData.type === 'update') {
        <div class="eq-card card-bwd">
          <div class="eq-card-title text-orange-400">Update Rule (all layers)</div>
          <app-equation-display [latex]="updateRuleLatex" [displayMode]="true"/>
          <div class="mt-3 text-xs text-slate-400">
            Learning rate &alpha;&nbsp;=&nbsp;
            <span class="text-orange-300 font-mono font-bold">{{ network?.config?.learningRate }}</span>
          </div>
        </div>
      }

    </div>
  }

  @if (!network || !dataset) {
    <div class="empty-state">Configure a network and select a dataset to begin.</div>
  }
</div>
  `,
  styles: [`
    :host { display: block; height: 100%; }

    .mm-root {
      display: flex; flex-direction: column; height: 100%; gap: 10px;
    }

    /* ── Controls ── */
    .ctrl-bar {
      display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
      background: #1e293b; border-radius: 12px; padding: 8px 14px;
      border: 1px solid #334155;
    }
    .ctrl-label { font-size: 12px; font-weight: 700; color: #93c5fd; margin-right: 4px; }
    .mini-sel {
      background: #0f172a; border: 1px solid #334155; border-radius: 4px;
      color: #f1f5f9; padding: 2px 6px; font-size: 11px; outline: none;
    }
    .ctrl-btn {
      padding: 4px 11px; border-radius: 6px; border: 1px solid #334155;
      background: #0f172a; color: #94a3b8; font-size: 12px; cursor: pointer; transition: all 0.15s;
    }
    .ctrl-btn:hover:not(:disabled) { border-color: #3b82f6; color: #93c5fd; }
    .ctrl-btn:disabled { opacity: 0.35; cursor: not-allowed; }
    .ctrl-btn.playing { background: #1c2940; border-color: #3b82f6; color: #60a5fa; }
    .reset-btn { border-color: #475569; }
    .step-counter { font-size: 11px; color: #64748b; white-space: nowrap; }
    .kbd-hint { font-size: 10px; color: #334155; white-space: nowrap; margin-left: 4px;
      background: #0f172a; border: 1px solid #1e293b; border-radius: 4px; padding: 2px 6px; }

    /* ── Progress bar ── */
    .progress-track {
      height: 4px; background: #1e293b; border-radius: 2px; overflow: hidden; flex-shrink: 0;
    }
    .progress-fill {
      height: 100%; border-radius: 2px; transition: width 0.3s ease, background 0.4s ease;
    }
    .progress-fwd  { background: linear-gradient(90deg, #1d4ed8, #3b82f6); }
    .progress-loss { background: #eab308; }
    .progress-bwd  { background: linear-gradient(90deg, #ea580c, #f97316); }
    .progress-upd  { background: #8b5cf6; }

    /* ── Pipeline timeline ── */
    .pipeline-row {
      display: flex; align-items: center; gap: 3px; flex-wrap: nowrap;
      overflow-x: auto; padding: 6px 8px;
      background: #0f172a; border-radius: 10px; border: 1px solid #1e293b;
      scrollbar-width: none;
    }
    .pipeline-row::-webkit-scrollbar { display: none; }
    .pipeline-arrow {
      color: #334155; font-size: 11px; flex-shrink: 0; padding: 0 1px;
    }
    .pipeline-item {
      padding: 3px 8px; border-radius: 20px; font-size: 10px; font-weight: 700;
      font-family: ui-monospace, monospace; white-space: nowrap; cursor: pointer;
      border: 1px solid transparent; transition: all 0.15s; flex-shrink: 0;
      background: #1e293b; color: #475569; border-color: #334155;
    }
    .pipeline-item:hover { opacity: 0.9; transform: scale(1.05); }
    .pipeline-item.pipeline-done { opacity: 0.45; }
    .pipeline-item.pipeline-active {
      transform: scale(1.1);
      box-shadow: 0 0 8px 2px currentColor;
    }
    /* type colors */
    .pipeline-input  { background: #1e3a5f; color: #93c5fd; border-color: #2563eb; }
    .pipeline-fwd    { background: #1d3a5e; color: #60a5fa; border-color: #2563eb; }
    .pipeline-loss   { background: #422006; color: #fde047; border-color: #92400e; }
    .pipeline-bwd    { background: #431407; color: #fb923c; border-color: #c2410c; }
    .pipeline-upd    { background: #1e1b4b; color: #a78bfa; border-color: #6d28d9; }

    /* ── Dots ── */
    .dot-row { display: flex; gap: 4px; padding: 0 4px; flex-wrap: wrap; align-items: center; }
    .dot {
      width: 9px; height: 9px; border-radius: 50%; background: #1e293b;
      border: 1px solid #334155; cursor: pointer; transition: all 0.15s; flex-shrink: 0;
    }
    .dot:hover { transform: scale(1.3); }
    .dot-fwd  { background: #1d3a5e; border-color: #2563eb; }
    .dot-loss { background: #422006; border-color: #92400e; }
    .dot-bwd  { background: #431407; border-color: #c2410c; }
    .dot-active { transform: scale(1.5) !important; box-shadow: 0 0 6px 1px currentColor; }
    .dot-fwd.dot-active  { background: #3b82f6; box-shadow: 0 0 6px 2px #3b82f6; }
    .dot-loss.dot-active { background: #eab308; box-shadow: 0 0 6px 2px #eab308; }
    .dot-bwd.dot-active  { background: #f97316; box-shadow: 0 0 6px 2px #f97316; }
    .dot-done { opacity: 0.5; }

    /* ── Step body animation ── */
    .step-body { display: flex; flex-direction: column; gap: 10px; flex: 1; overflow-y: auto; }

    @keyframes eq-reveal {
      from { opacity: 0; transform: translateY(16px) scale(0.98); }
      to   { opacity: 1; transform: translateY(0) scale(1); }
    }
    .anim-enter .eq-reveal-card {
      animation: eq-reveal 0.3s ease-out both;
    }

    /* ── Step banner ── */
    .step-banner {
      border-radius: 10px; padding: 10px 16px;
      border: 1px solid transparent; display: flex; justify-content: space-between; align-items: center;
    }
    .banner-fwd  { background: #0c1e3d; border-color: #1d4ed8; }
    .banner-loss { background: #1c1000; border-color: #78350f; }
    .banner-bwd  { background: #1c0900; border-color: #9a3412; }
    .banner-title { font-size: 14px; font-weight: 700; display: flex; align-items: center; gap: 6px; }
    .banner-fwd  .banner-title { color: #93c5fd; }
    .banner-loss .banner-title { color: #fde047; }
    .banner-bwd  .banner-title { color: #fb923c; }
    .banner-sub { font-size: 11px; color: #64748b; text-align: right; max-width: 55%; }
    .step-icon { font-size: 16px; }

    /* ── Diagram cross-reference ── */
    .diag-ref {
      display: flex; align-items: flex-start; gap: 8px; flex-wrap: wrap;
      background: #0f1f38; border: 1px solid #1e3a5f; border-radius: 8px;
      padding: 8px 12px; font-size: 11.5px;
    }
    .diag-ref-icon { color: #3b82f6; font-weight: 700; white-space: nowrap; font-size: 11px; padding-top: 1px; }
    .diag-ref-text { color: #93c5fd; flex: 1; line-height: 1.5; }
    .diag-badges   { display: flex; gap: 4px; flex-wrap: wrap; align-items: center; }
    .diag-badge {
      font-size: 10px; font-weight: 700; font-family: ui-monospace,monospace;
      padding: 1px 7px; border-radius: 20px; white-space: nowrap;
    }

    /* ── Equation cards ── */
    .eq-card {
      background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px 18px;
    }
    .card-fwd-hdr { border-color: #1e40af; background: #0c1b36; }
    .card-loss-hdr { border-color: #78350f; background: #1a0f00; }
    .card-fwd  { border-left: 3px solid #2563eb; }
    .card-loss { border-left: 3px solid #d97706; }
    .card-bwd  { border-left: 3px solid #ea580c; }
    .eq-card-title { margin-bottom: 12px; border-bottom: 1px solid #1e293b; padding-bottom: 10px; }
    /* KaTeX inside the title — override its default sizing to be readable */
    .eq-card-title ::ng-deep .katex { font-size: 1.05em !important; }
    .eq-card-title ::ng-deep .katex .text { font-weight: 600; letter-spacing: 0.02em; }
    .title-fwd  ::ng-deep .katex, .title-fwd  ::ng-deep .katex .text { color: #93c5fd; }
    .title-loss ::ng-deep .katex, .title-loss ::ng-deep .katex .text { color: #fde047; }
    .title-bwd  ::ng-deep .katex, .title-bwd  ::ng-deep .katex .text { color: #fb923c; }
    .eq-row { display: flex; flex-direction: column; gap: 4px; }
    .eq-row-label { font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; }

    /* ── Loss strip ── */
    .loss-strip {
      display: flex; gap: 20px; flex-wrap: wrap;
      background: #1c1a00; border: 1px solid #78350f; border-radius: 10px; padding: 12px 16px;
    }
    .loss-kv { display: flex; flex-direction: column; gap: 2px; }
    .loss-key { font-size: 10px; color: #78716c; text-transform: uppercase; letter-spacing: 0.05em; }
    .loss-val { font-size: 15px; font-weight: 700; transition: color 0.3s ease; }

    .empty-state {
      flex: 1; display: flex; align-items: center; justify-content: center;
      color: #475569; font-size: 13px;
    }
  `]
})
export class MathModeComponent implements OnInit, OnChanges, OnDestroy {
  @Input() network: NeuralNetwork | null = null;
  @Input() dataset: Dataset | null = null;
  @Output() activeLayerChange = new EventEmitter<number>();
  @Output() stepCompleted = new EventEmitter<{ forwardResult: ForwardPassResult | null; activeLayer: number; mode: 'forward' | 'backward' | 'idle' }>();

  sampleIdx = 0;
  currentStep = 0;
  steps: StepKind[] = [];
  forwardResult: ForwardPassResult | null = null;
  backwardResult: BackwardPassResult | null = null;
  currentTarget: number[] = [];

  // Data flash state
  dataFlash = false;
  private flashTimer: ReturnType<typeof setTimeout> | null = null;

  // Auto-play
  playing = false;
  playSpeed = 1200;
  private playTimer: ReturnType<typeof setTimeout> | null = null;

  // Trigger re-animation
  animKey = false;
  private animTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(private cdr: ChangeDetectorRef) {}

  @HostListener('document:keydown', ['$event'])
  onKeyDown(e: KeyboardEvent): void {
    // Only handle when no input/select is focused
    const tag = (document.activeElement?.tagName ?? '').toLowerCase();
    if (tag === 'input' || tag === 'select' || tag === 'textarea') return;
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { e.preventDefault(); this.nextStep(); }
    if (e.key === 'ArrowLeft'  || e.key === 'ArrowUp')   { e.preventDefault(); this.prevStep(); }
    if (e.key === ' ')                                    { e.preventDefault(); this.togglePlay(); }
  }

  ngOnInit(): void {}

  get currentStepData(): StepKind | null { return this.steps[this.currentStep] ?? null; }
  get lossLatex(): string { return this.network ? getLossLatex(this.network.config.lossFunction) : ''; }

  get progressPct(): number {
    if (!this.steps.length) return 0;
    return (this.currentStep / (this.steps.length - 1)) * 100;
  }

  get progressPhase(): 'forward' | 'loss' | 'backward' | 'update' {
    const s = this.currentStepData;
    if (!s) return 'forward';
    if (s.type === 'loss') return 'loss';
    if (s.type === 'backward') return 'backward';
    if (s.type === 'update') return 'update';
    return 'forward';
  }
  readonly updateRuleLatex = 'W^{[l]} \\leftarrow W^{[l]} - \\alpha \\cdot \\frac{\\partial L}{\\partial W^{[l]}}';
  readonly inputLayerLatex = 'a^{[0]} = x';

  /** Computed pipeline stages from steps array */
  get pipelineStages(): PipelineStage[] {
    return this.steps.map((s, i): PipelineStage => {
      switch (s.type) {
        case 'intro':
          return { label: '→INPUT', type: 'input', stepIdx: i };
        case 'forward':
          return { label: `FWD[${s.layerIdx + 1}]`, type: 'forward', layerIdx: s.layerIdx, stepIdx: i };
        case 'loss':
          return { label: '⚖LOSS', type: 'loss', stepIdx: i };
        case 'backward':
          return { label: `←BWD[${s.layerIdx + 1}]`, type: 'backward', layerIdx: s.layerIdx, stepIdx: i };
        case 'update':
          return { label: '✦UPDATE', type: 'update', stepIdx: i };
      }
    });
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (this.network && this.dataset) {
      this.runPass();
      // Trigger value flash when forwardResult changes
      if (changes['network'] || changes['dataset']) {
        this.triggerDataFlash();
      }
    }
  }

  ngOnDestroy(): void {
    this.stopPlay();
    if (this.flashTimer) clearTimeout(this.flashTimer);
  }

  private triggerDataFlash(): void {
    if (this.flashTimer) clearTimeout(this.flashTimer);
    this.dataFlash = true;
    this.flashTimer = setTimeout(() => {
      this.dataFlash = false;
      this.cdr.markForCheck();
    }, 350);
  }

  // ── Pass execution ────────────────────────────────────────────────────
  runPass(): void {
    if (!this.network || !this.dataset) return;
    const idx = Math.min(this.sampleIdx, this.dataset.inputs.length - 1);
    this.currentTarget = this.dataset.targets[idx];
    this.forwardResult  = this.network.forward(this.dataset.inputs[idx]);
    this.backwardResult = this.network.backward(this.currentTarget, this.forwardResult);
    this.buildSteps();
    this.currentStep = 0;
    this.triggerAnim();
    this.triggerDataFlash();
    this.emitState();
  }

  buildSteps(): void {
    if (!this.network || !this.forwardResult || !this.backwardResult) return;
    const steps: StepKind[] = [];
    const L = this.network.weights.length;

    // Intro
    steps.push({
      type: 'intro', title: 'Input Layer',
      equations: [{
        label: 'Input vector x',
        symbolic: `a^{[0]} = x = \\begin{bmatrix}${this.forwardResult.input.map((_,i) => `x_{${i+1}}`).join('\\\\')}\\end{bmatrix}`,
        numeric:  `a^{[0]} = \\begin{bmatrix}${this.forwardResult.input.map(v => v.toFixed(4)).join('\\\\')}\\end{bmatrix}`,
      }],
    });

    // Forward layers
    for (let l = 0; l < L; l++) {
      steps.push({ type: 'forward', layerIdx: l, equations: this.network.getForwardEquations(l, this.forwardResult) });
    }

    // Loss
    steps.push({
      type: 'loss',
      equations: [{
        label: 'Compute loss',
        symbolic: getLossLatex(this.network.config.lossFunction),
        numeric: `L = ${this.backwardResult.loss.toFixed(6)}`,
      }],
    });

    // Backward layers (reverse)
    for (let l = L - 1; l >= 0; l--) {
      steps.push({ type: 'backward', layerIdx: l, equations: this.network.getBackwardEquations(l, this.backwardResult, this.forwardResult) });
    }

    // Weight update
    steps.push({
      type: 'update',
      equations: [{
        label: 'Gradient descent update',
        symbolic: `W^{[l]} \\leftarrow W^{[l]} - \\alpha \\cdot \\frac{\\partial L}{\\partial W^{[l]}}`,
        numeric:  `\\alpha = ${this.network.config.learningRate}`,
      }],
    });

    this.steps = steps;
  }

  // ── Navigation ────────────────────────────────────────────────────────
  prevStep(): void { if (this.currentStep > 0) { this.currentStep--; this.onStepChange(); } }
  nextStep(): void { if (this.currentStep < this.steps.length - 1) { this.currentStep++; this.onStepChange(); } }
  goToStep(i: number): void { this.currentStep = i; this.onStepChange(); }

  private onStepChange(): void { this.triggerAnim(); this.emitState(); }

  // ── Auto-play ────────────────────────────────────────────────────────
  togglePlay(): void {
    this.playing ? this.stopPlay() : this.startPlay();
  }

  private startPlay(): void {
    this.playing = true;
    this.scheduleNext();
  }

  private scheduleNext(): void {
    if (!this.playing) return;
    this.playTimer = setTimeout(() => {
      if (this.currentStep < this.steps.length - 1) {
        this.currentStep++;
        this.triggerAnim();
        this.emitState();
        this.cdr.markForCheck();
        this.scheduleNext();
      } else {
        this.stopPlay();
        this.cdr.markForCheck();
      }
    }, +this.playSpeed);
  }

  private stopPlay(): void {
    this.playing = false;
    if (this.playTimer) { clearTimeout(this.playTimer); this.playTimer = null; }
  }

  // ── Animation trigger ─────────────────────────────────────────────────
  private triggerAnim(): void {
    this.animKey = false;
    if (this.animTimer) clearTimeout(this.animTimer);
    this.animTimer = setTimeout(() => { this.animKey = true; this.cdr.markForCheck(); }, 10);
  }

  // ── Emit to parent (diagram sync) ────────────────────────────────────
  emitState(): void {
    const step = this.currentStepData;
    let layer = -1;
    let diagramMode: 'forward' | 'backward' | 'idle' = 'idle';
    if (step?.type === 'intro')    { layer = 0;               diagramMode = 'forward'; }
    if (step?.type === 'forward')  { layer = step.layerIdx + 1; diagramMode = 'forward'; }
    // backward: layerIdx is the weight-layer index l; edges l flow when activeLayer === l
    if (step?.type === 'backward') { layer = step.layerIdx;     diagramMode = 'backward'; }
    this.activeLayerChange.emit(layer);
    this.stepCompleted.emit({ forwardResult: this.forwardResult, activeLayer: layer, mode: diagramMode });
  }

  // ── Display helpers ───────────────────────────────────────────────────
  stepTitle(step: StepKind): string {
    switch (step.type) {
      case 'intro':    return 'Input Layer';
      case 'forward':  return `Forward Pass — Layer ${step.layerIdx + 1}`;
      case 'loss':     return 'Loss Computation';
      case 'backward': return `Backward Pass — Layer ${step.layerIdx + 1}`;
      case 'update':   return 'Weight Update';
    }
  }

  stepSubtitle(step: StepKind): string {
    switch (step.type) {
      case 'intro':    return 'Feed input into network';
      case 'forward':  return `z = W·a + b  →  a = ${this.layerActivation(step)}(z)`;
      case 'loss':     return `Measure prediction error`;
      case 'backward': return `Chain rule: ∂L/∂W[${step.layerIdx + 1}]`;
      case 'update':   return `W ← W − α·∇W, b ← b − α·∇b`;
    }
  }

  stepIcon(step: StepKind): string {
    switch (step.type) {
      case 'intro':    return '→';
      case 'forward':  return '⟶';
      case 'loss':     return '⚖';
      case 'backward': return '⟵';
      case 'update':   return '✦';
    }
  }

  layerActivation(step: StepKind): string {
    if (step.type === 'forward')
      return this.network?.config?.layers[step.layerIdx + 1]?.activation ?? '';
    if (step.type === 'intro') return 'linear';
    return '';
  }

  activationLatex(step: StepKind): string {
    if (step.type === 'intro')   return this.inputLayerLatex;
    if (step.type === 'forward') {
      const act = this.network?.config?.layers[step.layerIdx + 1]?.activation ?? 'sigmoid';
      return getActivationLatex(act);
    }
    return '';
  }

  diagramRef(step: StepKind): string {
    switch (step.type) {
      case 'intro':
        return 'Blue a[0] column — input neurons receive raw feature values.';
      case 'forward':
        return `Purple-glowing a[${step.layerIdx + 1}] column is active. Edges feeding it carry W[${step.layerIdx + 1}] weights. First z=W·a+b is computed, then the activation squashes it to produce a.`;
      case 'loss':
        return 'Green output column ŷ vs. target y — the loss quantifies their difference.';
      case 'backward':
        return `Orange-glowing a[${step.layerIdx + 1}] — gradients flow right→left. ∂L/∂W[${step.layerIdx + 1}] is the slope of loss w.r.t. every incoming edge weight.`;
      case 'update':
        return 'All edge weights update simultaneously. Thickness & colour in the diagram will reflect the new values after this step.';
    }
  }

  diagramBadges(step: StepKind): { label: string; bg: string; fg: string }[] {
    switch (step.type) {
      case 'intro':
        return [{ label: 'a[0]', bg: '#1e3a5f', fg: '#93c5fd' }];
      case 'forward': {
        const l = step.layerIdx + 1;
        return [
          { label: `a[${l-1}]`, bg: '#1e3a5f', fg: '#93c5fd' },
          { label: `W[${l}]`,   bg: '#1e1b4b', fg: '#a78bfa' },
          { label: `a[${l}]`,   bg: '#2e1065', fg: '#c4b5fd' },
        ];
      }
      case 'loss':
        return [
          { label: 'ŷ (output)', bg: '#14532d', fg: '#86efac' },
          { label: 'y (target)', bg: '#1a2e05', fg: '#a3e635' },
          { label: 'L',          bg: '#422006', fg: '#fde047' },
        ];
      case 'backward': {
        const l = step.layerIdx + 1;
        return [
          { label: `∂L/∂W[${l}]`, bg: '#431407', fg: '#fb923c' },
          { label: `∂L/∂b[${l}]`, bg: '#431407', fg: '#fdba74' },
        ];
      }
      case 'update':
        return [
          { label: 'W ← W − α∇W', bg: '#1c0b00', fg: '#f97316' },
          { label: `α=${this.network?.config?.learningRate}`, bg: '#1c0b00', fg: '#fdba74' },
        ];
    }
  }
}
