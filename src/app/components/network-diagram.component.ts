import {
  Component,
  Input,
  Output,
  EventEmitter,
  OnChanges,
  SimpleChanges,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { NetworkConfig, ForwardPassResult, BackwardPassResult } from '../engine/types';

interface NodeData {
  x: number; y: number; layer: number; neuron: number;
  value: number | null; bias: number | null; zValue: number | null; label: string;
  id: string;
}
interface EdgeData {
  x1: number; y1: number; x2: number; y2: number;
  weight: number; layerIdx: number; fromNeuron: number; toNeuron: number;
  flowing: boolean; id: string;
  mx: number; my: number; // midpoint for label
}
interface TooltipState {
  visible: boolean; x: number; y: number; lines: string[];
}
interface LayerMeta {
  x: number; kind: 'input' | 'hidden' | 'output'; idx: number;
  aLabel: string; activation: string;
}
interface WeightMeta {
  mx: number; my: number; label: string;
}

// ─── coordinate constants ────────────────────────────────────────────────────
const TOP     = 34;   // height of top badge strip
const BOTTOM  = 38;   // height of bottom label strip
const PAD_X   = 72;   // horizontal padding
const SVG_W   = 720;
const SVG_H   = 310;  // total: TOP + node area + BOTTOM

/** Interpolate two #rrggbb hex colors by factor t (0..1) */
function lerpHex(a: string, b: string, t: number): string {
  const clamp = (v: number) => Math.min(255, Math.max(0, Math.round(v)));
  const pa = parseInt(a.slice(1), 16);
  const pb = parseInt(b.slice(1), 16);
  const ar = (pa >> 16) & 0xff, ag = (pa >> 8) & 0xff, ab = pa & 0xff;
  const br = (pb >> 16) & 0xff, bg = (pb >> 8) & 0xff, bb = pb & 0xff;
  const rr = clamp(ar + (br - ar) * t);
  const rg = clamp(ag + (bg - ag) * t);
  const rb = clamp(ab + (bb - ab) * t);
  return `#${rr.toString(16).padStart(2, '0')}${rg.toString(16).padStart(2, '0')}${rb.toString(16).padStart(2, '0')}`;
}

@Component({
  selector: 'app-network-diagram',
  standalone: true,
  imports: [CommonModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
<div class="nn-wrap" (mouseleave)="hideTooltip()">
  <svg [attr.viewBox]="'0 0 ' + W + ' ' + H" class="nn-svg" style="overflow:visible">

    <!-- ── Top badge strip ── -->
    @for (lm of layerMetas; track lm.idx) {
      <rect [attr.x]="lm.x - badgeW/2" [attr.y]="2" [attr.width]="badgeW" height="28" rx="7"
        [attr.fill]="layerColor(lm.kind)" fill-opacity="0.18"
        [attr.stroke]="layerColor(lm.kind)" stroke-width="1.2" stroke-opacity="0.6"/>
      <text [attr.x]="lm.x" y="20" text-anchor="middle" font-size="13" font-weight="700"
        font-family="ui-monospace,monospace" font-style="italic"
        [attr.fill]="layerColor(lm.kind)">{{ lm.aLabel }}</text>
    }

    <!-- ── Edges ── -->
    @for (e of edges; track e.id) {
      <line
        [attr.x1]="e.x1" [attr.y1]="e.y1" [attr.x2]="e.x2" [attr.y2]="e.y2"
        [attr.stroke]="edgeStroke(e)"
        [attr.stroke-width]="edgeW(e)"
        [attr.stroke-opacity]="edgeOpacity(e)"
        [attr.stroke-dasharray]="e.flowing ? '7 4' : 'none'"
        [class.edge-flow]="e.flowing && mode !== 'backward'"
        [class.edge-flow-bwd]="e.flowing && mode === 'backward'"
        [class.edge-hov]="hoveredEdge === e.id"
        stroke-linecap="round"
        (mouseenter)="onEdgeHover($event, e)"
        (mouseleave)="hideTooltip()"
      />
      <!-- Weight value mid-label (only when not too many edges) -->
      @if (edges.length <= 20 && !e.flowing) {
        <text [attr.x]="e.mx" [attr.y]="e.my - 5"
          text-anchor="middle" font-size="9" font-family="ui-monospace,monospace"
          [attr.fill]="edgeStroke(e)" opacity="0.65">{{ fmtW(e.weight) }}</text>
      }
    }

    <!-- ── Traveling signal dots (animateMotion) ── -->
    @for (e of edges; track e.id) {
      @if (e.flowing && mode === 'forward') {
        <!-- Dot 1 forward -->
        <g [attr.transform]="'translate(' + e.x1 + ',' + e.y1 + ')'">
          <circle r="4" fill="#93c5fd" opacity="0.85">
            <animateMotion
              [attr.path]="'M 0 0 L ' + (e.x2-e.x1) + ' ' + (e.y2-e.y1)"
              dur="0.75s" repeatCount="indefinite" calcMode="linear"
              [attr.begin]="dotBegin(e, 0)"/>
          </circle>
          <circle r="1.5" fill="white" opacity="0.9">
            <animateMotion
              [attr.path]="'M 0 0 L ' + (e.x2-e.x1) + ' ' + (e.y2-e.y1)"
              dur="0.75s" repeatCount="indefinite" calcMode="linear"
              [attr.begin]="dotBegin(e, 0)"/>
          </circle>
        </g>
        <!-- Dot 2 forward (staggered) -->
        <g [attr.transform]="'translate(' + e.x1 + ',' + e.y1 + ')'">
          <circle r="4" fill="#93c5fd" opacity="0.85">
            <animateMotion
              [attr.path]="'M 0 0 L ' + (e.x2-e.x1) + ' ' + (e.y2-e.y1)"
              dur="0.75s" repeatCount="indefinite" calcMode="linear"
              [attr.begin]="dotBegin(e, 1)"/>
          </circle>
          <circle r="1.5" fill="white" opacity="0.9">
            <animateMotion
              [attr.path]="'M 0 0 L ' + (e.x2-e.x1) + ' ' + (e.y2-e.y1)"
              dur="0.75s" repeatCount="indefinite" calcMode="linear"
              [attr.begin]="dotBegin(e, 1)"/>
          </circle>
        </g>
      }
      @if (e.flowing && mode === 'backward') {
        <!-- Dot 1 backward -->
        <g [attr.transform]="'translate(' + e.x2 + ',' + e.y2 + ')'">
          <circle r="4" fill="#fdba74" opacity="0.85">
            <animateMotion
              [attr.path]="'M 0 0 L ' + (e.x1-e.x2) + ' ' + (e.y1-e.y2)"
              dur="0.75s" repeatCount="indefinite" calcMode="linear"
              [attr.begin]="dotBegin(e, 0)"/>
          </circle>
          <circle r="1.5" fill="white" opacity="0.9">
            <animateMotion
              [attr.path]="'M 0 0 L ' + (e.x1-e.x2) + ' ' + (e.y1-e.y2)"
              dur="0.75s" repeatCount="indefinite" calcMode="linear"
              [attr.begin]="dotBegin(e, 0)"/>
          </circle>
        </g>
        <!-- Dot 2 backward (staggered) -->
        <g [attr.transform]="'translate(' + e.x2 + ',' + e.y2 + ')'">
          <circle r="4" fill="#fdba74" opacity="0.85">
            <animateMotion
              [attr.path]="'M 0 0 L ' + (e.x1-e.x2) + ' ' + (e.y1-e.y2)"
              dur="0.75s" repeatCount="indefinite" calcMode="linear"
              [attr.begin]="dotBegin(e, 1)"/>
          </circle>
          <circle r="1.5" fill="white" opacity="0.9">
            <animateMotion
              [attr.path]="'M 0 0 L ' + (e.x1-e.x2) + ' ' + (e.y1-e.y2)"
              dur="0.75s" repeatCount="indefinite" calcMode="linear"
              [attr.begin]="dotBegin(e, 1)"/>
          </circle>
        </g>
      }
    }

    <!-- ── W[l] labels between layers (vertical center of node area) ── -->
    @for (wm of weightMetas; track $index) {
      <rect [attr.x]="wm.mx - 22" [attr.y]="wm.my - 11" width="44" height="20" rx="5"
        fill="#1e1b4b" fill-opacity="0.7"
        stroke="#6d28d9" stroke-width="1" stroke-opacity="0.5"/>
      <text [attr.x]="wm.mx" [attr.y]="wm.my + 4"
        text-anchor="middle" font-size="12" font-weight="600"
        font-family="ui-monospace,monospace" font-style="italic" fill="#a78bfa">
        {{ wm.label }}
      </text>
    }

    <!-- ── Nodes ── -->
    @for (nd of nodes; track nd.id) {
      <!-- Active layer halo -->
      @if (isActive(nd.layer)) {
        <circle [attr.cx]="nd.x" [attr.cy]="nd.y" [attr.r]="R + 12"
          [attr.fill]="activeColor()" fill-opacity="0.12"
          [class.pulse-ring]="isActive(nd.layer)"/>
        <circle [attr.cx]="nd.x" [attr.cy]="nd.y" [attr.r]="R + 6"
          fill="none" [attr.stroke]="activeColor()" stroke-width="1.5"
          stroke-opacity="0.5" [class.pulse-ring]="isActive(nd.layer)"/>
      }

      <!-- Backward pass delta rings -->
      @if (mode === 'backward' && isActive(nd.layer) && getDeltaRingRadius(nd) > 0) {
        <circle [attr.cx]="nd.x" [attr.cy]="nd.y" [attr.r]="getDeltaRingRadius(nd)"
          fill="none" stroke="#f97316" stroke-width="1.5"
          stroke-dasharray="5 3" opacity="0.6"/>
      }

      <!-- Hovered halo -->
      @if (hoveredNode === nd.id) {
        <circle [attr.cx]="nd.x" [attr.cy]="nd.y" [attr.r]="R + 8"
          fill="none" stroke="#facc15" stroke-width="2" stroke-opacity="0.7"/>
      }

      <!-- Main node -->
      <circle [attr.cx]="nd.x" [attr.cy]="nd.y" [attr.r]="R"
        [attr.fill]="nodeHeatFill(nd)"
        [attr.stroke]="nodeStroke(nd)"
        stroke-width="2.5"
        style="cursor:pointer"
        (mouseenter)="onNodeHover($event, nd)"
        (mouseleave)="hideTooltip()"
        (click)="onNodeClick(nd)"
      />

      <!-- Value inside node (top half) or neuron index -->
      @if (nd.value !== null) {
        <text [attr.x]="nd.x" [attr.y]="nd.y - 2"
          text-anchor="middle" dominant-baseline="middle"
          [attr.font-size]="vFontSize"
          [attr.fill]="valueFill(nd)"
          font-family="ui-monospace,monospace"
          font-weight="600" style="pointer-events:none">{{ fmtV(nd.value) }}</text>
        <!-- Activation symbol below value (only for non-input layers with enough space) -->
        @if (nd.layer > 0 && R >= 16) {
          <text [attr.x]="nd.x" [attr.y]="nd.y + R - 7"
            text-anchor="middle" dominant-baseline="auto"
            font-size="9" [attr.fill]="actSymbolColor(nd.layer)"
            font-family="ui-monospace,monospace" style="pointer-events:none">
            {{ actSymbol(nd.layer) }}
          </text>
        }
      } @else {
        <text [attr.x]="nd.x" [attr.y]="nd.y + 1"
          text-anchor="middle" dominant-baseline="middle"
          [attr.font-size]="vFontSize" fill="#64748b" font-family="ui-monospace,monospace"
          style="pointer-events:none">{{ nd.label }}</text>
      }

      <!-- Bias indicator dot (small circle off the bottom-right of node) -->
      @if (nd.layer > 0 && nd.bias !== null && R >= 16) {
        <circle [attr.cx]="nd.x + R - 2" [attr.cy]="nd.y - R + 2" r="5"
          [attr.fill]="biasColor(nd.bias)"
          stroke="#0f172a" stroke-width="1" opacity="0.85"
          style="pointer-events:none"/>
        <text [attr.x]="nd.x + R - 2" [attr.y]="nd.y - R + 2"
          text-anchor="middle" dominant-baseline="middle"
          font-size="6" fill="#f1f5f9" font-weight="700" style="pointer-events:none">b</text>
      }

      <!-- Input side label x₀ᵢ -->
      @if (nd.layer === 0) {
        <text [attr.x]="nd.x - R - 8" [attr.y]="nd.y + 1"
          text-anchor="end" dominant-baseline="middle"
          font-size="11" fill="#93c5fd" font-family="ui-monospace,monospace" font-style="italic">
          x{{ nd.neuron }}
        </text>
      }
      <!-- Output side label ŷᵢ -->
      @if (nd.layer === lastL) {
        <text [attr.x]="nd.x + R + 8" [attr.y]="nd.y + 1"
          text-anchor="start" dominant-baseline="middle"
          font-size="11" fill="#86efac" font-family="ui-monospace,monospace" font-style="italic">
          ŷ{{ nd.neuron }}
        </text>
      }
    }

    <!-- ── Active layer scan-line flash ── -->
    @for (lm of layerMetas; track lm.idx) {
      @if (isActive(lm.idx)) {
        <rect [attr.x]="lm.x - R - 18" [attr.y]="TOP + R + 8"
          [attr.width]="(R + 18) * 2"
          [attr.height]="H - BOTTOM - TOP - R - 8 - R - 4"
          rx="6"
          [attr.fill]="mode === 'backward' ? '#7c2d12' : '#1e1b4b'"
          fill-opacity="0.18"
          class="scan-col"
          style="pointer-events:none"/>
      }
    }

    <!-- ── Bottom layer labels ── -->
    @for (lm of layerMetas; track lm.idx) {
      <text [attr.x]="lm.x" [attr.y]="H - 6"
        text-anchor="middle" font-size="11" fill="#64748b" font-family="ui-sans-serif,sans-serif">
        {{ lm.kind === 'input' ? 'Input' : lm.kind === 'output' ? 'Output' : 'Hidden ' + lm.idx }}
      </text>
      <text [attr.x]="lm.x" [attr.y]="H - 20"
        text-anchor="middle" font-size="10" fill="#475569" font-family="ui-monospace,monospace">
        {{ lm.activation }}
      </text>
    }

    <!-- ── Legend ── -->
    <g [attr.transform]="'translate(' + (W - 145) + ', ' + (H - 34) + ')'">
      <line x1="0" y1="5" x2="16" y2="5" stroke="#3b82f6" stroke-width="2"/>
      <text x="20" y="9" font-size="10" fill="#475569">positive w</text>
      <line x1="0" y1="18" x2="16" y2="18" stroke="#f97316" stroke-width="2"/>
      <text x="20" y="22" font-size="10" fill="#475569">negative w</text>
    </g>

    <!-- ── SVG Tooltip ── -->
    @if (tip.visible) {
      <g [attr.transform]="'translate(' + tip.x + ',' + tip.y + ')'">
        <rect [attr.width]="tipW" [attr.height]="tip.lines.length * 16 + 10"
          rx="6" fill="#0f172a" stroke="#334155" stroke-width="1.2"/>
        @for (line of tip.lines; track $index) {
          <text [attr.x]="6" [attr.y]="($index + 1) * 16"
            font-size="10.5" fill="#cbd5e1" font-family="ui-monospace,monospace">{{ line }}</text>
        }
      </g>
    }
  </svg>
</div>
  `,
  styles: [`
    :host { display: block; width: 100%; height: 100%; }
    .nn-wrap { width: 100%; height: 100%; padding: 6px 8px; box-sizing: border-box; }
    .nn-svg { width: 100%; height: 100%; cursor: default; }

    @keyframes pulse-ring {
      0%   { transform: scale(1);   opacity: 0.6; }
      50%  { transform: scale(1.08); opacity: 0.3; }
      100% { transform: scale(1);   opacity: 0.6; }
    }
    .pulse-ring { animation: pulse-ring 1.4s ease-in-out infinite; transform-box: fill-box; transform-origin: center; }

    @keyframes dash-flow {
      to { stroke-dashoffset: -22; }
    }
    .edge-flow     { animation: dash-flow     0.55s linear infinite; }

    @keyframes dash-flow-bwd {
      to { stroke-dashoffset: 22; }
    }
    .edge-flow-bwd { animation: dash-flow-bwd 0.55s linear infinite; }
    .edge-hov  { filter: brightness(1.6); }

    @keyframes scan-col-in {
      from { opacity: 0; }
      to   { opacity: 1; }
    }
    .scan-col { animation: scan-col-in 0.25s ease-out; }
  `]
})
export class NetworkDiagramComponent implements OnChanges {
  @Input() config!: NetworkConfig;
  @Input() weights: number[][][] = [];
  @Input() biases: number[][] = [];
  @Input() forwardResult: ForwardPassResult | null = null;
  @Input() backwardResult: BackwardPassResult | null = null;
  @Input() activeLayer = -1;
  @Input() mode: 'forward' | 'backward' | 'idle' = 'idle';

  @Output() nodeClicked = new EventEmitter<{ layer: number; neuron: number }>();

  // SVG constants exposed to template
  readonly W      = SVG_W;
  readonly H      = SVG_H;
  readonly TOP    = TOP;
  readonly BOTTOM = BOTTOM;

  nodes: NodeData[] = [];
  edges: EdgeData[] = [];
  layerMetas: LayerMeta[] = [];
  weightMetas: WeightMeta[] = [];

  R = 22;           // node radius — recomputed based on network size
  vFontSize = 10;   // value font size inside nodes
  badgeW = 52;

  get lastL(): number { return (this.config?.layers?.length ?? 1) - 1; }

  // Interaction state
  hoveredNode = '';
  hoveredEdge = '';
  tip: TooltipState = { visible: false, x: 0, y: 0, lines: [] };
  tipW = 160;

  constructor(private cdr: ChangeDetectorRef) {}

  ngOnChanges(_: SimpleChanges): void { this.rebuild(); }

  rebuild(): void {
    if (!this.config?.layers?.length) return;
    const layers = this.config.layers;
    const L = layers.length;

    // Dynamic radius
    const maxN = Math.max(...layers.map(l => l.neurons));
    this.R    = maxN <= 3 ? 24 : maxN <= 5 ? 20 : maxN <= 8 ? 16 : maxN <= 12 ? 13 : 10;
    this.vFontSize = this.R >= 20 ? 10 : this.R >= 14 ? 9 : 8;
    this.badgeW = Math.max(50, this.R * 2 + 12);

    const usableW = SVG_W - 2 * PAD_X;
    const xStep = L > 1 ? usableW / (L - 1) : 0;
    const nodeTop    = TOP + this.R + 8;
    const nodeBottom = SVG_H - BOTTOM - this.R - 4;
    const nodeAreaH  = Math.max(0, nodeBottom - nodeTop);

    this.nodes      = [];
    this.edges      = [];
    this.layerMetas = [];
    this.weightMetas = [];

    const layerXs: number[] = [];

    for (let l = 0; l < L; l++) {
      const n = layers[l].neurons;
      const xPos = PAD_X + l * xStep;
      layerXs.push(xPos);

      const yStep = n > 1 ? nodeAreaH / (n - 1) : 0;
      const yStart = n > 1 ? nodeTop : nodeTop + nodeAreaH / 2;

      for (let i = 0; i < n; i++) {
        const yPos = n > 1 ? yStart + i * yStep : yStart;
        const aVal = this.forwardResult?.aValues[l]?.[i] ?? null;
        const zVal = l > 0 ? (this.forwardResult?.zValues[l - 1]?.[i] ?? null) : null;
        const bias = l > 0 ? (this.biases[l - 1]?.[i] ?? null) : null;
        this.nodes.push({
          x: xPos, y: yPos, layer: l, neuron: i,
          value: aVal, bias, zValue: zVal,
          label: n <= 9 ? String(i + 1) : '',
          id: `n-${l}-${i}`,
        });
      }

      // Layer meta
      const kind: 'input' | 'hidden' | 'output' =
        l === 0 ? 'input' : l === L - 1 ? 'output' : 'hidden';
      this.layerMetas.push({
        x: xPos, kind, idx: l,
        aLabel: `a[${l}]`,
        activation: layers[l].activation,
      });
    }

    // Edges
    for (let l = 0; l < this.weights.length; l++) {
      const W = this.weights[l];
      const fromN = layers[l].neurons;
      const toN   = layers[l + 1].neurons;
      const flowing = this.mode !== 'idle' && (
        (this.mode === 'forward'  && this.activeLayer === l + 1) ||
        (this.mode === 'backward' && this.activeLayer === l)
      );
      for (let j = 0; j < toN; j++) {
        for (let k = 0; k < fromN; k++) {
          const fn = this.nodes.find(nd => nd.layer === l     && nd.neuron === k);
          const tn = this.nodes.find(nd => nd.layer === l + 1 && nd.neuron === j);
          if (!fn || !tn) continue;
          const w = W?.[j]?.[k] ?? 0;
          this.edges.push({
            x1: fn.x, y1: fn.y, x2: tn.x, y2: tn.y,
            weight: w, layerIdx: l, fromNeuron: k, toNeuron: j,
            flowing, id: `e-${l}-${k}-${j}`,
            mx: (fn.x + tn.x) / 2,
            my: (fn.y + tn.y) / 2,
          });
        }
      }
    }

    // Weight matrix labels — vertically centered between adjacent layers
    for (let l = 1; l < L; l++) {
      const mx = (layerXs[l - 1] + layerXs[l]) / 2;
      const my = TOP + nodeAreaH / 2;
      this.weightMetas.push({ mx, my, label: `W[${l}]` });
    }
  }

  // ── Colors ──────────────────────────────────────────────────────────────
  layerColor(kind: 'input' | 'hidden' | 'output'): string {
    return kind === 'input' ? '#3b82f6' : kind === 'output' ? '#22c55e' : '#8b5cf6';
  }

  /** Node fill using activation heatmap */
  nodeHeatFill(nd: NodeData): string {
    const kind = nd.layer === 0 ? 'input' : nd.layer === this.lastL ? 'output' : 'hidden';
    const v = nd.value;

    if (this.isActive(nd.layer)) {
      // Active layer: still use heatmap but tinted toward active color
      if (v === null) return '#2e1065';
    }

    if (v === null) {
      if (kind === 'input')  return '#1e3a5f';
      if (kind === 'output') return '#14532d';
      return '#1e293b';
    }

    const t = Math.min(1, Math.abs(v));

    if (kind === 'input') {
      return lerpHex('#0d1f38', '#3b82f6', t);
    }
    if (kind === 'output') {
      return lerpHex('#0a1f15', '#22c55e', t);
    }
    // hidden: positive → purple, negative → orange
    if (v >= 0) {
      return lerpHex('#0f172a', '#8b5cf6', t);
    } else {
      return lerpHex('#0f172a', '#f97316', t);
    }
  }

  nodeStroke(nd: NodeData): string {
    if (this.hoveredNode === nd.id) return '#facc15';
    if (this.isActive(nd.layer))   return '#a78bfa';
    if (nd.layer === 0)            return '#3b82f6';
    if (nd.layer === this.lastL)   return '#22c55e';
    return '#475569';
  }

  /** Text color reflecting activation strength */
  valueFill(nd: NodeData): string {
    if (nd.value === null) return '#64748b';
    const t = Math.min(1, Math.abs(nd.value));
    // bright white at high activation, dim slate at near-zero
    return lerpHex('#475569', '#f1f5f9', t);
  }

  edgeStroke(e: EdgeData): string {
    if (this.hoveredEdge === e.id) return '#facc15';
    if (e.weight >  0.05) return '#3b82f6';
    if (e.weight < -0.05) return '#f97316';
    return '#475569';
  }

  edgeW(e: EdgeData): number {
    if (this.hoveredEdge === e.id) return 3;
    return Math.min(3.5, Math.max(0.5, Math.abs(e.weight) * 2));
  }

  edgeOpacity(e: EdgeData): number {
    if (this.hoveredEdge === e.id) return 1;
    return Math.min(0.95, Math.max(0.12, Math.abs(e.weight) * 0.65 + 0.15));
  }

  activeColor(): string {
    return this.mode === 'backward' ? '#f97316' : '#a78bfa';
  }

  isActive(layer: number): boolean {
    return this.activeLayer === layer && this.mode !== 'idle';
  }

  /** Get delta ring radius for backward pass node */
  getDeltaRingRadius(nd: NodeData): number {
    if (!this.backwardResult || this.mode !== 'backward') return 0;
    // layerGradients index: 0 = weight layer between input (0) and hidden (1)
    // activeLayer corresponds to a node layer; weight layer index = activeLayer - 1
    const wIdx = nd.layer - 1;
    if (wIdx < 0 || wIdx >= this.backwardResult.layerGradients.length) return 0;
    const delta = this.backwardResult.layerGradients[wIdx]?.delta[nd.neuron] ?? 0;
    const absDelta = Math.abs(delta);
    const raw = this.R + 6 + absDelta * 40;
    return Math.min(this.R + 24, Math.max(this.R + 8, raw));
  }

  /** Small activation function symbol rendered inside the node bottom */
  actSymbol(layer: number): string {
    const act = this.config?.layers[layer]?.activation ?? '';
    switch (act) {
      case 'sigmoid': return 'σ';
      case 'relu':    return 'R';
      case 'tanh':    return 'th';
      case 'softmax': return 'sm';
      case 'linear':  return '—';
      default:        return '';
    }
  }

  /** Color of the activation symbol — dim, matching the layer tone */
  actSymbolColor(layer: number): string {
    if (layer === this.lastL) return '#4ade80';
    return '#7c6fe0';
  }

  /** Bias indicator dot color — warm for positive, cool for negative, grey for zero */
  biasColor(bias: number): string {
    if (bias > 0.05)  return '#f97316';  // orange-warm
    if (bias < -0.05) return '#3b82f6';  // blue-cool
    return '#475569';                     // neutral
  }

  /** Compute the begin offset for traveling dots */
  dotBegin(e: EdgeData, dotIndex: number): string {
    const base = (e.fromNeuron * 0.07 + e.toNeuron * 0.05) % 0.5;
    const offset = dotIndex === 0 ? base : base + 0.33;
    return `${offset.toFixed(2)}s`;
  }

  // ── Formatting ──────────────────────────────────────────────────────────
  fmtV(v: number): string {
    if (v === null || isNaN(v)) return '';
    const a = Math.abs(v);
    return a >= 10 ? v.toFixed(1) : a >= 1 ? v.toFixed(2) : v.toFixed(3);
  }

  fmtW(w: number): string {
    return (w >= 0 ? '+' : '') + w.toFixed(2);
  }

  // ── Interaction ──────────────────────────────────────────────────────────
  onNodeHover(evt: MouseEvent, nd: NodeData): void {
    this.hoveredNode = nd.id;
    const lines: string[] = [];
    const kindStr = nd.layer === 0 ? 'Input' : nd.layer === this.lastL ? 'Output' : 'Hidden';
    lines.push(`${kindStr} Layer ${nd.layer}, Neuron ${nd.neuron}`);
    if (nd.value !== null)  lines.push(`a[${nd.layer}]_${nd.neuron} = ${nd.value.toFixed(5)}`);
    if (nd.zValue !== null) lines.push(`z[${nd.layer}]_${nd.neuron} = ${nd.zValue.toFixed(5)}`);
    if (nd.bias  !== null)  lines.push(`bias = ${nd.bias.toFixed(5)}`);
    this.tipW = Math.max(160, lines.reduce((m, l) => Math.max(m, l.length * 6.5), 0) + 12);
    this.showTip(evt, lines);
  }

  onEdgeHover(evt: MouseEvent, e: EdgeData): void {
    this.hoveredEdge = e.id;
    const lines = [
      `W[${e.layerIdx + 1}] [${e.toNeuron}][${e.fromNeuron}]`,
      `weight = ${e.weight.toFixed(5)}`,
      `from neuron ${e.fromNeuron}  →  to neuron ${e.toNeuron}`,
    ];
    this.tipW = 185;
    this.showTip(evt, lines);
  }

  onNodeClick(nd: NodeData): void {
    this.nodeClicked.emit({ layer: nd.layer, neuron: nd.neuron });
  }

  private showTip(evt: MouseEvent, lines: string[]): void {
    const svgEl = (evt.target as SVGElement).closest('svg') as SVGSVGElement;
    const ctm = svgEl?.getScreenCTM?.();
    let tx = 0, ty = 0;
    if (ctm) {
      const pt = svgEl.createSVGPoint();
      pt.x = evt.clientX; pt.y = evt.clientY;
      const sp = pt.matrixTransform(ctm.inverse());
      tx = sp.x + 10; ty = sp.y - 10;
      if (tx + this.tipW > SVG_W - 10) tx = sp.x - this.tipW - 10;
      if (ty < 0) ty = sp.y + 20;
    }
    this.tip = { visible: true, x: tx, y: ty, lines };
    this.cdr.markForCheck();
  }

  hideTooltip(): void {
    this.hoveredNode = '';
    this.hoveredEdge = '';
    this.tip = { ...this.tip, visible: false };
    this.cdr.markForCheck();
  }
}
