import {
  Component,
  Input,
  OnChanges,
  ElementRef,
  ViewChild,
  AfterViewInit,
  SimpleChanges,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import katex from 'katex';

@Component({
  selector: 'app-equation-display',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div #container class="katex-container" [class]="containerClass"></div>
  `,
  styles: [`
    .katex-container {
      overflow-x: auto;
      padding: 4px 0;
    }
  `]
})
export class EquationDisplayComponent implements OnChanges, AfterViewInit {
  @Input() latex = '';
  @Input() displayMode = true;
  @Input() containerClass = '';
  @ViewChild('container') container!: ElementRef;

  ngAfterViewInit(): void {
    this.render();
  }

  ngOnChanges(_changes: SimpleChanges): void {
    if (this.container) {
      this.render();
    }
  }

  private render(): void {
    if (!this.container || !this.latex) return;
    try {
      katex.render(this.latex, this.container.nativeElement, {
        displayMode: this.displayMode,
        throwOnError: false,
        strict: false,
        output: 'html',
      });
    } catch (e) {
      this.container.nativeElement.textContent = this.latex;
    }
  }
}
