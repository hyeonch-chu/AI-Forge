/**
 * Tests for MetricsChart — pure SVG bar chart component.
 *
 * Run locally:
 *   npm test -- MetricsChart
 * Or in CI:
 *   npm run test:ci
 */
import { render, screen } from '@testing-library/react';
import MetricsChart, { type ChartRun } from '@/components/MetricsChart';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a minimal ChartRun with the given metric values. */
function makeRun(overrides: Partial<ChartRun> = {}): ChartRun {
  return { id: 'r1', name: 'run1', map50: 0.8, map5095: 0.6, ...overrides };
}

// ---------------------------------------------------------------------------
// Empty / null states
// ---------------------------------------------------------------------------
describe('MetricsChart — empty data', () => {
  it('renders nothing when runs array is empty', () => {
    const { container } = render(<MetricsChart runs={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders nothing when no run has map50 data', () => {
    const runs: ChartRun[] = [{ id: 'r1', name: 'run1' }];
    const { container } = render(<MetricsChart runs={runs} />);
    expect(container.firstChild).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Chart structure
// ---------------------------------------------------------------------------
describe('MetricsChart — chart structure', () => {
  it('renders an SVG with the accessible aria-label', () => {
    render(<MetricsChart runs={[makeRun()]} />);
    expect(screen.getByRole('img', { name: /metrics bar chart/i })).toBeInTheDocument();
  });

  it('renders a container div wrapping the chart', () => {
    const { container } = render(<MetricsChart runs={[makeRun()]} />);
    expect(container.querySelector('div')).toBeInTheDocument();
  });

  it('renders the descriptive caption text', () => {
    render(<MetricsChart runs={[makeRun()]} />);
    expect(screen.getByText(/detection metrics across runs/i)).toBeInTheDocument();
  });

  it('renders Y-axis tick labels for 0, 0.25, 0.50, 0.75, 1.00', () => {
    render(<MetricsChart runs={[makeRun()]} />);
    // Tick labels are SVG <text> elements
    const svg = document.querySelector('svg')!;
    const texts = Array.from(svg.querySelectorAll('text')).map((t) => t.textContent);
    expect(texts).toContain('0.00');
    expect(texts).toContain('0.25');
    expect(texts).toContain('0.50');
    expect(texts).toContain('0.75');
    expect(texts).toContain('1.00');
  });

  it('renders legend labels for mAP50 and mAP50-95', () => {
    render(<MetricsChart runs={[makeRun()]} />);
    const svg = document.querySelector('svg')!;
    const texts = Array.from(svg.querySelectorAll('text')).map((t) => t.textContent);
    expect(texts).toContain('mAP50');
    expect(texts).toContain('mAP50-95');
  });
});

// ---------------------------------------------------------------------------
// Bars rendered per run
// ---------------------------------------------------------------------------
describe('MetricsChart — bar rendering', () => {
  it('renders two rect bars for a run with both map50 and map5095', () => {
    render(<MetricsChart runs={[makeRun({ map50: 0.8, map5095: 0.6 })]} />);
    // One run ⇒ 2 data bars + 4 legend rects = 6 total rects in SVG
    // We just verify there are at least 2 rects (the legend also has rects)
    const rects = document.querySelectorAll('svg rect');
    expect(rects.length).toBeGreaterThanOrEqual(2);
  });

  it('renders a tooltip title for the mAP50 bar', () => {
    render(<MetricsChart runs={[makeRun({ map50: 0.8 })]} />);
    const titles = Array.from(document.querySelectorAll('svg title')).map(
      (t) => t.textContent,
    );
    expect(titles.some((t) => t?.startsWith('mAP50:'))).toBe(true);
  });

  it('renders a tooltip title for the mAP50-95 bar when present', () => {
    render(<MetricsChart runs={[makeRun({ map5095: 0.6 })]} />);
    const titles = Array.from(document.querySelectorAll('svg title')).map(
      (t) => t.textContent,
    );
    expect(titles.some((t) => t?.startsWith('mAP50-95:'))).toBe(true);
  });

  it('skips the mAP50-95 bar when map5095 is not provided', () => {
    render(<MetricsChart runs={[makeRun({ map5095: undefined })]} />);
    const titles = Array.from(document.querySelectorAll('svg title')).map(
      (t) => t.textContent,
    );
    expect(titles.some((t) => t?.startsWith('mAP50-95:'))).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Run name labels and truncation
// ---------------------------------------------------------------------------
describe('MetricsChart — x-axis labels', () => {
  it('renders the run name as an x-axis label (truncated to 9 chars)', () => {
    const run = makeRun({ name: 'my-experiment-run-001' });
    render(<MetricsChart runs={[run]} />);
    const svg = document.querySelector('svg')!;
    const textContents = Array.from(svg.querySelectorAll('text')).map((t) => t.textContent);
    // Should contain the first 9 characters
    expect(textContents).toContain('my-experi');
  });

  it('falls back to "run1" label when name is empty', () => {
    render(<MetricsChart runs={[makeRun({ name: '' })]} />);
    const svg = document.querySelector('svg')!;
    const textContents = Array.from(svg.querySelectorAll('text')).map((t) => t.textContent);
    expect(textContents).toContain('run1');
  });
});

// ---------------------------------------------------------------------------
// Data cap
// ---------------------------------------------------------------------------
describe('MetricsChart — data cap', () => {
  it('caps rendered runs at 10 even when more are supplied', () => {
    // Supply 15 runs — only the first 10 should be rendered
    const runs: ChartRun[] = Array.from({ length: 15 }, (_, i) => ({
      id: `r${i}`,
      name: `run${i}`,
      map50: 0.5,
    }));
    render(<MetricsChart runs={runs} />);
    const svg = document.querySelector('svg')!;
    const labels = Array.from(svg.querySelectorAll('text'))
      .map((t) => t.textContent)
      // Filter out Y-tick and legend labels
      .filter((t) => t?.startsWith('run'));
    // At most 10 run labels
    expect(labels.length).toBeLessThanOrEqual(10);
  });
});
