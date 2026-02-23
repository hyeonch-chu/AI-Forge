/**
 * Tests for Sidebar — navigation component with internal routes and external service links.
 *
 * Run locally:
 *   npm test -- Sidebar
 * Or in CI:
 *   npm run test:ci
 */
import { render, screen } from '@testing-library/react';
import Sidebar from '@/components/Sidebar';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

// Mock next/navigation since it reads browser state unavailable in jsdom
jest.mock('next/navigation', () => ({
  usePathname: jest.fn(() => '/'),
}));

// Mock next/link to render a plain <a> so we can assert hrefs easily
jest.mock('next/link', () => {
  const MockLink = ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  );
  MockLink.displayName = 'MockLink';
  return MockLink;
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Re-import usePathname mock after jest.mock() is set up. */
function setPathname(path: string) {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const { usePathname } = require('next/navigation');
  (usePathname as jest.Mock).mockReturnValue(path);
}

// ---------------------------------------------------------------------------
// Brand / identity
// ---------------------------------------------------------------------------
describe('Sidebar — brand', () => {
  beforeEach(() => setPathname('/'));

  it('renders the application title "AI-Forge"', () => {
    render(<Sidebar />);
    expect(screen.getByText('AI-Forge')).toBeInTheDocument();
  });

  it('renders the platform subtitle', () => {
    render(<Sidebar />);
    expect(screen.getByText(/VLM MLOps Platform/i)).toBeInTheDocument();
  });

  it('renders the version footer', () => {
    render(<Sidebar />);
    expect(screen.getByText('v1.0.0')).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Internal navigation links
// ---------------------------------------------------------------------------
describe('Sidebar — internal navigation', () => {
  beforeEach(() => setPathname('/'));

  it('renders the Dashboard navigation link', () => {
    render(<Sidebar />);
    expect(screen.getByRole('link', { name: /dashboard/i })).toBeInTheDocument();
  });

  it('renders the Experiments navigation link', () => {
    render(<Sidebar />);
    expect(screen.getByRole('link', { name: /experiments/i })).toBeInTheDocument();
  });

  it('renders the Models navigation link', () => {
    render(<Sidebar />);
    expect(screen.getByRole('link', { name: /models/i })).toBeInTheDocument();
  });

  it('renders the Inference navigation link', () => {
    render(<Sidebar />);
    expect(screen.getByRole('link', { name: /inference/i })).toBeInTheDocument();
  });

  it('Dashboard link href points to "/"', () => {
    render(<Sidebar />);
    expect(screen.getByRole('link', { name: /dashboard/i })).toHaveAttribute('href', '/');
  });

  it('Experiments link href points to "/experiments"', () => {
    render(<Sidebar />);
    expect(screen.getByRole('link', { name: /experiments/i })).toHaveAttribute(
      'href',
      '/experiments',
    );
  });

  it('Models link href points to "/models"', () => {
    render(<Sidebar />);
    expect(screen.getByRole('link', { name: /models/i })).toHaveAttribute('href', '/models');
  });

  it('Inference link href points to "/inference"', () => {
    render(<Sidebar />);
    expect(screen.getByRole('link', { name: /inference/i })).toHaveAttribute(
      'href',
      '/inference',
    );
  });
});

// ---------------------------------------------------------------------------
// Active link highlighting
// ---------------------------------------------------------------------------
describe('Sidebar — active link highlighting', () => {
  it('applies active class to the Dashboard link when pathname is "/"', () => {
    setPathname('/');
    render(<Sidebar />);
    const dashboardLink = screen.getByRole('link', { name: /dashboard/i });
    // Active links contain the "text-blue-300" class in our implementation
    expect(dashboardLink.className).toMatch(/text-blue-300/);
  });

  it('applies active class to Experiments link when pathname is "/experiments"', () => {
    setPathname('/experiments');
    render(<Sidebar />);
    const expLink = screen.getByRole('link', { name: /experiments/i });
    expect(expLink.className).toMatch(/text-blue-300/);
  });

  it('does NOT apply active class to Dashboard when on "/experiments"', () => {
    setPathname('/experiments');
    render(<Sidebar />);
    const dashboardLink = screen.getByRole('link', { name: /dashboard/i });
    expect(dashboardLink.className).not.toMatch(/text-blue-300/);
  });

  it('applies active class to Models link when pathname is "/models"', () => {
    setPathname('/models');
    render(<Sidebar />);
    const modelsLink = screen.getByRole('link', { name: /models/i });
    expect(modelsLink.className).toMatch(/text-blue-300/);
  });

  it('applies active class to Inference link when pathname is "/inference"', () => {
    setPathname('/inference');
    render(<Sidebar />);
    const inferenceLink = screen.getByRole('link', { name: /inference/i });
    expect(inferenceLink.className).toMatch(/text-blue-300/);
  });
});

// ---------------------------------------------------------------------------
// External service links
// ---------------------------------------------------------------------------
describe('Sidebar — external service links', () => {
  beforeEach(() => setPathname('/'));

  it('renders the "Services" section heading', () => {
    render(<Sidebar />);
    expect(screen.getByText(/services/i)).toBeInTheDocument();
  });

  it('renders an "MLflow UI" external link', () => {
    render(<Sidebar />);
    expect(screen.getByText('MLflow UI')).toBeInTheDocument();
  });

  it('renders an "API Docs" external link', () => {
    render(<Sidebar />);
    expect(screen.getByText('API Docs')).toBeInTheDocument();
  });

  it('renders a "Grafana" external link', () => {
    render(<Sidebar />);
    expect(screen.getByText('Grafana')).toBeInTheDocument();
  });

  it('renders a "MinIO" external link', () => {
    render(<Sidebar />);
    expect(screen.getByText('MinIO')).toBeInTheDocument();
  });

  it('MLflow link opens in a new tab', () => {
    render(<Sidebar />);
    // The <a> element containing "MLflow UI" text should have target="_blank"
    const link = screen.getByText('MLflow UI').closest('a')!;
    expect(link).toHaveAttribute('target', '_blank');
  });

  it('all external links include rel="noopener noreferrer" for security', () => {
    render(<Sidebar />);
    const externalAnchors = document
      .querySelectorAll('a[target="_blank"]');
    externalAnchors.forEach((a) => {
      expect(a).toHaveAttribute('rel', 'noopener noreferrer');
    });
  });

  it('MLflow link href includes port 5000', () => {
    render(<Sidebar />);
    const link = screen.getByText('MLflow UI').closest('a')!;
    expect(link.getAttribute('href')).toContain(':5000');
  });

  it('API Docs link href includes port 8000 and /docs path', () => {
    render(<Sidebar />);
    const link = screen.getByText('API Docs').closest('a')!;
    const href = link.getAttribute('href')!;
    expect(href).toContain(':8000');
    expect(href).toContain('/docs');
  });

  it('Grafana link href includes port 3001', () => {
    render(<Sidebar />);
    const link = screen.getByText('Grafana').closest('a')!;
    expect(link.getAttribute('href')).toContain(':3001');
  });

  it('MinIO link href includes port 9001', () => {
    render(<Sidebar />);
    const link = screen.getByText('MinIO').closest('a')!;
    expect(link.getAttribute('href')).toContain(':9001');
  });
});
