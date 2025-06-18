// WS6-P2 Comprehensive Testing Framework
// Advanced testing suite for enhanced interface and visualization components

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { jest } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

// Import components to test
import { DataVisualization } from '../components/charts/DataVisualization';
import { AccountManagement } from '../components/advanced/AccountManagement';
import { MarketAnalysisDashboard } from '../components/advanced/MarketAnalysis';
import { ProtocolComplianceMonitor } from '../components/advanced/ProtocolCompliance';
import { EnhancedAppShell } from '../components/enhanced/EnhancedAppShell';
import { UIComponents } from '../components/advanced/UIComponents';

// Mock data for testing
const mockPortfolioData = [
  { date: '2024-01-01', generation: 10000, revenue: 15000, compounding: 25000 },
  { date: '2024-01-02', generation: 10500, revenue: 15200, compounding: 25800 },
  { date: '2024-01-03', generation: 9800, revenue: 14900, compounding: 25200 }
];

const mockMarketData = [
  { symbol: 'SPY', price: 450.25, change: 2.15, changePercent: 0.0048, volume: 50000000 },
  { symbol: 'QQQ', price: 380.50, change: -1.25, changePercent: -0.0033, volume: 30000000 }
];

const mockTradingSignals = [
  {
    id: 'signal-1',
    symbol: 'SPY',
    type: 'buy' as const,
    strategy: 'Momentum',
    confidence: 0.85,
    targetPrice: 455.00,
    stopLoss: 445.00,
    timeframe: '1D',
    reasoning: 'Strong momentum with volume confirmation',
    riskLevel: 'Medium' as const,
    expectedReturn: 0.12,
    probability: 0.78,
    timestamp: new Date().toISOString()
  }
];

const mockUser = {
  name: 'John Doe',
  email: 'john@example.com',
  avatar: 'https://example.com/avatar.jpg'
};

// Test Suite 1: Data Visualization Components
describe('Data Visualization Components', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders portfolio performance chart correctly', async () => {
    render(
      <DataVisualization
        data={mockPortfolioData}
        type="portfolio"
        title="Portfolio Performance"
      />
    );

    expect(screen.getByText('Portfolio Performance')).toBeInTheDocument();
    expect(screen.getByText('Generation')).toBeInTheDocument();
    expect(screen.getByText('Revenue')).toBeInTheDocument();
    expect(screen.getByText('Compounding')).toBeInTheDocument();
  });

  test('handles timeframe changes correctly', async () => {
    const user = userEvent.setup();
    render(
      <DataVisualization
        data={mockPortfolioData}
        type="portfolio"
        title="Portfolio Performance"
      />
    );

    const timeframeSelect = screen.getByRole('combobox', { name: /timeframe/i });
    await user.selectOptions(timeframeSelect, '1M');

    await waitFor(() => {
      expect(timeframeSelect).toHaveValue('1M');
    });
  });

  test('exports data correctly', async () => {
    const user = userEvent.setup();
    const mockExport = jest.fn();
    
    render(
      <DataVisualization
        data={mockPortfolioData}
        type="portfolio"
        title="Portfolio Performance"
        onExport={mockExport}
      />
    );

    const exportButton = screen.getByRole('button', { name: /export/i });
    await user.click(exportButton);

    expect(mockExport).toHaveBeenCalledWith(mockPortfolioData, 'csv');
  });

  test('handles empty data gracefully', () => {
    render(
      <DataVisualization
        data={[]}
        type="portfolio"
        title="Portfolio Performance"
      />
    );

    expect(screen.getByText(/no data available/i)).toBeInTheDocument();
  });

  test('responsive design works correctly', () => {
    // Mock window resize
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 768,
    });

    render(
      <DataVisualization
        data={mockPortfolioData}
        type="portfolio"
        title="Portfolio Performance"
      />
    );

    // Chart should adapt to mobile view
    const chartContainer = screen.getByTestId('chart-container');
    expect(chartContainer).toHaveClass('mobile-responsive');
  });
});

// Test Suite 2: Account Management Components
describe('Account Management Components', () => {
  test('renders account overview correctly', () => {
    render(<AccountManagement accountId="test-account" />);

    expect(screen.getByText('Portfolio Rebalancing')).toBeInTheDocument();
    expect(screen.getByText('Position Management')).toBeInTheDocument();
    expect(screen.getByText('Risk Analysis')).toBeInTheDocument();
  });

  test('handles portfolio rebalancing', async () => {
    const user = userEvent.setup();
    const mockRebalance = jest.fn();

    render(
      <AccountManagement 
        accountId="test-account" 
        onRebalance={mockRebalance}
      />
    );

    // Adjust allocation
    const allocationSlider = screen.getByRole('slider', { name: /generation allocation/i });
    await user.click(allocationSlider);
    
    // Trigger rebalancing
    const rebalanceButton = screen.getByRole('button', { name: /rebalance portfolio/i });
    await user.click(rebalanceButton);

    expect(mockRebalance).toHaveBeenCalled();
  });

  test('validates allocation limits', async () => {
    const user = userEvent.setup();
    render(<AccountManagement accountId="test-account" />);

    const allocationInput = screen.getByRole('spinbutton', { name: /generation allocation/i });
    await user.clear(allocationInput);
    await user.type(allocationInput, '150');

    await waitFor(() => {
      expect(screen.getByText(/allocation cannot exceed 100%/i)).toBeInTheDocument();
    });
  });

  test('displays position details correctly', async () => {
    const user = userEvent.setup();
    render(<AccountManagement accountId="test-account" />);

    const positionRow = screen.getByTestId('position-SPY');
    await user.click(positionRow);

    await waitFor(() => {
      expect(screen.getByText('Position Details')).toBeInTheDocument();
      expect(screen.getByText('P&L Analysis')).toBeInTheDocument();
    });
  });
});

// Test Suite 3: Market Analysis Components
describe('Market Analysis Components', () => {
  test('renders market analysis dashboard', () => {
    render(<MarketAnalysisDashboard symbols={['SPY', 'QQQ']} />);

    expect(screen.getByText('Market Analysis')).toBeInTheDocument();
    expect(screen.getByText('Market Overview')).toBeInTheDocument();
    expect(screen.getByText('Technical Levels')).toBeInTheDocument();
  });

  test('handles tab navigation', async () => {
    const user = userEvent.setup();
    render(<MarketAnalysisDashboard symbols={['SPY', 'QQQ']} />);

    const technicalTab = screen.getByRole('tab', { name: /technical levels/i });
    await user.click(technicalTab);

    await waitFor(() => {
      expect(screen.getByText('Support & Resistance Levels')).toBeInTheDocument();
    });
  });

  test('refreshes market data', async () => {
    const user = userEvent.setup();
    const mockRefresh = jest.fn();

    render(
      <MarketAnalysisDashboard 
        symbols={['SPY', 'QQQ']} 
        onRefresh={mockRefresh}
      />
    );

    const refreshButton = screen.getByRole('button', { name: /refresh/i });
    await user.click(refreshButton);

    expect(mockRefresh).toHaveBeenCalled();
  });

  test('displays economic indicators', () => {
    render(<MarketAnalysisDashboard symbols={['SPY', 'QQQ']} />);

    const economicTab = screen.getByRole('tab', { name: /economic data/i });
    fireEvent.click(economicTab);

    expect(screen.getByText('GDP Growth')).toBeInTheDocument();
    expect(screen.getByText('Inflation')).toBeInTheDocument();
    expect(screen.getByText('Unemployment')).toBeInTheDocument();
  });
});

// Test Suite 4: Protocol Compliance Components
describe('Protocol Compliance Components', () => {
  test('renders compliance monitor', () => {
    render(<ProtocolComplianceMonitor accountId="test-account" />);

    expect(screen.getByText('Protocol Compliance')).toBeInTheDocument();
    expect(screen.getByText('Compliance Score')).toBeInTheDocument();
    expect(screen.getByText('Week Classification')).toBeInTheDocument();
  });

  test('handles violation details', async () => {
    const user = userEvent.setup();
    render(<ProtocolComplianceMonitor accountId="test-account" />);

    // Assuming there's a violation to click
    const violationItem = screen.getByTestId('violation-item-0');
    await user.click(violationItem);

    await waitFor(() => {
      expect(screen.getByText('Violation Details')).toBeInTheDocument();
    });
  });

  test('exports compliance report', async () => {
    const user = userEvent.setup();
    const mockExport = jest.fn();

    render(
      <ProtocolComplianceMonitor 
        accountId="test-account" 
        onExportReport={mockExport}
      />
    );

    const exportButton = screen.getByRole('button', { name: /export report/i });
    await user.click(exportButton);

    expect(mockExport).toHaveBeenCalled();
  });
});

// Test Suite 5: Enhanced App Shell
describe('Enhanced App Shell', () => {
  test('renders app shell with user info', () => {
    render(
      <EnhancedAppShell title="ALL-USE Dashboard" user={mockUser}>
        <div>Test Content</div>
      </EnhancedAppShell>
    );

    expect(screen.getByText('ALL-USE Dashboard')).toBeInTheDocument();
    expect(screen.getByText('John Doe')).toBeInTheDocument();
    expect(screen.getByText('Test Content')).toBeInTheDocument();
  });

  test('handles theme toggle', async () => {
    const user = userEvent.setup();
    render(
      <EnhancedAppShell title="ALL-USE Dashboard" user={mockUser}>
        <div>Test Content</div>
      </EnhancedAppShell>
    );

    const themeToggle = screen.getByRole('button', { name: /switch to.*theme/i });
    await user.click(themeToggle);

    // Check if theme class is applied
    await waitFor(() => {
      expect(document.documentElement).toHaveClass('dark');
    });
  });

  test('opens settings modal', async () => {
    const user = userEvent.setup();
    render(
      <EnhancedAppShell title="ALL-USE Dashboard" user={mockUser}>
        <div>Test Content</div>
      </EnhancedAppShell>
    );

    const settingsButton = screen.getByRole('button', { name: /open settings/i });
    await user.click(settingsButton);

    await waitFor(() => {
      expect(screen.getByText('Settings')).toBeInTheDocument();
      expect(screen.getByText('Appearance')).toBeInTheDocument();
      expect(screen.getByText('Accessibility')).toBeInTheDocument();
    });
  });

  test('handles notifications', async () => {
    const user = userEvent.setup();
    render(
      <EnhancedAppShell title="ALL-USE Dashboard" user={mockUser}>
        <div>Test Content</div>
      </EnhancedAppShell>
    );

    // Trigger a notification (this would typically come from an action)
    act(() => {
      // Simulate notification trigger
      const event = new CustomEvent('notification', {
        detail: {
          type: 'success',
          title: 'Test Notification',
          message: 'This is a test notification'
        }
      });
      window.dispatchEvent(event);
    });

    await waitFor(() => {
      expect(screen.getByText('Test Notification')).toBeInTheDocument();
    });
  });
});

// Test Suite 6: UI Components
describe('UI Components', () => {
  test('button component renders correctly', () => {
    render(
      <UIComponents.Button variant="primary" size="md">
        Test Button
      </UIComponents.Button>
    );

    const button = screen.getByRole('button', { name: /test button/i });
    expect(button).toBeInTheDocument();
    expect(button).toHaveClass('btn-primary');
  });

  test('modal component works correctly', async () => {
    const user = userEvent.setup();
    const mockClose = jest.fn();

    render(
      <UIComponents.Modal isOpen={true} onClose={mockClose} title="Test Modal">
        <div>Modal Content</div>
      </UIComponents.Modal>
    );

    expect(screen.getByText('Test Modal')).toBeInTheDocument();
    expect(screen.getByText('Modal Content')).toBeInTheDocument();

    const closeButton = screen.getByRole('button', { name: /close/i });
    await user.click(closeButton);

    expect(mockClose).toHaveBeenCalled();
  });

  test('progress bar displays correctly', () => {
    render(
      <UIComponents.ProgressBar 
        value={75} 
        max={100} 
        color="blue" 
        size="md"
      />
    );

    const progressBar = screen.getByRole('progressbar');
    expect(progressBar).toBeInTheDocument();
    expect(progressBar).toHaveAttribute('aria-valuenow', '75');
    expect(progressBar).toHaveAttribute('aria-valuemax', '100');
  });

  test('tabs component handles navigation', async () => {
    const user = userEvent.setup();
    const mockChange = jest.fn();

    const tabs = [
      { id: 'tab1', label: 'Tab 1' },
      { id: 'tab2', label: 'Tab 2' }
    ];

    render(
      <UIComponents.Tabs 
        tabs={tabs} 
        activeTab="tab1" 
        onChange={mockChange}
      />
    );

    const tab2 = screen.getByRole('tab', { name: /tab 2/i });
    await user.click(tab2);

    expect(mockChange).toHaveBeenCalledWith('tab2');
  });
});

// Test Suite 7: Performance and Accessibility
describe('Performance and Accessibility', () => {
  test('components are accessible', () => {
    render(
      <EnhancedAppShell title="ALL-USE Dashboard" user={mockUser}>
        <DataVisualization
          data={mockPortfolioData}
          type="portfolio"
          title="Portfolio Performance"
        />
      </EnhancedAppShell>
    );

    // Check for proper ARIA labels
    expect(screen.getByRole('main')).toBeInTheDocument();
    expect(screen.getByRole('banner')).toBeInTheDocument();
    
    // Check for keyboard navigation
    const firstFocusableElement = screen.getAllByRole('button')[0];
    expect(firstFocusableElement).toBeInTheDocument();
  });

  test('components handle reduced motion preference', () => {
    // Mock reduced motion preference
    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      value: jest.fn().mockImplementation(query => ({
        matches: query === '(prefers-reduced-motion: reduce)',
        media: query,
        onchange: null,
        addListener: jest.fn(),
        removeListener: jest.fn(),
        addEventListener: jest.fn(),
        removeEventListener: jest.fn(),
        dispatchEvent: jest.fn(),
      })),
    });

    render(
      <EnhancedAppShell title="ALL-USE Dashboard" user={mockUser}>
        <div>Test Content</div>
      </EnhancedAppShell>
    );

    expect(document.documentElement).toHaveClass('reduced-motion');
  });

  test('components handle high contrast mode', () => {
    // Mock high contrast preference
    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      value: jest.fn().mockImplementation(query => ({
        matches: query === '(prefers-contrast: high)',
        media: query,
        onchange: null,
        addListener: jest.fn(),
        removeListener: jest.fn(),
        addEventListener: jest.fn(),
        removeEventListener: jest.fn(),
        dispatchEvent: jest.fn(),
      })),
    });

    render(
      <EnhancedAppShell title="ALL-USE Dashboard" user={mockUser}>
        <div>Test Content</div>
      </EnhancedAppShell>
    );

    expect(document.documentElement).toHaveClass('high-contrast');
  });
});

// Test Suite 8: Integration Tests
describe('Integration Tests', () => {
  test('complete user workflow', async () => {
    const user = userEvent.setup();
    
    render(
      <EnhancedAppShell title="ALL-USE Dashboard" user={mockUser}>
        <MarketAnalysisDashboard symbols={['SPY', 'QQQ']} />
        <AccountManagement accountId="test-account" />
        <ProtocolComplianceMonitor accountId="test-account" />
      </EnhancedAppShell>
    );

    // 1. Check market analysis
    expect(screen.getByText('Market Analysis')).toBeInTheDocument();
    
    // 2. Navigate to account management
    const accountTab = screen.getByText('Portfolio Rebalancing');
    await user.click(accountTab);
    
    // 3. Check compliance
    expect(screen.getByText('Protocol Compliance')).toBeInTheDocument();
    
    // 4. Open settings
    const settingsButton = screen.getByRole('button', { name: /open settings/i });
    await user.click(settingsButton);
    
    await waitFor(() => {
      expect(screen.getByText('Settings')).toBeInTheDocument();
    });
  });

  test('error handling across components', async () => {
    // Mock console.error to avoid noise in tests
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    
    // Component that throws an error
    const ErrorComponent = () => {
      throw new Error('Test error');
    };

    render(
      <EnhancedAppShell title="ALL-USE Dashboard" user={mockUser}>
        <ErrorComponent />
      </EnhancedAppShell>
    );

    await waitFor(() => {
      expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    });

    consoleSpy.mockRestore();
  });
});

// Test utilities and helpers
export const testUtils = {
  // Mock API responses
  mockApiResponse: (data: any, delay: number = 100) => {
    return new Promise(resolve => {
      setTimeout(() => resolve(data), delay);
    });
  },

  // Mock WebSocket connection
  mockWebSocket: () => {
    const mockWS = {
      send: jest.fn(),
      close: jest.fn(),
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      readyState: WebSocket.OPEN
    };
    return mockWS;
  },

  // Performance testing helper
  measurePerformance: async (component: React.ReactElement) => {
    const startTime = performance.now();
    render(component);
    const endTime = performance.now();
    return endTime - startTime;
  },

  // Accessibility testing helper
  checkAccessibility: (container: HTMLElement) => {
    const focusableElements = container.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    return {
      focusableCount: focusableElements.length,
      hasProperLabels: Array.from(focusableElements).every(el => 
        el.getAttribute('aria-label') || 
        el.getAttribute('aria-labelledby') || 
        (el as HTMLElement).textContent
      )
    };
  }
};

// Export test configuration
export const testConfig = {
  testTimeout: 10000,
  setupFilesAfterEnv: ['<rootDir>/src/setupTests.ts'],
  testEnvironment: 'jsdom',
  moduleNameMapping: {
    '^@/(.*)$': '<rootDir>/src/$1'
  },
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/**/*.d.ts',
    '!src/tests/**/*'
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  }
};

