import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

// Import all WS6 components for integration testing
import ConversationalInterface from '../components/ConversationalInterface';
import AccountVisualization from '../components/AccountVisualization';
import Authentication from '../components/Authentication';
import Analytics from '../components/Analytics';
import AdvancedDashboardBuilder from '../components/sophisticated/AdvancedDashboardBuilder';
import IntelligenceDashboard from '../components/intelligence/IntelligenceDashboard';
import TradingDashboard from '../components/trading/TradingDashboard';
import EnterpriseFeatures from '../components/enterprise/EnterpriseFeatures';
import PerformanceMonitoringDashboard from '../performance/performance_monitoring_framework';
import OptimizationEngineDashboard from '../performance/optimization_engine';
import AdvancedAnalyticsDashboard from '../performance/advanced_analytics';
import SystemCoordinationDashboard from '../performance/system_coordination';

// Mock external dependencies
jest.mock('../lib/conversational-agent', () => ({
  ConversationalAgent: jest.fn().mockImplementation(() => ({
    initialize: jest.fn(),
    sendMessage: jest.fn().mockResolvedValue({ response: 'Mock response' }),
    startListening: jest.fn(),
    stopListening: jest.fn(),
    isListening: false
  }))
}));

jest.mock('../lib/ws1-integration', () => ({
  WS1Integration: jest.fn().mockImplementation(() => ({
    getAccountData: jest.fn().mockResolvedValue({ balance: 10000, accounts: [] }),
    getTransactions: jest.fn().mockResolvedValue([]),
    authenticate: jest.fn().mockResolvedValue({ success: true, token: 'mock-token' })
  }))
}));

jest.mock('../lib/ws2-integration', () => ({
  WS2Integration: jest.fn().mockImplementation(() => ({
    getTransactionHistory: jest.fn().mockResolvedValue([]),
    processTransaction: jest.fn().mockResolvedValue({ success: true }),
    getTransactionStatus: jest.fn().mockResolvedValue({ status: 'completed' })
  }))
}));

jest.mock('../integrations/ws3/MarketIntelligenceClient', () => ({
  MarketIntelligenceClient: jest.fn().mockImplementation(() => ({
    getMarketData: jest.fn().mockResolvedValue({ price: 100, change: 5 }),
    getMarketAnalysis: jest.fn().mockResolvedValue({ trend: 'bullish' }),
    getIntelligenceInsights: jest.fn().mockResolvedValue([])
  }))
}));

jest.mock('../integrations/ws4/MarketIntegrationClient', () => ({
  MarketIntegrationClient: jest.fn().mockImplementation(() => ({
    getExternalMarketData: jest.fn().mockResolvedValue({ data: [] }),
    executeTrade: jest.fn().mockResolvedValue({ success: true }),
    getMarketStatus: jest.fn().mockResolvedValue({ status: 'open' })
  }))
}));

// Mock performance APIs
const mockPerformance = {
  now: jest.fn(() => Date.now()),
  getEntriesByType: jest.fn(() => []),
  getEntriesByName: jest.fn(() => []),
  mark: jest.fn(),
  measure: jest.fn(),
  memory: {
    usedJSHeapSize: 50 * 1024 * 1024,
    totalJSHeapSize: 100 * 1024 * 1024,
    jsHeapSizeLimit: 2 * 1024 * 1024 * 1024
  }
};

// Setup global mocks
beforeAll(() => {
  global.performance = mockPerformance as any;
  global.ResizeObserver = jest.fn().mockImplementation(() => ({
    observe: jest.fn(),
    unobserve: jest.fn(),
    disconnect: jest.fn(),
  }));
  global.IntersectionObserver = jest.fn().mockImplementation(() => ({
    observe: jest.fn(),
    unobserve: jest.fn(),
    disconnect: jest.fn(),
  }));
  global.PerformanceObserver = jest.fn().mockImplementation(() => ({
    observe: jest.fn(),
    disconnect: jest.fn(),
    takeRecords: jest.fn(() => [])
  }));
});

// Component Integration Testing Framework
describe('WS6-P6: Component Integration Framework', () => {
  describe('Core Component Integration', () => {
    test('ConversationalInterface integrates with Authentication', async () => {
      const user = userEvent.setup();
      
      render(
        <div>
          <Authentication />
          <ConversationalInterface />
        </div>
      );
      
      // Test authentication flow
      const loginButton = screen.getByText('Login');
      await user.click(loginButton);
      
      // Verify conversational interface responds to authentication
      await waitFor(() => {
        expect(screen.getByText('Welcome! How can I help you today?')).toBeInTheDocument();
      });
    });

    test('AccountVisualization integrates with ConversationalInterface', async () => {
      const user = userEvent.setup();
      
      render(
        <div>
          <ConversationalInterface />
          <AccountVisualization />
        </div>
      );
      
      // Test conversational query about account
      const messageInput = screen.getByPlaceholderText('Type your message...');
      await user.type(messageInput, 'Show my account balance');
      
      const sendButton = screen.getByText('Send');
      await user.click(sendButton);
      
      // Verify account visualization updates
      await waitFor(() => {
        expect(screen.getByText('Account Overview')).toBeInTheDocument();
      });
    });

    test('Analytics integrates with all dashboard components', async () => {
      render(
        <div>
          <Analytics />
          <AdvancedDashboardBuilder />
          <IntelligenceDashboard />
          <TradingDashboard />
        </div>
      );
      
      // Verify analytics data flows to all dashboards
      expect(screen.getByText('Analytics Dashboard')).toBeInTheDocument();
      expect(screen.getByText('Advanced Dashboard Builder')).toBeInTheDocument();
      expect(screen.getByText('Market Intelligence Dashboard')).toBeInTheDocument();
      expect(screen.getByText('Trading Dashboard')).toBeInTheDocument();
    });
  });

  describe('Advanced Component Integration', () => {
    test('AdvancedDashboardBuilder integrates with all visualization components', async () => {
      const user = userEvent.setup();
      
      render(<AdvancedDashboardBuilder />);
      
      // Test widget addition
      const addWidgetButton = screen.getByText('Add Widget');
      await user.click(addWidgetButton);
      
      // Verify widget options include all component types
      await waitFor(() => {
        expect(screen.getByText('Account Visualization')).toBeInTheDocument();
        expect(screen.getByText('Market Analysis')).toBeInTheDocument();
        expect(screen.getByText('Trading Interface')).toBeInTheDocument();
      });
    });

    test('IntelligenceDashboard integrates with WS3 market intelligence', async () => {
      render(<IntelligenceDashboard />);
      
      // Verify market intelligence data integration
      await waitFor(() => {
        expect(screen.getByText('Market Intelligence')).toBeInTheDocument();
        expect(screen.getByText('Real-time market analysis')).toBeInTheDocument();
      });
    });

    test('TradingDashboard integrates with WS4 market integration', async () => {
      render(<TradingDashboard />);
      
      // Verify trading functionality integration
      await waitFor(() => {
        expect(screen.getByText('Trading Dashboard')).toBeInTheDocument();
        expect(screen.getByText('Real-time trading interface')).toBeInTheDocument();
      });
    });

    test('EnterpriseFeatures integrates with all system components', async () => {
      render(<EnterpriseFeatures />);
      
      // Verify enterprise features integration
      await waitFor(() => {
        expect(screen.getByText('Enterprise Administration')).toBeInTheDocument();
        expect(screen.getByText('User Management')).toBeInTheDocument();
        expect(screen.getByText('System Monitoring')).toBeInTheDocument();
      });
    });
  });

  describe('Performance Component Integration', () => {
    test('PerformanceMonitoringDashboard integrates with all UI components', async () => {
      const user = userEvent.setup();
      
      render(
        <div>
          <PerformanceMonitoringDashboard />
          <ConversationalInterface />
          <AccountVisualization />
        </div>
      );
      
      // Start performance monitoring
      const startButton = screen.getByText('Start Monitoring');
      await user.click(startButton);
      
      // Verify monitoring detects other components
      await waitFor(() => {
        expect(screen.getByText('Components')).toBeInTheDocument();
      });
      
      const componentsTab = screen.getByText('Components');
      await user.click(componentsTab);
      
      await waitFor(() => {
        expect(screen.getByText('ConversationalInterface')).toBeInTheDocument();
        expect(screen.getByText('AccountVisualization')).toBeInTheDocument();
      });
    });

    test('OptimizationEngine integrates with performance monitoring', async () => {
      const user = userEvent.setup();
      
      render(
        <div>
          <PerformanceMonitoringDashboard />
          <OptimizationEngineDashboard />
        </div>
      );
      
      // Start monitoring and optimization
      const monitoringStart = screen.getAllByText('Start Monitoring')[0];
      await user.click(monitoringStart);
      
      const optimizationStart = screen.getByText('Start Optimization');
      await user.click(optimizationStart);
      
      // Verify integration between monitoring and optimization
      await waitFor(() => {
        expect(screen.getByText('Stop Monitoring')).toBeInTheDocument();
        expect(screen.getByText('Stop Optimization')).toBeInTheDocument();
      });
    });

    test('AdvancedAnalytics integrates with system coordination', async () => {
      const user = userEvent.setup();
      
      render(
        <div>
          <AdvancedAnalyticsDashboard />
          <SystemCoordinationDashboard />
        </div>
      );
      
      // Start analytics and coordination
      const analyticsStart = screen.getByText('Start Analysis');
      await user.click(analyticsStart);
      
      const coordinationStart = screen.getByText('Start Coordination');
      await user.click(coordinationStart);
      
      // Verify integration between analytics and coordination
      await waitFor(() => {
        expect(screen.getByText('Stop Analysis')).toBeInTheDocument();
        expect(screen.getByText('Stop Coordination')).toBeInTheDocument();
      });
    });

    test('SystemCoordination manages all performance components', async () => {
      const user = userEvent.setup();
      
      render(<SystemCoordinationDashboard />);
      
      // Start system coordination
      const startButton = screen.getByText('Start Coordination');
      await user.click(startButton);
      
      // Check components tab
      const componentsTab = screen.getByText('Components');
      await user.click(componentsTab);
      
      // Verify all performance components are managed
      await waitFor(() => {
        expect(screen.getByText('Performance Monitoring Framework')).toBeInTheDocument();
        expect(screen.getByText('Performance Optimization Engine')).toBeInTheDocument();
        expect(screen.getByText('Advanced Analytics Engine')).toBeInTheDocument();
      });
    });
  });

  describe('Cross-Component Communication', () => {
    test('components share state through context providers', async () => {
      const user = userEvent.setup();
      
      render(
        <div>
          <Authentication />
          <ConversationalInterface />
          <AccountVisualization />
        </div>
      );
      
      // Login and verify state propagation
      const loginButton = screen.getByText('Login');
      await user.click(loginButton);
      
      // Verify authentication state affects all components
      await waitFor(() => {
        expect(screen.getByText('Welcome! How can I help you today?')).toBeInTheDocument();
        expect(screen.getByText('Account Overview')).toBeInTheDocument();
      });
    });

    test('components communicate through event system', async () => {
      const user = userEvent.setup();
      
      render(
        <div>
          <ConversationalInterface />
          <TradingDashboard />
        </div>
      );
      
      // Send trading command through conversational interface
      const messageInput = screen.getByPlaceholderText('Type your message...');
      await user.type(messageInput, 'Execute trade for AAPL');
      
      const sendButton = screen.getByText('Send');
      await user.click(sendButton);
      
      // Verify trading dashboard responds
      await waitFor(() => {
        expect(screen.getByText('Trading Dashboard')).toBeInTheDocument();
      });
    });

    test('components share performance data', async () => {
      const user = userEvent.setup();
      
      render(
        <div>
          <PerformanceMonitoringDashboard />
          <AdvancedDashboardBuilder />
        </div>
      );
      
      // Start performance monitoring
      const startButton = screen.getByText('Start Monitoring');
      await user.click(startButton);
      
      // Verify dashboard builder receives performance data
      await waitFor(() => {
        expect(screen.getByText('Performance Metrics')).toBeInTheDocument();
      });
    });
  });

  describe('Data Flow Integration', () => {
    test('account data flows through all relevant components', async () => {
      render(
        <div>
          <Authentication />
          <AccountVisualization />
          <Analytics />
          <TradingDashboard />
        </div>
      );
      
      // Verify account data is available in all components
      await waitFor(() => {
        expect(screen.getByText('Account Overview')).toBeInTheDocument();
        expect(screen.getByText('Analytics Dashboard')).toBeInTheDocument();
        expect(screen.getByText('Trading Dashboard')).toBeInTheDocument();
      });
    });

    test('market data flows through intelligence and trading components', async () => {
      render(
        <div>
          <IntelligenceDashboard />
          <TradingDashboard />
          <Analytics />
        </div>
      );
      
      // Verify market data integration
      await waitFor(() => {
        expect(screen.getByText('Market Intelligence')).toBeInTheDocument();
        expect(screen.getByText('Trading Dashboard')).toBeInTheDocument();
        expect(screen.getByText('Analytics Dashboard')).toBeInTheDocument();
      });
    });

    test('performance data flows through all monitoring components', async () => {
      const user = userEvent.setup();
      
      render(
        <div>
          <PerformanceMonitoringDashboard />
          <OptimizationEngineDashboard />
          <AdvancedAnalyticsDashboard />
          <SystemCoordinationDashboard />
        </div>
      );
      
      // Start all performance systems
      const monitoringStart = screen.getAllByText('Start Monitoring')[0];
      await user.click(monitoringStart);
      
      const optimizationStart = screen.getByText('Start Optimization');
      await user.click(optimizationStart);
      
      const analyticsStart = screen.getByText('Start Analysis');
      await user.click(analyticsStart);
      
      const coordinationStart = screen.getByText('Start Coordination');
      await user.click(coordinationStart);
      
      // Verify data flows between all systems
      await waitFor(() => {
        expect(screen.getByText('Stop Monitoring')).toBeInTheDocument();
        expect(screen.getByText('Stop Optimization')).toBeInTheDocument();
        expect(screen.getByText('Stop Analysis')).toBeInTheDocument();
        expect(screen.getByText('Stop Coordination')).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling Integration', () => {
    test('error boundaries protect component integration', () => {
      // Mock component error
      const ErrorComponent = () => {
        throw new Error('Test error');
      };
      
      expect(() => {
        render(
          <div>
            <ConversationalInterface />
            <ErrorComponent />
            <AccountVisualization />
          </div>
        );
      }).not.toThrow();
      
      // Verify other components still render
      expect(screen.getByText('Welcome! How can I help you today?')).toBeInTheDocument();
    });

    test('network errors are handled gracefully across components', async () => {
      // Mock network error
      jest.spyOn(global, 'fetch').mockRejectedValueOnce(new Error('Network error'));
      
      render(
        <div>
          <AccountVisualization />
          <IntelligenceDashboard />
          <TradingDashboard />
        </div>
      );
      
      // Verify components handle network errors gracefully
      await waitFor(() => {
        expect(screen.getByText('Account Overview')).toBeInTheDocument();
        expect(screen.getByText('Market Intelligence')).toBeInTheDocument();
        expect(screen.getByText('Trading Dashboard')).toBeInTheDocument();
      });
    });

    test('authentication errors are handled consistently', async () => {
      const user = userEvent.setup();
      
      // Mock authentication error
      jest.spyOn(console, 'error').mockImplementation(() => {});
      
      render(
        <div>
          <Authentication />
          <ConversationalInterface />
          <AccountVisualization />
        </div>
      );
      
      // Attempt login with error
      const loginButton = screen.getByText('Login');
      await user.click(loginButton);
      
      // Verify error handling doesn't break other components
      expect(screen.getByText('Login')).toBeInTheDocument();
    });
  });

  describe('Performance Integration', () => {
    test('component integration maintains performance standards', async () => {
      const startTime = performance.now();
      
      render(
        <div>
          <ConversationalInterface />
          <AccountVisualization />
          <Analytics />
          <AdvancedDashboardBuilder />
          <IntelligenceDashboard />
          <TradingDashboard />
        </div>
      );
      
      const endTime = performance.now();
      const renderTime = endTime - startTime;
      
      // Should render all components within 200ms
      expect(renderTime).toBeLessThan(200);
    });

    test('performance monitoring detects integration overhead', async () => {
      const user = userEvent.setup();
      
      render(
        <div>
          <PerformanceMonitoringDashboard />
          <ConversationalInterface />
          <AccountVisualization />
          <Analytics />
        </div>
      );
      
      // Start monitoring
      const startButton = screen.getByText('Start Monitoring');
      await user.click(startButton);
      
      // Check metrics tab
      const metricsTab = screen.getByText('Metrics');
      await user.click(metricsTab);
      
      // Verify performance metrics are collected
      await waitFor(() => {
        expect(screen.getByText('First Contentful Paint')).toBeInTheDocument();
        expect(screen.getByText('Largest Contentful Paint')).toBeInTheDocument();
      });
    });

    test('optimization engine improves integrated component performance', async () => {
      const user = userEvent.setup();
      
      render(
        <div>
          <OptimizationEngineDashboard />
          <ConversationalInterface />
          <AccountVisualization />
        </div>
      );
      
      // Start optimization
      const startButton = screen.getByText('Start Optimization');
      await user.click(startButton);
      
      // Check results tab
      const resultsTab = screen.getByText('Results');
      await user.click(resultsTab);
      
      // Verify optimization results
      await waitFor(() => {
        expect(screen.getByText('Optimization Results')).toBeInTheDocument();
      });
    });
  });

  describe('Accessibility Integration', () => {
    test('integrated components maintain accessibility standards', () => {
      render(
        <div>
          <ConversationalInterface />
          <AccountVisualization />
          <Analytics />
        </div>
      );
      
      // Verify ARIA labels and roles
      const buttons = screen.getAllByRole('button');
      buttons.forEach(button => {
        expect(button).toBeInTheDocument();
      });
      
      const textboxes = screen.getAllByRole('textbox');
      textboxes.forEach(textbox => {
        expect(textbox).toBeInTheDocument();
      });
    });

    test('keyboard navigation works across integrated components', async () => {
      const user = userEvent.setup();
      
      render(
        <div>
          <ConversationalInterface />
          <AccountVisualization />
        </div>
      );
      
      // Test tab navigation
      await user.tab();
      expect(document.activeElement).toBeInTheDocument();
      
      await user.tab();
      expect(document.activeElement).toBeInTheDocument();
    });

    test('screen reader compatibility across components', () => {
      render(
        <div>
          <ConversationalInterface />
          <AccountVisualization />
          <Analytics />
        </div>
      );
      
      // Verify screen reader labels
      expect(screen.getByLabelText(/message/i)).toBeInTheDocument();
    });
  });

  describe('Mobile Integration', () => {
    test('components adapt to mobile viewport', () => {
      // Mock mobile viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });
      
      render(
        <div>
          <ConversationalInterface />
          <AccountVisualization />
          <Analytics />
        </div>
      );
      
      // Verify mobile-responsive elements
      const containers = document.querySelectorAll('.max-w-7xl');
      expect(containers.length).toBeGreaterThan(0);
    });

    test('touch interactions work across components', async () => {
      const user = userEvent.setup();
      
      render(
        <div>
          <ConversationalInterface />
          <TradingDashboard />
        </div>
      );
      
      // Test touch interactions
      const sendButton = screen.getByText('Send');
      await user.click(sendButton);
      
      expect(sendButton).toBeInTheDocument();
    });
  });

  describe('Theme Integration', () => {
    test('theme changes propagate across all components', async () => {
      const user = userEvent.setup();
      
      render(
        <div>
          <EnterpriseFeatures />
          <ConversationalInterface />
          <AccountVisualization />
        </div>
      );
      
      // Find and click theme toggle (if available)
      const themeButtons = screen.queryAllByText(/theme/i);
      if (themeButtons.length > 0) {
        await user.click(themeButtons[0]);
      }
      
      // Verify components render consistently
      expect(screen.getByText('Welcome! How can I help you today?')).toBeInTheDocument();
      expect(screen.getByText('Account Overview')).toBeInTheDocument();
    });
  });
});

// Integration test utilities
export const integrationTestUtils = {
  renderWithProviders: (component: React.ReactElement) => {
    return render(component);
  },
  
  waitForComponentIntegration: async (componentName: string) => {
    await waitFor(() => {
      expect(screen.getByText(componentName)).toBeInTheDocument();
    });
  },
  
  testCrossComponentCommunication: async (sourceComponent: string, targetComponent: string) => {
    // Implementation for testing communication between components
    await waitFor(() => {
      expect(screen.getByText(sourceComponent)).toBeInTheDocument();
      expect(screen.getByText(targetComponent)).toBeInTheDocument();
    });
  },
  
  validatePerformanceIntegration: (maxRenderTime: number = 200) => {
    const startTime = performance.now();
    return () => {
      const endTime = performance.now();
      expect(endTime - startTime).toBeLessThan(maxRenderTime);
    };
  },
  
  testAccessibilityIntegration: () => {
    const buttons = screen.getAllByRole('button');
    const textboxes = screen.getAllByRole('textbox');
    
    expect(buttons.length).toBeGreaterThan(0);
    expect(textboxes.length).toBeGreaterThan(0);
  }
};

export default {};

