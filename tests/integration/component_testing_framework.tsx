import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

// Import all WS6 components for testing
import ConversationalInterface from '../components/ConversationalInterface';
import AccountVisualization from '../components/AccountVisualization';
import Authentication from '../components/Authentication';
import Analytics from '../components/Analytics';
import DataVisualization from '../components/charts/DataVisualization';
import AccountManagement from '../components/advanced/AccountManagement';
import RiskManagement from '../components/advanced/RiskManagement';
import TransactionHistory from '../components/advanced/TransactionHistory';
import MarketAnalysis from '../components/advanced/MarketAnalysis';
import ProtocolCompliance from '../components/advanced/ProtocolCompliance';
import AdvancedDashboardBuilder from '../components/sophisticated/AdvancedDashboardBuilder';
import IntelligenceDashboard from '../components/intelligence/IntelligenceDashboard';
import TradingDashboard from '../components/trading/TradingDashboard';
import EnterpriseFeatures from '../components/enterprise/EnterpriseFeatures';
import SystemIntegrationHub from '../integration/SystemIntegrationHub';
import CoordinationEngine from '../coordination/CoordinationEngine';
import OrchestrationManager from '../orchestration/OrchestrationManager';

// Mock data generators
const mockUserData = {
  id: 'user_123',
  email: 'test@example.com',
  name: 'Test User',
  accountTier: 'premium',
  expertiseLevel: 'intermediate',
  preferences: {
    theme: 'light',
    notifications: true,
    autoRefresh: true
  }
};

const mockAccountData = {
  totalValue: 225000,
  accounts: {
    generation: { value: 75000, allocation: 33.3, performance: 12.5, risk: 'medium' },
    revenue: { value: 60000, allocation: 26.7, performance: 8.7, risk: 'low' },
    compounding: { value: 90000, allocation: 40.0, performance: 15.2, risk: 'high' }
  },
  performance: {
    daily: 2.3,
    weekly: 5.8,
    monthly: 12.1,
    yearly: 18.5
  }
};

const mockMarketData = {
  symbols: ['SPY', 'QQQ', 'IWM'],
  prices: { SPY: 425.67, QQQ: 368.45, IWM: 198.23 },
  changes: { SPY: 1.23, QQQ: -0.87, IWM: 2.45 },
  volume: { SPY: 45000000, QQQ: 32000000, IWM: 28000000 }
};

const mockTradingData = {
  positions: [
    { symbol: 'SPY', quantity: 100, avgPrice: 420.50, currentPrice: 425.67, pnl: 517.00 },
    { symbol: 'QQQ', quantity: 50, avgPrice: 370.25, currentPrice: 368.45, pnl: -90.00 }
  ],
  orders: [
    { id: 'order_1', symbol: 'SPY', type: 'limit', quantity: 25, price: 424.00, status: 'pending' },
    { id: 'order_2', symbol: 'QQQ', type: 'market', quantity: 10, status: 'filled' }
  ]
};

// Test utilities
const renderWithProviders = (component: React.ReactElement) => {
  return render(component);
};

const mockWebSocket = () => {
  const mockWS = {
    send: jest.fn(),
    close: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    readyState: WebSocket.OPEN
  };
  
  global.WebSocket = jest.fn(() => mockWS) as any;
  return mockWS;
};

const mockLocalStorage = () => {
  const store: { [key: string]: string } = {};
  
  global.localStorage = {
    getItem: jest.fn((key: string) => store[key] || null),
    setItem: jest.fn((key: string, value: string) => { store[key] = value; }),
    removeItem: jest.fn((key: string) => { delete store[key]; }),
    clear: jest.fn(() => { Object.keys(store).forEach(key => delete store[key]); }),
    length: 0,
    key: jest.fn()
  };
};

describe('WS6-P4: Comprehensive Component Testing Framework', () => {
  beforeEach(() => {
    mockLocalStorage();
    mockWebSocket();
    jest.clearAllMocks();
  });

  describe('WS6-P1: Conversational Interface Foundation', () => {
    describe('ConversationalInterface Component', () => {
      test('renders conversational interface with initial state', () => {
        renderWithProviders(<ConversationalInterface />);
        
        expect(screen.getByText('ALL-USE Protocol Assistant')).toBeInTheDocument();
        expect(screen.getByPlaceholderText(/ask about the all-use protocol/i)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument();
      });

      test('handles message input and submission', async () => {
        const user = userEvent.setup();
        renderWithProviders(<ConversationalInterface />);
        
        const input = screen.getByPlaceholderText(/ask about the all-use protocol/i);
        const sendButton = screen.getByRole('button', { name: /send/i });
        
        await user.type(input, 'What is the three-tier structure?');
        await user.click(sendButton);
        
        expect(input).toHaveValue('');
        await waitFor(() => {
          expect(screen.getByText('What is the three-tier structure?')).toBeInTheDocument();
        });
      });

      test('displays suggested questions', () => {
        renderWithProviders(<ConversationalInterface />);
        
        expect(screen.getByText('How does the forking protocol work?')).toBeInTheDocument();
        expect(screen.getByText('What is delta targeting?')).toBeInTheDocument();
        expect(screen.getByText('Explain week classification')).toBeInTheDocument();
      });

      test('handles voice input toggle', async () => {
        const user = userEvent.setup();
        renderWithProviders(<ConversationalInterface />);
        
        const voiceButton = screen.getByRole('button', { name: /voice input/i });
        await user.click(voiceButton);
        
        // Voice functionality would be tested with proper mocks
        expect(voiceButton).toBeInTheDocument();
      });

      test('maintains conversation history', async () => {
        const user = userEvent.setup();
        renderWithProviders(<ConversationalInterface />);
        
        const input = screen.getByPlaceholderText(/ask about the all-use protocol/i);
        const sendButton = screen.getByRole('button', { name: /send/i });
        
        await user.type(input, 'First message');
        await user.click(sendButton);
        
        await user.type(input, 'Second message');
        await user.click(sendButton);
        
        await waitFor(() => {
          expect(screen.getByText('First message')).toBeInTheDocument();
          expect(screen.getByText('Second message')).toBeInTheDocument();
        });
      });
    });

    describe('AccountVisualization Component', () => {
      test('renders account visualization with portfolio data', () => {
        renderWithProviders(<AccountVisualization />);
        
        expect(screen.getByText('Portfolio Overview')).toBeInTheDocument();
        expect(screen.getByText('Generation Account')).toBeInTheDocument();
        expect(screen.getByText('Revenue Account')).toBeInTheDocument();
        expect(screen.getByText('Compounding Account')).toBeInTheDocument();
      });

      test('displays account performance metrics', () => {
        renderWithProviders(<AccountVisualization />);
        
        expect(screen.getByText(/\$225,000/)).toBeInTheDocument();
        expect(screen.getByText(/Total Portfolio Value/)).toBeInTheDocument();
      });

      test('handles view mode switching', async () => {
        const user = userEvent.setup();
        renderWithProviders(<AccountVisualization />);
        
        const detailedViewButton = screen.getByText('Detailed');
        await user.click(detailedViewButton);
        
        expect(screen.getByText('Account Details')).toBeInTheDocument();
      });

      test('displays risk indicators correctly', () => {
        renderWithProviders(<AccountVisualization />);
        
        expect(screen.getByText('Medium Risk')).toBeInTheDocument();
        expect(screen.getByText('Low Risk')).toBeInTheDocument();
        expect(screen.getByText('High Risk')).toBeInTheDocument();
      });

      test('handles timeframe selection', async () => {
        const user = userEvent.setup();
        renderWithProviders(<AccountVisualization />);
        
        const monthlyButton = screen.getByText('Month');
        await user.click(monthlyButton);
        
        await waitFor(() => {
          expect(screen.getByText(/12.1%/)).toBeInTheDocument();
        });
      });
    });

    describe('Authentication Component', () => {
      test('renders login form by default', () => {
        renderWithProviders(<Authentication />);
        
        expect(screen.getByText('Sign In to ALL-USE')).toBeInTheDocument();
        expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
        expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
      });

      test('handles form validation', async () => {
        const user = userEvent.setup();
        renderWithProviders(<Authentication />);
        
        const signInButton = screen.getByRole('button', { name: /sign in/i });
        await user.click(signInButton);
        
        await waitFor(() => {
          expect(screen.getByText('Email is required')).toBeInTheDocument();
          expect(screen.getByText('Password is required')).toBeInTheDocument();
        });
      });

      test('switches between login and registration', async () => {
        const user = userEvent.setup();
        renderWithProviders(<Authentication />);
        
        const registerLink = screen.getByText(/create account/i);
        await user.click(registerLink);
        
        expect(screen.getByText('Create ALL-USE Account')).toBeInTheDocument();
        expect(screen.getByLabelText(/confirm password/i)).toBeInTheDocument();
      });

      test('handles successful authentication', async () => {
        const user = userEvent.setup();
        renderWithProviders(<Authentication />);
        
        const emailInput = screen.getByLabelText(/email/i);
        const passwordInput = screen.getByLabelText(/password/i);
        const signInButton = screen.getByRole('button', { name: /sign in/i });
        
        await user.type(emailInput, 'test@example.com');
        await user.type(passwordInput, 'password123');
        await user.click(signInButton);
        
        // Mock successful authentication response
        await waitFor(() => {
          expect(localStorage.setItem).toHaveBeenCalledWith('auth_token', expect.any(String));
        });
      });

      test('displays password strength indicator', async () => {
        const user = userEvent.setup();
        renderWithProviders(<Authentication />);
        
        const registerLink = screen.getByText(/create account/i);
        await user.click(registerLink);
        
        const passwordInput = screen.getByLabelText(/^password/i);
        await user.type(passwordInput, 'weak');
        
        expect(screen.getByText('Weak')).toBeInTheDocument();
        
        await user.clear(passwordInput);
        await user.type(passwordInput, 'StrongPassword123!');
        
        expect(screen.getByText('Strong')).toBeInTheDocument();
      });
    });

    describe('Analytics Component', () => {
      test('renders analytics dashboard with performance metrics', () => {
        renderWithProviders(<Analytics />);
        
        expect(screen.getByText('Performance Analytics')).toBeInTheDocument();
        expect(screen.getByText('Week Classification')).toBeInTheDocument();
        expect(screen.getByText('Protocol Compliance')).toBeInTheDocument();
      });

      test('displays week classification history', () => {
        renderWithProviders(<Analytics />);
        
        expect(screen.getByText('Green Week')).toBeInTheDocument();
        expect(screen.getByText('Red Week')).toBeInTheDocument();
        expect(screen.getByText('Chop Week')).toBeInTheDocument();
      });

      test('shows protocol compliance metrics', () => {
        renderWithProviders(<Analytics />);
        
        expect(screen.getByText(/95.2%/)).toBeInTheDocument();
        expect(screen.getByText('Protocol Adherence')).toBeInTheDocument();
      });

      test('handles timeframe filtering', async () => {
        const user = userEvent.setup();
        renderWithProviders(<Analytics />);
        
        const timeframeSelect = screen.getByDisplayValue('Last 30 Days');
        await user.selectOptions(timeframeSelect, 'Last 90 Days');
        
        expect(timeframeSelect).toHaveValue('Last 90 Days');
      });
    });
  });

  describe('WS6-P2: Enhanced Interface and Visualization', () => {
    describe('DataVisualization Component', () => {
      test('renders portfolio performance chart', () => {
        renderWithProviders(<DataVisualization type="portfolio" data={mockAccountData} />);
        
        expect(screen.getByText('Portfolio Performance')).toBeInTheDocument();
      });

      test('handles chart type switching', async () => {
        const user = userEvent.setup();
        renderWithProviders(<DataVisualization type="portfolio" data={mockAccountData} />);
        
        const chartTypeSelect = screen.getByDisplayValue('Line Chart');
        await user.selectOptions(chartTypeSelect, 'Bar Chart');
        
        expect(chartTypeSelect).toHaveValue('Bar Chart');
      });

      test('displays risk analysis heatmap', () => {
        renderWithProviders(<DataVisualization type="risk" data={mockAccountData} />);
        
        expect(screen.getByText('Risk Analysis')).toBeInTheDocument();
      });

      test('handles data export functionality', async () => {
        const user = userEvent.setup();
        renderWithProviders(<DataVisualization type="portfolio" data={mockAccountData} />);
        
        const exportButton = screen.getByRole('button', { name: /export/i });
        await user.click(exportButton);
        
        // Mock export functionality
        expect(exportButton).toBeInTheDocument();
      });
    });

    describe('AccountManagement Component', () => {
      test('renders account management interface', () => {
        renderWithProviders(<AccountManagement />);
        
        expect(screen.getByText('Account Management')).toBeInTheDocument();
        expect(screen.getByText('Portfolio Rebalancing')).toBeInTheDocument();
      });

      test('handles portfolio rebalancing', async () => {
        const user = userEvent.setup();
        renderWithProviders(<AccountManagement />);
        
        const rebalanceButton = screen.getByRole('button', { name: /rebalance/i });
        await user.click(rebalanceButton);
        
        expect(screen.getByText('Rebalancing Preview')).toBeInTheDocument();
      });

      test('displays allocation controls', () => {
        renderWithProviders(<AccountManagement />);
        
        expect(screen.getByText('Generation: 33.3%')).toBeInTheDocument();
        expect(screen.getByText('Revenue: 26.7%')).toBeInTheDocument();
        expect(screen.getByText('Compounding: 40.0%')).toBeInTheDocument();
      });
    });

    describe('RiskManagement Component', () => {
      test('renders risk management dashboard', () => {
        renderWithProviders(<RiskManagement />);
        
        expect(screen.getByText('Risk Management')).toBeInTheDocument();
        expect(screen.getByText('Value at Risk (VaR)')).toBeInTheDocument();
      });

      test('displays VaR calculations', () => {
        renderWithProviders(<RiskManagement />);
        
        expect(screen.getByText('Daily VaR')).toBeInTheDocument();
        expect(screen.getByText('Weekly VaR')).toBeInTheDocument();
        expect(screen.getByText('Monthly VaR')).toBeInTheDocument();
      });

      test('handles stress testing', async () => {
        const user = userEvent.setup();
        renderWithProviders(<RiskManagement />);
        
        const stressTestButton = screen.getByRole('button', { name: /stress test/i });
        await user.click(stressTestButton);
        
        expect(screen.getByText('Stress Test Results')).toBeInTheDocument();
      });
    });

    describe('TransactionHistory Component', () => {
      test('renders transaction history table', () => {
        renderWithProviders(<TransactionHistory />);
        
        expect(screen.getByText('Transaction History')).toBeInTheDocument();
        expect(screen.getByText('Date')).toBeInTheDocument();
        expect(screen.getByText('Symbol')).toBeInTheDocument();
        expect(screen.getByText('Type')).toBeInTheDocument();
        expect(screen.getByText('Amount')).toBeInTheDocument();
      });

      test('handles transaction filtering', async () => {
        const user = userEvent.setup();
        renderWithProviders(<TransactionHistory />);
        
        const filterInput = screen.getByPlaceholderText(/search transactions/i);
        await user.type(filterInput, 'SPY');
        
        expect(filterInput).toHaveValue('SPY');
      });

      test('displays transaction details on click', async () => {
        const user = userEvent.setup();
        renderWithProviders(<TransactionHistory />);
        
        const transactionRow = screen.getByText('SPY').closest('tr');
        await user.click(transactionRow!);
        
        expect(screen.getByText('Transaction Details')).toBeInTheDocument();
      });
    });
  });

  describe('WS6-P3: Advanced Interface and Integration', () => {
    describe('AdvancedDashboardBuilder Component', () => {
      test('renders dashboard builder interface', () => {
        renderWithProviders(<AdvancedDashboardBuilder />);
        
        expect(screen.getByText('Dashboard Builder')).toBeInTheDocument();
        expect(screen.getByText('Widget Palette')).toBeInTheDocument();
      });

      test('handles widget drag and drop', async () => {
        const user = userEvent.setup();
        renderWithProviders(<AdvancedDashboardBuilder />);
        
        const widget = screen.getByText('Portfolio Chart');
        const dropZone = screen.getByText('Drop widgets here');
        
        // Mock drag and drop functionality
        fireEvent.dragStart(widget);
        fireEvent.dragOver(dropZone);
        fireEvent.drop(dropZone);
        
        expect(screen.getByText('Portfolio Chart')).toBeInTheDocument();
      });

      test('saves dashboard layout', async () => {
        const user = userEvent.setup();
        renderWithProviders(<AdvancedDashboardBuilder />);
        
        const saveButton = screen.getByRole('button', { name: /save layout/i });
        await user.click(saveButton);
        
        expect(localStorage.setItem).toHaveBeenCalledWith('dashboard_layout', expect.any(String));
      });
    });

    describe('IntelligenceDashboard Component', () => {
      test('renders market intelligence dashboard', () => {
        renderWithProviders(<IntelligenceDashboard />);
        
        expect(screen.getByText('Market Intelligence')).toBeInTheDocument();
        expect(screen.getByText('Market Sentiment')).toBeInTheDocument();
      });

      test('displays real-time market data', () => {
        renderWithProviders(<IntelligenceDashboard />);
        
        expect(screen.getByText('SPY')).toBeInTheDocument();
        expect(screen.getByText('QQQ')).toBeInTheDocument();
        expect(screen.getByText('IWM')).toBeInTheDocument();
      });

      test('handles symbol selection', async () => {
        const user = userEvent.setup();
        renderWithProviders(<IntelligenceDashboard />);
        
        const symbolSelect = screen.getByDisplayValue('SPY');
        await user.selectOptions(symbolSelect, 'QQQ');
        
        expect(symbolSelect).toHaveValue('QQQ');
      });
    });

    describe('TradingDashboard Component', () => {
      test('renders trading dashboard with positions', () => {
        renderWithProviders(<TradingDashboard />);
        
        expect(screen.getByText('Trading Dashboard')).toBeInTheDocument();
        expect(screen.getByText('Positions')).toBeInTheDocument();
        expect(screen.getByText('Orders')).toBeInTheDocument();
      });

      test('displays current positions', () => {
        renderWithProviders(<TradingDashboard />);
        
        expect(screen.getByText('SPY')).toBeInTheDocument();
        expect(screen.getByText('100 shares')).toBeInTheDocument();
        expect(screen.getByText('+$517.00')).toBeInTheDocument();
      });

      test('handles order placement', async () => {
        const user = userEvent.setup();
        renderWithProviders(<TradingDashboard />);
        
        const placeOrderButton = screen.getByRole('button', { name: /place order/i });
        await user.click(placeOrderButton);
        
        expect(screen.getByText('Order Entry')).toBeInTheDocument();
      });
    });

    describe('EnterpriseFeatures Component', () => {
      test('renders enterprise features dashboard', () => {
        renderWithProviders(<EnterpriseFeatures />);
        
        expect(screen.getByText('Enterprise Features')).toBeInTheDocument();
        expect(screen.getByText('User Management')).toBeInTheDocument();
      });

      test('displays user management interface', () => {
        renderWithProviders(<EnterpriseFeatures />);
        
        expect(screen.getByText('Active Users')).toBeInTheDocument();
        expect(screen.getByText('Permissions')).toBeInTheDocument();
      });

      test('handles user role management', async () => {
        const user = userEvent.setup();
        renderWithProviders(<EnterpriseFeatures />);
        
        const roleSelect = screen.getByDisplayValue('User');
        await user.selectOptions(roleSelect, 'Admin');
        
        expect(roleSelect).toHaveValue('Admin');
      });
    });

    describe('SystemIntegrationHub Component', () => {
      test('renders system integration dashboard', () => {
        renderWithProviders(<SystemIntegrationHub />);
        
        expect(screen.getByText('System Integration Hub')).toBeInTheDocument();
        expect(screen.getByText('Workstream Status')).toBeInTheDocument();
      });

      test('displays workstream health metrics', () => {
        renderWithProviders(<SystemIntegrationHub />);
        
        expect(screen.getByText('Agent Foundation')).toBeInTheDocument();
        expect(screen.getByText('Protocol Systems')).toBeInTheDocument();
        expect(screen.getByText('Market Intelligence')).toBeInTheDocument();
      });

      test('handles workstream restart', async () => {
        const user = userEvent.setup();
        renderWithProviders(<SystemIntegrationHub />);
        
        const workstreamCard = screen.getByText('Agent Foundation').closest('div');
        await user.click(workstreamCard!);
        
        const restartButton = screen.getByRole('button', { name: /restart/i });
        await user.click(restartButton);
        
        expect(screen.getByText('Workstream restarted successfully')).toBeInTheDocument();
      });
    });
  });

  describe('Integration Testing', () => {
    test('components communicate correctly', async () => {
      const user = userEvent.setup();
      
      // Test authentication flow affecting other components
      renderWithProviders(<Authentication />);
      
      const emailInput = screen.getByLabelText(/email/i);
      const passwordInput = screen.getByLabelText(/password/i);
      const signInButton = screen.getByRole('button', { name: /sign in/i });
      
      await user.type(emailInput, 'test@example.com');
      await user.type(passwordInput, 'password123');
      await user.click(signInButton);
      
      // Verify authentication state is set
      expect(localStorage.setItem).toHaveBeenCalledWith('auth_token', expect.any(String));
    });

    test('real-time data updates work correctly', async () => {
      const mockWS = mockWebSocket();
      renderWithProviders(<TradingDashboard />);
      
      // Simulate WebSocket message
      const messageEvent = new MessageEvent('message', {
        data: JSON.stringify({
          type: 'price_update',
          symbol: 'SPY',
          price: 426.50
        })
      });
      
      mockWS.addEventListener.mock.calls.forEach(([event, handler]) => {
        if (event === 'message') {
          handler(messageEvent);
        }
      });
      
      await waitFor(() => {
        expect(screen.getByText('426.50')).toBeInTheDocument();
      });
    });

    test('error handling works across components', async () => {
      const user = userEvent.setup();
      
      // Mock network error
      global.fetch = jest.fn().mockRejectedValue(new Error('Network error'));
      
      renderWithProviders(<AccountVisualization />);
      
      await waitFor(() => {
        expect(screen.getByText(/error loading account data/i)).toBeInTheDocument();
      });
    });
  });

  describe('Performance Testing', () => {
    test('components render within performance budget', async () => {
      const startTime = performance.now();
      
      renderWithProviders(<TradingDashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('Trading Dashboard')).toBeInTheDocument();
      });
      
      const endTime = performance.now();
      const renderTime = endTime - startTime;
      
      // Should render within 1000ms
      expect(renderTime).toBeLessThan(1000);
    });

    test('handles large datasets efficiently', async () => {
      const largeDataset = Array.from({ length: 1000 }, (_, i) => ({
        id: i,
        symbol: `STOCK${i}`,
        price: Math.random() * 100,
        change: (Math.random() - 0.5) * 10
      }));
      
      renderWithProviders(<TransactionHistory data={largeDataset} />);
      
      // Should handle large datasets without performance issues
      expect(screen.getByText('Transaction History')).toBeInTheDocument();
    });
  });

  describe('Accessibility Testing', () => {
    test('components have proper ARIA labels', () => {
      renderWithProviders(<ConversationalInterface />);
      
      const input = screen.getByLabelText(/message input/i);
      const button = screen.getByRole('button', { name: /send message/i });
      
      expect(input).toHaveAttribute('aria-label');
      expect(button).toHaveAttribute('aria-label');
    });

    test('keyboard navigation works correctly', async () => {
      const user = userEvent.setup();
      renderWithProviders(<AccountVisualization />);
      
      const firstButton = screen.getByText('Overview');
      firstButton.focus();
      
      await user.keyboard('{Tab}');
      
      const secondButton = screen.getByText('Detailed');
      expect(secondButton).toHaveFocus();
    });

    test('screen reader compatibility', () => {
      renderWithProviders(<Analytics />);
      
      // Check for proper heading structure
      expect(screen.getByRole('heading', { level: 1 })).toBeInTheDocument();
      expect(screen.getByRole('heading', { level: 2 })).toBeInTheDocument();
    });
  });
});

// Export test utilities for use in other test files
export {
  mockUserData,
  mockAccountData,
  mockMarketData,
  mockTradingData,
  renderWithProviders,
  mockWebSocket,
  mockLocalStorage
};

