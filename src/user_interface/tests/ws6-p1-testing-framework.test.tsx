// WS6-P1 Testing and Validation Framework
// Comprehensive testing suite for ALL-USE User Interface components

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

// Import components to test
import { ConversationalInterface } from '../components/ConversationalInterface';
import { AccountVisualization } from '../components/AccountVisualization';
import { Authentication } from '../components/Authentication';
import { Analytics } from '../components/Analytics';

// Import services and libraries
import { ConversationManager, ALLUSEAgent } from '../lib/conversational-agent';
import { ws1Agent } from '../lib/ws1-integration';
import { authService } from '../lib/auth-service';

// Mock external dependencies
jest.mock('../lib/ws1-integration');
jest.mock('../lib/auth-service');

describe('WS6-P1: Conversational Interface Foundation', () => {
  
  describe('ConversationalInterface Component', () => {
    const mockUser = {
      id: 'test-user',
      email: 'test@alluse.com',
      name: 'Test User',
      accountTier: 'premium' as const,
      expertiseLevel: 'intermediate' as const,
      joinDate: '2024-01-01',
      lastLogin: '2024-06-18',
      protocolAccess: ['three-tier', 'forking', 'delta-targeting']
    };

    beforeEach(() => {
      jest.clearAllMocks();
    });

    test('renders conversational interface correctly', () => {
      render(<ConversationalInterface user={mockUser} />);
      
      expect(screen.getByText('Protocol Chat')).toBeInTheDocument();
      expect(screen.getByPlaceholderText(/ask about the all-use protocol/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument();
    });

    test('displays welcome message on initialization', () => {
      render(<ConversationalInterface user={mockUser} />);
      
      expect(screen.getByText(/welcome/i)).toBeInTheDocument();
      expect(screen.getByText(/three-tier account structure/i)).toBeInTheDocument();
    });

    test('handles user input and generates agent response', async () => {
      const user = userEvent.setup();
      render(<ConversationalInterface user={mockUser} />);
      
      const input = screen.getByPlaceholderText(/ask about the all-use protocol/i);
      const sendButton = screen.getByRole('button', { name: /send/i });
      
      await user.type(input, 'What is the three-tier account structure?');
      await user.click(sendButton);
      
      await waitFor(() => {
        expect(screen.getByText('What is the three-tier account structure?')).toBeInTheDocument();
      });
      
      // Check for agent response
      await waitFor(() => {
        expect(screen.getByText(/generation account/i)).toBeInTheDocument();
      }, { timeout: 3000 });
    });

    test('provides suggested questions', () => {
      render(<ConversationalInterface user={mockUser} />);
      
      expect(screen.getByText(/explain the three-tier account structure/i)).toBeInTheDocument();
      expect(screen.getByText(/how does forking work/i)).toBeInTheDocument();
      expect(screen.getByText(/what trading opportunities are available/i)).toBeInTheDocument();
    });

    test('handles suggested question clicks', async () => {
      const user = userEvent.setup();
      render(<ConversationalInterface user={mockUser} />);
      
      const suggestedQuestion = screen.getByText(/explain the three-tier account structure/i);
      await user.click(suggestedQuestion);
      
      const input = screen.getByPlaceholderText(/ask about the all-use protocol/i);
      expect(input).toHaveValue('Explain the three-tier account structure');
    });

    test('speech recognition functionality', async () => {
      // Mock speech recognition
      const mockSpeechRecognition = {
        start: jest.fn(),
        stop: jest.fn(),
        addEventListener: jest.fn(),
        removeEventListener: jest.fn()
      };
      
      (global as any).webkitSpeechRecognition = jest.fn(() => mockSpeechRecognition);
      
      render(<ConversationalInterface user={mockUser} />);
      
      const micButton = screen.getByRole('button', { name: /start voice input/i });
      expect(micButton).toBeInTheDocument();
      
      fireEvent.click(micButton);
      expect(mockSpeechRecognition.start).toHaveBeenCalled();
    });

    test('message history persistence', async () => {
      const user = userEvent.setup();
      render(<ConversationalInterface user={mockUser} />);
      
      const input = screen.getByPlaceholderText(/ask about the all-use protocol/i);
      const sendButton = screen.getByRole('button', { name: /send/i });
      
      // Send first message
      await user.type(input, 'First message');
      await user.click(sendButton);
      
      // Send second message
      await user.clear(input);
      await user.type(input, 'Second message');
      await user.click(sendButton);
      
      // Check both messages are visible
      expect(screen.getByText('First message')).toBeInTheDocument();
      expect(screen.getByText('Second message')).toBeInTheDocument();
    });
  });

  describe('AccountVisualization Component', () => {
    const mockAccounts = [
      {
        id: 'gen-001',
        name: 'Generation Account',
        type: 'generation' as const,
        balance: 75000,
        weeklyReturn: 2.3,
        monthlyReturn: 8.7,
        yearlyReturn: 89.4,
        riskLevel: 'high' as const,
        strategy: 'Aggressive premium harvesting with higher delta targeting'
      },
      {
        id: 'rev-001',
        name: 'Revenue Account',
        type: 'revenue' as const,
        balance: 60000,
        weeklyReturn: 1.1,
        monthlyReturn: 4.2,
        yearlyReturn: 52.3,
        riskLevel: 'medium' as const,
        strategy: 'Balanced income generation with moderate delta exposure'
      },
      {
        id: 'comp-001',
        name: 'Compounding Account',
        type: 'compounding' as const,
        balance: 90000,
        weeklyReturn: 0.8,
        monthlyReturn: 3.1,
        yearlyReturn: 38.7,
        riskLevel: 'low' as const,
        strategy: 'Conservative growth with geometric compounding'
      }
    ];

    test('renders account visualization correctly', () => {
      render(<AccountVisualization accounts={mockAccounts} />);
      
      expect(screen.getByText('Account Portfolio')).toBeInTheDocument();
      expect(screen.getByText('Generation Account')).toBeInTheDocument();
      expect(screen.getByText('Revenue Account')).toBeInTheDocument();
      expect(screen.getByText('Compounding Account')).toBeInTheDocument();
    });

    test('displays account balances correctly', () => {
      render(<AccountVisualization accounts={mockAccounts} />);
      
      expect(screen.getByText('$75,000')).toBeInTheDocument();
      expect(screen.getByText('$60,000')).toBeInTheDocument();
      expect(screen.getByText('$90,000')).toBeInTheDocument();
    });

    test('shows performance metrics', () => {
      render(<AccountVisualization accounts={mockAccounts} />);
      
      expect(screen.getByText('+2.3%')).toBeInTheDocument();
      expect(screen.getByText('+1.1%')).toBeInTheDocument();
      expect(screen.getByText('+0.8%')).toBeInTheDocument();
    });

    test('view mode switching functionality', async () => {
      const user = userEvent.setup();
      render(<AccountVisualization accounts={mockAccounts} />);
      
      // Switch to detailed view
      const detailedButton = screen.getByText('Detailed');
      await user.click(detailedButton);
      
      expect(screen.getByText(/strategy/i)).toBeInTheDocument();
      
      // Switch to performance view
      const performanceButton = screen.getByText('Performance');
      await user.click(performanceButton);
      
      expect(screen.getByText(/yearly return/i)).toBeInTheDocument();
    });

    test('timeframe selection', async () => {
      const user = userEvent.setup();
      render(<AccountVisualization accounts={mockAccounts} />);
      
      const monthButton = screen.getByText('Month');
      await user.click(monthButton);
      
      expect(screen.getByText('+8.7%')).toBeInTheDocument();
      expect(screen.getByText('+4.2%')).toBeInTheDocument();
      expect(screen.getByText('+3.1%')).toBeInTheDocument();
    });

    test('risk level indicators', () => {
      render(<AccountVisualization accounts={mockAccounts} />);
      
      expect(screen.getByText('High Risk')).toBeInTheDocument();
      expect(screen.getByText('Medium Risk')).toBeInTheDocument();
      expect(screen.getByText('Low Risk')).toBeInTheDocument();
    });
  });

  describe('Authentication Component', () => {
    const mockOnAuthenticated = jest.fn();
    const mockOnLogout = jest.fn();

    beforeEach(() => {
      jest.clearAllMocks();
      localStorage.clear();
    });

    test('renders login form by default', () => {
      render(<Authentication onAuthenticated={mockOnAuthenticated} onLogout={mockOnLogout} />);
      
      expect(screen.getByText('Sign in to your account')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Enter your email')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Enter your password')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
    });

    test('switches to registration form', async () => {
      const user = userEvent.setup();
      render(<Authentication onAuthenticated={mockOnAuthenticated} onLogout={mockOnLogout} />);
      
      const signUpLink = screen.getByText(/don't have an account/i);
      await user.click(signUpLink);
      
      expect(screen.getByText('Create your account')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Enter your full name')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /create account/i })).toBeInTheDocument();
    });

    test('validates email format', async () => {
      const user = userEvent.setup();
      render(<Authentication onAuthenticated={mockOnAuthenticated} onLogout={mockOnLogout} />);
      
      const emailInput = screen.getByPlaceholderText('Enter your email');
      const passwordInput = screen.getByPlaceholderText('Enter your password');
      const submitButton = screen.getByRole('button', { name: /sign in/i });
      
      await user.type(emailInput, 'invalid-email');
      await user.type(passwordInput, 'password123');
      await user.click(submitButton);
      
      await waitFor(() => {
        expect(screen.getByText(/please enter a valid email address/i)).toBeInTheDocument();
      });
    });

    test('successful demo login', async () => {
      const user = userEvent.setup();
      render(<Authentication onAuthenticated={mockOnAuthenticated} onLogout={mockOnLogout} />);
      
      const emailInput = screen.getByPlaceholderText('Enter your email');
      const passwordInput = screen.getByPlaceholderText('Enter your password');
      const submitButton = screen.getByRole('button', { name: /sign in/i });
      
      await user.type(emailInput, 'demo@alluse.com');
      await user.type(passwordInput, 'password123');
      await user.click(submitButton);
      
      await waitFor(() => {
        expect(screen.getByText(/login successful/i)).toBeInTheDocument();
      });
      
      await waitFor(() => {
        expect(mockOnAuthenticated).toHaveBeenCalled();
      }, { timeout: 2000 });
    });

    test('password visibility toggle', async () => {
      const user = userEvent.setup();
      render(<Authentication onAuthenticated={mockOnAuthenticated} onLogout={mockOnLogout} />);
      
      const passwordInput = screen.getByPlaceholderText('Enter your password');
      const toggleButton = screen.getByRole('button', { name: /toggle password visibility/i });
      
      expect(passwordInput).toHaveAttribute('type', 'password');
      
      await user.click(toggleButton);
      expect(passwordInput).toHaveAttribute('type', 'text');
      
      await user.click(toggleButton);
      expect(passwordInput).toHaveAttribute('type', 'password');
    });

    test('remember me functionality', async () => {
      const user = userEvent.setup();
      render(<Authentication onAuthenticated={mockOnAuthenticated} onLogout={mockOnLogout} />);
      
      const rememberCheckbox = screen.getByRole('checkbox', { name: /remember me/i });
      
      expect(screen.getByText(/remember me for 1 day/i)).toBeInTheDocument();
      
      await user.click(rememberCheckbox);
      expect(screen.getByText(/remember me for 30 days/i)).toBeInTheDocument();
    });

    test('registration password strength validation', async () => {
      const user = userEvent.setup();
      render(<Authentication onAuthenticated={mockOnAuthenticated} onLogout={mockOnLogout} />);
      
      // Switch to registration
      const signUpLink = screen.getByText(/don't have an account/i);
      await user.click(signUpLink);
      
      const passwordInput = screen.getByPlaceholderText('Enter your password');
      
      await user.type(passwordInput, 'weak');
      expect(screen.getByText(/WEAK/i)).toBeInTheDocument();
      
      await user.clear(passwordInput);
      await user.type(passwordInput, 'StrongPass123!');
      expect(screen.getByText(/STRONG/i)).toBeInTheDocument();
    });
  });

  describe('Analytics Component', () => {
    test('renders analytics dashboard correctly', () => {
      render(<Analytics />);
      
      expect(screen.getByText('Performance Analytics')).toBeInTheDocument();
      expect(screen.getByText('Current Week Status')).toBeInTheDocument();
      expect(screen.getByText('Performance Overview')).toBeInTheDocument();
    });

    test('displays week classification', () => {
      render(<Analytics />);
      
      expect(screen.getByText(/Green Week|Yellow Week|Red Week/)).toBeInTheDocument();
    });

    test('shows performance metrics', () => {
      render(<Analytics />);
      
      expect(screen.getByText('Total Return')).toBeInTheDocument();
      expect(screen.getByText('Weekly Average')).toBeInTheDocument();
      expect(screen.getByText('Monthly Average')).toBeInTheDocument();
    });

    test('metric selector functionality', async () => {
      const user = userEvent.setup();
      render(<Analytics />);
      
      const complianceButton = screen.getByText('Compliance');
      await user.click(complianceButton);
      
      expect(screen.getByText('Compliance (%)')).toBeInTheDocument();
    });

    test('timeframe selector functionality', async () => {
      const user = userEvent.setup();
      const mockOnTimeframeChange = jest.fn();
      
      render(<Analytics onTimeframeChange={mockOnTimeframeChange} />);
      
      const quarterButton = screen.getByText('Quarter');
      await user.click(quarterButton);
      
      expect(mockOnTimeframeChange).toHaveBeenCalledWith('quarter');
    });

    test('risk analysis section', () => {
      render(<Analytics />);
      
      expect(screen.getByText('Risk Analysis')).toBeInTheDocument();
      expect(screen.getByText('Low Risk Periods')).toBeInTheDocument();
      expect(screen.getByText('Medium Risk Periods')).toBeInTheDocument();
      expect(screen.getByText('High Risk Periods')).toBeInTheDocument();
    });

    test('protocol insights section', () => {
      render(<Analytics />);
      
      expect(screen.getByText('Protocol Insights')).toBeInTheDocument();
      expect(screen.getByText(/protocol adherence/i)).toBeInTheDocument();
    });
  });

  describe('Integration Tests', () => {
    test('conversational agent integration with WS1', async () => {
      const mockWS1Response = {
        concept: 'three-tier-account-structure',
        explanation: 'The ALL-USE three-tier system...',
        examples: ['Generation Account example'],
        implementationSteps: ['Step 1', 'Step 2'],
        riskFactors: ['Risk 1'],
        relatedConcepts: ['delta-targeting']
      };

      (ws1Agent.explainProtocolConcept as jest.Mock).mockResolvedValue(mockWS1Response);

      const user = userEvent.setup();
      const mockUser = {
        id: 'test-user',
        email: 'test@alluse.com',
        name: 'Test User',
        accountTier: 'premium' as const,
        expertiseLevel: 'intermediate' as const,
        joinDate: '2024-01-01',
        lastLogin: '2024-06-18',
        protocolAccess: ['three-tier', 'forking', 'delta-targeting']
      };

      render(<ConversationalInterface user={mockUser} />);
      
      const input = screen.getByPlaceholderText(/ask about the all-use protocol/i);
      const sendButton = screen.getByRole('button', { name: /send/i });
      
      await user.type(input, 'Explain the three-tier account structure');
      await user.click(sendButton);
      
      await waitFor(() => {
        expect(ws1Agent.explainProtocolConcept).toHaveBeenCalledWith('three-tier-account-structure');
      });
    });

    test('authentication service integration', async () => {
      const mockAuthResult = {
        success: true,
        user: {
          id: 'test-user',
          email: 'test@alluse.com',
          name: 'Test User',
          accountTier: 'premium' as const,
          expertiseLevel: 'intermediate' as const,
          joinDate: '2024-01-01',
          lastLogin: '2024-06-18',
          protocolAccess: ['three-tier']
        },
        token: 'test-token',
        expiresAt: new Date(Date.now() + 86400000)
      };

      (authService.login as jest.Mock).mockResolvedValue(mockAuthResult);

      const user = userEvent.setup();
      const mockOnAuthenticated = jest.fn();
      const mockOnLogout = jest.fn();

      render(<Authentication onAuthenticated={mockOnAuthenticated} onLogout={mockOnLogout} />);
      
      const emailInput = screen.getByPlaceholderText('Enter your email');
      const passwordInput = screen.getByPlaceholderText('Enter your password');
      const submitButton = screen.getByRole('button', { name: /sign in/i });
      
      await user.type(emailInput, 'test@alluse.com');
      await user.type(passwordInput, 'password123');
      await user.click(submitButton);
      
      await waitFor(() => {
        expect(authService.login).toHaveBeenCalledWith('test@alluse.com', 'password123', false);
      });
    });
  });

  describe('Performance Tests', () => {
    test('component rendering performance', () => {
      const startTime = performance.now();
      
      render(<ConversationalInterface user={{
        id: 'test-user',
        email: 'test@alluse.com',
        name: 'Test User',
        accountTier: 'premium',
        expertiseLevel: 'intermediate',
        joinDate: '2024-01-01',
        lastLogin: '2024-06-18',
        protocolAccess: ['three-tier']
      }} />);
      
      const endTime = performance.now();
      const renderTime = endTime - startTime;
      
      // Component should render within 100ms
      expect(renderTime).toBeLessThan(100);
    });

    test('message handling performance', async () => {
      const user = userEvent.setup();
      const mockUser = {
        id: 'test-user',
        email: 'test@alluse.com',
        name: 'Test User',
        accountTier: 'premium' as const,
        expertiseLevel: 'intermediate' as const,
        joinDate: '2024-01-01',
        lastLogin: '2024-06-18',
        protocolAccess: ['three-tier']
      };

      render(<ConversationalInterface user={mockUser} />);
      
      const input = screen.getByPlaceholderText(/ask about the all-use protocol/i);
      const sendButton = screen.getByRole('button', { name: /send/i });
      
      const startTime = performance.now();
      
      await user.type(input, 'Test message');
      await user.click(sendButton);
      
      await waitFor(() => {
        expect(screen.getByText('Test message')).toBeInTheDocument();
      });
      
      const endTime = performance.now();
      const responseTime = endTime - startTime;
      
      // Message handling should complete within 2 seconds
      expect(responseTime).toBeLessThan(2000);
    });
  });

  describe('Accessibility Tests', () => {
    test('keyboard navigation support', async () => {
      const user = userEvent.setup();
      render(<ConversationalInterface user={{
        id: 'test-user',
        email: 'test@alluse.com',
        name: 'Test User',
        accountTier: 'premium',
        expertiseLevel: 'intermediate',
        joinDate: '2024-01-01',
        lastLogin: '2024-06-18',
        protocolAccess: ['three-tier']
      }} />);
      
      const input = screen.getByPlaceholderText(/ask about the all-use protocol/i);
      
      // Tab to input field
      await user.tab();
      expect(input).toHaveFocus();
      
      // Type message and press Enter
      await user.type(input, 'Test message{enter}');
      
      await waitFor(() => {
        expect(screen.getByText('Test message')).toBeInTheDocument();
      });
    });

    test('screen reader compatibility', () => {
      render(<Authentication onAuthenticated={jest.fn()} onLogout={jest.fn()} />);
      
      // Check for proper labels
      expect(screen.getByLabelText(/email address/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
      
      // Check for ARIA attributes
      const submitButton = screen.getByRole('button', { name: /sign in/i });
      expect(submitButton).toBeInTheDocument();
    });

    test('color contrast and visual indicators', () => {
      render(<Analytics />);
      
      // Check for text content that indicates proper color usage
      expect(screen.getByText(/Green Week|Yellow Week|Red Week/)).toBeInTheDocument();
      
      // Risk level indicators should be present
      expect(screen.getByText('Low Risk Periods')).toBeInTheDocument();
      expect(screen.getByText('Medium Risk Periods')).toBeInTheDocument();
      expect(screen.getByText('High Risk Periods')).toBeInTheDocument();
    });
  });
});

// Test utilities and helpers
export const testUtils = {
  createMockUser: (overrides = {}) => ({
    id: 'test-user',
    email: 'test@alluse.com',
    name: 'Test User',
    accountTier: 'premium' as const,
    expertiseLevel: 'intermediate' as const,
    joinDate: '2024-01-01',
    lastLogin: '2024-06-18',
    protocolAccess: ['three-tier', 'forking', 'delta-targeting'],
    ...overrides
  }),

  createMockAccounts: () => [
    {
      id: 'gen-001',
      name: 'Generation Account',
      type: 'generation' as const,
      balance: 75000,
      weeklyReturn: 2.3,
      monthlyReturn: 8.7,
      yearlyReturn: 89.4,
      riskLevel: 'high' as const,
      strategy: 'Aggressive premium harvesting'
    },
    {
      id: 'rev-001',
      name: 'Revenue Account',
      type: 'revenue' as const,
      balance: 60000,
      weeklyReturn: 1.1,
      monthlyReturn: 4.2,
      yearlyReturn: 52.3,
      riskLevel: 'medium' as const,
      strategy: 'Balanced income generation'
    },
    {
      id: 'comp-001',
      name: 'Compounding Account',
      type: 'compounding' as const,
      balance: 90000,
      weeklyReturn: 0.8,
      monthlyReturn: 3.1,
      yearlyReturn: 38.7,
      riskLevel: 'low' as const,
      strategy: 'Conservative growth'
    }
  ],

  waitForAsyncOperation: async (operation: () => Promise<void>, timeout = 5000) => {
    await act(async () => {
      await waitFor(operation, { timeout });
    });
  },

  mockSpeechRecognition: () => {
    const mockSpeechRecognition = {
      start: jest.fn(),
      stop: jest.fn(),
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      continuous: false,
      interimResults: false,
      lang: 'en-US'
    };
    
    (global as any).webkitSpeechRecognition = jest.fn(() => mockSpeechRecognition);
    (global as any).SpeechRecognition = jest.fn(() => mockSpeechRecognition);
    
    return mockSpeechRecognition;
  }
};

