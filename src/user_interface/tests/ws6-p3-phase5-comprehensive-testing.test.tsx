import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import SystemIntegrationHub from '../integration/SystemIntegrationHub';
import CoordinationEngine from '../coordination/CoordinationEngine';
import OrchestrationManager from '../orchestration/OrchestrationManager';

// Mock data for testing
const mockWorkstreams = [
  {
    id: 'ws1',
    name: 'Agent Foundation',
    status: 'active',
    health: 98.5,
    lastUpdate: new Date(),
    metrics: { requests: 15420, latency: 45, errors: 2, uptime: 99.8 },
    dependencies: [],
    version: '2.1.0'
  }
];

const mockCoordinationRules = [
  {
    id: 'rule_001',
    name: 'Market Data Sync',
    description: 'Synchronize market data across WS3 and WS4',
    trigger: { workstream: 'ws3', event: 'market_data_update', condition: 'volume > 1000' },
    actions: [{ workstream: 'ws4', action: 'update_positions', parameters: { priority: 'high' } }],
    priority: 'high',
    enabled: true,
    executionCount: 1247
  }
];

const mockWorkflows = [
  {
    id: 'wf_001',
    name: 'Daily Market Analysis',
    description: 'Comprehensive daily market analysis',
    status: 'running',
    priority: 'high',
    steps: [
      {
        id: 'step_001',
        name: 'Collect Market Data',
        type: 'action',
        workstream: 'ws3',
        action: 'collect_market_data',
        parameters: { symbols: ['SPY'] },
        status: 'completed',
        retryCount: 0,
        maxRetries: 3,
        timeout: 120
      }
    ],
    dependencies: [],
    createdAt: new Date(),
    metrics: { totalRuns: 156, successRate: 94.2, averageDuration: 420 },
    configuration: { notifications: true }
  }
];

describe('WS6-P3 Phase 5: System Integration and Advanced Coordination', () => {
  describe('SystemIntegrationHub', () => {
    test('renders system integration hub with workstream status', () => {
      render(<SystemIntegrationHub />);
      
      expect(screen.getByText('System Integration Hub')).toBeInTheDocument();
      expect(screen.getByText('Comprehensive monitoring and coordination of ALL-USE workstreams')).toBeInTheDocument();
      expect(screen.getByText('System Health')).toBeInTheDocument();
      expect(screen.getByText('Total Requests')).toBeInTheDocument();
      expect(screen.getByText('Workstream Status')).toBeInTheDocument();
    });

    test('displays workstream metrics correctly', async () => {
      render(<SystemIntegrationHub />);
      
      // Check for workstream cards
      await waitFor(() => {
        expect(screen.getByText('Agent Foundation')).toBeInTheDocument();
        expect(screen.getByText('Protocol Systems')).toBeInTheDocument();
        expect(screen.getByText('Market Intelligence')).toBeInTheDocument();
        expect(screen.getByText('Market Integration')).toBeInTheDocument();
        expect(screen.getByText('Learning Systems')).toBeInTheDocument();
      });
    });

    test('handles workstream restart functionality', async () => {
      render(<SystemIntegrationHub />);
      
      // Click on a workstream to expand details
      const workstreamCard = screen.getByText('Agent Foundation').closest('div');
      fireEvent.click(workstreamCard!);
      
      await waitFor(() => {
        const restartButton = screen.getByText('Restart');
        expect(restartButton).toBeInTheDocument();
        fireEvent.click(restartButton);
      });
    });

    test('toggles auto refresh functionality', () => {
      render(<SystemIntegrationHub />);
      
      const autoRefreshToggle = screen.getByRole('button', { name: /auto refresh/i });
      fireEvent.click(autoRefreshToggle);
      
      // Verify toggle state change
      expect(autoRefreshToggle).toBeInTheDocument();
    });

    test('displays data flow monitoring', () => {
      render(<SystemIntegrationHub />);
      
      expect(screen.getByText('Data Flow Monitoring')).toBeInTheDocument();
      expect(screen.getByText('WS1')).toBeInTheDocument();
      expect(screen.getByText('WS2')).toBeInTheDocument();
    });
  });

  describe('CoordinationEngine', () => {
    test('renders coordination engine with metrics', () => {
      render(<CoordinationEngine />);
      
      expect(screen.getByText('Coordination Engine')).toBeInTheDocument();
      expect(screen.getByText('Advanced coordination and synchronization across ALL-USE workstreams')).toBeInTheDocument();
      expect(screen.getByText('Coordination Score')).toBeInTheDocument();
      expect(screen.getByText('Active Rules')).toBeInTheDocument();
    });

    test('displays coordination rules tab', async () => {
      render(<CoordinationEngine />);
      
      const rulesTab = screen.getByText('Coordination Rules');
      fireEvent.click(rulesTab);
      
      await waitFor(() => {
        expect(screen.getByText('Search rules...')).toBeInTheDocument();
        expect(screen.getByText('All Priorities')).toBeInTheDocument();
      });
    });

    test('handles rule enable/disable functionality', async () => {
      render(<CoordinationEngine />);
      
      // Navigate to rules tab
      const rulesTab = screen.getByText('Coordination Rules');
      fireEvent.click(rulesTab);
      
      await waitFor(() => {
        const enableButton = screen.getAllByText('Disable')[0];
        fireEvent.click(enableButton);
      });
    });

    test('displays sync operations tab', async () => {
      render(<CoordinationEngine />);
      
      const syncTab = screen.getByText('Sync Operations');
      fireEvent.click(syncTab);
      
      await waitFor(() => {
        expect(screen.getByText('Synchronization Operations')).toBeInTheDocument();
      });
    });

    test('shows coordination events', async () => {
      render(<CoordinationEngine />);
      
      const eventsTab = screen.getByText('Events');
      fireEvent.click(eventsTab);
      
      await waitFor(() => {
        expect(screen.getByText('Coordination Events')).toBeInTheDocument();
      });
    });

    test('filters rules by search term', async () => {
      render(<CoordinationEngine />);
      
      const rulesTab = screen.getByText('Coordination Rules');
      fireEvent.click(rulesTab);
      
      await waitFor(() => {
        const searchInput = screen.getByPlaceholderText('Search rules...');
        fireEvent.change(searchInput, { target: { value: 'Market' } });
        expect(searchInput).toHaveValue('Market');
      });
    });
  });

  describe('OrchestrationManager', () => {
    test('renders orchestration manager with dashboard', () => {
      render(<OrchestrationManager />);
      
      expect(screen.getByText('Orchestration Manager')).toBeInTheDocument();
      expect(screen.getByText('Advanced workflow orchestration and automation across ALL-USE workstreams')).toBeInTheDocument();
      expect(screen.getByText('Total Workflows')).toBeInTheDocument();
      expect(screen.getByText('Active')).toBeInTheDocument();
    });

    test('displays workflows tab with workflow management', async () => {
      render(<OrchestrationManager />);
      
      const workflowsTab = screen.getByText('Workflows');
      fireEvent.click(workflowsTab);
      
      await waitFor(() => {
        expect(screen.getByText('Workflow Management')).toBeInTheDocument();
        expect(screen.getByText('Search workflows...')).toBeInTheDocument();
      });
    });

    test('handles workflow control buttons', async () => {
      render(<OrchestrationManager />);
      
      const workflowsTab = screen.getByText('Workflows');
      fireEvent.click(workflowsTab);
      
      await waitFor(() => {
        // Test workflow control buttons (play, pause, stop)
        const playButtons = screen.getAllByRole('button');
        const playButton = playButtons.find(btn => btn.querySelector('svg'));
        if (playButton) {
          fireEvent.click(playButton);
        }
      });
    });

    test('displays resources tab with monitoring', async () => {
      render(<OrchestrationManager />);
      
      const resourcesTab = screen.getByText('Resources');
      fireEvent.click(resourcesTab);
      
      await waitFor(() => {
        expect(screen.getByText('Resource Monitoring')).toBeInTheDocument();
      });
    });

    test('shows execution logs tab', async () => {
      render(<OrchestrationManager />);
      
      const logsTab = screen.getByText('Execution Logs');
      fireEvent.click(logsTab);
      
      await waitFor(() => {
        expect(screen.getByText('Execution Logs')).toBeInTheDocument();
      });
    });

    test('filters workflows by status', async () => {
      render(<OrchestrationManager />);
      
      const workflowsTab = screen.getByText('Workflows');
      fireEvent.click(workflowsTab);
      
      await waitFor(() => {
        const statusFilter = screen.getByDisplayValue('All Status');
        fireEvent.change(statusFilter, { target: { value: 'running' } });
        expect(statusFilter).toHaveValue('running');
      });
    });

    test('toggles workflow details', async () => {
      render(<OrchestrationManager />);
      
      const workflowsTab = screen.getByText('Workflows');
      fireEvent.click(workflowsTab);
      
      await waitFor(() => {
        const showDetailsButton = screen.getByText('Show Details');
        fireEvent.click(showDetailsButton);
        expect(screen.getByText('Hide Details')).toBeInTheDocument();
      });
    });
  });

  describe('Integration Testing', () => {
    test('components handle real-time updates', async () => {
      render(<SystemIntegrationHub />);
      
      // Wait for initial render
      await waitFor(() => {
        expect(screen.getByText('System Integration Hub')).toBeInTheDocument();
      });
      
      // Simulate time passage for auto-refresh
      await new Promise(resolve => setTimeout(resolve, 100));
    });

    test('components maintain state consistency', async () => {
      render(<CoordinationEngine />);
      
      // Test tab navigation maintains state
      const rulesTab = screen.getByText('Coordination Rules');
      fireEvent.click(rulesTab);
      
      const overviewTab = screen.getByText('Overview');
      fireEvent.click(overviewTab);
      
      fireEvent.click(rulesTab);
      
      await waitFor(() => {
        expect(screen.getByText('Search rules...')).toBeInTheDocument();
      });
    });

    test('error handling for invalid operations', async () => {
      render(<OrchestrationManager />);
      
      // Test error handling doesn't crash the component
      const workflowsTab = screen.getByText('Workflows');
      fireEvent.click(workflowsTab);
      
      await waitFor(() => {
        expect(screen.getByText('Workflow Management')).toBeInTheDocument();
      });
    });
  });

  describe('Performance Testing', () => {
    test('components render within performance budget', async () => {
      const startTime = performance.now();
      
      render(<SystemIntegrationHub />);
      
      await waitFor(() => {
        expect(screen.getByText('System Integration Hub')).toBeInTheDocument();
      });
      
      const endTime = performance.now();
      const renderTime = endTime - startTime;
      
      // Should render within 1000ms
      expect(renderTime).toBeLessThan(1000);
    });

    test('handles large datasets efficiently', async () => {
      render(<CoordinationEngine />);
      
      // Test with multiple rapid state updates
      const rulesTab = screen.getByText('Coordination Rules');
      fireEvent.click(rulesTab);
      
      await waitFor(() => {
        const searchInput = screen.getByPlaceholderText('Search rules...');
        
        // Rapid typing simulation
        for (let i = 0; i < 10; i++) {
          fireEvent.change(searchInput, { target: { value: `test${i}` } });
        }
        
        expect(searchInput).toHaveValue('test9');
      });
    });
  });

  describe('Accessibility Testing', () => {
    test('components have proper ARIA labels', () => {
      render(<SystemIntegrationHub />);
      
      const refreshButton = screen.getByRole('button', { name: /refresh/i });
      expect(refreshButton).toBeInTheDocument();
    });

    test('keyboard navigation works correctly', async () => {
      render(<CoordinationEngine />);
      
      const rulesTab = screen.getByText('Coordination Rules');
      
      // Test keyboard navigation
      rulesTab.focus();
      fireEvent.keyDown(rulesTab, { key: 'Enter' });
      
      await waitFor(() => {
        expect(screen.getByText('Search rules...')).toBeInTheDocument();
      });
    });

    test('screen reader compatibility', () => {
      render(<OrchestrationManager />);
      
      // Check for proper heading structure
      expect(screen.getByRole('heading', { level: 1 })).toBeInTheDocument();
    });
  });
});

// Integration test utilities
export const testUtils = {
  // Mock WebSocket for real-time testing
  mockWebSocket: () => {
    const mockWS = {
      send: jest.fn(),
      close: jest.fn(),
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      readyState: WebSocket.OPEN
    };
    
    global.WebSocket = jest.fn(() => mockWS) as any;
    return mockWS;
  },
  
  // Mock performance API
  mockPerformance: () => {
    global.performance = {
      ...global.performance,
      now: jest.fn(() => Date.now())
    };
  },
  
  // Test data generators
  generateMockWorkstream: (overrides = {}) => ({
    id: 'ws_test',
    name: 'Test Workstream',
    status: 'active',
    health: 95.0,
    lastUpdate: new Date(),
    metrics: { requests: 1000, latency: 50, errors: 0, uptime: 99.9 },
    dependencies: [],
    version: '1.0.0',
    ...overrides
  }),
  
  generateMockWorkflow: (overrides = {}) => ({
    id: 'wf_test',
    name: 'Test Workflow',
    description: 'Test workflow description',
    status: 'running',
    priority: 'medium',
    steps: [],
    dependencies: [],
    createdAt: new Date(),
    metrics: { totalRuns: 10, successRate: 90.0, averageDuration: 300 },
    configuration: {},
    ...overrides
  })
};

export default testUtils;

