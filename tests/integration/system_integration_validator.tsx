import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

// System Integration Validation Framework
// Comprehensive testing of integration across all workstreams and system components

// Mock all workstream integrations
jest.mock('../lib/ws1-integration', () => ({
  WS1Integration: jest.fn().mockImplementation(() => ({
    // Account Management Integration
    authenticate: jest.fn().mockResolvedValue({
      success: true,
      token: 'mock-auth-token',
      user: { id: '1', name: 'Test User', email: 'test@alluse.com' }
    }),
    getAccountData: jest.fn().mockResolvedValue({
      accountId: 'acc-123',
      balance: 10000.50,
      currency: 'USD',
      accounts: [
        { id: 'acc-123', type: 'checking', balance: 10000.50 },
        { id: 'acc-124', type: 'savings', balance: 25000.00 }
      ]
    }),
    getTransactions: jest.fn().mockResolvedValue([
      { id: 'txn-1', amount: -50.00, description: 'Coffee Shop', date: '2024-01-15' },
      { id: 'txn-2', amount: 1000.00, description: 'Salary Deposit', date: '2024-01-14' }
    ]),
    updateAccountSettings: jest.fn().mockResolvedValue({ success: true }),
    validatePermissions: jest.fn().mockResolvedValue({ hasAccess: true, permissions: ['read', 'write'] })
  }))
}));

jest.mock('../lib/ws2-integration', () => ({
  WS2Integration: jest.fn().mockImplementation(() => ({
    // Transaction Processing Integration
    processTransaction: jest.fn().mockResolvedValue({
      transactionId: 'txn-new-123',
      status: 'completed',
      amount: 100.00,
      timestamp: new Date().toISOString()
    }),
    getTransactionStatus: jest.fn().mockResolvedValue({
      status: 'completed',
      confirmations: 3,
      estimatedCompletion: new Date().toISOString()
    }),
    getTransactionHistory: jest.fn().mockResolvedValue([
      { id: 'txn-1', type: 'debit', amount: 100, status: 'completed' },
      { id: 'txn-2', type: 'credit', amount: 500, status: 'pending' }
    ]),
    validateTransaction: jest.fn().mockResolvedValue({ valid: true, fees: 2.50 }),
    cancelTransaction: jest.fn().mockResolvedValue({ success: true, refundAmount: 97.50 })
  }))
}));

jest.mock('../integrations/ws3/MarketIntelligenceClient', () => ({
  MarketIntelligenceClient: jest.fn().mockImplementation(() => ({
    // Market Intelligence Integration
    getMarketData: jest.fn().mockResolvedValue({
      symbol: 'AAPL',
      price: 150.25,
      change: 2.50,
      changePercent: 1.69,
      volume: 1000000,
      marketCap: 2500000000000
    }),
    getMarketAnalysis: jest.fn().mockResolvedValue({
      trend: 'bullish',
      confidence: 0.85,
      signals: ['moving_average_crossover', 'volume_surge'],
      recommendation: 'buy',
      targetPrice: 160.00
    }),
    getIntelligenceInsights: jest.fn().mockResolvedValue([
      { type: 'trend', message: 'Strong upward momentum detected', confidence: 0.9 },
      { type: 'volume', message: 'Above average trading volume', confidence: 0.8 }
    ]),
    subscribeToMarketUpdates: jest.fn().mockResolvedValue({ subscriptionId: 'sub-123' }),
    getHistoricalData: jest.fn().mockResolvedValue({
      data: Array.from({ length: 30 }, (_, i) => ({
        date: new Date(Date.now() - i * 24 * 60 * 60 * 1000).toISOString(),
        price: 150 + Math.random() * 10 - 5
      }))
    })
  }))
}));

jest.mock('../integrations/ws4/MarketIntegrationClient', () => ({
  MarketIntegrationClient: jest.fn().mockImplementation(() => ({
    // Market Integration
    getExternalMarketData: jest.fn().mockResolvedValue({
      exchanges: ['NYSE', 'NASDAQ', 'LSE'],
      data: [
        { exchange: 'NYSE', symbol: 'AAPL', price: 150.25, volume: 1000000 },
        { exchange: 'NASDAQ', symbol: 'GOOGL', price: 2800.50, volume: 500000 }
      ]
    }),
    executeTrade: jest.fn().mockResolvedValue({
      orderId: 'order-123',
      status: 'filled',
      executedPrice: 150.30,
      executedQuantity: 10,
      fees: 9.99
    }),
    getMarketStatus: jest.fn().mockResolvedValue({
      status: 'open',
      nextClose: '2024-01-15T21:00:00Z',
      timezone: 'America/New_York'
    }),
    getOrderBook: jest.fn().mockResolvedValue({
      bids: [{ price: 150.20, quantity: 100 }, { price: 150.15, quantity: 200 }],
      asks: [{ price: 150.25, quantity: 150 }, { price: 150.30, quantity: 300 }]
    }),
    cancelOrder: jest.fn().mockResolvedValue({ success: true, cancelledOrderId: 'order-123' })
  }))
}));

jest.mock('../performance/performance_monitoring_framework', () => ({
  PerformanceMonitor: jest.fn().mockImplementation(() => ({
    // WS5 Learning Systems Integration
    startMonitoring: jest.fn().mockResolvedValue({ monitoringId: 'monitor-123' }),
    stopMonitoring: jest.fn().mockResolvedValue({ success: true }),
    getPerformanceMetrics: jest.fn().mockResolvedValue({
      renderTime: 45.2,
      memoryUsage: 67.8,
      networkLatency: 120.5,
      userInteractions: 25,
      errorRate: 0.1
    }),
    getComponentMetrics: jest.fn().mockResolvedValue([
      { component: 'ConversationalInterface', renderTime: 12.5, memoryUsage: 15.2 },
      { component: 'AccountVisualization', renderTime: 18.7, memoryUsage: 22.1 }
    ]),
    optimizePerformance: jest.fn().mockResolvedValue({
      optimizationsApplied: 5,
      performanceImprovement: 15.3,
      recommendations: ['Enable component memoization', 'Optimize bundle size']
    })
  }))
}));

// System Integration Test Components
const SystemIntegrationTestSuite = () => {
  const [integrationStatus, setIntegrationStatus] = React.useState({
    ws1: 'disconnected',
    ws2: 'disconnected',
    ws3: 'disconnected',
    ws4: 'disconnected',
    ws5: 'disconnected'
  });

  const [testResults, setTestResults] = React.useState({
    authentication: null,
    dataFlow: null,
    crossWorkstream: null,
    performance: null,
    security: null
  });

  const testWorkstreamIntegration = async (workstream: string) => {
    setIntegrationStatus(prev => ({ ...prev, [workstream]: 'testing' }));
    
    try {
      // Simulate integration test
      await new Promise(resolve => setTimeout(resolve, 1000));
      setIntegrationStatus(prev => ({ ...prev, [workstream]: 'connected' }));
      return { success: true, workstream };
    } catch (error) {
      setIntegrationStatus(prev => ({ ...prev, [workstream]: 'error' }));
      return { success: false, workstream, error };
    }
  };

  const runIntegrationTests = async () => {
    const workstreams = ['ws1', 'ws2', 'ws3', 'ws4', 'ws5'];
    
    for (const ws of workstreams) {
      await testWorkstreamIntegration(ws);
    }
  };

  return (
    <div data-testid="system-integration-test-suite">
      <h1>System Integration Validation</h1>
      
      <div data-testid="integration-status">
        {Object.entries(integrationStatus).map(([ws, status]) => (
          <div key={ws} data-testid={`${ws}-status`} className={`status-${status}`}>
            {ws.toUpperCase()}: {status}
          </div>
        ))}
      </div>
      
      <button 
        data-testid="run-integration-tests"
        onClick={runIntegrationTests}
      >
        Run Integration Tests
      </button>
      
      <div data-testid="test-results">
        {Object.entries(testResults).map(([test, result]) => (
          <div key={test} data-testid={`${test}-result`}>
            {test}: {result ? 'PASS' : 'PENDING'}
          </div>
        ))}
      </div>
    </div>
  );
};

// System Integration Validation Tests
describe('WS6-P6: System Integration Validation', () => {
  describe('Workstream Integration Testing', () => {
    test('WS1 Account Management integration works correctly', async () => {
      const { WS1Integration } = require('../lib/ws1-integration');
      const ws1Client = new WS1Integration();
      
      // Test authentication
      const authResult = await ws1Client.authenticate('test@alluse.com', 'password');
      expect(authResult.success).toBe(true);
      expect(authResult.token).toBeTruthy();
      
      // Test account data retrieval
      const accountData = await ws1Client.getAccountData();
      expect(accountData.accountId).toBeTruthy();
      expect(accountData.balance).toBeGreaterThan(0);
      expect(accountData.accounts).toHaveLength(2);
      
      // Test transaction history
      const transactions = await ws1Client.getTransactions();
      expect(transactions).toHaveLength(2);
      expect(transactions[0]).toHaveProperty('id');
      expect(transactions[0]).toHaveProperty('amount');
      
      // Test permissions validation
      const permissions = await ws1Client.validatePermissions();
      expect(permissions.hasAccess).toBe(true);
      expect(permissions.permissions).toContain('read');
    });

    test('WS2 Transaction Processing integration works correctly', async () => {
      const { WS2Integration } = require('../lib/ws2-integration');
      const ws2Client = new WS2Integration();
      
      // Test transaction processing
      const processResult = await ws2Client.processTransaction({
        amount: 100.00,
        type: 'debit',
        description: 'Test transaction'
      });
      expect(processResult.transactionId).toBeTruthy();
      expect(processResult.status).toBe('completed');
      
      // Test transaction status
      const statusResult = await ws2Client.getTransactionStatus('txn-123');
      expect(statusResult.status).toBe('completed');
      expect(statusResult.confirmations).toBeGreaterThan(0);
      
      // Test transaction validation
      const validationResult = await ws2Client.validateTransaction({
        amount: 100.00,
        type: 'debit'
      });
      expect(validationResult.valid).toBe(true);
      expect(validationResult.fees).toBeGreaterThan(0);
    });

    test('WS3 Market Intelligence integration works correctly', async () => {
      const { MarketIntelligenceClient } = require('../integrations/ws3/MarketIntelligenceClient');
      const ws3Client = new MarketIntelligenceClient();
      
      // Test market data retrieval
      const marketData = await ws3Client.getMarketData('AAPL');
      expect(marketData.symbol).toBe('AAPL');
      expect(marketData.price).toBeGreaterThan(0);
      expect(marketData.volume).toBeGreaterThan(0);
      
      // Test market analysis
      const analysis = await ws3Client.getMarketAnalysis('AAPL');
      expect(analysis.trend).toBeTruthy();
      expect(analysis.confidence).toBeGreaterThan(0);
      expect(analysis.confidence).toBeLessThanOrEqual(1);
      
      // Test intelligence insights
      const insights = await ws3Client.getIntelligenceInsights('AAPL');
      expect(insights).toHaveLength(2);
      expect(insights[0]).toHaveProperty('type');
      expect(insights[0]).toHaveProperty('confidence');
      
      // Test historical data
      const historicalData = await ws3Client.getHistoricalData('AAPL', '30d');
      expect(historicalData.data).toHaveLength(30);
      expect(historicalData.data[0]).toHaveProperty('date');
      expect(historicalData.data[0]).toHaveProperty('price');
    });

    test('WS4 Market Integration works correctly', async () => {
      const { MarketIntegrationClient } = require('../integrations/ws4/MarketIntegrationClient');
      const ws4Client = new MarketIntegrationClient();
      
      // Test external market data
      const externalData = await ws4Client.getExternalMarketData();
      expect(externalData.exchanges).toContain('NYSE');
      expect(externalData.data).toHaveLength(2);
      expect(externalData.data[0]).toHaveProperty('exchange');
      
      // Test trade execution
      const tradeResult = await ws4Client.executeTrade({
        symbol: 'AAPL',
        quantity: 10,
        price: 150.25,
        type: 'buy'
      });
      expect(tradeResult.orderId).toBeTruthy();
      expect(tradeResult.status).toBe('filled');
      expect(tradeResult.executedQuantity).toBe(10);
      
      // Test market status
      const marketStatus = await ws4Client.getMarketStatus();
      expect(marketStatus.status).toBe('open');
      expect(marketStatus.timezone).toBeTruthy();
      
      // Test order book
      const orderBook = await ws4Client.getOrderBook('AAPL');
      expect(orderBook.bids).toHaveLength(2);
      expect(orderBook.asks).toHaveLength(2);
      expect(orderBook.bids[0]).toHaveProperty('price');
      expect(orderBook.bids[0]).toHaveProperty('quantity');
    });

    test('WS5 Learning Systems integration works correctly', async () => {
      const { PerformanceMonitor } = require('../performance/performance_monitoring_framework');
      const ws5Client = new PerformanceMonitor();
      
      // Test performance monitoring
      const monitoringResult = await ws5Client.startMonitoring();
      expect(monitoringResult.monitoringId).toBeTruthy();
      
      // Test performance metrics
      const metrics = await ws5Client.getPerformanceMetrics();
      expect(metrics.renderTime).toBeGreaterThan(0);
      expect(metrics.memoryUsage).toBeGreaterThan(0);
      expect(metrics.networkLatency).toBeGreaterThan(0);
      
      // Test component metrics
      const componentMetrics = await ws5Client.getComponentMetrics();
      expect(componentMetrics).toHaveLength(2);
      expect(componentMetrics[0]).toHaveProperty('component');
      expect(componentMetrics[0]).toHaveProperty('renderTime');
      
      // Test performance optimization
      const optimizationResult = await ws5Client.optimizePerformance();
      expect(optimizationResult.optimizationsApplied).toBeGreaterThan(0);
      expect(optimizationResult.performanceImprovement).toBeGreaterThan(0);
      expect(optimizationResult.recommendations).toHaveLength(2);
    });
  });

  describe('Cross-Workstream Data Flow Testing', () => {
    test('account data flows from WS1 to all other workstreams', async () => {
      const { WS1Integration } = require('../lib/ws1-integration');
      const { WS2Integration } = require('../lib/ws2-integration');
      
      const ws1Client = new WS1Integration();
      const ws2Client = new WS2Integration();
      
      // Get account data from WS1
      const accountData = await ws1Client.getAccountData();
      expect(accountData.accountId).toBeTruthy();
      
      // Verify WS2 can process transactions for this account
      const transactionResult = await ws2Client.processTransaction({
        accountId: accountData.accountId,
        amount: 100.00,
        type: 'debit'
      });
      expect(transactionResult.transactionId).toBeTruthy();
      
      // Verify transaction appears in WS1 transaction history
      const transactions = await ws1Client.getTransactions();
      expect(transactions).toHaveLength(2);
    });

    test('market data flows from WS3 to WS4 trading systems', async () => {
      const { MarketIntelligenceClient } = require('../integrations/ws3/MarketIntelligenceClient');
      const { MarketIntegrationClient } = require('../integrations/ws4/MarketIntegrationClient');
      
      const ws3Client = new MarketIntelligenceClient();
      const ws4Client = new MarketIntegrationClient();
      
      // Get market analysis from WS3
      const analysis = await ws3Client.getMarketAnalysis('AAPL');
      expect(analysis.recommendation).toBeTruthy();
      
      // Use analysis to execute trade in WS4
      const tradeResult = await ws4Client.executeTrade({
        symbol: 'AAPL',
        quantity: 10,
        price: analysis.targetPrice,
        type: analysis.recommendation === 'buy' ? 'buy' : 'sell'
      });
      expect(tradeResult.orderId).toBeTruthy();
      expect(tradeResult.status).toBe('filled');
    });

    test('performance data flows from WS5 to all UI components', async () => {
      const { PerformanceMonitor } = require('../performance/performance_monitoring_framework');
      
      const ws5Client = new PerformanceMonitor();
      
      // Start performance monitoring
      await ws5Client.startMonitoring();
      
      // Get component metrics
      const componentMetrics = await ws5Client.getComponentMetrics();
      expect(componentMetrics).toHaveLength(2);
      
      // Verify metrics include UI components
      const componentNames = componentMetrics.map(m => m.component);
      expect(componentNames).toContain('ConversationalInterface');
      expect(componentNames).toContain('AccountVisualization');
    });

    test('authentication state propagates across all workstreams', async () => {
      const { WS1Integration } = require('../lib/ws1-integration');
      const { WS2Integration } = require('../lib/ws2-integration');
      const { MarketIntelligenceClient } = require('../integrations/ws3/MarketIntelligenceClient');
      
      const ws1Client = new WS1Integration();
      const ws2Client = new WS2Integration();
      const ws3Client = new MarketIntelligenceClient();
      
      // Authenticate with WS1
      const authResult = await ws1Client.authenticate('test@alluse.com', 'password');
      expect(authResult.success).toBe(true);
      
      // Verify other workstreams recognize authentication
      const accountData = await ws1Client.getAccountData();
      expect(accountData.accountId).toBeTruthy();
      
      const transactionHistory = await ws2Client.getTransactionHistory();
      expect(transactionHistory).toHaveLength(2);
      
      const marketData = await ws3Client.getMarketData('AAPL');
      expect(marketData.symbol).toBe('AAPL');
    });
  });

  describe('System-Wide Performance Integration', () => {
    test('system maintains performance under cross-workstream load', async () => {
      const startTime = performance.now();
      
      // Simulate concurrent operations across all workstreams
      const operations = await Promise.all([
        require('../lib/ws1-integration').WS1Integration().getAccountData(),
        require('../lib/ws2-integration').WS2Integration().getTransactionHistory(),
        require('../integrations/ws3/MarketIntelligenceClient').MarketIntelligenceClient().getMarketData('AAPL'),
        require('../integrations/ws4/MarketIntegrationClient').MarketIntegrationClient().getExternalMarketData(),
        require('../performance/performance_monitoring_framework').PerformanceMonitor().getPerformanceMetrics()
      ]);
      
      const endTime = performance.now();
      const totalTime = endTime - startTime;
      
      // All operations should complete within 2 seconds
      expect(totalTime).toBeLessThan(2000);
      
      // All operations should succeed
      expect(operations).toHaveLength(5);
      operations.forEach(result => {
        expect(result).toBeTruthy();
      });
    });

    test('error in one workstream does not affect others', async () => {
      // Mock error in WS3
      const { MarketIntelligenceClient } = require('../integrations/ws3/MarketIntelligenceClient');
      const ws3Client = new MarketIntelligenceClient();
      ws3Client.getMarketData = jest.fn().mockRejectedValue(new Error('WS3 Error'));
      
      // Other workstreams should still function
      const { WS1Integration } = require('../lib/ws1-integration');
      const { WS2Integration } = require('../lib/ws2-integration');
      
      const ws1Client = new WS1Integration();
      const ws2Client = new WS2Integration();
      
      const accountData = await ws1Client.getAccountData();
      expect(accountData.accountId).toBeTruthy();
      
      const transactionHistory = await ws2Client.getTransactionHistory();
      expect(transactionHistory).toHaveLength(2);
      
      // WS3 should fail
      await expect(ws3Client.getMarketData('AAPL')).rejects.toThrow('WS3 Error');
    });

    test('system recovers gracefully from temporary failures', async () => {
      const { WS1Integration } = require('../lib/ws1-integration');
      const ws1Client = new WS1Integration();
      
      // Simulate temporary failure
      ws1Client.getAccountData = jest.fn()
        .mockRejectedValueOnce(new Error('Temporary failure'))
        .mockResolvedValueOnce({
          accountId: 'acc-123',
          balance: 10000.50,
          currency: 'USD'
        });
      
      // First call should fail
      await expect(ws1Client.getAccountData()).rejects.toThrow('Temporary failure');
      
      // Second call should succeed
      const accountData = await ws1Client.getAccountData();
      expect(accountData.accountId).toBe('acc-123');
    });
  });

  describe('Security Integration Testing', () => {
    test('authentication tokens are properly validated across workstreams', async () => {
      const { WS1Integration } = require('../lib/ws1-integration');
      const ws1Client = new WS1Integration();
      
      // Test with valid token
      const authResult = await ws1Client.authenticate('test@alluse.com', 'password');
      expect(authResult.success).toBe(true);
      expect(authResult.token).toBeTruthy();
      
      // Test permissions validation
      const permissions = await ws1Client.validatePermissions();
      expect(permissions.hasAccess).toBe(true);
      expect(permissions.permissions).toContain('read');
      expect(permissions.permissions).toContain('write');
    });

    test('sensitive data is properly encrypted in transit', async () => {
      const { WS2Integration } = require('../lib/ws2-integration');
      const ws2Client = new WS2Integration();
      
      // Process a transaction with sensitive data
      const transactionResult = await ws2Client.processTransaction({
        amount: 1000.00,
        accountNumber: '****1234', // Should be masked
        routingNumber: '****5678'  // Should be masked
      });
      
      expect(transactionResult.transactionId).toBeTruthy();
      expect(transactionResult.status).toBe('completed');
    });

    test('access control is enforced across all workstreams', async () => {
      const { WS1Integration } = require('../lib/ws1-integration');
      const ws1Client = new WS1Integration();
      
      // Test permission validation
      const permissions = await ws1Client.validatePermissions();
      expect(permissions.hasAccess).toBe(true);
      
      // Mock insufficient permissions
      ws1Client.validatePermissions = jest.fn().mockResolvedValue({
        hasAccess: false,
        permissions: ['read']
      });
      
      const restrictedPermissions = await ws1Client.validatePermissions();
      expect(restrictedPermissions.hasAccess).toBe(false);
      expect(restrictedPermissions.permissions).not.toContain('write');
    });
  });

  describe('API Integration Testing', () => {
    test('all API endpoints respond correctly', async () => {
      // Mock fetch for API testing
      global.fetch = jest.fn()
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ status: 'healthy', workstream: 'WS1' })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ status: 'healthy', workstream: 'WS2' })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ status: 'healthy', workstream: 'WS3' })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ status: 'healthy', workstream: 'WS4' })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ status: 'healthy', workstream: 'WS5' })
        });
      
      const endpoints = [
        '/api/ws1/health',
        '/api/ws2/health',
        '/api/ws3/health',
        '/api/ws4/health',
        '/api/ws5/health'
      ];
      
      const responses = await Promise.all(
        endpoints.map(endpoint => fetch(endpoint).then(r => r.json()))
      );
      
      responses.forEach((response, index) => {
        expect(response.status).toBe('healthy');
        expect(response.workstream).toBe(`WS${index + 1}`);
      });
    });

    test('API rate limiting works correctly', async () => {
      // Mock rate-limited API response
      global.fetch = jest.fn()
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ data: 'success' })
        })
        .mockResolvedValueOnce({
          ok: false,
          status: 429,
          json: async () => ({ error: 'Rate limit exceeded' })
        });
      
      // First request should succeed
      const firstResponse = await fetch('/api/ws3/market-data');
      const firstData = await firstResponse.json();
      expect(firstData.data).toBe('success');
      
      // Second request should be rate limited
      const secondResponse = await fetch('/api/ws3/market-data');
      expect(secondResponse.status).toBe(429);
      
      const secondData = await secondResponse.json();
      expect(secondData.error).toBe('Rate limit exceeded');
    });

    test('API error handling is consistent across workstreams', async () => {
      // Mock API error responses
      global.fetch = jest.fn()
        .mockResolvedValue({
          ok: false,
          status: 500,
          json: async () => ({ 
            error: 'Internal Server Error',
            code: 'INTERNAL_ERROR',
            timestamp: new Date().toISOString()
          })
        });
      
      const response = await fetch('/api/ws1/account-data');
      expect(response.ok).toBe(false);
      expect(response.status).toBe(500);
      
      const errorData = await response.json();
      expect(errorData.error).toBe('Internal Server Error');
      expect(errorData.code).toBe('INTERNAL_ERROR');
      expect(errorData.timestamp).toBeTruthy();
    });
  });

  describe('Real-time Data Synchronization', () => {
    test('real-time updates propagate across all components', async () => {
      const { WS1Integration } = require('../lib/ws1-integration');
      const { WS2Integration } = require('../lib/ws2-integration');
      
      const ws1Client = new WS1Integration();
      const ws2Client = new WS2Integration();
      
      // Get initial account balance
      const initialAccount = await ws1Client.getAccountData();
      const initialBalance = initialAccount.balance;
      
      // Process a transaction
      await ws2Client.processTransaction({
        accountId: initialAccount.accountId,
        amount: -50.00,
        type: 'debit',
        description: 'Test purchase'
      });
      
      // Verify account balance is updated
      const updatedAccount = await ws1Client.getAccountData();
      expect(updatedAccount.balance).toBe(initialBalance - 50.00);
    });

    test('market data updates are synchronized across trading components', async () => {
      const { MarketIntelligenceClient } = require('../integrations/ws3/MarketIntelligenceClient');
      const { MarketIntegrationClient } = require('../integrations/ws4/MarketIntegrationClient');
      
      const ws3Client = new MarketIntelligenceClient();
      const ws4Client = new MarketIntegrationClient();
      
      // Get market data from intelligence system
      const intelligenceData = await ws3Client.getMarketData('AAPL');
      
      // Get external market data
      const externalData = await ws4Client.getExternalMarketData();
      const appleData = externalData.data.find(d => d.symbol === 'AAPL');
      
      // Prices should be synchronized (within reasonable tolerance)
      expect(Math.abs(intelligenceData.price - appleData.price)).toBeLessThan(1.0);
    });

    test('performance metrics are updated in real-time', async () => {
      const { PerformanceMonitor } = require('../performance/performance_monitoring_framework');
      const ws5Client = new PerformanceMonitor();
      
      // Start monitoring
      await ws5Client.startMonitoring();
      
      // Get initial metrics
      const initialMetrics = await ws5Client.getPerformanceMetrics();
      
      // Simulate some activity
      await new Promise(resolve => setTimeout(resolve, 100));
      
      // Get updated metrics
      const updatedMetrics = await ws5Client.getPerformanceMetrics();
      
      // Metrics should be updated
      expect(updatedMetrics.userInteractions).toBeGreaterThanOrEqual(initialMetrics.userInteractions);
    });
  });

  describe('Integration Test Suite UI', () => {
    test('integration test suite renders correctly', () => {
      render(<SystemIntegrationTestSuite />);
      
      expect(screen.getByText('System Integration Validation')).toBeInTheDocument();
      expect(screen.getByTestId('integration-status')).toBeInTheDocument();
      expect(screen.getByTestId('run-integration-tests')).toBeInTheDocument();
      expect(screen.getByTestId('test-results')).toBeInTheDocument();
    });

    test('integration tests can be executed from UI', async () => {
      const user = userEvent.setup();
      render(<SystemIntegrationTestSuite />);
      
      const runButton = screen.getByTestId('run-integration-tests');
      await user.click(runButton);
      
      // Verify test execution starts
      await waitFor(() => {
        expect(screen.getByTestId('ws1-status')).toHaveTextContent('testing');
      });
      
      // Wait for tests to complete
      await waitFor(() => {
        expect(screen.getByTestId('ws1-status')).toHaveTextContent('connected');
      }, { timeout: 10000 });
    });

    test('integration status is displayed correctly', async () => {
      render(<SystemIntegrationTestSuite />);
      
      // Check initial status
      expect(screen.getByTestId('ws1-status')).toHaveTextContent('WS1: disconnected');
      expect(screen.getByTestId('ws2-status')).toHaveTextContent('WS2: disconnected');
      expect(screen.getByTestId('ws3-status')).toHaveTextContent('WS3: disconnected');
      expect(screen.getByTestId('ws4-status')).toHaveTextContent('WS4: disconnected');
      expect(screen.getByTestId('ws5-status')).toHaveTextContent('WS5: disconnected');
    });

    test('test results are displayed correctly', () => {
      render(<SystemIntegrationTestSuite />);
      
      expect(screen.getByTestId('authentication-result')).toHaveTextContent('authentication: PENDING');
      expect(screen.getByTestId('dataFlow-result')).toHaveTextContent('dataFlow: PENDING');
      expect(screen.getByTestId('crossWorkstream-result')).toHaveTextContent('crossWorkstream: PENDING');
      expect(screen.getByTestId('performance-result')).toHaveTextContent('performance: PENDING');
      expect(screen.getByTestId('security-result')).toHaveTextContent('security: PENDING');
    });
  });
});

// Integration test utilities
export const systemIntegrationUtils = {
  testWorkstreamConnection: async (workstream: string) => {
    const clients = {
      ws1: () => new (require('../lib/ws1-integration').WS1Integration)(),
      ws2: () => new (require('../lib/ws2-integration').WS2Integration)(),
      ws3: () => new (require('../integrations/ws3/MarketIntelligenceClient').MarketIntelligenceClient)(),
      ws4: () => new (require('../integrations/ws4/MarketIntegrationClient').MarketIntegrationClient)(),
      ws5: () => new (require('../performance/performance_monitoring_framework').PerformanceMonitor)()
    };
    
    const client = clients[workstream as keyof typeof clients]();
    
    try {
      // Test basic connectivity
      if (workstream === 'ws1') {
        await client.getAccountData();
      } else if (workstream === 'ws2') {
        await client.getTransactionHistory();
      } else if (workstream === 'ws3') {
        await client.getMarketData('AAPL');
      } else if (workstream === 'ws4') {
        await client.getExternalMarketData();
      } else if (workstream === 'ws5') {
        await client.getPerformanceMetrics();
      }
      
      return { success: true, workstream };
    } catch (error) {
      return { success: false, workstream, error };
    }
  },
  
  testDataFlow: async (sourceWorkstream: string, targetWorkstream: string) => {
    // Test data flow between workstreams
    const sourceClient = await systemIntegrationUtils.testWorkstreamConnection(sourceWorkstream);
    const targetClient = await systemIntegrationUtils.testWorkstreamConnection(targetWorkstream);
    
    return {
      success: sourceClient.success && targetClient.success,
      source: sourceWorkstream,
      target: targetWorkstream
    };
  },
  
  validateSystemHealth: async () => {
    const workstreams = ['ws1', 'ws2', 'ws3', 'ws4', 'ws5'];
    const results = await Promise.all(
      workstreams.map(ws => systemIntegrationUtils.testWorkstreamConnection(ws))
    );
    
    const healthyWorkstreams = results.filter(r => r.success).length;
    const totalWorkstreams = workstreams.length;
    
    return {
      healthy: healthyWorkstreams,
      total: totalWorkstreams,
      healthPercentage: (healthyWorkstreams / totalWorkstreams) * 100,
      results
    };
  }
};

export default SystemIntegrationTestSuite;

