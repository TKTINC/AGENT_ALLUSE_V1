import { test, expect, Page, BrowserContext } from '@playwright/test';

// Test configuration and utilities
const BASE_URL = process.env.BASE_URL || 'http://localhost:3000';
const TEST_USER = {
  email: 'test@alluse.com',
  password: 'TestPassword123!',
  name: 'Test User'
};

// Page object models for better test organization
class AuthenticationPage {
  constructor(private page: Page) {}

  async navigateToLogin() {
    await this.page.goto(`${BASE_URL}/login`);
  }

  async login(email: string, password: string) {
    await this.page.fill('[data-testid="email-input"]', email);
    await this.page.fill('[data-testid="password-input"]', password);
    await this.page.click('[data-testid="login-button"]');
  }

  async register(email: string, password: string, name: string) {
    await this.page.click('[data-testid="register-link"]');
    await this.page.fill('[data-testid="name-input"]', name);
    await this.page.fill('[data-testid="email-input"]', email);
    await this.page.fill('[data-testid="password-input"]', password);
    await this.page.fill('[data-testid="confirm-password-input"]', password);
    await this.page.click('[data-testid="register-button"]');
  }

  async logout() {
    await this.page.click('[data-testid="user-menu"]');
    await this.page.click('[data-testid="logout-button"]');
  }
}

class DashboardPage {
  constructor(private page: Page) {}

  async navigateToDashboard() {
    await this.page.goto(`${BASE_URL}/dashboard`);
  }

  async switchTab(tabName: string) {
    await this.page.click(`[data-testid="tab-${tabName.toLowerCase()}"]`);
  }

  async getPortfolioValue() {
    return await this.page.textContent('[data-testid="portfolio-value"]');
  }

  async getAccountPerformance(account: string) {
    return await this.page.textContent(`[data-testid="${account}-performance"]`);
  }
}

class TradingPage {
  constructor(private page: Page) {}

  async navigateToTrading() {
    await this.page.goto(`${BASE_URL}/trading`);
  }

  async placeOrder(symbol: string, quantity: number, orderType: string, price?: number) {
    await this.page.click('[data-testid="place-order-button"]');
    await this.page.fill('[data-testid="symbol-input"]', symbol);
    await this.page.fill('[data-testid="quantity-input"]', quantity.toString());
    await this.page.selectOption('[data-testid="order-type-select"]', orderType);
    
    if (price && orderType === 'limit') {
      await this.page.fill('[data-testid="price-input"]', price.toString());
    }
    
    await this.page.click('[data-testid="submit-order-button"]');
  }

  async getPositions() {
    await this.page.waitForSelector('[data-testid="positions-table"]');
    return await this.page.$$eval('[data-testid="position-row"]', rows => 
      rows.map(row => ({
        symbol: row.querySelector('[data-testid="position-symbol"]')?.textContent,
        quantity: row.querySelector('[data-testid="position-quantity"]')?.textContent,
        pnl: row.querySelector('[data-testid="position-pnl"]')?.textContent
      }))
    );
  }

  async cancelOrder(orderId: string) {
    await this.page.click(`[data-testid="cancel-order-${orderId}"]`);
    await this.page.click('[data-testid="confirm-cancel"]');
  }
}

class AnalyticsPage {
  constructor(private page: Page) {}

  async navigateToAnalytics() {
    await this.page.goto(`${BASE_URL}/analytics`);
  }

  async selectTimeframe(timeframe: string) {
    await this.page.selectOption('[data-testid="timeframe-select"]', timeframe);
  }

  async getPerformanceMetrics() {
    return {
      totalReturn: await this.page.textContent('[data-testid="total-return"]'),
      sharpeRatio: await this.page.textContent('[data-testid="sharpe-ratio"]'),
      maxDrawdown: await this.page.textContent('[data-testid="max-drawdown"]'),
      winRate: await this.page.textContent('[data-testid="win-rate"]')
    };
  }

  async exportData(format: string) {
    await this.page.click('[data-testid="export-button"]');
    await this.page.selectOption('[data-testid="export-format"]', format);
    await this.page.click('[data-testid="confirm-export"]');
  }
}

class ConversationalPage {
  constructor(private page: Page) {}

  async navigateToChat() {
    await this.page.goto(`${BASE_URL}/chat`);
  }

  async sendMessage(message: string) {
    await this.page.fill('[data-testid="message-input"]', message);
    await this.page.click('[data-testid="send-button"]');
  }

  async getLastResponse() {
    await this.page.waitForSelector('[data-testid="agent-response"]:last-child');
    return await this.page.textContent('[data-testid="agent-response"]:last-child');
  }

  async useSuggestedQuestion(questionIndex: number) {
    await this.page.click(`[data-testid="suggested-question-${questionIndex}"]`);
  }

  async toggleVoiceInput() {
    await this.page.click('[data-testid="voice-toggle"]');
  }
}

// Test suites
test.describe('WS6-P4: End-to-End Testing Framework', () => {
  let context: BrowserContext;
  let page: Page;
  let authPage: AuthenticationPage;
  let dashboardPage: DashboardPage;
  let tradingPage: TradingPage;
  let analyticsPage: AnalyticsPage;
  let conversationalPage: ConversationalPage;

  test.beforeAll(async ({ browser }) => {
    context = await browser.newContext();
    page = await context.newPage();
    
    // Initialize page objects
    authPage = new AuthenticationPage(page);
    dashboardPage = new DashboardPage(page);
    tradingPage = new TradingPage(page);
    analyticsPage = new AnalyticsPage(page);
    conversationalPage = new ConversationalPage(page);
  });

  test.afterAll(async () => {
    await context.close();
  });

  test.describe('Authentication Flow', () => {
    test('should complete user registration flow', async () => {
      await authPage.navigateToLogin();
      
      await authPage.register(
        `test+${Date.now()}@alluse.com`,
        TEST_USER.password,
        TEST_USER.name
      );
      
      // Should redirect to dashboard after successful registration
      await expect(page).toHaveURL(/.*dashboard/);
      await expect(page.locator('[data-testid="welcome-message"]')).toBeVisible();
    });

    test('should complete user login flow', async () => {
      await authPage.navigateToLogin();
      await authPage.login(TEST_USER.email, TEST_USER.password);
      
      // Should redirect to dashboard after successful login
      await expect(page).toHaveURL(/.*dashboard/);
      await expect(page.locator('[data-testid="portfolio-overview"]')).toBeVisible();
    });

    test('should handle invalid login credentials', async () => {
      await authPage.navigateToLogin();
      await authPage.login('invalid@email.com', 'wrongpassword');
      
      await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
      await expect(page.locator('[data-testid="error-message"]')).toContainText('Invalid credentials');
    });

    test('should complete logout flow', async () => {
      // First login
      await authPage.navigateToLogin();
      await authPage.login(TEST_USER.email, TEST_USER.password);
      
      // Then logout
      await authPage.logout();
      
      // Should redirect to login page
      await expect(page).toHaveURL(/.*login/);
    });
  });

  test.describe('Dashboard and Portfolio Management', () => {
    test.beforeEach(async () => {
      await authPage.navigateToLogin();
      await authPage.login(TEST_USER.email, TEST_USER.password);
    });

    test('should display portfolio overview correctly', async () => {
      await dashboardPage.navigateToDashboard();
      
      const portfolioValue = await dashboardPage.getPortfolioValue();
      expect(portfolioValue).toMatch(/\$[\d,]+/);
      
      // Check all three account types are displayed
      await expect(page.locator('[data-testid="generation-account"]')).toBeVisible();
      await expect(page.locator('[data-testid="revenue-account"]')).toBeVisible();
      await expect(page.locator('[data-testid="compounding-account"]')).toBeVisible();
    });

    test('should switch between account views', async () => {
      await dashboardPage.navigateToDashboard();
      
      // Switch to detailed view
      await dashboardPage.switchTab('detailed');
      await expect(page.locator('[data-testid="account-details"]')).toBeVisible();
      
      // Switch to performance view
      await dashboardPage.switchTab('performance');
      await expect(page.locator('[data-testid="performance-charts"]')).toBeVisible();
    });

    test('should display real-time performance updates', async () => {
      await dashboardPage.navigateToDashboard();
      
      const initialPerformance = await dashboardPage.getAccountPerformance('generation');
      
      // Wait for real-time update (mock WebSocket data)
      await page.waitForTimeout(5000);
      
      const updatedPerformance = await dashboardPage.getAccountPerformance('generation');
      
      // Performance should update (values may change)
      expect(updatedPerformance).toBeDefined();
    });

    test('should handle portfolio rebalancing workflow', async () => {
      await dashboardPage.navigateToDashboard();
      
      await page.click('[data-testid="rebalance-button"]');
      await expect(page.locator('[data-testid="rebalancing-modal"]')).toBeVisible();
      
      // Adjust allocation sliders
      await page.fill('[data-testid="generation-allocation"]', '35');
      await page.fill('[data-testid="revenue-allocation"]', '30');
      await page.fill('[data-testid="compounding-allocation"]', '35');
      
      await page.click('[data-testid="preview-rebalance"]');
      await expect(page.locator('[data-testid="rebalance-preview"]')).toBeVisible();
      
      await page.click('[data-testid="confirm-rebalance"]');
      await expect(page.locator('[data-testid="rebalance-success"]')).toBeVisible();
    });
  });

  test.describe('Trading Interface', () => {
    test.beforeEach(async () => {
      await authPage.navigateToLogin();
      await authPage.login(TEST_USER.email, TEST_USER.password);
    });

    test('should display current positions and orders', async () => {
      await tradingPage.navigateToTrading();
      
      await expect(page.locator('[data-testid="positions-table"]')).toBeVisible();
      await expect(page.locator('[data-testid="orders-table"]')).toBeVisible();
      
      const positions = await tradingPage.getPositions();
      expect(positions.length).toBeGreaterThan(0);
    });

    test('should place a market order successfully', async () => {
      await tradingPage.navigateToTrading();
      
      await tradingPage.placeOrder('SPY', 10, 'market');
      
      await expect(page.locator('[data-testid="order-success"]')).toBeVisible();
      await expect(page.locator('[data-testid="order-success"]')).toContainText('Order placed successfully');
    });

    test('should place a limit order successfully', async () => {
      await tradingPage.navigateToTrading();
      
      await tradingPage.placeOrder('QQQ', 5, 'limit', 365.00);
      
      await expect(page.locator('[data-testid="order-success"]')).toBeVisible();
      
      // Check that order appears in orders table
      await expect(page.locator('[data-testid="orders-table"]')).toContainText('QQQ');
      await expect(page.locator('[data-testid="orders-table"]')).toContainText('365.00');
    });

    test('should cancel an existing order', async () => {
      await tradingPage.navigateToTrading();
      
      // First place an order
      await tradingPage.placeOrder('IWM', 15, 'limit', 200.00);
      
      // Then cancel it
      const orderId = await page.getAttribute('[data-testid="latest-order"]', 'data-order-id');
      await tradingPage.cancelOrder(orderId!);
      
      await expect(page.locator('[data-testid="cancel-success"]')).toBeVisible();
    });

    test('should handle order validation errors', async () => {
      await tradingPage.navigateToTrading();
      
      await page.click('[data-testid="place-order-button"]');
      await page.click('[data-testid="submit-order-button"]'); // Submit without filling required fields
      
      await expect(page.locator('[data-testid="validation-error"]')).toBeVisible();
      await expect(page.locator('[data-testid="validation-error"]')).toContainText('Symbol is required');
    });

    test('should display real-time market data', async () => {
      await tradingPage.navigateToTrading();
      
      await expect(page.locator('[data-testid="market-data"]')).toBeVisible();
      await expect(page.locator('[data-testid="spy-price"]')).toBeVisible();
      await expect(page.locator('[data-testid="qqq-price"]')).toBeVisible();
      
      // Prices should be updating (check for price format)
      const spyPrice = await page.textContent('[data-testid="spy-price"]');
      expect(spyPrice).toMatch(/\$\d+\.\d{2}/);
    });
  });

  test.describe('Analytics and Reporting', () => {
    test.beforeEach(async () => {
      await authPage.navigateToLogin();
      await authPage.login(TEST_USER.email, TEST_USER.password);
    });

    test('should display performance analytics', async () => {
      await analyticsPage.navigateToAnalytics();
      
      const metrics = await analyticsPage.getPerformanceMetrics();
      
      expect(metrics.totalReturn).toMatch(/[\+\-]?\d+\.\d+%/);
      expect(metrics.sharpeRatio).toMatch(/\d+\.\d+/);
      expect(metrics.maxDrawdown).toMatch(/\d+\.\d+%/);
      expect(metrics.winRate).toMatch(/\d+\.\d+%/);
    });

    test('should filter analytics by timeframe', async () => {
      await analyticsPage.navigateToAnalytics();
      
      await analyticsPage.selectTimeframe('30d');
      await page.waitForTimeout(1000); // Wait for data to update
      
      const monthlyMetrics = await analyticsPage.getPerformanceMetrics();
      
      await analyticsPage.selectTimeframe('90d');
      await page.waitForTimeout(1000);
      
      const quarterlyMetrics = await analyticsPage.getPerformanceMetrics();
      
      // Metrics should be different for different timeframes
      expect(monthlyMetrics.totalReturn).not.toBe(quarterlyMetrics.totalReturn);
    });

    test('should export analytics data', async () => {
      await analyticsPage.navigateToAnalytics();
      
      // Mock download functionality
      const downloadPromise = page.waitForEvent('download');
      await analyticsPage.exportData('csv');
      const download = await downloadPromise;
      
      expect(download.suggestedFilename()).toContain('.csv');
    });

    test('should display week classification history', async () => {
      await analyticsPage.navigateToAnalytics();
      
      await expect(page.locator('[data-testid="week-classification"]')).toBeVisible();
      await expect(page.locator('[data-testid="green-weeks"]')).toBeVisible();
      await expect(page.locator('[data-testid="red-weeks"]')).toBeVisible();
      await expect(page.locator('[data-testid="chop-weeks"]')).toBeVisible();
    });

    test('should show protocol compliance metrics', async () => {
      await analyticsPage.navigateToAnalytics();
      
      await expect(page.locator('[data-testid="protocol-compliance"]')).toBeVisible();
      
      const complianceScore = await page.textContent('[data-testid="compliance-score"]');
      expect(complianceScore).toMatch(/\d+\.\d+%/);
    });
  });

  test.describe('Conversational Interface', () => {
    test.beforeEach(async () => {
      await authPage.navigateToLogin();
      await authPage.login(TEST_USER.email, TEST_USER.password);
    });

    test('should handle basic conversation flow', async () => {
      await conversationalPage.navigateToChat();
      
      await conversationalPage.sendMessage('What is the three-tier structure?');
      
      const response = await conversationalPage.getLastResponse();
      expect(response).toContain('three-tier');
      expect(response).toContain('Generation');
      expect(response).toContain('Revenue');
      expect(response).toContain('Compounding');
    });

    test('should use suggested questions', async () => {
      await conversationalPage.navigateToChat();
      
      await conversationalPage.useSuggestedQuestion(0);
      
      const response = await conversationalPage.getLastResponse();
      expect(response).toBeDefined();
      expect(response!.length).toBeGreaterThan(0);
    });

    test('should maintain conversation context', async () => {
      await conversationalPage.navigateToChat();
      
      await conversationalPage.sendMessage('Tell me about the forking protocol');
      await conversationalPage.getLastResponse();
      
      await conversationalPage.sendMessage('How often does this happen?');
      
      const response = await conversationalPage.getLastResponse();
      expect(response).toContain('forking'); // Should reference previous context
    });

    test('should handle voice input toggle', async () => {
      await conversationalPage.navigateToChat();
      
      await conversationalPage.toggleVoiceInput();
      
      await expect(page.locator('[data-testid="voice-indicator"]')).toBeVisible();
      await expect(page.locator('[data-testid="voice-indicator"]')).toContainText('Listening');
    });

    test('should provide protocol education', async () => {
      await conversationalPage.navigateToChat();
      
      await conversationalPage.sendMessage('Explain delta targeting');
      
      const response = await conversationalPage.getLastResponse();
      expect(response).toContain('delta');
      expect(response).toContain('target');
      expect(response).toContain('risk');
    });
  });

  test.describe('Advanced Features and Enterprise', () => {
    test.beforeEach(async () => {
      await authPage.navigateToLogin();
      await authPage.login(TEST_USER.email, TEST_USER.password);
    });

    test('should access dashboard builder', async () => {
      await page.goto(`${BASE_URL}/dashboard-builder`);
      
      await expect(page.locator('[data-testid="dashboard-builder"]')).toBeVisible();
      await expect(page.locator('[data-testid="widget-palette"]')).toBeVisible();
    });

    test('should customize dashboard layout', async () => {
      await page.goto(`${BASE_URL}/dashboard-builder`);
      
      // Drag widget to dashboard
      const widget = page.locator('[data-testid="portfolio-widget"]');
      const dropZone = page.locator('[data-testid="dashboard-grid"]');
      
      await widget.dragTo(dropZone);
      
      await expect(page.locator('[data-testid="dashboard-grid"] [data-testid="portfolio-widget"]')).toBeVisible();
      
      // Save layout
      await page.click('[data-testid="save-layout"]');
      await expect(page.locator('[data-testid="save-success"]')).toBeVisible();
    });

    test('should access system integration hub', async () => {
      await page.goto(`${BASE_URL}/integration`);
      
      await expect(page.locator('[data-testid="integration-hub"]')).toBeVisible();
      await expect(page.locator('[data-testid="workstream-status"]')).toBeVisible();
    });

    test('should monitor system health', async () => {
      await page.goto(`${BASE_URL}/integration`);
      
      await expect(page.locator('[data-testid="system-health"]')).toBeVisible();
      
      const healthScore = await page.textContent('[data-testid="health-score"]');
      expect(healthScore).toMatch(/\d+\.\d+%/);
    });

    test('should access coordination engine', async () => {
      await page.goto(`${BASE_URL}/coordination`);
      
      await expect(page.locator('[data-testid="coordination-engine"]')).toBeVisible();
      await expect(page.locator('[data-testid="coordination-rules"]')).toBeVisible();
    });
  });

  test.describe('Cross-Browser Compatibility', () => {
    ['chromium', 'firefox', 'webkit'].forEach(browserName => {
      test(`should work correctly in ${browserName}`, async ({ browser }) => {
        const context = await browser.newContext();
        const page = await context.newPage();
        
        await page.goto(`${BASE_URL}/login`);
        await expect(page.locator('[data-testid="login-form"]')).toBeVisible();
        
        await context.close();
      });
    });
  });

  test.describe('Mobile Responsiveness', () => {
    test('should work on mobile viewport', async () => {
      await page.setViewportSize({ width: 375, height: 667 }); // iPhone SE
      
      await authPage.navigateToLogin();
      await authPage.login(TEST_USER.email, TEST_USER.password);
      
      await expect(page.locator('[data-testid="mobile-menu"]')).toBeVisible();
      await expect(page.locator('[data-testid="portfolio-overview"]')).toBeVisible();
    });

    test('should handle tablet viewport', async () => {
      await page.setViewportSize({ width: 768, height: 1024 }); // iPad
      
      await dashboardPage.navigateToDashboard();
      
      await expect(page.locator('[data-testid="tablet-layout"]')).toBeVisible();
      await expect(page.locator('[data-testid="portfolio-overview"]')).toBeVisible();
    });
  });

  test.describe('Accessibility Compliance', () => {
    test('should have proper heading structure', async () => {
      await dashboardPage.navigateToDashboard();
      
      const h1 = await page.locator('h1').count();
      const h2 = await page.locator('h2').count();
      
      expect(h1).toBeGreaterThan(0);
      expect(h2).toBeGreaterThan(0);
    });

    test('should have proper ARIA labels', async () => {
      await conversationalPage.navigateToChat();
      
      const messageInput = page.locator('[data-testid="message-input"]');
      const sendButton = page.locator('[data-testid="send-button"]');
      
      await expect(messageInput).toHaveAttribute('aria-label');
      await expect(sendButton).toHaveAttribute('aria-label');
    });

    test('should support keyboard navigation', async () => {
      await dashboardPage.navigateToDashboard();
      
      await page.keyboard.press('Tab');
      await page.keyboard.press('Tab');
      await page.keyboard.press('Enter');
      
      // Should navigate through interactive elements
      const focusedElement = await page.evaluate(() => document.activeElement?.tagName);
      expect(['BUTTON', 'A', 'INPUT']).toContain(focusedElement);
    });
  });

  test.describe('Performance Testing', () => {
    test('should load pages within performance budget', async () => {
      const startTime = Date.now();
      
      await dashboardPage.navigateToDashboard();
      await page.waitForLoadState('networkidle');
      
      const loadTime = Date.now() - startTime;
      expect(loadTime).toBeLessThan(3000); // Should load within 3 seconds
    });

    test('should handle concurrent users', async () => {
      // Simulate multiple concurrent sessions
      const contexts = await Promise.all([
        context.browser()?.newContext(),
        context.browser()?.newContext(),
        context.browser()?.newContext()
      ]);
      
      const pages = await Promise.all(
        contexts.map(ctx => ctx?.newPage())
      );
      
      // All pages should load successfully
      await Promise.all(
        pages.map(page => page?.goto(`${BASE_URL}/dashboard`))
      );
      
      // Cleanup
      await Promise.all(contexts.map(ctx => ctx?.close()));
    });
  });
});

// Export page objects for reuse
export {
  AuthenticationPage,
  DashboardPage,
  TradingPage,
  AnalyticsPage,
  ConversationalPage
};

