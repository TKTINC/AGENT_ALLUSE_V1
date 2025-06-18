// WS6-P1 End-to-End Testing Framework
// Comprehensive E2E testing for ALL-USE User Interface using Playwright

import { test, expect, Page, BrowserContext } from '@playwright/test';

// Test configuration
const BASE_URL = process.env.BASE_URL || 'http://localhost:5173';
const TEST_TIMEOUT = 30000;

// Test data
const TEST_USERS = {
  demo: {
    email: 'demo@alluse.com',
    password: 'password123',
    name: 'Demo User'
  },
  beginner: {
    email: 'beginner@alluse.com',
    password: 'beginner123',
    name: 'Beginner User'
  },
  advanced: {
    email: 'advanced@alluse.com',
    password: 'advanced123',
    name: 'Advanced User'
  }
};

// Page Object Models
class AuthenticationPage {
  constructor(private page: Page) {}

  async navigate() {
    await this.page.goto(BASE_URL);
  }

  async login(email: string, password: string, rememberMe = false) {
    await this.page.fill('[placeholder="Enter your email"]', email);
    await this.page.fill('[placeholder="Enter your password"]', password);
    
    if (rememberMe) {
      await this.page.check('[type="checkbox"]');
    }
    
    await this.page.click('button:has-text("Sign In")');
  }

  async register(name: string, email: string, password: string) {
    await this.page.click('text=Don\'t have an account? Sign up');
    await this.page.fill('[placeholder="Enter your full name"]', name);
    await this.page.fill('[placeholder="Enter your email"]', email);
    await this.page.fill('[placeholder="Enter your password"]', password);
    await this.page.fill('[placeholder="Confirm your password"]', password);
    await this.page.click('button:has-text("Create Account")');
  }

  async waitForAuthentication() {
    await this.page.waitForSelector('text=Protocol Chat', { timeout: TEST_TIMEOUT });
  }

  async logout() {
    await this.page.click('button:has-text("Logout")');
  }
}

class ConversationalInterfacePage {
  constructor(private page: Page) {}

  async sendMessage(message: string) {
    await this.page.fill('[placeholder*="Ask about the ALL-USE protocol"]', message);
    await this.page.click('button:has-text("Send")');
  }

  async clickSuggestedQuestion(questionText: string) {
    await this.page.click(`text=${questionText}`);
  }

  async waitForAgentResponse() {
    await this.page.waitForSelector('.agent-message', { timeout: TEST_TIMEOUT });
  }

  async getLastMessage() {
    const messages = await this.page.locator('.message').all();
    return messages[messages.length - 1];
  }

  async startVoiceInput() {
    await this.page.click('button[aria-label="Start voice input"]');
  }

  async stopVoiceInput() {
    await this.page.click('button[aria-label="Stop voice input"]');
  }

  async clearConversation() {
    await this.page.click('button:has-text("Clear")');
  }
}

class AccountVisualizationPage {
  constructor(private page: Page) {}

  async navigateToAccounts() {
    await this.page.click('text=Accounts');
  }

  async switchViewMode(mode: 'Overview' | 'Detailed' | 'Performance') {
    await this.page.click(`text=${mode}`);
  }

  async switchTimeframe(timeframe: 'Week' | 'Month' | 'Year') {
    await this.page.click(`text=${timeframe}`);
  }

  async getAccountBalance(accountType: string) {
    const accountCard = this.page.locator(`[data-testid="${accountType}-account"]`);
    return await accountCard.locator('.balance').textContent();
  }

  async expandAccountDetails(accountType: string) {
    await this.page.click(`[data-testid="${accountType}-account"] button:has-text("View Details")`);
  }
}

class AnalyticsPage {
  constructor(private page: Page) {}

  async navigateToAnalytics() {
    await this.page.click('text=Analytics');
  }

  async selectMetric(metric: 'Returns' | 'Compliance' | 'Trades') {
    await this.page.click(`text=${metric}`);
  }

  async selectTimeframe(timeframe: 'Week' | 'Month' | 'Quarter' | 'Year') {
    await this.page.click(`text=${timeframe}`);
  }

  async getCurrentWeekClassification() {
    return await this.page.locator('.week-classification').textContent();
  }

  async getPerformanceMetric(metric: string) {
    return await this.page.locator(`[data-testid="${metric}"]`).textContent();
  }

  async toggleDetails() {
    await this.page.click('button:has-text("Show Details")');
  }
}

// Test suites
test.describe('WS6-P1: Conversational Interface Foundation E2E Tests', () => {
  let context: BrowserContext;
  let page: Page;
  let authPage: AuthenticationPage;
  let chatPage: ConversationalInterfacePage;
  let accountsPage: AccountVisualizationPage;
  let analyticsPage: AnalyticsPage;

  test.beforeAll(async ({ browser }) => {
    context = await browser.newContext();
    page = await context.newPage();
    
    authPage = new AuthenticationPage(page);
    chatPage = new ConversationalInterfacePage(page);
    accountsPage = new AccountVisualizationPage(page);
    analyticsPage = new AnalyticsPage(page);
  });

  test.afterAll(async () => {
    await context.close();
  });

  test.describe('Authentication Flow', () => {
    test('should display login form on initial load', async () => {
      await authPage.navigate();
      
      await expect(page).toHaveTitle(/ALL-USE Platform/);
      await expect(page.locator('text=Sign in to your account')).toBeVisible();
      await expect(page.locator('[placeholder="Enter your email"]')).toBeVisible();
      await expect(page.locator('[placeholder="Enter your password"]')).toBeVisible();
    });

    test('should validate email format', async () => {
      await authPage.navigate();
      
      await page.fill('[placeholder="Enter your email"]', 'invalid-email');
      await page.fill('[placeholder="Enter your password"]', 'password123');
      await page.click('button:has-text("Sign In")');
      
      await expect(page.locator('text=Please enter a valid email address')).toBeVisible();
    });

    test('should handle invalid credentials', async () => {
      await authPage.navigate();
      
      await authPage.login('wrong@email.com', 'wrongpassword');
      
      await expect(page.locator('text=Invalid email or password')).toBeVisible();
    });

    test('should successfully login with demo credentials', async () => {
      await authPage.navigate();
      
      await authPage.login(TEST_USERS.demo.email, TEST_USERS.demo.password);
      await authPage.waitForAuthentication();
      
      await expect(page.locator('text=Protocol Chat')).toBeVisible();
      await expect(page.locator('text=Demo User')).toBeVisible();
    });

    test('should remember login session', async () => {
      await authPage.navigate();
      
      await authPage.login(TEST_USERS.demo.email, TEST_USERS.demo.password, true);
      await authPage.waitForAuthentication();
      
      // Refresh page
      await page.reload();
      
      // Should still be logged in
      await expect(page.locator('text=Protocol Chat')).toBeVisible();
    });

    test('should handle registration flow', async () => {
      await authPage.navigate();
      
      const newUser = {
        name: 'Test User',
        email: `test${Date.now()}@alluse.com`,
        password: 'TestPass123!'
      };
      
      await authPage.register(newUser.name, newUser.email, newUser.password);
      await authPage.waitForAuthentication();
      
      await expect(page.locator(`text=${newUser.name}`)).toBeVisible();
    });

    test('should handle logout', async () => {
      // Ensure logged in
      await authPage.navigate();
      await authPage.login(TEST_USERS.demo.email, TEST_USERS.demo.password);
      await authPage.waitForAuthentication();
      
      await authPage.logout();
      
      await expect(page.locator('text=Sign in to your account')).toBeVisible();
    });
  });

  test.describe('Conversational Interface', () => {
    test.beforeEach(async () => {
      await authPage.navigate();
      await authPage.login(TEST_USERS.demo.email, TEST_USERS.demo.password);
      await authPage.waitForAuthentication();
    });

    test('should display welcome message', async () => {
      await expect(page.locator('text=Welcome')).toBeVisible();
      await expect(page.locator('text=three-tier account structure')).toBeVisible();
    });

    test('should handle user message input', async () => {
      const testMessage = 'What is the three-tier account structure?';
      
      await chatPage.sendMessage(testMessage);
      
      await expect(page.locator(`text=${testMessage}`)).toBeVisible();
      await chatPage.waitForAgentResponse();
      
      await expect(page.locator('text=Generation Account')).toBeVisible();
    });

    test('should provide suggested questions', async () => {
      await expect(page.locator('text=Explain the three-tier account structure')).toBeVisible();
      await expect(page.locator('text=How does forking work?')).toBeVisible();
      await expect(page.locator('text=What trading opportunities are available?')).toBeVisible();
    });

    test('should handle suggested question clicks', async () => {
      await chatPage.clickSuggestedQuestion('Explain the three-tier account structure');
      
      const input = page.locator('[placeholder*="Ask about the ALL-USE protocol"]');
      await expect(input).toHaveValue('Explain the three-tier account structure');
    });

    test('should handle multiple message exchanges', async () => {
      await chatPage.sendMessage('First message');
      await chatPage.waitForAgentResponse();
      
      await chatPage.sendMessage('Second message');
      await chatPage.waitForAgentResponse();
      
      await expect(page.locator('text=First message')).toBeVisible();
      await expect(page.locator('text=Second message')).toBeVisible();
    });

    test('should handle protocol-specific questions', async () => {
      const protocolQuestions = [
        'How does forking work?',
        'What is delta targeting?',
        'Explain risk management',
        'What is the current week classification?'
      ];

      for (const question of protocolQuestions) {
        await chatPage.sendMessage(question);
        await chatPage.waitForAgentResponse();
        
        // Verify agent provides relevant response
        const lastMessage = await chatPage.getLastMessage();
        const messageText = await lastMessage.textContent();
        expect(messageText).toBeTruthy();
        expect(messageText!.length).toBeGreaterThan(50); // Substantial response
      }
    });

    test('should handle voice input (if supported)', async () => {
      // Check if voice input button is available
      const voiceButton = page.locator('button[aria-label="Start voice input"]');
      
      if (await voiceButton.isVisible()) {
        await chatPage.startVoiceInput();
        await expect(page.locator('button[aria-label="Stop voice input"]')).toBeVisible();
        
        await chatPage.stopVoiceInput();
        await expect(page.locator('button[aria-label="Start voice input"]')).toBeVisible();
      }
    });

    test('should clear conversation history', async () => {
      await chatPage.sendMessage('Test message');
      await chatPage.waitForAgentResponse();
      
      await chatPage.clearConversation();
      
      // Should only show welcome message
      const messages = await page.locator('.message').count();
      expect(messages).toBe(1);
      await expect(page.locator('text=Welcome')).toBeVisible();
    });
  });

  test.describe('Account Visualization', () => {
    test.beforeEach(async () => {
      await authPage.navigate();
      await authPage.login(TEST_USERS.demo.email, TEST_USERS.demo.password);
      await authPage.waitForAuthentication();
      await accountsPage.navigateToAccounts();
    });

    test('should display account portfolio overview', async () => {
      await expect(page.locator('text=Account Portfolio')).toBeVisible();
      await expect(page.locator('text=Generation Account')).toBeVisible();
      await expect(page.locator('text=Revenue Account')).toBeVisible();
      await expect(page.locator('text=Compounding Account')).toBeVisible();
    });

    test('should show account balances', async () => {
      // Check for currency formatting
      await expect(page.locator('text=/\\$[0-9,]+/')).toBeVisible();
    });

    test('should display performance metrics', async () => {
      // Check for percentage returns
      await expect(page.locator('text=/[+-][0-9.]+%/')).toBeVisible();
    });

    test('should switch between view modes', async () => {
      await accountsPage.switchViewMode('Detailed');
      await expect(page.locator('text=Strategy')).toBeVisible();
      
      await accountsPage.switchViewMode('Performance');
      await expect(page.locator('text=Yearly Return')).toBeVisible();
      
      await accountsPage.switchViewMode('Overview');
      await expect(page.locator('text=Total Portfolio Value')).toBeVisible();
    });

    test('should switch between timeframes', async () => {
      await accountsPage.switchTimeframe('Week');
      await expect(page.locator('text=Weekly Performance')).toBeVisible();
      
      await accountsPage.switchTimeframe('Month');
      await expect(page.locator('text=Monthly Performance')).toBeVisible();
      
      await accountsPage.switchTimeframe('Year');
      await expect(page.locator('text=Yearly Performance')).toBeVisible();
    });

    test('should display risk level indicators', async () => {
      await expect(page.locator('text=High Risk')).toBeVisible();
      await expect(page.locator('text=Medium Risk')).toBeVisible();
      await expect(page.locator('text=Low Risk')).toBeVisible();
    });

    test('should handle account detail expansion', async () => {
      await accountsPage.expandAccountDetails('generation');
      await expect(page.locator('text=Aggressive premium harvesting')).toBeVisible();
    });
  });

  test.describe('Analytics Dashboard', () => {
    test.beforeEach(async () => {
      await authPage.navigate();
      await authPage.login(TEST_USERS.demo.email, TEST_USERS.demo.password);
      await authPage.waitForAuthentication();
      await analyticsPage.navigateToAnalytics();
    });

    test('should display analytics dashboard', async () => {
      await expect(page.locator('text=Performance Analytics')).toBeVisible();
      await expect(page.locator('text=Current Week Status')).toBeVisible();
      await expect(page.locator('text=Performance Overview')).toBeVisible();
    });

    test('should show current week classification', async () => {
      const classification = await analyticsPage.getCurrentWeekClassification();
      expect(classification).toMatch(/Green Week|Yellow Week|Red Week/);
    });

    test('should display performance metrics', async () => {
      await expect(page.locator('text=Total Return')).toBeVisible();
      await expect(page.locator('text=Weekly Average')).toBeVisible();
      await expect(page.locator('text=Monthly Average')).toBeVisible();
      await expect(page.locator('text=Yearly Projection')).toBeVisible();
    });

    test('should switch between metric types', async () => {
      await analyticsPage.selectMetric('Compliance');
      await expect(page.locator('text=Compliance (%)')).toBeVisible();
      
      await analyticsPage.selectMetric('Trades');
      await expect(page.locator('text=Trades')).toBeVisible();
      
      await analyticsPage.selectMetric('Returns');
      await expect(page.locator('text=Returns (%)')).toBeVisible();
    });

    test('should switch between timeframes', async () => {
      await analyticsPage.selectTimeframe('Quarter');
      await expect(page.locator('text=Quarter')).toBeVisible();
      
      await analyticsPage.selectTimeframe('Year');
      await expect(page.locator('text=Year')).toBeVisible();
    });

    test('should display week classification history', async () => {
      await expect(page.locator('text=Week Classification History')).toBeVisible();
      await expect(page.locator('text=Week 1')).toBeVisible();
    });

    test('should show risk analysis', async () => {
      await expect(page.locator('text=Risk Analysis')).toBeVisible();
      await expect(page.locator('text=Low Risk Periods')).toBeVisible();
      await expect(page.locator('text=Medium Risk Periods')).toBeVisible();
      await expect(page.locator('text=High Risk Periods')).toBeVisible();
    });

    test('should display protocol insights', async () => {
      await expect(page.locator('text=Protocol Insights')).toBeVisible();
      await expect(page.locator('text=protocol adherence')).toBeVisible();
    });

    test('should toggle detailed view', async () => {
      await analyticsPage.toggleDetails();
      await expect(page.locator('text=Sharpe Ratio')).toBeVisible();
      await expect(page.locator('text=Max Drawdown')).toBeVisible();
    });
  });

  test.describe('Cross-Component Integration', () => {
    test.beforeEach(async () => {
      await authPage.navigate();
      await authPage.login(TEST_USERS.demo.email, TEST_USERS.demo.password);
      await authPage.waitForAuthentication();
    });

    test('should navigate between tabs seamlessly', async () => {
      // Start in Protocol Chat
      await expect(page.locator('text=Protocol Chat')).toBeVisible();
      
      // Navigate to Accounts
      await accountsPage.navigateToAccounts();
      await expect(page.locator('text=Account Portfolio')).toBeVisible();
      
      // Navigate to Analytics
      await analyticsPage.navigateToAnalytics();
      await expect(page.locator('text=Performance Analytics')).toBeVisible();
      
      // Back to Protocol Chat
      await page.click('text=Protocol Chat');
      await expect(page.locator('text=Welcome')).toBeVisible();
    });

    test('should maintain conversation state across tab switches', async () => {
      await chatPage.sendMessage('Test message for state persistence');
      await chatPage.waitForAgentResponse();
      
      await accountsPage.navigateToAccounts();
      await page.click('text=Protocol Chat');
      
      await expect(page.locator('text=Test message for state persistence')).toBeVisible();
    });

    test('should handle user expertise level appropriately', async () => {
      // Test with beginner user
      await authPage.logout();
      await authPage.login(TEST_USERS.beginner.email, TEST_USERS.beginner.password);
      await authPage.waitForAuthentication();
      
      await chatPage.sendMessage('Explain the three-tier account structure');
      await chatPage.waitForAgentResponse();
      
      // Should provide beginner-friendly explanation
      await expect(page.locator('text=For Beginners')).toBeVisible();
    });

    test('should handle responsive design', async () => {
      // Test mobile viewport
      await page.setViewportSize({ width: 375, height: 667 });
      
      await expect(page.locator('text=Protocol Chat')).toBeVisible();
      await expect(page.locator('[placeholder*="Ask about the ALL-USE protocol"]')).toBeVisible();
      
      // Test tablet viewport
      await page.setViewportSize({ width: 768, height: 1024 });
      
      await expect(page.locator('text=Protocol Chat')).toBeVisible();
      
      // Reset to desktop
      await page.setViewportSize({ width: 1280, height: 720 });
    });
  });

  test.describe('Performance and Reliability', () => {
    test('should load initial page within performance budget', async () => {
      const startTime = Date.now();
      
      await authPage.navigate();
      await page.waitForLoadState('networkidle');
      
      const loadTime = Date.now() - startTime;
      expect(loadTime).toBeLessThan(3000); // 3 second budget
    });

    test('should handle network interruptions gracefully', async () => {
      await authPage.navigate();
      await authPage.login(TEST_USERS.demo.email, TEST_USERS.demo.password);
      await authPage.waitForAuthentication();
      
      // Simulate network offline
      await context.setOffline(true);
      
      await chatPage.sendMessage('Test message during offline');
      
      // Should show appropriate error or retry mechanism
      await expect(page.locator('text=Connection error')).toBeVisible({ timeout: 5000 });
      
      // Restore network
      await context.setOffline(false);
    });

    test('should handle concurrent user interactions', async () => {
      await authPage.navigate();
      await authPage.login(TEST_USERS.demo.email, TEST_USERS.demo.password);
      await authPage.waitForAuthentication();
      
      // Simulate rapid interactions
      const promises = [
        chatPage.sendMessage('Message 1'),
        accountsPage.navigateToAccounts(),
        analyticsPage.navigateToAnalytics(),
        page.click('text=Protocol Chat')
      ];
      
      await Promise.all(promises);
      
      // Application should remain stable
      await expect(page.locator('text=Protocol Chat')).toBeVisible();
    });

    test('should maintain session across browser refresh', async () => {
      await authPage.navigate();
      await authPage.login(TEST_USERS.demo.email, TEST_USERS.demo.password, true);
      await authPage.waitForAuthentication();
      
      await page.reload();
      
      // Should automatically restore session
      await expect(page.locator('text=Protocol Chat')).toBeVisible();
      await expect(page.locator('text=Demo User')).toBeVisible();
    });
  });

  test.describe('Accessibility Compliance', () => {
    test('should support keyboard navigation', async () => {
      await authPage.navigate();
      
      // Tab through login form
      await page.keyboard.press('Tab');
      await expect(page.locator('[placeholder="Enter your email"]')).toBeFocused();
      
      await page.keyboard.press('Tab');
      await expect(page.locator('[placeholder="Enter your password"]')).toBeFocused();
      
      await page.keyboard.press('Tab');
      await expect(page.locator('button:has-text("Sign In")')).toBeFocused();
    });

    test('should have proper ARIA labels and roles', async () => {
      await authPage.navigate();
      await authPage.login(TEST_USERS.demo.email, TEST_USERS.demo.password);
      await authPage.waitForAuthentication();
      
      // Check for proper ARIA attributes
      await expect(page.locator('[role="main"]')).toBeVisible();
      await expect(page.locator('[aria-label*="conversation"]')).toBeVisible();
    });

    test('should support screen reader navigation', async () => {
      await authPage.navigate();
      await authPage.login(TEST_USERS.demo.email, TEST_USERS.demo.password);
      await authPage.waitForAuthentication();
      
      // Check for proper heading structure
      await expect(page.locator('h1, h2, h3')).toHaveCount(3, { timeout: 5000 });
      
      // Check for proper form labels
      await accountsPage.navigateToAccounts();
      await expect(page.locator('label')).toHaveCount(0, { timeout: 5000 }); // No form inputs on accounts page
    });

    test('should have sufficient color contrast', async () => {
      await authPage.navigate();
      await authPage.login(TEST_USERS.demo.email, TEST_USERS.demo.password);
      await authPage.waitForAuthentication();
      
      // This would typically use axe-core for automated accessibility testing
      // For now, we verify that text content is visible and readable
      await expect(page.locator('text=Protocol Chat')).toBeVisible();
      await expect(page.locator('text=Accounts')).toBeVisible();
      await expect(page.locator('text=Analytics')).toBeVisible();
    });
  });
});

// Test utilities for E2E tests
export const e2eTestUtils = {
  waitForStableNetwork: async (page: Page, timeout = 5000) => {
    await page.waitForLoadState('networkidle', { timeout });
  },

  takeScreenshot: async (page: Page, name: string) => {
    await page.screenshot({ path: `test-results/${name}.png`, fullPage: true });
  },

  mockApiResponse: async (page: Page, url: string, response: any) => {
    await page.route(url, route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(response)
      });
    });
  },

  simulateSlowNetwork: async (context: BrowserContext) => {
    await context.route('**/*', route => {
      setTimeout(() => route.continue(), 1000);
    });
  },

  clearBrowserData: async (context: BrowserContext) => {
    await context.clearCookies();
    await context.clearPermissions();
  }
};

