// Authentication Service Library
// Provides secure authentication and session management for ALL-USE platform

export interface User {
  id: string;
  email: string;
  name: string;
  accountTier: 'basic' | 'standard' | 'premium';
  expertiseLevel: 'beginner' | 'intermediate' | 'advanced';
  joinDate: string;
  lastLogin: string;
  protocolAccess: string[];
  preferences: {
    theme: 'light' | 'dark';
    notifications: boolean;
    speechEnabled: boolean;
    defaultTimeframe: 'week' | 'month' | 'quarter' | 'year';
  };
}

export interface AuthenticationResult {
  success: boolean;
  user?: User;
  token?: string;
  error?: string;
  expiresAt?: Date;
}

export interface SessionInfo {
  isAuthenticated: boolean;
  user: User | null;
  expiresAt: Date | null;
  timeRemaining: number; // in milliseconds
}

export class AuthenticationService {
  private static instance: AuthenticationService;
  private currentUser: User | null = null;
  private sessionExpiry: Date | null = null;
  private sessionCheckInterval: NodeJS.Timeout | null = null;

  private constructor() {
    this.initializeSessionCheck();
  }

  static getInstance(): AuthenticationService {
    if (!AuthenticationService.instance) {
      AuthenticationService.instance = new AuthenticationService();
    }
    return AuthenticationService.instance;
  }

  private initializeSessionCheck(): void {
    // Check session every minute
    this.sessionCheckInterval = setInterval(() => {
      this.validateSession();
    }, 60000);

    // Check for existing session on initialization
    this.loadSessionFromStorage();
  }

  private loadSessionFromStorage(): void {
    try {
      const savedUser = localStorage.getItem('alluse_user');
      const sessionExpiry = localStorage.getItem('alluse_session_expiry');

      if (savedUser && sessionExpiry) {
        const expiry = new Date(parseInt(sessionExpiry));
        const now = new Date();

        if (now < expiry) {
          this.currentUser = JSON.parse(savedUser);
          this.sessionExpiry = expiry;
        } else {
          this.clearSession();
        }
      }
    } catch (error) {
      console.error('Error loading session from storage:', error);
      this.clearSession();
    }
  }

  private saveSessionToStorage(user: User, expiresAt: Date): void {
    try {
      localStorage.setItem('alluse_user', JSON.stringify(user));
      localStorage.setItem('alluse_session_expiry', expiresAt.getTime().toString());
    } catch (error) {
      console.error('Error saving session to storage:', error);
    }
  }

  private clearSession(): void {
    this.currentUser = null;
    this.sessionExpiry = null;
    localStorage.removeItem('alluse_user');
    localStorage.removeItem('alluse_session_expiry');
  }

  private validateSession(): boolean {
    if (!this.currentUser || !this.sessionExpiry) {
      return false;
    }

    const now = new Date();
    if (now >= this.sessionExpiry) {
      this.clearSession();
      return false;
    }

    return true;
  }

  async login(email: string, password: string, rememberMe: boolean = false): Promise<AuthenticationResult> {
    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Demo authentication logic
      if (email === 'demo@alluse.com' && password === 'password123') {
        const user: User = {
          id: 'demo-user-001',
          email: 'demo@alluse.com',
          name: 'Demo User',
          accountTier: 'premium',
          expertiseLevel: 'intermediate',
          joinDate: '2024-01-15T00:00:00Z',
          lastLogin: new Date().toISOString(),
          protocolAccess: ['three-tier', 'forking', 'delta-targeting', 'risk-management', 'advanced-analytics'],
          preferences: {
            theme: 'light',
            notifications: true,
            speechEnabled: false,
            defaultTimeframe: 'month'
          }
        };

        const sessionDuration = rememberMe ? 30 * 24 * 60 * 60 * 1000 : 24 * 60 * 60 * 1000; // 30 days or 1 day
        const expiresAt = new Date(Date.now() + sessionDuration);

        this.currentUser = user;
        this.sessionExpiry = expiresAt;
        this.saveSessionToStorage(user, expiresAt);

        return {
          success: true,
          user,
          token: `demo-token-${Date.now()}`,
          expiresAt
        };
      }

      // Additional demo users for testing
      const demoUsers: { [key: string]: { password: string; user: Omit<User, 'lastLogin'> } } = {
        'beginner@alluse.com': {
          password: 'beginner123',
          user: {
            id: 'beginner-user-001',
            email: 'beginner@alluse.com',
            name: 'Beginner User',
            accountTier: 'basic',
            expertiseLevel: 'beginner',
            joinDate: '2025-06-01T00:00:00Z',
            protocolAccess: ['three-tier'],
            preferences: {
              theme: 'light',
              notifications: true,
              speechEnabled: true,
              defaultTimeframe: 'week'
            }
          }
        },
        'advanced@alluse.com': {
          password: 'advanced123',
          user: {
            id: 'advanced-user-001',
            email: 'advanced@alluse.com',
            name: 'Advanced User',
            accountTier: 'premium',
            expertiseLevel: 'advanced',
            joinDate: '2023-03-10T00:00:00Z',
            protocolAccess: ['three-tier', 'forking', 'delta-targeting', 'risk-management', 'advanced-analytics', 'portfolio-optimization'],
            preferences: {
              theme: 'dark',
              notifications: false,
              speechEnabled: false,
              defaultTimeframe: 'quarter'
            }
          }
        }
      };

      const demoUser = demoUsers[email];
      if (demoUser && demoUser.password === password) {
        const user: User = {
          ...demoUser.user,
          lastLogin: new Date().toISOString()
        };

        const sessionDuration = rememberMe ? 30 * 24 * 60 * 60 * 1000 : 24 * 60 * 60 * 1000;
        const expiresAt = new Date(Date.now() + sessionDuration);

        this.currentUser = user;
        this.sessionExpiry = expiresAt;
        this.saveSessionToStorage(user, expiresAt);

        return {
          success: true,
          user,
          token: `token-${user.id}-${Date.now()}`,
          expiresAt
        };
      }

      return {
        success: false,
        error: 'Invalid email or password'
      };
    } catch (error) {
      return {
        success: false,
        error: 'Authentication service temporarily unavailable'
      };
    }
  }

  async register(email: string, password: string, name: string, rememberMe: boolean = false): Promise<AuthenticationResult> {
    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1200));

      // Basic validation
      if (!this.isValidEmail(email)) {
        return {
          success: false,
          error: 'Invalid email format'
        };
      }

      if (!this.isValidPassword(password)) {
        return {
          success: false,
          error: 'Password does not meet requirements'
        };
      }

      if (!name.trim()) {
        return {
          success: false,
          error: 'Name is required'
        };
      }

      // Check if user already exists (demo logic)
      const existingEmails = ['demo@alluse.com', 'beginner@alluse.com', 'advanced@alluse.com'];
      if (existingEmails.includes(email)) {
        return {
          success: false,
          error: 'User with this email already exists'
        };
      }

      // Create new user
      const user: User = {
        id: `user-${Date.now()}`,
        email,
        name: name.trim(),
        accountTier: 'basic',
        expertiseLevel: 'beginner',
        joinDate: new Date().toISOString(),
        lastLogin: new Date().toISOString(),
        protocolAccess: ['three-tier'],
        preferences: {
          theme: 'light',
          notifications: true,
          speechEnabled: false,
          defaultTimeframe: 'month'
        }
      };

      const sessionDuration = rememberMe ? 30 * 24 * 60 * 60 * 1000 : 24 * 60 * 60 * 1000;
      const expiresAt = new Date(Date.now() + sessionDuration);

      this.currentUser = user;
      this.sessionExpiry = expiresAt;
      this.saveSessionToStorage(user, expiresAt);

      return {
        success: true,
        user,
        token: `token-${user.id}-${Date.now()}`,
        expiresAt
      };
    } catch (error) {
      return {
        success: false,
        error: 'Registration service temporarily unavailable'
      };
    }
  }

  logout(): void {
    this.clearSession();
  }

  getCurrentUser(): User | null {
    if (!this.validateSession()) {
      return null;
    }
    return this.currentUser;
  }

  getSessionInfo(): SessionInfo {
    const isAuthenticated = this.validateSession();
    const timeRemaining = this.sessionExpiry ? Math.max(0, this.sessionExpiry.getTime() - Date.now()) : 0;

    return {
      isAuthenticated,
      user: this.currentUser,
      expiresAt: this.sessionExpiry,
      timeRemaining
    };
  }

  async updateUserPreferences(preferences: Partial<User['preferences']>): Promise<boolean> {
    if (!this.currentUser) {
      return false;
    }

    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 300));

      this.currentUser.preferences = {
        ...this.currentUser.preferences,
        ...preferences
      };

      // Update storage
      if (this.sessionExpiry) {
        this.saveSessionToStorage(this.currentUser, this.sessionExpiry);
      }

      return true;
    } catch (error) {
      console.error('Error updating user preferences:', error);
      return false;
    }
  }

  async changePassword(currentPassword: string, newPassword: string): Promise<{ success: boolean; error?: string }> {
    if (!this.currentUser) {
      return { success: false, error: 'Not authenticated' };
    }

    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 800));

      // Validate current password (demo logic)
      if (this.currentUser.email === 'demo@alluse.com' && currentPassword !== 'password123') {
        return { success: false, error: 'Current password is incorrect' };
      }

      if (!this.isValidPassword(newPassword)) {
        return { success: false, error: 'New password does not meet requirements' };
      }

      // In real implementation, this would update the password on the server
      return { success: true };
    } catch (error) {
      return { success: false, error: 'Password change service temporarily unavailable' };
    }
  }

  extendSession(duration: number = 24 * 60 * 60 * 1000): boolean {
    if (!this.validateSession() || !this.currentUser) {
      return false;
    }

    const newExpiry = new Date(Date.now() + duration);
    this.sessionExpiry = newExpiry;
    this.saveSessionToStorage(this.currentUser, newExpiry);
    return true;
  }

  private isValidEmail(email: string): boolean {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }

  private isValidPassword(password: string): boolean {
    // Minimum 6 characters
    return password.length >= 6;
  }

  destroy(): void {
    if (this.sessionCheckInterval) {
      clearInterval(this.sessionCheckInterval);
      this.sessionCheckInterval = null;
    }
    this.clearSession();
  }
}

// Singleton instance for global access
export const authService = AuthenticationService.getInstance();

