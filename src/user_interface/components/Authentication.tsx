import React, { useState, useEffect } from 'react';
import { User, Mail, Lock, Eye, EyeOff, Shield, CheckCircle, AlertCircle, LogIn, UserPlus } from 'lucide-react';

interface AuthenticatedUser {
  id: string;
  email: string;
  name: string;
  accountTier: 'basic' | 'standard' | 'premium';
  expertiseLevel: 'beginner' | 'intermediate' | 'advanced';
  joinDate: string;
  lastLogin: string;
  protocolAccess: string[];
}

interface AuthenticationProps {
  onAuthenticated: (user: AuthenticatedUser) => void;
  onLogout: () => void;
  className?: string;
}

export const Authentication: React.FC<AuthenticationProps> = ({
  onAuthenticated,
  onLogout,
  className = ''
}) => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [name, setName] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [rememberMe, setRememberMe] = useState(false);

  // Check for existing session on component mount
  useEffect(() => {
    const checkExistingSession = () => {
      const savedUser = localStorage.getItem('alluse_user');
      const sessionExpiry = localStorage.getItem('alluse_session_expiry');
      
      if (savedUser && sessionExpiry) {
        const now = new Date().getTime();
        const expiry = parseInt(sessionExpiry);
        
        if (now < expiry) {
          const user = JSON.parse(savedUser);
          onAuthenticated(user);
        } else {
          // Session expired, clear storage
          localStorage.removeItem('alluse_user');
          localStorage.removeItem('alluse_session_expiry');
        }
      }
    };

    checkExistingSession();
  }, [onAuthenticated]);

  const validateEmail = (email: string): boolean => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const validatePassword = (password: string): { isValid: boolean; strength: string; issues: string[] } => {
    const issues: string[] = [];
    let strength = 'weak';

    if (password.length < 6) {
      issues.push('At least 6 characters required');
    }
    if (!/[A-Z]/.test(password)) {
      issues.push('Include uppercase letter');
    }
    if (!/[a-z]/.test(password)) {
      issues.push('Include lowercase letter');
    }
    if (!/[0-9]/.test(password)) {
      issues.push('Include number');
    }
    if (!/[!@#$%^&*]/.test(password)) {
      issues.push('Include special character');
    }

    if (issues.length === 0) {
      strength = 'strong';
    } else if (issues.length <= 2) {
      strength = 'medium';
    }

    return {
      isValid: password.length >= 6,
      strength,
      issues
    };
  };

  const getPasswordStrengthColor = (strength: string): string => {
    switch (strength) {
      case 'strong': return 'text-green-600';
      case 'medium': return 'text-yellow-600';
      default: return 'text-red-600';
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setIsLoading(true);

    // Validation
    if (!validateEmail(email)) {
      setError('Please enter a valid email address');
      setIsLoading(false);
      return;
    }

    const passwordValidation = validatePassword(password);
    if (!passwordValidation.isValid) {
      setError('Password does not meet requirements');
      setIsLoading(false);
      return;
    }

    if (!isLogin) {
      if (password !== confirmPassword) {
        setError('Passwords do not match');
        setIsLoading(false);
        return;
      }
      if (!name.trim()) {
        setError('Name is required for registration');
        setIsLoading(false);
        return;
      }
    }

    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1500));

      if (isLogin) {
        // Login logic
        if (email === 'demo@alluse.com' && password === 'password123') {
          const user: AuthenticatedUser = {
            id: 'demo-user-001',
            email: 'demo@alluse.com',
            name: 'Demo User',
            accountTier: 'premium',
            expertiseLevel: 'intermediate',
            joinDate: '2024-01-15',
            lastLogin: new Date().toISOString(),
            protocolAccess: ['three-tier', 'forking', 'delta-targeting', 'risk-management']
          };

          // Save session
          const sessionExpiry = new Date().getTime() + (rememberMe ? 30 * 24 * 60 * 60 * 1000 : 24 * 60 * 60 * 1000); // 30 days or 1 day
          localStorage.setItem('alluse_user', JSON.stringify(user));
          localStorage.setItem('alluse_session_expiry', sessionExpiry.toString());

          setSuccess('Login successful! Welcome back.');
          setTimeout(() => onAuthenticated(user), 1000);
        } else {
          setError('Invalid email or password. Try demo@alluse.com / password123');
        }
      } else {
        // Registration logic
        const user: AuthenticatedUser = {
          id: `user-${Date.now()}`,
          email,
          name,
          accountTier: 'basic',
          expertiseLevel: 'beginner',
          joinDate: new Date().toISOString(),
          lastLogin: new Date().toISOString(),
          protocolAccess: ['three-tier']
        };

        // Save session
        const sessionExpiry = new Date().getTime() + (rememberMe ? 30 * 24 * 60 * 60 * 1000 : 24 * 60 * 60 * 1000);
        localStorage.setItem('alluse_user', JSON.stringify(user));
        localStorage.setItem('alluse_session_expiry', sessionExpiry.toString());

        setSuccess('Registration successful! Welcome to ALL-USE.');
        setTimeout(() => onAuthenticated(user), 1000);
      }
    } catch (error) {
      setError('An error occurred. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('alluse_user');
    localStorage.removeItem('alluse_session_expiry');
    onLogout();
  };

  const passwordValidation = validatePassword(password);

  return (
    <div className={`min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4 ${className}`}>
      <div className="max-w-md w-full">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
            <Shield className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">ALL-USE Platform</h1>
          <p className="text-gray-600">
            {isLogin ? 'Sign in to your account' : 'Create your account'}
          </p>
        </div>

        {/* Form */}
        <div className="bg-white rounded-lg shadow-lg p-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Name field (registration only) */}
            {!isLogin && (
              <div>
                <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-2">
                  Full Name
                </label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <input
                    id="name"
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Enter your full name"
                    required={!isLogin}
                  />
                </div>
              </div>
            )}

            {/* Email field */}
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-2">
                Email Address
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Enter your email"
                  required
                />
              </div>
            </div>

            {/* Password field */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-2">
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full pl-10 pr-12 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Enter your password"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
              
              {/* Password strength indicator (registration only) */}
              {!isLogin && password && (
                <div className="mt-2">
                  <div className="flex items-center gap-2">
                    <div className="text-xs">Strength:</div>
                    <div className={`text-xs font-medium ${getPasswordStrengthColor(passwordValidation.strength)}`}>
                      {passwordValidation.strength.toUpperCase()}
                    </div>
                  </div>
                  {passwordValidation.issues.length > 0 && (
                    <div className="mt-1 text-xs text-gray-600">
                      Missing: {passwordValidation.issues.join(', ')}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Confirm password field (registration only) */}
            {!isLogin && (
              <div>
                <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700 mb-2">
                  Confirm Password
                </label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <input
                    id="confirmPassword"
                    type={showConfirmPassword ? 'text' : 'password'}
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    className="w-full pl-10 pr-12 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Confirm your password"
                    required={!isLogin}
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
                  >
                    {showConfirmPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
                {confirmPassword && password !== confirmPassword && (
                  <div className="mt-1 text-xs text-red-600">Passwords do not match</div>
                )}
              </div>
            )}

            {/* Remember me checkbox */}
            <div className="flex items-center justify-between">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={rememberMe}
                  onChange={(e) => setRememberMe(e.target.checked)}
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <span className="ml-2 text-sm text-gray-600">
                  Remember me for {rememberMe ? '30 days' : '1 day'}
                </span>
              </label>
              {isLogin && (
                <button
                  type="button"
                  className="text-sm text-blue-600 hover:text-blue-500"
                >
                  Forgot password?
                </button>
              )}
            </div>

            {/* Error message */}
            {error && (
              <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-lg">
                <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0" />
                <span className="text-sm text-red-700">{error}</span>
              </div>
            )}

            {/* Success message */}
            {success && (
              <div className="flex items-center gap-2 p-3 bg-green-50 border border-green-200 rounded-lg">
                <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0" />
                <span className="text-sm text-green-700">{success}</span>
              </div>
            )}

            {/* Submit button */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  {isLogin ? 'Signing in...' : 'Creating account...'}
                </>
              ) : (
                <>
                  {isLogin ? <LogIn className="w-5 h-5" /> : <UserPlus className="w-5 h-5" />}
                  {isLogin ? 'Sign In' : 'Create Account'}
                </>
              )}
            </button>
          </form>

          {/* Toggle between login and registration */}
          <div className="mt-6 text-center">
            <button
              onClick={() => {
                setIsLogin(!isLogin);
                setError('');
                setSuccess('');
                setPassword('');
                setConfirmPassword('');
                setName('');
              }}
              className="text-sm text-blue-600 hover:text-blue-500"
            >
              {isLogin ? "Don't have an account? Sign up" : 'Already have an account? Sign in'}
            </button>
          </div>

          {/* Demo credentials hint */}
          {isLogin && (
            <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="text-xs text-blue-700">
                <strong>Demo Credentials:</strong><br />
                Email: demo@alluse.com<br />
                Password: password123
              </div>
            </div>
          )}

          {/* Security notice */}
          <div className="mt-6 text-center">
            <div className="flex items-center justify-center gap-2 text-xs text-gray-500">
              <Shield className="w-4 h-4" />
              <span>Protected by bank-level encryption</span>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-8 text-sm text-gray-600">
          <p>By signing in, you agree to our Terms of Service and Privacy Policy</p>
        </div>
      </div>
    </div>
  );
};

