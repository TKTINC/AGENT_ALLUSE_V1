// Advanced Customization System for WS6-P3
// Comprehensive user preferences, themes, layouts, and personalization

import React, { useState, useEffect, useCallback, createContext, useContext } from 'react';
import { 
  Palette, 
  Layout, 
  Settings, 
  Monitor, 
  Smartphone, 
  Tablet, 
  Sun, 
  Moon, 
  Eye, 
  EyeOff,
  Grid,
  List,
  BarChart3,
  PieChart,
  TrendingUp,
  Zap,
  Shield,
  Bell,
  Volume2,
  VolumeX,
  Keyboard,
  Mouse,
  Accessibility,
  Languages,
  Clock,
  MapPin,
  Download,
  Upload,
  RotateCcw,
  Save,
  Share2,
  Copy,
  Check
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// Customization Types
export interface Theme {
  id: string;
  name: string;
  description: string;
  colors: {
    primary: string;
    secondary: string;
    accent: string;
    background: string;
    surface: string;
    text: string;
    textSecondary: string;
    border: string;
    success: string;
    warning: string;
    error: string;
    info: string;
  };
  fonts: {
    primary: string;
    secondary: string;
    mono: string;
  };
  spacing: {
    xs: string;
    sm: string;
    md: string;
    lg: string;
    xl: string;
  };
  borderRadius: {
    sm: string;
    md: string;
    lg: string;
  };
  shadows: {
    sm: string;
    md: string;
    lg: string;
  };
}

export interface LayoutPreferences {
  sidebarPosition: 'left' | 'right' | 'hidden';
  sidebarWidth: number;
  headerHeight: number;
  footerVisible: boolean;
  compactMode: boolean;
  gridDensity: 'comfortable' | 'compact' | 'spacious';
  cardStyle: 'elevated' | 'outlined' | 'filled';
  animationsEnabled: boolean;
  transitionSpeed: 'slow' | 'normal' | 'fast';
}

export interface DisplayPreferences {
  colorMode: 'light' | 'dark' | 'auto';
  highContrast: boolean;
  reducedMotion: boolean;
  fontSize: 'small' | 'medium' | 'large' | 'extra-large';
  lineHeight: 'tight' | 'normal' | 'relaxed';
  colorBlindnessSupport: 'none' | 'protanopia' | 'deuteranopia' | 'tritanopia';
  screenReaderOptimized: boolean;
}

export interface NotificationPreferences {
  enabled: boolean;
  sound: boolean;
  desktop: boolean;
  email: boolean;
  sms: boolean;
  orderUpdates: boolean;
  marketAlerts: boolean;
  newsUpdates: boolean;
  systemMessages: boolean;
  quietHours: {
    enabled: boolean;
    start: string;
    end: string;
  };
}

export interface TradingPreferences {
  defaultOrderType: 'market' | 'limit' | 'stop' | 'stop_limit';
  defaultTimeInForce: 'day' | 'gtc' | 'ioc' | 'fok';
  confirmationRequired: boolean;
  riskWarnings: boolean;
  advancedFeatures: boolean;
  paperTradingMode: boolean;
  autoRefreshInterval: number;
  chartType: 'candlestick' | 'line' | 'bar' | 'area';
  chartTimeframe: '1m' | '5m' | '15m' | '1h' | '4h' | '1d';
}

export interface WorkspaceLayout {
  id: string;
  name: string;
  description: string;
  isDefault: boolean;
  layout: {
    type: 'grid' | 'tabs' | 'split';
    columns: number;
    rows: number;
    widgets: WorkspaceWidget[];
  };
  createdAt: number;
  updatedAt: number;
}

export interface WorkspaceWidget {
  id: string;
  type: 'chart' | 'watchlist' | 'orders' | 'positions' | 'news' | 'analytics' | 'calculator';
  position: { x: number; y: number; width: number; height: number };
  config: Record<string, any>;
  visible: boolean;
}

export interface UserPreferences {
  theme: Theme;
  layout: LayoutPreferences;
  display: DisplayPreferences;
  notifications: NotificationPreferences;
  trading: TradingPreferences;
  workspaces: WorkspaceLayout[];
  activeWorkspace: string;
  language: string;
  timezone: string;
  currency: string;
  dateFormat: string;
  timeFormat: '12h' | '24h';
  numberFormat: 'US' | 'EU' | 'UK';
  shortcuts: Record<string, string>;
  lastUpdated: number;
}

// Default Themes
export const defaultThemes: Theme[] = [
  {
    id: 'alluse-light',
    name: 'ALL-USE Light',
    description: 'Clean and professional light theme',
    colors: {
      primary: '#2563eb',
      secondary: '#64748b',
      accent: '#0ea5e9',
      background: '#ffffff',
      surface: '#f8fafc',
      text: '#1e293b',
      textSecondary: '#64748b',
      border: '#e2e8f0',
      success: '#10b981',
      warning: '#f59e0b',
      error: '#ef4444',
      info: '#3b82f6'
    },
    fonts: {
      primary: 'Inter, system-ui, sans-serif',
      secondary: 'Inter, system-ui, sans-serif',
      mono: 'JetBrains Mono, Consolas, monospace'
    },
    spacing: {
      xs: '0.25rem',
      sm: '0.5rem',
      md: '1rem',
      lg: '1.5rem',
      xl: '2rem'
    },
    borderRadius: {
      sm: '0.25rem',
      md: '0.5rem',
      lg: '0.75rem'
    },
    shadows: {
      sm: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
      md: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
      lg: '0 10px 15px -3px rgb(0 0 0 / 0.1)'
    }
  },
  {
    id: 'alluse-dark',
    name: 'ALL-USE Dark',
    description: 'Elegant dark theme for extended use',
    colors: {
      primary: '#3b82f6',
      secondary: '#6b7280',
      accent: '#06b6d4',
      background: '#0f172a',
      surface: '#1e293b',
      text: '#f1f5f9',
      textSecondary: '#94a3b8',
      border: '#334155',
      success: '#10b981',
      warning: '#f59e0b',
      error: '#ef4444',
      info: '#3b82f6'
    },
    fonts: {
      primary: 'Inter, system-ui, sans-serif',
      secondary: 'Inter, system-ui, sans-serif',
      mono: 'JetBrains Mono, Consolas, monospace'
    },
    spacing: {
      xs: '0.25rem',
      sm: '0.5rem',
      md: '1rem',
      lg: '1.5rem',
      xl: '2rem'
    },
    borderRadius: {
      sm: '0.25rem',
      md: '0.5rem',
      lg: '0.75rem'
    },
    shadows: {
      sm: '0 1px 2px 0 rgb(0 0 0 / 0.3)',
      md: '0 4px 6px -1px rgb(0 0 0 / 0.3)',
      lg: '0 10px 15px -3px rgb(0 0 0 / 0.3)'
    }
  },
  {
    id: 'professional',
    name: 'Professional',
    description: 'Corporate-friendly professional theme',
    colors: {
      primary: '#1f2937',
      secondary: '#6b7280',
      accent: '#059669',
      background: '#ffffff',
      surface: '#f9fafb',
      text: '#111827',
      textSecondary: '#6b7280',
      border: '#d1d5db',
      success: '#059669',
      warning: '#d97706',
      error: '#dc2626',
      info: '#2563eb'
    },
    fonts: {
      primary: 'system-ui, -apple-system, sans-serif',
      secondary: 'system-ui, -apple-system, sans-serif',
      mono: 'SF Mono, Consolas, monospace'
    },
    spacing: {
      xs: '0.25rem',
      sm: '0.5rem',
      md: '1rem',
      lg: '1.5rem',
      xl: '2rem'
    },
    borderRadius: {
      sm: '0.125rem',
      md: '0.25rem',
      lg: '0.5rem'
    },
    shadows: {
      sm: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
      md: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
      lg: '0 10px 15px -3px rgb(0 0 0 / 0.1)'
    }
  }
];

// Default Preferences
export const createDefaultPreferences = (): UserPreferences => ({
  theme: defaultThemes[0],
  layout: {
    sidebarPosition: 'left',
    sidebarWidth: 280,
    headerHeight: 64,
    footerVisible: true,
    compactMode: false,
    gridDensity: 'comfortable',
    cardStyle: 'elevated',
    animationsEnabled: true,
    transitionSpeed: 'normal'
  },
  display: {
    colorMode: 'auto',
    highContrast: false,
    reducedMotion: false,
    fontSize: 'medium',
    lineHeight: 'normal',
    colorBlindnessSupport: 'none',
    screenReaderOptimized: false
  },
  notifications: {
    enabled: true,
    sound: true,
    desktop: true,
    email: false,
    sms: false,
    orderUpdates: true,
    marketAlerts: true,
    newsUpdates: false,
    systemMessages: true,
    quietHours: {
      enabled: false,
      start: '22:00',
      end: '08:00'
    }
  },
  trading: {
    defaultOrderType: 'limit',
    defaultTimeInForce: 'day',
    confirmationRequired: true,
    riskWarnings: true,
    advancedFeatures: false,
    paperTradingMode: false,
    autoRefreshInterval: 5000,
    chartType: 'candlestick',
    chartTimeframe: '1h'
  },
  workspaces: [],
  activeWorkspace: 'default',
  language: 'en-US',
  timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
  currency: 'USD',
  dateFormat: 'MM/DD/YYYY',
  timeFormat: '12h',
  numberFormat: 'US',
  shortcuts: {
    'new-order': 'Ctrl+N',
    'cancel-all': 'Ctrl+Shift+C',
    'refresh': 'F5',
    'search': 'Ctrl+K',
    'settings': 'Ctrl+,',
    'help': 'F1'
  },
  lastUpdated: Date.now()
});

// Customization Context
interface CustomizationContextType {
  preferences: UserPreferences;
  updatePreferences: (updates: Partial<UserPreferences>) => void;
  resetPreferences: () => void;
  exportPreferences: () => string;
  importPreferences: (data: string) => boolean;
  applyTheme: (theme: Theme) => void;
  createWorkspace: (name: string, description: string) => WorkspaceLayout;
  updateWorkspace: (id: string, updates: Partial<WorkspaceLayout>) => void;
  deleteWorkspace: (id: string) => void;
  setActiveWorkspace: (id: string) => void;
}

const CustomizationContext = createContext<CustomizationContextType | null>(null);

export const useCustomization = () => {
  const context = useContext(CustomizationContext);
  if (!context) {
    throw new Error('useCustomization must be used within a CustomizationProvider');
  }
  return context;
};

// Customization Provider
export const CustomizationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [preferences, setPreferences] = useState<UserPreferences>(() => {
    const saved = localStorage.getItem('alluse-preferences');
    return saved ? JSON.parse(saved) : createDefaultPreferences();
  });

  // Save preferences to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('alluse-preferences', JSON.stringify(preferences));
  }, [preferences]);

  // Apply theme to document
  useEffect(() => {
    const root = document.documentElement;
    const theme = preferences.theme;
    
    // Apply CSS custom properties
    Object.entries(theme.colors).forEach(([key, value]) => {
      root.style.setProperty(`--color-${key}`, value);
    });
    
    Object.entries(theme.spacing).forEach(([key, value]) => {
      root.style.setProperty(`--spacing-${key}`, value);
    });
    
    Object.entries(theme.borderRadius).forEach(([key, value]) => {
      root.style.setProperty(`--radius-${key}`, value);
    });
    
    Object.entries(theme.shadows).forEach(([key, value]) => {
      root.style.setProperty(`--shadow-${key}`, value);
    });
    
    // Apply font families
    root.style.setProperty('--font-primary', theme.fonts.primary);
    root.style.setProperty('--font-secondary', theme.fonts.secondary);
    root.style.setProperty('--font-mono', theme.fonts.mono);
    
    // Apply display preferences
    if (preferences.display.colorMode === 'dark') {
      root.classList.add('dark');
    } else if (preferences.display.colorMode === 'light') {
      root.classList.remove('dark');
    } else {
      // Auto mode - use system preference
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      if (mediaQuery.matches) {
        root.classList.add('dark');
      } else {
        root.classList.remove('dark');
      }
    }
    
    // Apply accessibility preferences
    if (preferences.display.reducedMotion) {
      root.style.setProperty('--animation-duration', '0s');
    } else {
      const speed = preferences.layout.transitionSpeed;
      const duration = speed === 'slow' ? '0.5s' : speed === 'fast' ? '0.15s' : '0.3s';
      root.style.setProperty('--animation-duration', duration);
    }
    
    // Apply font size
    const fontSizeMap = {
      'small': '14px',
      'medium': '16px',
      'large': '18px',
      'extra-large': '20px'
    };
    root.style.setProperty('--base-font-size', fontSizeMap[preferences.display.fontSize]);
    
  }, [preferences.theme, preferences.display, preferences.layout.transitionSpeed]);

  const updatePreferences = useCallback((updates: Partial<UserPreferences>) => {
    setPreferences(prev => ({
      ...prev,
      ...updates,
      lastUpdated: Date.now()
    }));
  }, []);

  const resetPreferences = useCallback(() => {
    setPreferences(createDefaultPreferences());
  }, []);

  const exportPreferences = useCallback(() => {
    return JSON.stringify(preferences, null, 2);
  }, [preferences]);

  const importPreferences = useCallback((data: string) => {
    try {
      const imported = JSON.parse(data);
      // Validate the imported data structure
      if (imported && typeof imported === 'object' && imported.theme && imported.layout) {
        setPreferences({
          ...imported,
          lastUpdated: Date.now()
        });
        return true;
      }
      return false;
    } catch {
      return false;
    }
  }, []);

  const applyTheme = useCallback((theme: Theme) => {
    updatePreferences({ theme });
  }, [updatePreferences]);

  const createWorkspace = useCallback((name: string, description: string) => {
    const workspace: WorkspaceLayout = {
      id: `workspace-${Date.now()}`,
      name,
      description,
      isDefault: false,
      layout: {
        type: 'grid',
        columns: 3,
        rows: 2,
        widgets: []
      },
      createdAt: Date.now(),
      updatedAt: Date.now()
    };

    updatePreferences({
      workspaces: [...preferences.workspaces, workspace]
    });

    return workspace;
  }, [preferences.workspaces, updatePreferences]);

  const updateWorkspace = useCallback((id: string, updates: Partial<WorkspaceLayout>) => {
    const updatedWorkspaces = preferences.workspaces.map(workspace =>
      workspace.id === id
        ? { ...workspace, ...updates, updatedAt: Date.now() }
        : workspace
    );

    updatePreferences({ workspaces: updatedWorkspaces });
  }, [preferences.workspaces, updatePreferences]);

  const deleteWorkspace = useCallback((id: string) => {
    const updatedWorkspaces = preferences.workspaces.filter(workspace => workspace.id !== id);
    const newActiveWorkspace = preferences.activeWorkspace === id 
      ? (updatedWorkspaces[0]?.id || 'default')
      : preferences.activeWorkspace;

    updatePreferences({
      workspaces: updatedWorkspaces,
      activeWorkspace: newActiveWorkspace
    });
  }, [preferences.workspaces, preferences.activeWorkspace, updatePreferences]);

  const setActiveWorkspace = useCallback((id: string) => {
    updatePreferences({ activeWorkspace: id });
  }, [updatePreferences]);

  const contextValue: CustomizationContextType = {
    preferences,
    updatePreferences,
    resetPreferences,
    exportPreferences,
    importPreferences,
    applyTheme,
    createWorkspace,
    updateWorkspace,
    deleteWorkspace,
    setActiveWorkspace
  };

  return (
    <CustomizationContext.Provider value={contextValue}>
      {children}
    </CustomizationContext.Provider>
  );
};

// Customization Panel Component
export const CustomizationPanel: React.FC<{
  isOpen: boolean;
  onClose: () => void;
}> = ({ isOpen, onClose }) => {
  const { preferences, updatePreferences, resetPreferences, exportPreferences, importPreferences, applyTheme } = useCustomization();
  const [activeTab, setActiveTab] = useState<'appearance' | 'layout' | 'display' | 'notifications' | 'trading' | 'workspace' | 'advanced'>('appearance');
  const [showResetConfirm, setShowResetConfirm] = useState(false);
  const [exportData, setExportData] = useState<string>('');
  const [importData, setImportData] = useState<string>('');
  const [showExport, setShowExport] = useState(false);
  const [showImport, setShowImport] = useState(false);

  const handleExport = useCallback(() => {
    const data = exportPreferences();
    setExportData(data);
    setShowExport(true);
  }, [exportPreferences]);

  const handleImport = useCallback(() => {
    if (importData.trim()) {
      const success = importPreferences(importData);
      if (success) {
        setImportData('');
        setShowImport(false);
      } else {
        alert('Invalid preferences data');
      }
    }
  }, [importData, importPreferences]);

  const copyToClipboard = useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = text;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
    }
  }, []);

  if (!isOpen) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        className="bg-white dark:bg-gray-800 rounded-lg w-full max-w-4xl h-[80vh] mx-4 flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="border-b border-gray-200 dark:border-gray-700 p-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Settings className="w-6 h-6 text-blue-500" />
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
              Customization & Preferences
            </h2>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={handleExport}
              className="px-3 py-1 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
            >
              <Download className="w-4 h-4" />
            </button>
            <button
              onClick={() => setShowImport(true)}
              className="px-3 py-1 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
            >
              <Upload className="w-4 h-4" />
            </button>
            <button
              onClick={() => setShowResetConfirm(true)}
              className="px-3 py-1 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
            <button
              onClick={onClose}
              className="px-3 py-1 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
            >
              ✕
            </button>
          </div>
        </div>

        <div className="flex flex-1 overflow-hidden">
          {/* Sidebar */}
          <div className="w-64 border-r border-gray-200 dark:border-gray-700 p-4">
            <nav className="space-y-1">
              {[
                { id: 'appearance', label: 'Appearance', icon: Palette },
                { id: 'layout', label: 'Layout', icon: Layout },
                { id: 'display', label: 'Display', icon: Monitor },
                { id: 'notifications', label: 'Notifications', icon: Bell },
                { id: 'trading', label: 'Trading', icon: TrendingUp },
                { id: 'workspace', label: 'Workspaces', icon: Grid },
                { id: 'advanced', label: 'Advanced', icon: Settings }
              ].map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  onClick={() => setActiveTab(id as any)}
                  className={`w-full flex items-center gap-3 px-3 py-2 text-sm rounded-lg transition-colors ${
                    activeTab === id
                      ? 'bg-blue-500 text-white'
                      : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-700'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {label}
                </button>
              ))}
            </nav>
          </div>

          {/* Content */}
          <div className="flex-1 p-6 overflow-y-auto">
            <AnimatePresence mode="wait">
              {activeTab === 'appearance' && (
                <AppearanceSettings key="appearance" preferences={preferences} updatePreferences={updatePreferences} applyTheme={applyTheme} />
              )}
              {activeTab === 'layout' && (
                <LayoutSettings key="layout" preferences={preferences} updatePreferences={updatePreferences} />
              )}
              {activeTab === 'display' && (
                <DisplaySettings key="display" preferences={preferences} updatePreferences={updatePreferences} />
              )}
              {activeTab === 'notifications' && (
                <NotificationSettings key="notifications" preferences={preferences} updatePreferences={updatePreferences} />
              )}
              {activeTab === 'trading' && (
                <TradingSettings key="trading" preferences={preferences} updatePreferences={updatePreferences} />
              )}
              {activeTab === 'workspace' && (
                <WorkspaceSettings key="workspace" preferences={preferences} updatePreferences={updatePreferences} />
              )}
              {activeTab === 'advanced' && (
                <AdvancedSettings key="advanced" preferences={preferences} updatePreferences={updatePreferences} />
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Reset Confirmation Modal */}
        <AnimatePresence>
          {showResetConfirm && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center"
            >
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-md mx-4"
              >
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Reset All Preferences
                </h3>
                <p className="text-gray-600 dark:text-gray-400 mb-6">
                  This will reset all your customizations to default values. This action cannot be undone.
                </p>
                <div className="flex gap-3">
                  <button
                    onClick={() => setShowResetConfirm(false)}
                    className="flex-1 px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={() => {
                      resetPreferences();
                      setShowResetConfirm(false);
                    }}
                    className="flex-1 px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600"
                  >
                    Reset
                  </button>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Export Modal */}
        <AnimatePresence>
          {showExport && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center"
            >
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-2xl mx-4 w-full"
              >
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Export Preferences
                </h3>
                <textarea
                  value={exportData}
                  readOnly
                  className="w-full h-64 p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-white font-mono text-sm"
                />
                <div className="flex gap-3 mt-4">
                  <button
                    onClick={() => setShowExport(false)}
                    className="flex-1 px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700"
                  >
                    Close
                  </button>
                  <button
                    onClick={() => copyToClipboard(exportData)}
                    className="flex-1 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 flex items-center justify-center gap-2"
                  >
                    <Copy className="w-4 h-4" />
                    Copy to Clipboard
                  </button>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Import Modal */}
        <AnimatePresence>
          {showImport && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center"
            >
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-2xl mx-4 w-full"
              >
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Import Preferences
                </h3>
                <textarea
                  value={importData}
                  onChange={(e) => setImportData(e.target.value)}
                  placeholder="Paste your exported preferences here..."
                  className="w-full h-64 p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white font-mono text-sm"
                />
                <div className="flex gap-3 mt-4">
                  <button
                    onClick={() => setShowImport(false)}
                    className="flex-1 px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleImport}
                    className="flex-1 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
                  >
                    Import
                  </button>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </motion.div>
  );
};

// Individual Settings Components (simplified for brevity)
const AppearanceSettings: React.FC<{
  preferences: UserPreferences;
  updatePreferences: (updates: Partial<UserPreferences>) => void;
  applyTheme: (theme: Theme) => void;
}> = ({ preferences, updatePreferences, applyTheme }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="space-y-6"
    >
      <div>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Theme Selection</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {defaultThemes.map((theme) => (
            <button
              key={theme.id}
              onClick={() => applyTheme(theme)}
              className={`p-4 border-2 rounded-lg text-left transition-colors ${
                preferences.theme.id === theme.id
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
              }`}
            >
              <div className="flex items-center gap-3 mb-2">
                <div
                  className="w-6 h-6 rounded-full"
                  style={{ backgroundColor: theme.colors.primary }}
                />
                <h4 className="font-medium text-gray-900 dark:text-white">{theme.name}</h4>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">{theme.description}</p>
            </button>
          ))}
        </div>
      </div>
    </motion.div>
  );
};

const LayoutSettings: React.FC<{
  preferences: UserPreferences;
  updatePreferences: (updates: Partial<UserPreferences>) => void;
}> = ({ preferences, updatePreferences }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="space-y-6"
    >
      <div>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Layout Preferences</h3>
        {/* Layout settings implementation */}
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Sidebar Position
            </label>
            <select
              value={preferences.layout.sidebarPosition}
              onChange={(e) => updatePreferences({
                layout: { ...preferences.layout, sidebarPosition: e.target.value as any }
              })}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="left">Left</option>
              <option value="right">Right</option>
              <option value="hidden">Hidden</option>
            </select>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

// Additional settings components would be implemented similarly...
const DisplaySettings: React.FC<any> = () => <div>Display Settings</div>;
const NotificationSettings: React.FC<any> = () => <div>Notification Settings</div>;
const TradingSettings: React.FC<any> = () => <div>Trading Settings</div>;
const WorkspaceSettings: React.FC<any> = () => <div>Workspace Settings</div>;
const AdvancedSettings: React.FC<any> = () => <div>Advanced Settings</div>;

// Export all components and utilities
export {
  AppearanceSettings,
  LayoutSettings,
  DisplaySettings,
  NotificationSettings,
  TradingSettings,
  WorkspaceSettings,
  AdvancedSettings
};

