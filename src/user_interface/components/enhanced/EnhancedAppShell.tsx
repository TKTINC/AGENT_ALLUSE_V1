import React, { useState, useEffect, useMemo } from 'react';
import { 
  Settings, 
  Monitor, 
  Smartphone, 
  Tablet, 
  Zap, 
  Eye, 
  Volume2, 
  VolumeX,
  Sun,
  Moon,
  Type,
  Contrast,
  MousePointer,
  Keyboard,
  Accessibility
} from 'lucide-react';
import { Button, Card, Modal, Tabs, ProgressBar } from '../advanced/UIComponents';
import { 
  useAccessibility, 
  useTheme, 
  useNotifications,
  AnimatedWrapper,
  ErrorBoundary
} from '../../utils/userExperience';
import { 
  usePerformanceMonitor, 
  useMemoryMonitor, 
  collectPerformanceMetrics,
  monitorPerformanceBudget
} from '../../utils/performance';

// Enhanced App Shell with performance monitoring
interface EnhancedAppShellProps {
  children: React.ReactNode;
  title: string;
  user?: {
    name: string;
    email: string;
    avatar?: string;
  };
}

export const EnhancedAppShell: React.FC<EnhancedAppShellProps> = ({
  children,
  title,
  user
}) => {
  const [showSettings, setShowSettings] = useState(false);
  const [showPerformancePanel, setShowPerformancePanel] = useState(false);
  const { theme, effectiveTheme, toggleTheme } = useTheme();
  const { 
    isHighContrast, 
    isReducedMotion, 
    fontSize, 
    increaseFontSize, 
    decreaseFontSize, 
    resetFontSize 
  } = useAccessibility();
  const { notifications, addNotification, removeNotification } = useNotifications();
  const performanceMonitor = usePerformanceMonitor('AppShell');
  const memoryInfo = useMemoryMonitor();

  // Performance monitoring
  useEffect(() => {
    monitorPerformanceBudget({
      maxLoadTime: 3000,
      maxMemoryUsage: 100 * 1024 * 1024 // 100MB
    });
  }, []);

  // Apply accessibility settings
  useEffect(() => {
    document.documentElement.style.fontSize = `${fontSize}px`;
    document.documentElement.classList.toggle('high-contrast', isHighContrast);
    document.documentElement.classList.toggle('reduced-motion', isReducedMotion);
    document.documentElement.classList.toggle('dark', effectiveTheme === 'dark');
  }, [fontSize, isHighContrast, isReducedMotion, effectiveTheme]);

  const performanceMetrics = useMemo(() => {
    return collectPerformanceMetrics();
  }, []);

  return (
    <ErrorBoundary
      onError={(error, errorInfo) => {
        addNotification({
          type: 'error',
          title: 'Application Error',
          message: 'An unexpected error occurred. The page will reload automatically.',
          duration: 0
        });
      }}
    >
      <div className={`min-h-screen transition-colors duration-300 ${
        effectiveTheme === 'dark' ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'
      }`}>
        {/* Enhanced Header */}
        <header className={`sticky top-0 z-50 border-b transition-colors duration-300 ${
          effectiveTheme === 'dark' 
            ? 'bg-gray-800 border-gray-700' 
            : 'bg-white border-gray-200'
        }`}>
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              {/* Logo and Title */}
              <AnimatedWrapper animation="slideUp">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                    <span className="text-white font-bold text-sm">AU</span>
                  </div>
                  <h1 className="text-xl font-semibold">{title}</h1>
                </div>
              </AnimatedWrapper>

              {/* User Actions */}
              <div className="flex items-center gap-4">
                {/* Performance Indicator */}
                {process.env.NODE_ENV === 'development' && (
                  <Button
                    variant="ghost"
                    size="sm"
                    icon={<Monitor className="w-4 h-4" />}
                    onClick={() => setShowPerformancePanel(true)}
                    className="hidden md:flex"
                  >
                    Performance
                  </Button>
                )}

                {/* Theme Toggle */}
                <Button
                  variant="ghost"
                  size="sm"
                  icon={effectiveTheme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
                  onClick={toggleTheme}
                  aria-label={`Switch to ${effectiveTheme === 'dark' ? 'light' : 'dark'} theme`}
                />

                {/* Settings */}
                <Button
                  variant="ghost"
                  size="sm"
                  icon={<Settings className="w-4 h-4" />}
                  onClick={() => setShowSettings(true)}
                  aria-label="Open settings"
                />

                {/* User Menu */}
                {user && (
                  <div className="flex items-center gap-3">
                    <div className="hidden md:block text-right">
                      <div className="text-sm font-medium">{user.name}</div>
                      <div className="text-xs text-gray-500">{user.email}</div>
                    </div>
                    <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
                      {user.avatar ? (
                        <img src={user.avatar} alt={user.name} className="w-8 h-8 rounded-full" />
                      ) : (
                        <span className="text-white text-sm font-medium">
                          {user.name.charAt(0).toUpperCase()}
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <AnimatedWrapper animation="fadeIn" delay={200}>
            {children}
          </AnimatedWrapper>
        </main>

        {/* Notifications */}
        <div className="fixed top-4 right-4 z-50 space-y-2">
          {notifications.map(notification => (
            <NotificationCard
              key={notification.id}
              notification={notification}
              onClose={() => removeNotification(notification.id)}
            />
          ))}
        </div>

        {/* Settings Modal */}
        <SettingsModal
          isOpen={showSettings}
          onClose={() => setShowSettings(false)}
          theme={theme}
          onThemeChange={toggleTheme}
          fontSize={fontSize}
          onFontSizeIncrease={increaseFontSize}
          onFontSizeDecrease={decreaseFontSize}
          onFontSizeReset={resetFontSize}
          isHighContrast={isHighContrast}
          isReducedMotion={isReducedMotion}
        />

        {/* Performance Panel */}
        {process.env.NODE_ENV === 'development' && (
          <PerformancePanel
            isOpen={showPerformancePanel}
            onClose={() => setShowPerformancePanel(false)}
            metrics={performanceMetrics}
            memoryInfo={memoryInfo}
            renderCount={performanceMonitor.renderCount}
          />
        )}
      </div>
    </ErrorBoundary>
  );
};

// Notification Card Component
interface NotificationCardProps {
  notification: {
    type: 'success' | 'error' | 'warning' | 'info';
    title: string;
    message: string;
    action?: {
      label: string;
      onClick: () => void;
    };
  };
  onClose: () => void;
}

const NotificationCard: React.FC<NotificationCardProps> = ({ notification, onClose }) => {
  const getNotificationStyles = (type: string) => {
    switch (type) {
      case 'success':
        return 'bg-green-50 border-green-200 text-green-800';
      case 'error':
        return 'bg-red-50 border-red-200 text-red-800';
      case 'warning':
        return 'bg-yellow-50 border-yellow-200 text-yellow-800';
      case 'info':
        return 'bg-blue-50 border-blue-200 text-blue-800';
      default:
        return 'bg-gray-50 border-gray-200 text-gray-800';
    }
  };

  return (
    <AnimatedWrapper animation="slideDown">
      <Card className={`p-4 max-w-sm border ${getNotificationStyles(notification.type)}`}>
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <h4 className="font-medium text-sm">{notification.title}</h4>
            <p className="text-sm mt-1 opacity-90">{notification.message}</p>
            {notification.action && (
              <Button
                variant="ghost"
                size="sm"
                onClick={notification.action.onClick}
                className="mt-2 p-0 h-auto text-sm underline"
              >
                {notification.action.label}
              </Button>
            )}
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            className="ml-2 p-1 h-auto opacity-70 hover:opacity-100"
          >
            ×
          </Button>
        </div>
      </Card>
    </AnimatedWrapper>
  );
};

// Settings Modal Component
interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  theme: string;
  onThemeChange: () => void;
  fontSize: number;
  onFontSizeIncrease: () => void;
  onFontSizeDecrease: () => void;
  onFontSizeReset: () => void;
  isHighContrast: boolean;
  isReducedMotion: boolean;
}

const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  theme,
  onThemeChange,
  fontSize,
  onFontSizeIncrease,
  onFontSizeDecrease,
  onFontSizeReset,
  isHighContrast,
  isReducedMotion
}) => {
  const [activeTab, setActiveTab] = useState('appearance');

  const tabs = [
    { id: 'appearance', label: 'Appearance', icon: <Eye className="w-4 h-4" /> },
    { id: 'accessibility', label: 'Accessibility', icon: <Accessibility className="w-4 h-4" /> },
    { id: 'performance', label: 'Performance', icon: <Zap className="w-4 h-4" /> }
  ];

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Settings" size="lg">
      <div className="space-y-6">
        <Tabs
          tabs={tabs}
          activeTab={activeTab}
          onChange={setActiveTab}
          variant="pills"
        />

        {activeTab === 'appearance' && (
          <div className="space-y-6">
            {/* Theme Settings */}
            <div>
              <h4 className="font-medium mb-3">Theme</h4>
              <div className="grid grid-cols-3 gap-3">
                <Button
                  variant={theme === 'light' ? 'primary' : 'outline'}
                  size="sm"
                  icon={<Sun className="w-4 h-4" />}
                  onClick={() => theme !== 'light' && onThemeChange()}
                  fullWidth
                >
                  Light
                </Button>
                <Button
                  variant={theme === 'dark' ? 'primary' : 'outline'}
                  size="sm"
                  icon={<Moon className="w-4 h-4" />}
                  onClick={() => theme !== 'dark' && onThemeChange()}
                  fullWidth
                >
                  Dark
                </Button>
                <Button
                  variant={theme === 'auto' ? 'primary' : 'outline'}
                  size="sm"
                  icon={<Monitor className="w-4 h-4" />}
                  onClick={() => theme !== 'auto' && onThemeChange()}
                  fullWidth
                >
                  Auto
                </Button>
              </div>
            </div>

            {/* Font Size */}
            <div>
              <h4 className="font-medium mb-3">Font Size</h4>
              <div className="flex items-center gap-3">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onFontSizeDecrease}
                  disabled={fontSize <= 12}
                >
                  A-
                </Button>
                <div className="flex-1 text-center">
                  <span className="text-sm text-gray-600">Current: {fontSize}px</span>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onFontSizeIncrease}
                  disabled={fontSize >= 24}
                >
                  A+
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={onFontSizeReset}
                >
                  Reset
                </Button>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'accessibility' && (
          <div className="space-y-6">
            {/* Accessibility Status */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card className="p-4">
                <div className="flex items-center gap-3">
                  <Contrast className="w-5 h-5 text-blue-600" />
                  <div>
                    <div className="font-medium">High Contrast</div>
                    <div className="text-sm text-gray-600">
                      {isHighContrast ? 'Enabled' : 'Disabled'}
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-4">
                <div className="flex items-center gap-3">
                  <MousePointer className="w-5 h-5 text-green-600" />
                  <div>
                    <div className="font-medium">Reduced Motion</div>
                    <div className="text-sm text-gray-600">
                      {isReducedMotion ? 'Enabled' : 'Disabled'}
                    </div>
                  </div>
                </div>
              </Card>
            </div>

            {/* Keyboard Shortcuts */}
            <div>
              <h4 className="font-medium mb-3">Keyboard Shortcuts</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Toggle Settings</span>
                  <kbd className="px-2 py-1 bg-gray-100 rounded text-xs">Ctrl + ,</kbd>
                </div>
                <div className="flex justify-between">
                  <span>Toggle Theme</span>
                  <kbd className="px-2 py-1 bg-gray-100 rounded text-xs">Ctrl + Shift + T</kbd>
                </div>
                <div className="flex justify-between">
                  <span>Focus Search</span>
                  <kbd className="px-2 py-1 bg-gray-100 rounded text-xs">Ctrl + K</kbd>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'performance' && (
          <div className="space-y-6">
            {/* Performance Settings */}
            <div>
              <h4 className="font-medium mb-3">Performance Options</h4>
              <div className="space-y-3">
                <label className="flex items-center gap-3">
                  <input type="checkbox" defaultChecked className="rounded" />
                  <span>Enable animations</span>
                </label>
                <label className="flex items-center gap-3">
                  <input type="checkbox" defaultChecked className="rounded" />
                  <span>Preload images</span>
                </label>
                <label className="flex items-center gap-3">
                  <input type="checkbox" defaultChecked className="rounded" />
                  <span>Cache API responses</span>
                </label>
              </div>
            </div>

            {/* Data Usage */}
            <div>
              <h4 className="font-medium mb-3">Data Usage</h4>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Images</span>
                  <span>2.3 MB</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Scripts</span>
                  <span>1.8 MB</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>API Data</span>
                  <span>0.5 MB</span>
                </div>
                <div className="border-t pt-2 flex justify-between font-medium">
                  <span>Total</span>
                  <span>4.6 MB</span>
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="flex gap-3 pt-4 border-t">
          <Button variant="outline" fullWidth onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" fullWidth onClick={onClose}>
            Save Changes
          </Button>
        </div>
      </div>
    </Modal>
  );
};

// Performance Panel Component (Development Only)
interface PerformancePanelProps {
  isOpen: boolean;
  onClose: () => void;
  metrics: any;
  memoryInfo: any;
  renderCount: number;
}

const PerformancePanel: React.FC<PerformancePanelProps> = ({
  isOpen,
  onClose,
  metrics,
  memoryInfo,
  renderCount
}) => {
  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Performance Monitor" size="lg">
      <div className="space-y-6">
        {/* Render Performance */}
        <div>
          <h4 className="font-medium mb-3">Render Performance</h4>
          <div className="grid grid-cols-2 gap-4">
            <Card className="p-4">
              <div className="text-2xl font-bold text-blue-600">{renderCount}</div>
              <div className="text-sm text-gray-600">Total Renders</div>
            </Card>
            <Card className="p-4">
              <div className="text-2xl font-bold text-green-600">
                {metrics?.firstContentfulPaint?.toFixed(0) || '--'}ms
              </div>
              <div className="text-sm text-gray-600">First Contentful Paint</div>
            </Card>
          </div>
        </div>

        {/* Memory Usage */}
        {memoryInfo && (
          <div>
            <h4 className="font-medium mb-3">Memory Usage</h4>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>Used Heap</span>
                  <span>{(memoryInfo.usedJSHeapSize / 1024 / 1024).toFixed(1)} MB</span>
                </div>
                <ProgressBar
                  value={memoryInfo.usedJSHeapSize}
                  max={memoryInfo.totalJSHeapSize}
                  color="blue"
                />
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>Total Heap</span>
                  <span>{(memoryInfo.totalJSHeapSize / 1024 / 1024).toFixed(1)} MB</span>
                </div>
                <ProgressBar
                  value={memoryInfo.totalJSHeapSize}
                  max={memoryInfo.jsHeapSizeLimit}
                  color="orange"
                />
              </div>
            </div>
          </div>
        )}

        {/* Network Performance */}
        {metrics && (
          <div>
            <h4 className="font-medium mb-3">Network Performance</h4>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center">
                <div className="text-lg font-bold text-purple-600">
                  {metrics.domContentLoaded?.toFixed(0) || '--'}ms
                </div>
                <div className="text-sm text-gray-600">DOM Content Loaded</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-red-600">
                  {metrics.loadComplete?.toFixed(0) || '--'}ms
                </div>
                <div className="text-sm text-gray-600">Load Complete</div>
              </div>
            </div>
          </div>
        )}

        <Button variant="outline" fullWidth onClick={onClose}>
          Close
        </Button>
      </div>
    </Modal>
  );
};

