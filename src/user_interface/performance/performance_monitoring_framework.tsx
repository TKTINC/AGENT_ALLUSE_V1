import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Activity, BarChart3, Clock, Cpu, HardDrive, Monitor, Network, TrendingUp, Zap, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';

// Performance monitoring interfaces and types
interface PerformanceMetrics {
  componentRenderTime: number;
  memoryUsage: number;
  networkLatency: number;
  bundleSize: number;
  firstContentfulPaint: number;
  largestContentfulPaint: number;
  cumulativeLayoutShift: number;
  firstInputDelay: number;
  timeToInteractive: number;
  totalBlockingTime: number;
}

interface ComponentPerformance {
  componentName: string;
  renderTime: number;
  memoryUsage: number;
  updateCount: number;
  lastUpdate: Date;
  performanceScore: number;
  issues: PerformanceIssue[];
}

interface PerformanceIssue {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  type: 'memory' | 'render' | 'network' | 'layout' | 'interaction';
  description: string;
  recommendation: string;
  impact: number;
  detected: Date;
}

interface PerformanceAlert {
  id: string;
  type: 'warning' | 'error' | 'info';
  message: string;
  timestamp: Date;
  resolved: boolean;
  component?: string;
}

// Performance monitoring framework
class PerformanceMonitoringFramework {
  private metrics: PerformanceMetrics[] = [];
  private componentMetrics: Map<string, ComponentPerformance> = new Map();
  private observers: PerformanceObserver[] = [];
  private alerts: PerformanceAlert[] = [];
  private isMonitoring: boolean = false;
  private thresholds = {
    renderTime: 100, // ms
    memoryUsage: 50 * 1024 * 1024, // 50MB
    networkLatency: 200, // ms
    firstContentfulPaint: 1500, // ms
    largestContentfulPaint: 2500, // ms
    cumulativeLayoutShift: 0.1,
    firstInputDelay: 100, // ms
    timeToInteractive: 3000, // ms
    totalBlockingTime: 300 // ms
  };

  startMonitoring(): void {
    if (this.isMonitoring) return;
    
    this.isMonitoring = true;
    this.initializeObservers();
    this.startMetricsCollection();
    this.startComponentMonitoring();
  }

  stopMonitoring(): void {
    this.isMonitoring = false;
    this.observers.forEach(observer => observer.disconnect());
    this.observers = [];
  }

  private initializeObservers(): void {
    // Performance observer for navigation timing
    if (typeof PerformanceObserver !== 'undefined') {
      const navigationObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach(entry => this.processNavigationEntry(entry as PerformanceNavigationTiming));
      });
      navigationObserver.observe({ entryTypes: ['navigation'] });
      this.observers.push(navigationObserver);

      // Performance observer for paint timing
      const paintObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach(entry => this.processPaintEntry(entry as PerformancePaintTiming));
      });
      paintObserver.observe({ entryTypes: ['paint'] });
      this.observers.push(paintObserver);

      // Performance observer for layout shift
      const layoutShiftObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach(entry => this.processLayoutShiftEntry(entry as LayoutShift));
      });
      layoutShiftObserver.observe({ entryTypes: ['layout-shift'] });
      this.observers.push(layoutShiftObserver);

      // Performance observer for first input delay
      const firstInputObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach(entry => this.processFirstInputEntry(entry as PerformanceEventTiming));
      });
      firstInputObserver.observe({ entryTypes: ['first-input'] });
      this.observers.push(firstInputObserver);
    }
  }

  private startMetricsCollection(): void {
    const collectMetrics = () => {
      if (!this.isMonitoring) return;

      const currentMetrics: PerformanceMetrics = {
        componentRenderTime: this.measureComponentRenderTime(),
        memoryUsage: this.measureMemoryUsage(),
        networkLatency: this.measureNetworkLatency(),
        bundleSize: this.measureBundleSize(),
        firstContentfulPaint: this.getFirstContentfulPaint(),
        largestContentfulPaint: this.getLargestContentfulPaint(),
        cumulativeLayoutShift: this.getCumulativeLayoutShift(),
        firstInputDelay: this.getFirstInputDelay(),
        timeToInteractive: this.getTimeToInteractive(),
        totalBlockingTime: this.getTotalBlockingTime()
      };

      this.metrics.push(currentMetrics);
      this.analyzeMetrics(currentMetrics);
      
      // Keep only last 100 metrics for memory efficiency
      if (this.metrics.length > 100) {
        this.metrics = this.metrics.slice(-100);
      }

      setTimeout(collectMetrics, 5000); // Collect every 5 seconds
    };

    collectMetrics();
  }

  private startComponentMonitoring(): void {
    // Monitor React component performance
    if (typeof window !== 'undefined' && (window as any).React) {
      const originalCreateElement = (window as any).React.createElement;
      
      (window as any).React.createElement = (...args: any[]) => {
        const startTime = performance.now();
        const result = originalCreateElement.apply(this, args);
        const endTime = performance.now();
        
        if (args[0] && typeof args[0] === 'function') {
          const componentName = args[0].name || 'Anonymous';
          this.recordComponentPerformance(componentName, endTime - startTime);
        }
        
        return result;
      };
    }
  }

  private processNavigationEntry(entry: PerformanceNavigationTiming): void {
    const timeToInteractive = entry.loadEventEnd - entry.navigationStart;
    this.checkThreshold('timeToInteractive', timeToInteractive, 'Time to Interactive');
  }

  private processPaintEntry(entry: PerformancePaintTiming): void {
    if (entry.name === 'first-contentful-paint') {
      this.checkThreshold('firstContentfulPaint', entry.startTime, 'First Contentful Paint');
    }
  }

  private processLayoutShiftEntry(entry: LayoutShift): void {
    this.checkThreshold('cumulativeLayoutShift', entry.value, 'Cumulative Layout Shift');
  }

  private processFirstInputEntry(entry: PerformanceEventTiming): void {
    const firstInputDelay = entry.processingStart - entry.startTime;
    this.checkThreshold('firstInputDelay', firstInputDelay, 'First Input Delay');
  }

  private measureComponentRenderTime(): number {
    // Average render time from component metrics
    const renderTimes = Array.from(this.componentMetrics.values()).map(c => c.renderTime);
    return renderTimes.length > 0 ? renderTimes.reduce((a, b) => a + b, 0) / renderTimes.length : 0;
  }

  private measureMemoryUsage(): number {
    if ('memory' in performance) {
      return (performance as any).memory.usedJSHeapSize;
    }
    return 0;
  }

  private measureNetworkLatency(): number {
    const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[];
    if (resources.length === 0) return 0;
    
    const latencies = resources.map(resource => resource.responseEnd - resource.requestStart);
    return latencies.reduce((a, b) => a + b, 0) / latencies.length;
  }

  private measureBundleSize(): number {
    const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[];
    return resources.reduce((total, resource) => total + (resource.transferSize || 0), 0);
  }

  private getFirstContentfulPaint(): number {
    const paintEntries = performance.getEntriesByType('paint');
    const fcpEntry = paintEntries.find(entry => entry.name === 'first-contentful-paint');
    return fcpEntry ? fcpEntry.startTime : 0;
  }

  private getLargestContentfulPaint(): number {
    const lcpEntries = performance.getEntriesByType('largest-contentful-paint');
    return lcpEntries.length > 0 ? lcpEntries[lcpEntries.length - 1].startTime : 0;
  }

  private getCumulativeLayoutShift(): number {
    const layoutShiftEntries = performance.getEntriesByType('layout-shift') as LayoutShift[];
    return layoutShiftEntries.reduce((total, entry) => total + entry.value, 0);
  }

  private getFirstInputDelay(): number {
    const firstInputEntries = performance.getEntriesByType('first-input') as PerformanceEventTiming[];
    if (firstInputEntries.length === 0) return 0;
    
    const entry = firstInputEntries[0];
    return entry.processingStart - entry.startTime;
  }

  private getTimeToInteractive(): number {
    const navigationEntries = performance.getEntriesByType('navigation') as PerformanceNavigationTiming[];
    if (navigationEntries.length === 0) return 0;
    
    const entry = navigationEntries[0];
    return entry.loadEventEnd - entry.navigationStart;
  }

  private getTotalBlockingTime(): number {
    const longTaskEntries = performance.getEntriesByType('longtask');
    return longTaskEntries.reduce((total, entry) => total + Math.max(0, entry.duration - 50), 0);
  }

  private recordComponentPerformance(componentName: string, renderTime: number): void {
    const existing = this.componentMetrics.get(componentName);
    
    if (existing) {
      existing.renderTime = (existing.renderTime + renderTime) / 2; // Moving average
      existing.updateCount++;
      existing.lastUpdate = new Date();
      existing.performanceScore = this.calculatePerformanceScore(existing);
    } else {
      const newMetric: ComponentPerformance = {
        componentName,
        renderTime,
        memoryUsage: this.measureMemoryUsage(),
        updateCount: 1,
        lastUpdate: new Date(),
        performanceScore: 100,
        issues: []
      };
      newMetric.performanceScore = this.calculatePerformanceScore(newMetric);
      this.componentMetrics.set(componentName, newMetric);
    }

    this.checkComponentThresholds(componentName, renderTime);
  }

  private calculatePerformanceScore(component: ComponentPerformance): number {
    let score = 100;
    
    // Deduct points for slow render times
    if (component.renderTime > this.thresholds.renderTime) {
      score -= Math.min(30, (component.renderTime - this.thresholds.renderTime) / 10);
    }
    
    // Deduct points for high memory usage
    if (component.memoryUsage > this.thresholds.memoryUsage) {
      score -= Math.min(20, (component.memoryUsage - this.thresholds.memoryUsage) / (1024 * 1024));
    }
    
    // Deduct points for issues
    component.issues.forEach(issue => {
      switch (issue.severity) {
        case 'critical': score -= 25; break;
        case 'high': score -= 15; break;
        case 'medium': score -= 10; break;
        case 'low': score -= 5; break;
      }
    });
    
    return Math.max(0, score);
  }

  private checkThreshold(metric: keyof typeof this.thresholds, value: number, displayName: string): void {
    const threshold = this.thresholds[metric];
    if (value > threshold) {
      this.createAlert('warning', `${displayName} (${value.toFixed(2)}) exceeds threshold (${threshold})`);
    }
  }

  private checkComponentThresholds(componentName: string, renderTime: number): void {
    if (renderTime > this.thresholds.renderTime) {
      this.createAlert('warning', `Component ${componentName} render time (${renderTime.toFixed(2)}ms) exceeds threshold`, componentName);
    }
  }

  private analyzeMetrics(metrics: PerformanceMetrics): void {
    // Analyze trends and create alerts for performance degradation
    if (this.metrics.length >= 5) {
      const recentMetrics = this.metrics.slice(-5);
      const avgRenderTime = recentMetrics.reduce((sum, m) => sum + m.componentRenderTime, 0) / 5;
      
      if (avgRenderTime > this.thresholds.renderTime * 1.5) {
        this.createAlert('error', `Average render time (${avgRenderTime.toFixed(2)}ms) significantly exceeds threshold`);
      }
    }
  }

  private createAlert(type: PerformanceAlert['type'], message: string, component?: string): void {
    const alert: PerformanceAlert = {
      id: `alert-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type,
      message,
      timestamp: new Date(),
      resolved: false,
      component
    };
    
    this.alerts.push(alert);
    
    // Keep only last 50 alerts
    if (this.alerts.length > 50) {
      this.alerts = this.alerts.slice(-50);
    }
  }

  getMetrics(): PerformanceMetrics[] {
    return [...this.metrics];
  }

  getComponentMetrics(): ComponentPerformance[] {
    return Array.from(this.componentMetrics.values());
  }

  getAlerts(): PerformanceAlert[] {
    return [...this.alerts];
  }

  getUnresolvedAlerts(): PerformanceAlert[] {
    return this.alerts.filter(alert => !alert.resolved);
  }

  resolveAlert(alertId: string): void {
    const alert = this.alerts.find(a => a.id === alertId);
    if (alert) {
      alert.resolved = true;
    }
  }

  getPerformanceSummary(): {
    overallScore: number;
    totalComponents: number;
    criticalIssues: number;
    averageRenderTime: number;
    memoryUsage: number;
    networkLatency: number;
  } {
    const components = this.getComponentMetrics();
    const latestMetrics = this.metrics[this.metrics.length - 1];
    
    const overallScore = components.length > 0 
      ? components.reduce((sum, c) => sum + c.performanceScore, 0) / components.length 
      : 100;
    
    const criticalIssues = components.reduce((count, c) => 
      count + c.issues.filter(i => i.severity === 'critical').length, 0
    );
    
    return {
      overallScore,
      totalComponents: components.length,
      criticalIssues,
      averageRenderTime: latestMetrics?.componentRenderTime || 0,
      memoryUsage: latestMetrics?.memoryUsage || 0,
      networkLatency: latestMetrics?.networkLatency || 0
    };
  }
}

// Performance monitoring dashboard component
const PerformanceMonitoringDashboard: React.FC = () => {
  const [framework] = useState(() => new PerformanceMonitoringFramework());
  const [metrics, setMetrics] = useState<PerformanceMetrics[]>([]);
  const [componentMetrics, setComponentMetrics] = useState<ComponentPerformance[]>([]);
  const [alerts, setAlerts] = useState<PerformanceAlert[]>([]);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [selectedTab, setSelectedTab] = useState<'overview' | 'components' | 'metrics' | 'alerts'>('overview');
  const intervalRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      framework.stopMonitoring();
    };
  }, [framework]);

  const startMonitoring = useCallback(() => {
    framework.startMonitoring();
    setIsMonitoring(true);
    
    intervalRef.current = setInterval(() => {
      setMetrics(framework.getMetrics());
      setComponentMetrics(framework.getComponentMetrics());
      setAlerts(framework.getAlerts());
    }, 1000);
  }, [framework]);

  const stopMonitoring = useCallback(() => {
    framework.stopMonitoring();
    setIsMonitoring(false);
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
  }, [framework]);

  const resolveAlert = useCallback((alertId: string) => {
    framework.resolveAlert(alertId);
    setAlerts(framework.getAlerts());
  }, [framework]);

  const summary = framework.getPerformanceSummary();
  const unresolvedAlerts = framework.getUnresolvedAlerts();

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatTime = (ms: number): string => {
    return `${ms.toFixed(2)}ms`;
  };

  const getScoreColor = (score: number): string => {
    if (score >= 90) return 'text-green-600';
    if (score >= 70) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getAlertIcon = (type: PerformanceAlert['type']) => {
    switch (type) {
      case 'error': return <XCircle className="w-4 h-4 text-red-500" />;
      case 'warning': return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
      case 'info': return <CheckCircle className="w-4 h-4 text-blue-500" />;
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
              <Monitor className="w-8 h-8 text-blue-600" />
              Performance Monitoring Framework
            </h1>
            <p className="text-gray-600 mt-2">
              Real-time performance monitoring and optimization for ALL-USE user interface components
            </p>
          </div>
          <div className="flex gap-3">
            <button
              onClick={isMonitoring ? stopMonitoring : startMonitoring}
              className={`px-6 py-2 rounded-lg font-medium flex items-center gap-2 ${
                isMonitoring 
                  ? 'bg-red-600 hover:bg-red-700 text-white' 
                  : 'bg-green-600 hover:bg-green-700 text-white'
              }`}
            >
              {isMonitoring ? (
                <>
                  <XCircle className="w-4 h-4" />
                  Stop Monitoring
                </>
              ) : (
                <>
                  <Activity className="w-4 h-4" />
                  Start Monitoring
                </>
              )}
            </button>
          </div>
        </div>

        {/* Performance Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Overall Score</p>
                <p className={`text-2xl font-bold ${getScoreColor(summary.overallScore)}`}>
                  {summary.overallScore.toFixed(1)}%
                </p>
              </div>
              <TrendingUp className="w-8 h-8 text-blue-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Components</p>
                <p className="text-2xl font-bold text-gray-900">{summary.totalComponents}</p>
              </div>
              <BarChart3 className="w-8 h-8 text-green-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Critical Issues</p>
                <p className={`text-2xl font-bold ${summary.criticalIssues > 0 ? 'text-red-600' : 'text-green-600'}`}>
                  {summary.criticalIssues}
                </p>
              </div>
              <AlertTriangle className="w-8 h-8 text-red-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Memory Usage</p>
                <p className="text-2xl font-bold text-gray-900">{formatBytes(summary.memoryUsage)}</p>
              </div>
              <HardDrive className="w-8 h-8 text-purple-500" />
            </div>
          </div>
        </div>

        {/* Alerts Banner */}
        {unresolvedAlerts.length > 0 && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="w-5 h-5 text-yellow-600" />
              <h3 className="font-medium text-yellow-800">
                {unresolvedAlerts.length} Unresolved Alert{unresolvedAlerts.length !== 1 ? 's' : ''}
              </h3>
            </div>
            <div className="space-y-1">
              {unresolvedAlerts.slice(0, 3).map(alert => (
                <p key={alert.id} className="text-sm text-yellow-700">
                  {alert.message}
                </p>
              ))}
              {unresolvedAlerts.length > 3 && (
                <p className="text-sm text-yellow-600">
                  And {unresolvedAlerts.length - 3} more...
                </p>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'overview', label: 'Overview', icon: Monitor },
            { id: 'components', label: 'Components', icon: BarChart3 },
            { id: 'metrics', label: 'Metrics', icon: Activity },
            { id: 'alerts', label: 'Alerts', icon: AlertTriangle }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setSelectedTab(tab.id as any)}
              className={`flex items-center gap-2 py-2 px-1 border-b-2 font-medium text-sm ${
                selectedTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {selectedTab === 'overview' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Clock className="w-5 h-5 text-blue-500" />
                Performance Trends
              </h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Average Render Time</span>
                  <span className="font-medium">{formatTime(summary.averageRenderTime)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Network Latency</span>
                  <span className="font-medium">{formatTime(summary.networkLatency)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Memory Usage</span>
                  <span className="font-medium">{formatBytes(summary.memoryUsage)}</span>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Zap className="w-5 h-5 text-yellow-500" />
                Performance Status
              </h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Monitoring Status</span>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    isMonitoring ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                  }`}>
                    {isMonitoring ? 'Active' : 'Inactive'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Data Points</span>
                  <span className="font-medium">{metrics.length}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Last Update</span>
                  <span className="font-medium">
                    {isMonitoring ? 'Real-time' : 'Stopped'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {selectedTab === 'components' && (
        <div className="bg-white rounded-lg shadow-md">
          <div className="p-6 border-b border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900">Component Performance</h3>
            <p className="text-sm text-gray-600 mt-1">
              Performance metrics for individual components
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Component
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Render Time
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Memory Usage
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Updates
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Score
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Issues
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {componentMetrics.map(component => (
                  <tr key={component.componentName} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">
                        {component.componentName}
                      </div>
                      <div className="text-sm text-gray-500">
                        Last update: {component.lastUpdate.toLocaleTimeString()}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {formatTime(component.renderTime)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {formatBytes(component.memoryUsage)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {component.updateCount}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`text-sm font-medium ${getScoreColor(component.performanceScore)}`}>
                        {component.performanceScore.toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {component.issues.length > 0 ? (
                        <span className="px-2 py-1 text-xs font-medium bg-red-100 text-red-800 rounded-full">
                          {component.issues.length} issue{component.issues.length !== 1 ? 's' : ''}
                        </span>
                      ) : (
                        <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded-full">
                          No issues
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {componentMetrics.length === 0 && (
              <div className="text-center py-12">
                <BarChart3 className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500">No component metrics available</p>
                <p className="text-sm text-gray-400">Start monitoring to see component performance data</p>
              </div>
            )}
          </div>
        </div>
      )}

      {selectedTab === 'metrics' && (
        <div className="space-y-6">
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-blue-500" />
              Core Web Vitals
            </h3>
            {metrics.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {[
                  { label: 'First Contentful Paint', value: metrics[metrics.length - 1]?.firstContentfulPaint, unit: 'ms', threshold: 1500 },
                  { label: 'Largest Contentful Paint', value: metrics[metrics.length - 1]?.largestContentfulPaint, unit: 'ms', threshold: 2500 },
                  { label: 'Cumulative Layout Shift', value: metrics[metrics.length - 1]?.cumulativeLayoutShift, unit: '', threshold: 0.1 },
                  { label: 'First Input Delay', value: metrics[metrics.length - 1]?.firstInputDelay, unit: 'ms', threshold: 100 }
                ].map(metric => (
                  <div key={metric.label} className="border border-gray-200 rounded-lg p-4">
                    <p className="text-sm font-medium text-gray-600">{metric.label}</p>
                    <p className={`text-xl font-bold ${
                      metric.value <= metric.threshold ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {metric.value?.toFixed(metric.unit === 'ms' ? 0 : 3)}{metric.unit}
                    </p>
                    <p className="text-xs text-gray-500">
                      Threshold: {metric.threshold}{metric.unit}
                    </p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <Activity className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500">No metrics data available</p>
                <p className="text-sm text-gray-400">Start monitoring to collect performance metrics</p>
              </div>
            )}
          </div>
        </div>
      )}

      {selectedTab === 'alerts' && (
        <div className="bg-white rounded-lg shadow-md">
          <div className="p-6 border-b border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900">Performance Alerts</h3>
            <p className="text-sm text-gray-600 mt-1">
              Real-time alerts for performance issues and threshold violations
            </p>
          </div>
          <div className="divide-y divide-gray-200">
            {alerts.length > 0 ? (
              alerts.slice().reverse().map(alert => (
                <div key={alert.id} className={`p-6 ${alert.resolved ? 'bg-gray-50' : ''}`}>
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3">
                      {getAlertIcon(alert.type)}
                      <div>
                        <p className={`font-medium ${alert.resolved ? 'text-gray-600' : 'text-gray-900'}`}>
                          {alert.message}
                        </p>
                        <div className="flex items-center gap-4 mt-1 text-sm text-gray-500">
                          <span>{alert.timestamp.toLocaleString()}</span>
                          {alert.component && <span>Component: {alert.component}</span>}
                        </div>
                      </div>
                    </div>
                    {!alert.resolved && (
                      <button
                        onClick={() => resolveAlert(alert.id)}
                        className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
                      >
                        Resolve
                      </button>
                    )}
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-12">
                <CheckCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500">No alerts</p>
                <p className="text-sm text-gray-400">All systems are performing within normal parameters</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default PerformanceMonitoringDashboard;

