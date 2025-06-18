import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Settings, Zap, TrendingUp, Cpu, HardDrive, Network, Image, Code, Layers, RefreshCw, CheckCircle, AlertTriangle, BarChart3 } from 'lucide-react';

// Optimization engine interfaces and types
interface OptimizationRule {
  id: string;
  name: string;
  description: string;
  category: 'rendering' | 'memory' | 'network' | 'bundle' | 'caching' | 'lazy-loading';
  priority: 'low' | 'medium' | 'high' | 'critical';
  enabled: boolean;
  autoApply: boolean;
  conditions: OptimizationCondition[];
  actions: OptimizationAction[];
  metrics: OptimizationMetrics;
}

interface OptimizationCondition {
  metric: string;
  operator: '>' | '<' | '>=' | '<=' | '==' | '!=';
  value: number;
  unit: string;
}

interface OptimizationAction {
  type: 'code-split' | 'lazy-load' | 'cache' | 'compress' | 'preload' | 'defer' | 'virtualize' | 'memoize';
  target: string;
  parameters: Record<string, any>;
  impact: number; // Expected performance improvement percentage
}

interface OptimizationMetrics {
  applicationsCount: number;
  successRate: number;
  averageImprovement: number;
  lastApplied: Date | null;
  totalImpact: number;
}

interface OptimizationResult {
  id: string;
  ruleId: string;
  ruleName: string;
  applied: Date;
  success: boolean;
  improvement: number;
  details: string;
  beforeMetrics: PerformanceSnapshot;
  afterMetrics: PerformanceSnapshot;
}

interface PerformanceSnapshot {
  renderTime: number;
  memoryUsage: number;
  bundleSize: number;
  networkLatency: number;
  cacheHitRate: number;
  timestamp: Date;
}

interface BundleAnalysis {
  totalSize: number;
  chunks: BundleChunk[];
  duplicates: DuplicateModule[];
  unusedCode: UnusedCodeSegment[];
  optimizationOpportunities: OptimizationOpportunity[];
}

interface BundleChunk {
  name: string;
  size: number;
  modules: string[];
  loadTime: number;
  critical: boolean;
}

interface DuplicateModule {
  name: string;
  instances: number;
  totalSize: number;
  locations: string[];
}

interface UnusedCodeSegment {
  file: string;
  lines: number;
  size: number;
  percentage: number;
}

interface OptimizationOpportunity {
  type: 'code-splitting' | 'tree-shaking' | 'compression' | 'caching' | 'lazy-loading';
  description: string;
  potentialSavings: number;
  effort: 'low' | 'medium' | 'high';
  impact: 'low' | 'medium' | 'high';
}

// Performance optimization engine
class PerformanceOptimizationEngine {
  private rules: Map<string, OptimizationRule> = new Map();
  private results: OptimizationResult[] = [];
  private isRunning: boolean = false;
  private bundleAnalysis: BundleAnalysis | null = null;

  constructor() {
    this.initializeDefaultRules();
  }

  private initializeDefaultRules(): void {
    const defaultRules: OptimizationRule[] = [
      {
        id: 'slow-render-optimization',
        name: 'Slow Render Optimization',
        description: 'Optimize components with render times exceeding 100ms',
        category: 'rendering',
        priority: 'high',
        enabled: true,
        autoApply: true,
        conditions: [
          { metric: 'renderTime', operator: '>', value: 100, unit: 'ms' }
        ],
        actions: [
          { type: 'memoize', target: 'component', parameters: { deep: true }, impact: 30 },
          { type: 'virtualize', target: 'lists', parameters: { itemHeight: 50 }, impact: 50 }
        ],
        metrics: {
          applicationsCount: 0,
          successRate: 0,
          averageImprovement: 0,
          lastApplied: null,
          totalImpact: 0
        }
      },
      {
        id: 'memory-leak-prevention',
        name: 'Memory Leak Prevention',
        description: 'Prevent memory leaks in components with high memory usage',
        category: 'memory',
        priority: 'critical',
        enabled: true,
        autoApply: true,
        conditions: [
          { metric: 'memoryUsage', operator: '>', value: 100, unit: 'MB' }
        ],
        actions: [
          { type: 'cache', target: 'cleanup', parameters: { aggressive: true }, impact: 40 },
          { type: 'lazy-load', target: 'heavy-components', parameters: { threshold: 0.5 }, impact: 35 }
        ],
        metrics: {
          applicationsCount: 0,
          successRate: 0,
          averageImprovement: 0,
          lastApplied: null,
          totalImpact: 0
        }
      },
      {
        id: 'bundle-size-optimization',
        name: 'Bundle Size Optimization',
        description: 'Optimize bundle size when exceeding 2MB',
        category: 'bundle',
        priority: 'high',
        enabled: true,
        autoApply: false,
        conditions: [
          { metric: 'bundleSize', operator: '>', value: 2, unit: 'MB' }
        ],
        actions: [
          { type: 'code-split', target: 'routes', parameters: { strategy: 'route-based' }, impact: 45 },
          { type: 'compress', target: 'assets', parameters: { level: 9 }, impact: 25 }
        ],
        metrics: {
          applicationsCount: 0,
          successRate: 0,
          averageImprovement: 0,
          lastApplied: null,
          totalImpact: 0
        }
      },
      {
        id: 'network-optimization',
        name: 'Network Performance Optimization',
        description: 'Optimize network requests with high latency',
        category: 'network',
        priority: 'medium',
        enabled: true,
        autoApply: true,
        conditions: [
          { metric: 'networkLatency', operator: '>', value: 200, unit: 'ms' }
        ],
        actions: [
          { type: 'cache', target: 'api-responses', parameters: { ttl: 300 }, impact: 60 },
          { type: 'preload', target: 'critical-resources', parameters: { priority: 'high' }, impact: 30 }
        ],
        metrics: {
          applicationsCount: 0,
          successRate: 0,
          averageImprovement: 0,
          lastApplied: null,
          totalImpact: 0
        }
      },
      {
        id: 'image-optimization',
        name: 'Image Loading Optimization',
        description: 'Optimize image loading and rendering performance',
        category: 'lazy-loading',
        priority: 'medium',
        enabled: true,
        autoApply: true,
        conditions: [
          { metric: 'imageLoadTime', operator: '>', value: 500, unit: 'ms' }
        ],
        actions: [
          { type: 'lazy-load', target: 'images', parameters: { threshold: 0.1 }, impact: 40 },
          { type: 'compress', target: 'images', parameters: { quality: 0.8 }, impact: 35 }
        ],
        metrics: {
          applicationsCount: 0,
          successRate: 0,
          averageImprovement: 0,
          lastApplied: null,
          totalImpact: 0
        }
      },
      {
        id: 'caching-strategy',
        name: 'Intelligent Caching Strategy',
        description: 'Implement intelligent caching for frequently accessed data',
        category: 'caching',
        priority: 'high',
        enabled: true,
        autoApply: true,
        conditions: [
          { metric: 'cacheHitRate', operator: '<', value: 70, unit: '%' }
        ],
        actions: [
          { type: 'cache', target: 'computed-values', parameters: { strategy: 'lru', size: 100 }, impact: 50 },
          { type: 'preload', target: 'predictive', parameters: { confidence: 0.8 }, impact: 25 }
        ],
        metrics: {
          applicationsCount: 0,
          successRate: 0,
          averageImprovement: 0,
          lastApplied: null,
          totalImpact: 0
        }
      }
    ];

    defaultRules.forEach(rule => this.rules.set(rule.id, rule));
  }

  startOptimization(): void {
    this.isRunning = true;
    this.runOptimizationCycle();
  }

  stopOptimization(): void {
    this.isRunning = false;
  }

  private async runOptimizationCycle(): Promise<void> {
    if (!this.isRunning) return;

    try {
      // Analyze current performance
      const currentMetrics = await this.getCurrentPerformanceMetrics();
      
      // Check each rule
      for (const rule of this.rules.values()) {
        if (!rule.enabled) continue;
        
        if (this.shouldApplyRule(rule, currentMetrics)) {
          if (rule.autoApply) {
            await this.applyOptimization(rule, currentMetrics);
          }
        }
      }

      // Schedule next cycle
      setTimeout(() => this.runOptimizationCycle(), 10000); // Every 10 seconds
    } catch (error) {
      console.error('Optimization cycle error:', error);
      setTimeout(() => this.runOptimizationCycle(), 30000); // Retry in 30 seconds
    }
  }

  private async getCurrentPerformanceMetrics(): Promise<PerformanceSnapshot> {
    return {
      renderTime: this.measureRenderTime(),
      memoryUsage: this.measureMemoryUsage(),
      bundleSize: await this.measureBundleSize(),
      networkLatency: this.measureNetworkLatency(),
      cacheHitRate: this.measureCacheHitRate(),
      timestamp: new Date()
    };
  }

  private measureRenderTime(): number {
    // Simulate render time measurement
    const paintEntries = performance.getEntriesByType('paint');
    const fcpEntry = paintEntries.find(entry => entry.name === 'first-contentful-paint');
    return fcpEntry ? fcpEntry.startTime : 0;
  }

  private measureMemoryUsage(): number {
    if ('memory' in performance) {
      return (performance as any).memory.usedJSHeapSize / (1024 * 1024); // Convert to MB
    }
    return 0;
  }

  private async measureBundleSize(): Promise<number> {
    const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[];
    const totalSize = resources.reduce((total, resource) => total + (resource.transferSize || 0), 0);
    return totalSize / (1024 * 1024); // Convert to MB
  }

  private measureNetworkLatency(): number {
    const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[];
    if (resources.length === 0) return 0;
    
    const latencies = resources.map(resource => resource.responseEnd - resource.requestStart);
    return latencies.reduce((a, b) => a + b, 0) / latencies.length;
  }

  private measureCacheHitRate(): number {
    // Simulate cache hit rate measurement
    return Math.random() * 100; // Placeholder
  }

  private shouldApplyRule(rule: OptimizationRule, metrics: PerformanceSnapshot): boolean {
    return rule.conditions.every(condition => {
      const metricValue = this.getMetricValue(condition.metric, metrics);
      return this.evaluateCondition(metricValue, condition.operator, condition.value);
    });
  }

  private getMetricValue(metric: string, snapshot: PerformanceSnapshot): number {
    switch (metric) {
      case 'renderTime': return snapshot.renderTime;
      case 'memoryUsage': return snapshot.memoryUsage;
      case 'bundleSize': return snapshot.bundleSize;
      case 'networkLatency': return snapshot.networkLatency;
      case 'cacheHitRate': return snapshot.cacheHitRate;
      case 'imageLoadTime': return 300; // Placeholder
      default: return 0;
    }
  }

  private evaluateCondition(value: number, operator: string, threshold: number): boolean {
    switch (operator) {
      case '>': return value > threshold;
      case '<': return value < threshold;
      case '>=': return value >= threshold;
      case '<=': return value <= threshold;
      case '==': return value === threshold;
      case '!=': return value !== threshold;
      default: return false;
    }
  }

  private async applyOptimization(rule: OptimizationRule, beforeMetrics: PerformanceSnapshot): Promise<void> {
    try {
      // Simulate optimization application
      const optimizationPromises = rule.actions.map(action => this.executeOptimizationAction(action));
      await Promise.all(optimizationPromises);

      // Wait for optimization to take effect
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Measure after metrics
      const afterMetrics = await this.getCurrentPerformanceMetrics();

      // Calculate improvement
      const improvement = this.calculateImprovement(beforeMetrics, afterMetrics);

      // Record result
      const result: OptimizationResult = {
        id: `result-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        ruleId: rule.id,
        ruleName: rule.name,
        applied: new Date(),
        success: improvement > 0,
        improvement,
        details: `Applied ${rule.actions.length} optimization actions`,
        beforeMetrics,
        afterMetrics
      };

      this.results.push(result);

      // Update rule metrics
      rule.metrics.applicationsCount++;
      rule.metrics.lastApplied = new Date();
      rule.metrics.totalImpact += improvement;
      rule.metrics.averageImprovement = rule.metrics.totalImpact / rule.metrics.applicationsCount;
      rule.metrics.successRate = this.results.filter(r => r.ruleId === rule.id && r.success).length / 
                                 this.results.filter(r => r.ruleId === rule.id).length * 100;

      // Keep only last 100 results
      if (this.results.length > 100) {
        this.results = this.results.slice(-100);
      }

    } catch (error) {
      console.error(`Failed to apply optimization rule ${rule.name}:`, error);
    }
  }

  private async executeOptimizationAction(action: OptimizationAction): Promise<void> {
    // Simulate optimization action execution
    switch (action.type) {
      case 'memoize':
        await this.applyMemoization(action.target, action.parameters);
        break;
      case 'lazy-load':
        await this.applyLazyLoading(action.target, action.parameters);
        break;
      case 'code-split':
        await this.applyCodeSplitting(action.target, action.parameters);
        break;
      case 'cache':
        await this.applyCaching(action.target, action.parameters);
        break;
      case 'compress':
        await this.applyCompression(action.target, action.parameters);
        break;
      case 'preload':
        await this.applyPreloading(action.target, action.parameters);
        break;
      case 'virtualize':
        await this.applyVirtualization(action.target, action.parameters);
        break;
      default:
        console.warn(`Unknown optimization action: ${action.type}`);
    }
  }

  private async applyMemoization(target: string, parameters: any): Promise<void> {
    // Simulate memoization optimization
    console.log(`Applying memoization to ${target}`, parameters);
  }

  private async applyLazyLoading(target: string, parameters: any): Promise<void> {
    // Simulate lazy loading optimization
    console.log(`Applying lazy loading to ${target}`, parameters);
  }

  private async applyCodeSplitting(target: string, parameters: any): Promise<void> {
    // Simulate code splitting optimization
    console.log(`Applying code splitting to ${target}`, parameters);
  }

  private async applyCaching(target: string, parameters: any): Promise<void> {
    // Simulate caching optimization
    console.log(`Applying caching to ${target}`, parameters);
  }

  private async applyCompression(target: string, parameters: any): Promise<void> {
    // Simulate compression optimization
    console.log(`Applying compression to ${target}`, parameters);
  }

  private async applyPreloading(target: string, parameters: any): Promise<void> {
    // Simulate preloading optimization
    console.log(`Applying preloading to ${target}`, parameters);
  }

  private async applyVirtualization(target: string, parameters: any): Promise<void> {
    // Simulate virtualization optimization
    console.log(`Applying virtualization to ${target}`, parameters);
  }

  private calculateImprovement(before: PerformanceSnapshot, after: PerformanceSnapshot): number {
    // Calculate overall performance improvement percentage
    const renderImprovement = Math.max(0, (before.renderTime - after.renderTime) / before.renderTime * 100);
    const memoryImprovement = Math.max(0, (before.memoryUsage - after.memoryUsage) / before.memoryUsage * 100);
    const networkImprovement = Math.max(0, (before.networkLatency - after.networkLatency) / before.networkLatency * 100);
    
    return (renderImprovement + memoryImprovement + networkImprovement) / 3;
  }

  async analyzeBundleSize(): Promise<BundleAnalysis> {
    // Simulate bundle analysis
    const analysis: BundleAnalysis = {
      totalSize: 2.1 * 1024 * 1024, // 2.1MB
      chunks: [
        {
          name: 'main',
          size: 800 * 1024,
          modules: ['App.tsx', 'index.tsx', 'components/*'],
          loadTime: 1200,
          critical: true
        },
        {
          name: 'vendor',
          size: 1.2 * 1024 * 1024,
          modules: ['react', 'react-dom', 'lodash'],
          loadTime: 1800,
          critical: true
        },
        {
          name: 'dashboard',
          size: 100 * 1024,
          modules: ['Dashboard.tsx', 'Charts.tsx'],
          loadTime: 300,
          critical: false
        }
      ],
      duplicates: [
        {
          name: 'lodash',
          instances: 3,
          totalSize: 150 * 1024,
          locations: ['main', 'vendor', 'dashboard']
        }
      ],
      unusedCode: [
        {
          file: 'utils.ts',
          lines: 45,
          size: 12 * 1024,
          percentage: 60
        }
      ],
      optimizationOpportunities: [
        {
          type: 'code-splitting',
          description: 'Split dashboard components into separate chunk',
          potentialSavings: 100 * 1024,
          effort: 'medium',
          impact: 'high'
        },
        {
          type: 'tree-shaking',
          description: 'Remove unused lodash functions',
          potentialSavings: 80 * 1024,
          effort: 'low',
          impact: 'medium'
        }
      ]
    };

    this.bundleAnalysis = analysis;
    return analysis;
  }

  getRules(): OptimizationRule[] {
    return Array.from(this.rules.values());
  }

  getResults(): OptimizationResult[] {
    return [...this.results];
  }

  getBundleAnalysis(): BundleAnalysis | null {
    return this.bundleAnalysis;
  }

  updateRule(ruleId: string, updates: Partial<OptimizationRule>): void {
    const rule = this.rules.get(ruleId);
    if (rule) {
      Object.assign(rule, updates);
    }
  }

  addRule(rule: OptimizationRule): void {
    this.rules.set(rule.id, rule);
  }

  removeRule(ruleId: string): void {
    this.rules.delete(ruleId);
  }

  getOptimizationSummary(): {
    totalOptimizations: number;
    successRate: number;
    averageImprovement: number;
    totalImpact: number;
    activeRules: number;
  } {
    const totalOptimizations = this.results.length;
    const successfulOptimizations = this.results.filter(r => r.success).length;
    const successRate = totalOptimizations > 0 ? (successfulOptimizations / totalOptimizations) * 100 : 0;
    const averageImprovement = totalOptimizations > 0 
      ? this.results.reduce((sum, r) => sum + r.improvement, 0) / totalOptimizations 
      : 0;
    const totalImpact = this.results.reduce((sum, r) => sum + r.improvement, 0);
    const activeRules = Array.from(this.rules.values()).filter(r => r.enabled).length;

    return {
      totalOptimizations,
      successRate,
      averageImprovement,
      totalImpact,
      activeRules
    };
  }

  isOptimizationRunning(): boolean {
    return this.isRunning;
  }
}

// Optimization engine dashboard component
const OptimizationEngineDashboard: React.FC = () => {
  const [engine] = useState(() => new PerformanceOptimizationEngine());
  const [rules, setRules] = useState<OptimizationRule[]>([]);
  const [results, setResults] = useState<OptimizationResult[]>([]);
  const [bundleAnalysis, setBundleAnalysis] = useState<BundleAnalysis | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [selectedTab, setSelectedTab] = useState<'overview' | 'rules' | 'results' | 'bundle'>('overview');
  const intervalRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    setRules(engine.getRules());
    setResults(engine.getResults());
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      engine.stopOptimization();
    };
  }, [engine]);

  const startOptimization = useCallback(() => {
    engine.startOptimization();
    setIsRunning(true);
    
    intervalRef.current = setInterval(() => {
      setRules(engine.getRules());
      setResults(engine.getResults());
    }, 2000);
  }, [engine]);

  const stopOptimization = useCallback(() => {
    engine.stopOptimization();
    setIsRunning(false);
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
  }, [engine]);

  const analyzeBundleSize = useCallback(async () => {
    const analysis = await engine.analyzeBundleSize();
    setBundleAnalysis(analysis);
  }, [engine]);

  const toggleRule = useCallback((ruleId: string) => {
    const rule = rules.find(r => r.id === ruleId);
    if (rule) {
      engine.updateRule(ruleId, { enabled: !rule.enabled });
      setRules(engine.getRules());
    }
  }, [engine, rules]);

  const toggleAutoApply = useCallback((ruleId: string) => {
    const rule = rules.find(r => r.id === ruleId);
    if (rule) {
      engine.updateRule(ruleId, { autoApply: !rule.autoApply });
      setRules(engine.getRules());
    }
  }, [engine, rules]);

  const summary = engine.getOptimizationSummary();

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatPercentage = (value: number): string => {
    return `${value.toFixed(1)}%`;
  };

  const getPriorityColor = (priority: string): string => {
    switch (priority) {
      case 'critical': return 'text-red-600 bg-red-100';
      case 'high': return 'text-orange-600 bg-orange-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'low': return 'text-green-600 bg-green-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'rendering': return <Cpu className="w-4 h-4" />;
      case 'memory': return <HardDrive className="w-4 h-4" />;
      case 'network': return <Network className="w-4 h-4" />;
      case 'bundle': return <Code className="w-4 h-4" />;
      case 'caching': return <Layers className="w-4 h-4" />;
      case 'lazy-loading': return <Image className="w-4 h-4" />;
      default: return <Settings className="w-4 h-4" />;
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
              <Zap className="w-8 h-8 text-yellow-600" />
              Performance Optimization Engine
            </h1>
            <p className="text-gray-600 mt-2">
              Intelligent performance optimization with automated rule-based improvements
            </p>
          </div>
          <div className="flex gap-3">
            <button
              onClick={analyzeBundleSize}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
            >
              <BarChart3 className="w-4 h-4" />
              Analyze Bundle
            </button>
            <button
              onClick={isRunning ? stopOptimization : startOptimization}
              className={`px-6 py-2 rounded-lg font-medium flex items-center gap-2 ${
                isRunning 
                  ? 'bg-red-600 hover:bg-red-700 text-white' 
                  : 'bg-green-600 hover:bg-green-700 text-white'
              }`}
            >
              {isRunning ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Stop Optimization
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  Start Optimization
                </>
              )}
            </button>
          </div>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Optimizations</p>
                <p className="text-2xl font-bold text-gray-900">{summary.totalOptimizations}</p>
              </div>
              <TrendingUp className="w-8 h-8 text-blue-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Success Rate</p>
                <p className="text-2xl font-bold text-green-600">{formatPercentage(summary.successRate)}</p>
              </div>
              <CheckCircle className="w-8 h-8 text-green-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Avg Improvement</p>
                <p className="text-2xl font-bold text-blue-600">{formatPercentage(summary.averageImprovement)}</p>
              </div>
              <TrendingUp className="w-8 h-8 text-blue-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Impact</p>
                <p className="text-2xl font-bold text-purple-600">{formatPercentage(summary.totalImpact)}</p>
              </div>
              <Zap className="w-8 h-8 text-purple-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Active Rules</p>
                <p className="text-2xl font-bold text-gray-900">{summary.activeRules}</p>
              </div>
              <Settings className="w-8 h-8 text-gray-500" />
            </div>
          </div>
        </div>

        {/* Status Banner */}
        <div className={`rounded-lg p-4 mb-6 ${
          isRunning ? 'bg-green-50 border border-green-200' : 'bg-gray-50 border border-gray-200'
        }`}>
          <div className="flex items-center gap-2">
            {isRunning ? (
              <>
                <RefreshCw className="w-5 h-5 text-green-600 animate-spin" />
                <span className="font-medium text-green-800">Optimization engine is running</span>
                <span className="text-green-600">- Continuously monitoring and optimizing performance</span>
              </>
            ) : (
              <>
                <AlertTriangle className="w-5 h-5 text-gray-600" />
                <span className="font-medium text-gray-800">Optimization engine is stopped</span>
                <span className="text-gray-600">- Click "Start Optimization" to begin automatic optimization</span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'overview', label: 'Overview', icon: TrendingUp },
            { id: 'rules', label: 'Optimization Rules', icon: Settings },
            { id: 'results', label: 'Results', icon: CheckCircle },
            { id: 'bundle', label: 'Bundle Analysis', icon: BarChart3 }
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
                <TrendingUp className="w-5 h-5 text-blue-500" />
                Optimization Performance
              </h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Optimizations Applied</span>
                  <span className="font-medium">{summary.totalOptimizations}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Success Rate</span>
                  <span className="font-medium text-green-600">{formatPercentage(summary.successRate)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Average Improvement</span>
                  <span className="font-medium text-blue-600">{formatPercentage(summary.averageImprovement)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Cumulative Impact</span>
                  <span className="font-medium text-purple-600">{formatPercentage(summary.totalImpact)}</span>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Settings className="w-5 h-5 text-gray-500" />
                Engine Status
              </h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Engine Status</span>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    isRunning ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                  }`}>
                    {isRunning ? 'Running' : 'Stopped'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Active Rules</span>
                  <span className="font-medium">{summary.activeRules} / {rules.length}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Auto-Apply Rules</span>
                  <span className="font-medium">{rules.filter(r => r.autoApply).length}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Last Optimization</span>
                  <span className="font-medium">
                    {results.length > 0 ? results[results.length - 1].applied.toLocaleTimeString() : 'Never'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Recent Results */}
          <div className="bg-white rounded-lg shadow-md">
            <div className="p-6 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900">Recent Optimizations</h3>
              <p className="text-sm text-gray-600 mt-1">Latest optimization results and improvements</p>
            </div>
            <div className="p-6">
              {results.length > 0 ? (
                <div className="space-y-4">
                  {results.slice(-5).reverse().map(result => (
                    <div key={result.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                      <div className="flex items-center gap-3">
                        {result.success ? (
                          <CheckCircle className="w-5 h-5 text-green-500" />
                        ) : (
                          <AlertTriangle className="w-5 h-5 text-red-500" />
                        )}
                        <div>
                          <p className="font-medium text-gray-900">{result.ruleName}</p>
                          <p className="text-sm text-gray-600">{result.details}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className={`font-medium ${result.success ? 'text-green-600' : 'text-red-600'}`}>
                          {result.success ? '+' : ''}{formatPercentage(result.improvement)}
                        </p>
                        <p className="text-sm text-gray-500">{result.applied.toLocaleTimeString()}</p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <TrendingUp className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-500">No optimizations applied yet</p>
                  <p className="text-sm text-gray-400">Start the optimization engine to see results</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {selectedTab === 'rules' && (
        <div className="bg-white rounded-lg shadow-md">
          <div className="p-6 border-b border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900">Optimization Rules</h3>
            <p className="text-sm text-gray-600 mt-1">
              Configure and manage performance optimization rules
            </p>
          </div>
          <div className="divide-y divide-gray-200">
            {rules.map(rule => (
              <div key={rule.id} className="p-6">
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-4">
                    <div className="flex items-center gap-2 mt-1">
                      {getCategoryIcon(rule.category)}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h4 className="font-medium text-gray-900">{rule.name}</h4>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getPriorityColor(rule.priority)}`}>
                          {rule.priority}
                        </span>
                        <span className="px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                          {rule.category}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 mb-3">{rule.description}</p>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <span className="text-gray-500">Applications:</span>
                          <span className="ml-1 font-medium">{rule.metrics.applicationsCount}</span>
                        </div>
                        <div>
                          <span className="text-gray-500">Success Rate:</span>
                          <span className="ml-1 font-medium">{formatPercentage(rule.metrics.successRate)}</span>
                        </div>
                        <div>
                          <span className="text-gray-500">Avg Improvement:</span>
                          <span className="ml-1 font-medium">{formatPercentage(rule.metrics.averageImprovement)}</span>
                        </div>
                        <div>
                          <span className="text-gray-500">Last Applied:</span>
                          <span className="ml-1 font-medium">
                            {rule.metrics.lastApplied ? rule.metrics.lastApplied.toLocaleDateString() : 'Never'}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <label className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={rule.autoApply}
                        onChange={() => toggleAutoApply(rule.id)}
                        className="rounded border-gray-300"
                      />
                      Auto-apply
                    </label>
                    <label className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={rule.enabled}
                        onChange={() => toggleRule(rule.id)}
                        className="rounded border-gray-300"
                      />
                      <span className={`text-sm font-medium ${rule.enabled ? 'text-green-600' : 'text-gray-500'}`}>
                        {rule.enabled ? 'Enabled' : 'Disabled'}
                      </span>
                    </label>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {selectedTab === 'results' && (
        <div className="bg-white rounded-lg shadow-md">
          <div className="p-6 border-b border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900">Optimization Results</h3>
            <p className="text-sm text-gray-600 mt-1">
              Detailed results and performance improvements from applied optimizations
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Rule
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Applied
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Improvement
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Details
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {results.slice().reverse().map(result => (
                  <tr key={result.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">{result.ruleName}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {result.applied.toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {result.success ? (
                        <span className="flex items-center gap-1 text-green-600">
                          <CheckCircle className="w-4 h-4" />
                          Success
                        </span>
                      ) : (
                        <span className="flex items-center gap-1 text-red-600">
                          <AlertTriangle className="w-4 h-4" />
                          Failed
                        </span>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`font-medium ${result.success ? 'text-green-600' : 'text-red-600'}`}>
                        {result.success ? '+' : ''}{formatPercentage(result.improvement)}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-600">
                      {result.details}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {results.length === 0 && (
              <div className="text-center py-12">
                <CheckCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500">No optimization results yet</p>
                <p className="text-sm text-gray-400">Start the optimization engine to see results</p>
              </div>
            )}
          </div>
        </div>
      )}

      {selectedTab === 'bundle' && (
        <div className="space-y-6">
          {bundleAnalysis ? (
            <>
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-blue-500" />
                  Bundle Overview
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="text-center">
                    <p className="text-2xl font-bold text-gray-900">{formatBytes(bundleAnalysis.totalSize)}</p>
                    <p className="text-sm text-gray-600">Total Bundle Size</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold text-blue-600">{bundleAnalysis.chunks.length}</p>
                    <p className="text-sm text-gray-600">Chunks</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold text-orange-600">{bundleAnalysis.optimizationOpportunities.length}</p>
                    <p className="text-sm text-gray-600">Optimization Opportunities</p>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-white rounded-lg shadow-md">
                  <div className="p-6 border-b border-gray-200">
                    <h4 className="font-semibold text-gray-900">Bundle Chunks</h4>
                  </div>
                  <div className="p-6">
                    <div className="space-y-4">
                      {bundleAnalysis.chunks.map(chunk => (
                        <div key={chunk.name} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                          <div>
                            <p className="font-medium text-gray-900">{chunk.name}</p>
                            <p className="text-sm text-gray-600">{chunk.modules.length} modules</p>
                          </div>
                          <div className="text-right">
                            <p className="font-medium">{formatBytes(chunk.size)}</p>
                            <p className="text-sm text-gray-600">{chunk.loadTime}ms</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg shadow-md">
                  <div className="p-6 border-b border-gray-200">
                    <h4 className="font-semibold text-gray-900">Optimization Opportunities</h4>
                  </div>
                  <div className="p-6">
                    <div className="space-y-4">
                      {bundleAnalysis.optimizationOpportunities.map((opportunity, index) => (
                        <div key={index} className="p-3 border border-gray-200 rounded-lg">
                          <div className="flex items-center justify-between mb-2">
                            <p className="font-medium text-gray-900 capitalize">{opportunity.type.replace('-', ' ')}</p>
                            <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">
                              {formatBytes(opportunity.potentialSavings)} savings
                            </span>
                          </div>
                          <p className="text-sm text-gray-600 mb-2">{opportunity.description}</p>
                          <div className="flex gap-2">
                            <span className={`px-2 py-1 text-xs rounded-full ${
                              opportunity.effort === 'low' ? 'bg-green-100 text-green-800' :
                              opportunity.effort === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                              'bg-red-100 text-red-800'
                            }`}>
                              {opportunity.effort} effort
                            </span>
                            <span className={`px-2 py-1 text-xs rounded-full ${
                              opportunity.impact === 'high' ? 'bg-green-100 text-green-800' :
                              opportunity.impact === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                              'bg-red-100 text-red-800'
                            }`}>
                              {opportunity.impact} impact
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="bg-white rounded-lg shadow-md p-12 text-center">
              <BarChart3 className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500 mb-4">No bundle analysis available</p>
              <button
                onClick={analyzeBundleSize}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Analyze Bundle Size
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default OptimizationEngineDashboard;

