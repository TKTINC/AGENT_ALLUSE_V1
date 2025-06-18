import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Network, Layers, Zap, Activity, Settings, Monitor, Cpu, HardDrive, Globe, CheckCircle, AlertTriangle, TrendingUp, BarChart3 } from 'lucide-react';

// System coordination interfaces and types
interface SystemComponent {
  id: string;
  name: string;
  type: 'monitoring' | 'optimization' | 'analytics' | 'ui-component' | 'integration';
  status: 'active' | 'inactive' | 'error' | 'optimizing';
  performanceScore: number;
  lastUpdate: Date;
  dependencies: string[];
  metrics: ComponentMetrics;
  configuration: ComponentConfiguration;
}

interface ComponentMetrics {
  cpuUsage: number;
  memoryUsage: number;
  responseTime: number;
  throughput: number;
  errorRate: number;
  availability: number;
}

interface ComponentConfiguration {
  enabled: boolean;
  priority: 'low' | 'medium' | 'high' | 'critical';
  autoOptimize: boolean;
  thresholds: Record<string, number>;
  optimizationRules: string[];
}

interface PerformanceCoordination {
  id: string;
  name: string;
  description: string;
  components: string[];
  coordinationType: 'load-balancing' | 'resource-sharing' | 'optimization-sync' | 'data-flow';
  status: 'active' | 'inactive' | 'configuring';
  effectiveness: number;
  lastExecution: Date;
}

interface SystemHealthMetrics {
  overallScore: number;
  componentHealth: number;
  coordinationEfficiency: number;
  optimizationImpact: number;
  resourceUtilization: number;
  userExperience: number;
  timestamp: Date;
}

interface PerformanceAlert {
  id: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  component: string;
  message: string;
  timestamp: Date;
  resolved: boolean;
  autoResolvable: boolean;
}

interface OptimizationTask {
  id: string;
  name: string;
  targetComponents: string[];
  type: 'performance' | 'memory' | 'network' | 'rendering' | 'coordination';
  priority: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  estimatedCompletion: Date;
  impact: number;
}

// System coordination and performance integration engine
class SystemCoordinationEngine {
  private components: Map<string, SystemComponent> = new Map();
  private coordinations: Map<string, PerformanceCoordination> = new Map();
  private healthHistory: SystemHealthMetrics[] = [];
  private alerts: PerformanceAlert[] = [];
  private optimizationTasks: OptimizationTask[] = [];
  private isCoordinating: boolean = false;

  constructor() {
    this.initializeSystemComponents();
    this.initializeCoordinations();
    this.generateSampleData();
  }

  private initializeSystemComponents(): void {
    const components: SystemComponent[] = [
      {
        id: 'performance-monitor',
        name: 'Performance Monitoring Framework',
        type: 'monitoring',
        status: 'active',
        performanceScore: 94.5,
        lastUpdate: new Date(),
        dependencies: [],
        metrics: {
          cpuUsage: 15,
          memoryUsage: 45,
          responseTime: 12,
          throughput: 1200,
          errorRate: 0.1,
          availability: 99.9
        },
        configuration: {
          enabled: true,
          priority: 'critical',
          autoOptimize: true,
          thresholds: { responseTime: 50, errorRate: 1.0 },
          optimizationRules: ['auto-scale', 'cache-optimization']
        }
      },
      {
        id: 'optimization-engine',
        name: 'Performance Optimization Engine',
        type: 'optimization',
        status: 'active',
        performanceScore: 91.2,
        lastUpdate: new Date(),
        dependencies: ['performance-monitor'],
        metrics: {
          cpuUsage: 25,
          memoryUsage: 60,
          responseTime: 18,
          throughput: 800,
          errorRate: 0.2,
          availability: 99.7
        },
        configuration: {
          enabled: true,
          priority: 'high',
          autoOptimize: true,
          thresholds: { cpuUsage: 80, memoryUsage: 85 },
          optimizationRules: ['intelligent-caching', 'resource-optimization']
        }
      },
      {
        id: 'analytics-engine',
        name: 'Advanced Analytics Engine',
        type: 'analytics',
        status: 'active',
        performanceScore: 88.7,
        lastUpdate: new Date(),
        dependencies: ['performance-monitor', 'optimization-engine'],
        metrics: {
          cpuUsage: 35,
          memoryUsage: 75,
          responseTime: 25,
          throughput: 600,
          errorRate: 0.3,
          availability: 99.5
        },
        configuration: {
          enabled: true,
          priority: 'high',
          autoOptimize: true,
          thresholds: { responseTime: 100, memoryUsage: 90 },
          optimizationRules: ['predictive-scaling', 'data-optimization']
        }
      },
      {
        id: 'conversational-interface',
        name: 'Conversational Interface',
        type: 'ui-component',
        status: 'active',
        performanceScore: 92.8,
        lastUpdate: new Date(),
        dependencies: ['performance-monitor'],
        metrics: {
          cpuUsage: 20,
          memoryUsage: 40,
          responseTime: 15,
          throughput: 1000,
          errorRate: 0.1,
          availability: 99.8
        },
        configuration: {
          enabled: true,
          priority: 'high',
          autoOptimize: true,
          thresholds: { responseTime: 30, memoryUsage: 60 },
          optimizationRules: ['component-memoization', 'lazy-loading']
        }
      },
      {
        id: 'dashboard-builder',
        name: 'Advanced Dashboard Builder',
        type: 'ui-component',
        status: 'active',
        performanceScore: 89.3,
        lastUpdate: new Date(),
        dependencies: ['analytics-engine', 'optimization-engine'],
        metrics: {
          cpuUsage: 30,
          memoryUsage: 55,
          responseTime: 22,
          throughput: 750,
          errorRate: 0.2,
          availability: 99.6
        },
        configuration: {
          enabled: true,
          priority: 'medium',
          autoOptimize: true,
          thresholds: { responseTime: 50, cpuUsage: 70 },
          optimizationRules: ['virtualization', 'data-streaming']
        }
      },
      {
        id: 'market-integration',
        name: 'Market Integration System',
        type: 'integration',
        status: 'active',
        performanceScore: 86.1,
        lastUpdate: new Date(),
        dependencies: ['performance-monitor', 'analytics-engine'],
        metrics: {
          cpuUsage: 40,
          memoryUsage: 65,
          responseTime: 35,
          throughput: 500,
          errorRate: 0.4,
          availability: 99.2
        },
        configuration: {
          enabled: true,
          priority: 'high',
          autoOptimize: true,
          thresholds: { responseTime: 100, errorRate: 0.5 },
          optimizationRules: ['connection-pooling', 'request-batching']
        }
      }
    ];

    components.forEach(component => this.components.set(component.id, component));
  }

  private initializeCoordinations(): void {
    const coordinations: PerformanceCoordination[] = [
      {
        id: 'monitoring-optimization-sync',
        name: 'Monitoring-Optimization Synchronization',
        description: 'Coordinates performance monitoring data with optimization engine decisions',
        components: ['performance-monitor', 'optimization-engine'],
        coordinationType: 'data-flow',
        status: 'active',
        effectiveness: 94.2,
        lastExecution: new Date()
      },
      {
        id: 'analytics-ui-coordination',
        name: 'Analytics-UI Performance Coordination',
        description: 'Balances analytics processing load with UI responsiveness',
        components: ['analytics-engine', 'conversational-interface', 'dashboard-builder'],
        coordinationType: 'load-balancing',
        status: 'active',
        effectiveness: 87.5,
        lastExecution: new Date()
      },
      {
        id: 'resource-sharing-optimization',
        name: 'Cross-Component Resource Sharing',
        description: 'Optimizes resource allocation across all system components',
        components: ['performance-monitor', 'optimization-engine', 'analytics-engine'],
        coordinationType: 'resource-sharing',
        status: 'active',
        effectiveness: 91.8,
        lastExecution: new Date()
      },
      {
        id: 'integration-performance-sync',
        name: 'Integration Performance Synchronization',
        description: 'Coordinates external integrations with internal performance optimization',
        components: ['market-integration', 'optimization-engine', 'performance-monitor'],
        coordinationType: 'optimization-sync',
        status: 'active',
        effectiveness: 83.7,
        lastExecution: new Date()
      }
    ];

    coordinations.forEach(coordination => this.coordinations.set(coordination.id, coordination));
  }

  private generateSampleData(): void {
    // Generate health history
    const now = new Date();
    for (let i = 24; i >= 0; i--) {
      const timestamp = new Date(now.getTime() - i * 60 * 60 * 1000);
      this.healthHistory.push({
        overallScore: 85 + Math.random() * 10,
        componentHealth: 88 + Math.random() * 8,
        coordinationEfficiency: 82 + Math.random() * 12,
        optimizationImpact: 90 + Math.random() * 8,
        resourceUtilization: 75 + Math.random() * 15,
        userExperience: 92 + Math.random() * 6,
        timestamp
      });
    }

    // Generate sample alerts
    this.alerts = [
      {
        id: 'alert-1',
        severity: 'warning',
        component: 'market-integration',
        message: 'Response time exceeding threshold (35ms > 30ms)',
        timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
        resolved: false,
        autoResolvable: true
      },
      {
        id: 'alert-2',
        severity: 'info',
        component: 'analytics-engine',
        message: 'Memory usage optimization completed successfully',
        timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000),
        resolved: true,
        autoResolvable: false
      },
      {
        id: 'alert-3',
        severity: 'error',
        component: 'dashboard-builder',
        message: 'Component rendering performance degraded',
        timestamp: new Date(Date.now() - 6 * 60 * 60 * 1000),
        resolved: true,
        autoResolvable: true
      }
    ];

    // Generate sample optimization tasks
    this.optimizationTasks = [
      {
        id: 'task-1',
        name: 'Memory Usage Optimization',
        targetComponents: ['analytics-engine', 'dashboard-builder'],
        type: 'memory',
        priority: 8,
        status: 'running',
        progress: 65,
        estimatedCompletion: new Date(Date.now() + 30 * 60 * 1000),
        impact: 15
      },
      {
        id: 'task-2',
        name: 'Network Latency Reduction',
        targetComponents: ['market-integration'],
        type: 'network',
        priority: 9,
        status: 'pending',
        progress: 0,
        estimatedCompletion: new Date(Date.now() + 60 * 60 * 1000),
        impact: 25
      },
      {
        id: 'task-3',
        name: 'Component Coordination Enhancement',
        targetComponents: ['performance-monitor', 'optimization-engine', 'analytics-engine'],
        type: 'coordination',
        priority: 7,
        status: 'completed',
        progress: 100,
        estimatedCompletion: new Date(Date.now() - 30 * 60 * 1000),
        impact: 12
      }
    ];
  }

  startCoordination(): void {
    this.isCoordinating = true;
    this.runCoordinationCycle();
  }

  stopCoordination(): void {
    this.isCoordinating = false;
  }

  private async runCoordinationCycle(): Promise<void> {
    if (!this.isCoordinating) return;

    try {
      // Update component metrics
      this.updateComponentMetrics();
      
      // Execute coordinations
      await this.executeCoordinations();
      
      // Process optimization tasks
      this.processOptimizationTasks();
      
      // Update system health
      this.updateSystemHealth();
      
      // Check for alerts
      this.checkSystemAlerts();
      
      // Schedule next cycle
      setTimeout(() => this.runCoordinationCycle(), 15000); // Every 15 seconds
    } catch (error) {
      console.error('Coordination cycle error:', error);
      setTimeout(() => this.runCoordinationCycle(), 30000); // Retry in 30 seconds
    }
  }

  private updateComponentMetrics(): void {
    for (const component of this.components.values()) {
      // Simulate realistic metric updates
      const metrics = component.metrics;
      
      metrics.cpuUsage = Math.max(5, Math.min(95, metrics.cpuUsage + (Math.random() - 0.5) * 10));
      metrics.memoryUsage = Math.max(10, Math.min(90, metrics.memoryUsage + (Math.random() - 0.5) * 8));
      metrics.responseTime = Math.max(5, metrics.responseTime + (Math.random() - 0.5) * 5);
      metrics.throughput = Math.max(100, metrics.throughput + (Math.random() - 0.5) * 100);
      metrics.errorRate = Math.max(0, Math.min(5, metrics.errorRate + (Math.random() - 0.5) * 0.2));
      metrics.availability = Math.max(95, Math.min(100, metrics.availability + (Math.random() - 0.5) * 0.5));
      
      // Update performance score based on metrics
      component.performanceScore = this.calculatePerformanceScore(component);
      component.lastUpdate = new Date();
    }
  }

  private calculatePerformanceScore(component: SystemComponent): number {
    const metrics = component.metrics;
    let score = 100;
    
    // Deduct points based on various metrics
    score -= Math.max(0, (metrics.cpuUsage - 50) / 2); // CPU usage penalty
    score -= Math.max(0, (metrics.memoryUsage - 60) / 2); // Memory usage penalty
    score -= Math.max(0, (metrics.responseTime - 20) / 5); // Response time penalty
    score -= metrics.errorRate * 10; // Error rate penalty
    score -= Math.max(0, (100 - metrics.availability) * 5); // Availability penalty
    
    return Math.max(0, Math.min(100, score));
  }

  private async executeCoordinations(): Promise<void> {
    for (const coordination of this.coordinations.values()) {
      if (coordination.status !== 'active') continue;
      
      await this.executeCoordination(coordination);
    }
  }

  private async executeCoordination(coordination: PerformanceCoordination): Promise<void> {
    const components = coordination.components.map(id => this.components.get(id)).filter(Boolean) as SystemComponent[];
    
    switch (coordination.coordinationType) {
      case 'load-balancing':
        await this.executeLoadBalancing(coordination, components);
        break;
      case 'resource-sharing':
        await this.executeResourceSharing(coordination, components);
        break;
      case 'optimization-sync':
        await this.executeOptimizationSync(coordination, components);
        break;
      case 'data-flow':
        await this.executeDataFlow(coordination, components);
        break;
    }
    
    coordination.lastExecution = new Date();
    coordination.effectiveness = Math.max(70, Math.min(100, coordination.effectiveness + (Math.random() - 0.5) * 5));
  }

  private async executeLoadBalancing(coordination: PerformanceCoordination, components: SystemComponent[]): Promise<void> {
    // Simulate load balancing between components
    const totalLoad = components.reduce((sum, c) => sum + c.metrics.cpuUsage, 0);
    const averageLoad = totalLoad / components.length;
    
    components.forEach(component => {
      if (component.metrics.cpuUsage > averageLoad * 1.2) {
        // Reduce load on overloaded components
        component.metrics.cpuUsage *= 0.95;
        component.metrics.responseTime *= 0.98;
      }
    });
  }

  private async executeResourceSharing(coordination: PerformanceCoordination, components: SystemComponent[]): Promise<void> {
    // Simulate resource sharing optimization
    const highMemoryComponents = components.filter(c => c.metrics.memoryUsage > 70);
    const lowMemoryComponents = components.filter(c => c.metrics.memoryUsage < 50);
    
    if (highMemoryComponents.length > 0 && lowMemoryComponents.length > 0) {
      highMemoryComponents.forEach(component => {
        component.metrics.memoryUsage *= 0.97;
      });
    }
  }

  private async executeOptimizationSync(coordination: PerformanceCoordination, components: SystemComponent[]): Promise<void> {
    // Simulate optimization synchronization
    components.forEach(component => {
      if (component.configuration.autoOptimize) {
        component.metrics.responseTime *= 0.99;
        component.metrics.errorRate *= 0.98;
      }
    });
  }

  private async executeDataFlow(coordination: PerformanceCoordination, components: SystemComponent[]): Promise<void> {
    // Simulate data flow optimization
    components.forEach(component => {
      component.metrics.throughput *= 1.01;
      component.metrics.availability = Math.min(100, component.metrics.availability + 0.1);
    });
  }

  private processOptimizationTasks(): void {
    this.optimizationTasks.forEach(task => {
      if (task.status === 'running') {
        task.progress = Math.min(100, task.progress + Math.random() * 10);
        
        if (task.progress >= 100) {
          task.status = 'completed';
          this.applyOptimizationTaskResults(task);
        }
      } else if (task.status === 'pending' && Math.random() < 0.1) {
        task.status = 'running';
        task.progress = 5;
      }
    });
  }

  private applyOptimizationTaskResults(task: OptimizationTask): void {
    task.targetComponents.forEach(componentId => {
      const component = this.components.get(componentId);
      if (component) {
        // Apply optimization improvements
        switch (task.type) {
          case 'memory':
            component.metrics.memoryUsage *= 0.9;
            break;
          case 'network':
            component.metrics.responseTime *= 0.85;
            break;
          case 'performance':
            component.performanceScore += task.impact;
            break;
          case 'coordination':
            component.metrics.throughput *= 1.1;
            break;
        }
      }
    });
  }

  private updateSystemHealth(): void {
    const components = Array.from(this.components.values());
    const coordinations = Array.from(this.coordinations.values());
    
    const componentHealth = components.reduce((sum, c) => sum + c.performanceScore, 0) / components.length;
    const coordinationEfficiency = coordinations.reduce((sum, c) => sum + c.effectiveness, 0) / coordinations.length;
    const resourceUtilization = 100 - (components.reduce((sum, c) => sum + c.metrics.cpuUsage, 0) / components.length);
    const userExperience = components
      .filter(c => c.type === 'ui-component')
      .reduce((sum, c) => sum + c.performanceScore, 0) / components.filter(c => c.type === 'ui-component').length;
    
    const optimizationImpact = this.optimizationTasks
      .filter(t => t.status === 'completed')
      .reduce((sum, t) => sum + t.impact, 0) / Math.max(1, this.optimizationTasks.filter(t => t.status === 'completed').length);
    
    const overallScore = (componentHealth + coordinationEfficiency + resourceUtilization + userExperience + optimizationImpact) / 5;
    
    const healthMetrics: SystemHealthMetrics = {
      overallScore,
      componentHealth,
      coordinationEfficiency,
      optimizationImpact,
      resourceUtilization,
      userExperience,
      timestamp: new Date()
    };
    
    this.healthHistory.push(healthMetrics);
    
    // Keep only last 48 hours of data
    if (this.healthHistory.length > 48) {
      this.healthHistory = this.healthHistory.slice(-48);
    }
  }

  private checkSystemAlerts(): void {
    for (const component of this.components.values()) {
      const config = component.configuration;
      const metrics = component.metrics;
      
      // Check thresholds
      Object.entries(config.thresholds).forEach(([metric, threshold]) => {
        const value = (metrics as any)[metric];
        if (value > threshold) {
          this.createAlert('warning', component.id, `${metric} (${value.toFixed(2)}) exceeds threshold (${threshold})`);
        }
      });
      
      // Check component status
      if (component.performanceScore < 70) {
        this.createAlert('error', component.id, `Performance score (${component.performanceScore.toFixed(1)}) is critically low`);
      }
    }
  }

  private createAlert(severity: PerformanceAlert['severity'], component: string, message: string): void {
    // Check if similar alert already exists
    const existingAlert = this.alerts.find(a => 
      !a.resolved && a.component === component && a.message === message
    );
    
    if (!existingAlert) {
      const alert: PerformanceAlert = {
        id: `alert-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        severity,
        component,
        message,
        timestamp: new Date(),
        resolved: false,
        autoResolvable: severity === 'warning' || severity === 'info'
      };
      
      this.alerts.push(alert);
      
      // Keep only last 100 alerts
      if (this.alerts.length > 100) {
        this.alerts = this.alerts.slice(-100);
      }
    }
  }

  getComponents(): SystemComponent[] {
    return Array.from(this.components.values());
  }

  getCoordinations(): PerformanceCoordination[] {
    return Array.from(this.coordinations.values());
  }

  getHealthHistory(): SystemHealthMetrics[] {
    return [...this.healthHistory];
  }

  getAlerts(): PerformanceAlert[] {
    return [...this.alerts];
  }

  getOptimizationTasks(): OptimizationTask[] {
    return [...this.optimizationTasks];
  }

  resolveAlert(alertId: string): void {
    const alert = this.alerts.find(a => a.id === alertId);
    if (alert) {
      alert.resolved = true;
    }
  }

  updateComponentConfiguration(componentId: string, config: Partial<ComponentConfiguration>): void {
    const component = this.components.get(componentId);
    if (component) {
      Object.assign(component.configuration, config);
    }
  }

  getSystemSummary(): {
    totalComponents: number;
    activeComponents: number;
    overallHealth: number;
    coordinationEfficiency: number;
    activeOptimizations: number;
    unresolvedAlerts: number;
  } {
    const components = Array.from(this.components.values());
    const latestHealth = this.healthHistory[this.healthHistory.length - 1];
    
    return {
      totalComponents: components.length,
      activeComponents: components.filter(c => c.status === 'active').length,
      overallHealth: latestHealth?.overallScore || 0,
      coordinationEfficiency: latestHealth?.coordinationEfficiency || 0,
      activeOptimizations: this.optimizationTasks.filter(t => t.status === 'running').length,
      unresolvedAlerts: this.alerts.filter(a => !a.resolved).length
    };
  }

  isCoordinationRunning(): boolean {
    return this.isCoordinating;
  }
}

// System coordination dashboard component
const SystemCoordinationDashboard: React.FC = () => {
  const [engine] = useState(() => new SystemCoordinationEngine());
  const [components, setComponents] = useState<SystemComponent[]>([]);
  const [coordinations, setCoordinations] = useState<PerformanceCoordination[]>([]);
  const [healthHistory, setHealthHistory] = useState<SystemHealthMetrics[]>([]);
  const [alerts, setAlerts] = useState<PerformanceAlert[]>([]);
  const [optimizationTasks, setOptimizationTasks] = useState<OptimizationTask[]>([]);
  const [isCoordinating, setIsCoordinating] = useState(false);
  const [selectedTab, setSelectedTab] = useState<'overview' | 'components' | 'coordinations' | 'tasks' | 'alerts'>('overview');
  const intervalRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    setComponents(engine.getComponents());
    setCoordinations(engine.getCoordinations());
    setHealthHistory(engine.getHealthHistory());
    setAlerts(engine.getAlerts());
    setOptimizationTasks(engine.getOptimizationTasks());
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      engine.stopCoordination();
    };
  }, [engine]);

  const startCoordination = useCallback(() => {
    engine.startCoordination();
    setIsCoordinating(true);
    
    intervalRef.current = setInterval(() => {
      setComponents(engine.getComponents());
      setCoordinations(engine.getCoordinations());
      setHealthHistory(engine.getHealthHistory());
      setAlerts(engine.getAlerts());
      setOptimizationTasks(engine.getOptimizationTasks());
    }, 3000);
  }, [engine]);

  const stopCoordination = useCallback(() => {
    engine.stopCoordination();
    setIsCoordinating(false);
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
  }, [engine]);

  const resolveAlert = useCallback((alertId: string) => {
    engine.resolveAlert(alertId);
    setAlerts(engine.getAlerts());
  }, [engine]);

  const toggleComponent = useCallback((componentId: string) => {
    const component = components.find(c => c.id === componentId);
    if (component) {
      engine.updateComponentConfiguration(componentId, { 
        enabled: !component.configuration.enabled 
      });
      setComponents(engine.getComponents());
    }
  }, [engine, components]);

  const summary = engine.getSystemSummary();

  const formatPercentage = (value: number): string => `${value.toFixed(1)}%`;
  const formatTime = (ms: number): string => `${ms.toFixed(1)}ms`;

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'active': return 'text-green-600 bg-green-100';
      case 'inactive': return 'text-gray-600 bg-gray-100';
      case 'error': return 'text-red-600 bg-red-100';
      case 'optimizing': return 'text-blue-600 bg-blue-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'monitoring': return <Monitor className="w-4 h-4" />;
      case 'optimization': return <Zap className="w-4 h-4" />;
      case 'analytics': return <BarChart3 className="w-4 h-4" />;
      case 'ui-component': return <Layers className="w-4 h-4" />;
      case 'integration': return <Globe className="w-4 h-4" />;
      default: return <Settings className="w-4 h-4" />;
    }
  };

  const getAlertIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return <AlertTriangle className="w-4 h-4 text-red-500" />;
      case 'error': return <AlertTriangle className="w-4 h-4 text-orange-500" />;
      case 'warning': return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
      case 'info': return <CheckCircle className="w-4 h-4 text-blue-500" />;
      default: return <Activity className="w-4 h-4 text-gray-500" />;
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
              <Network className="w-8 h-8 text-indigo-600" />
              System Coordination & Performance Integration
            </h1>
            <p className="text-gray-600 mt-2">
              Comprehensive system coordination with intelligent performance integration across all components
            </p>
          </div>
          <div className="flex gap-3">
            <button
              onClick={isCoordinating ? stopCoordination : startCoordination}
              className={`px-6 py-2 rounded-lg font-medium flex items-center gap-2 ${
                isCoordinating 
                  ? 'bg-red-600 hover:bg-red-700 text-white' 
                  : 'bg-indigo-600 hover:bg-indigo-700 text-white'
              }`}
            >
              {isCoordinating ? (
                <>
                  <Activity className="w-4 h-4 animate-pulse" />
                  Stop Coordination
                </>
              ) : (
                <>
                  <Network className="w-4 h-4" />
                  Start Coordination
                </>
              )}
            </button>
          </div>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">System Health</p>
                <p className="text-2xl font-bold text-indigo-600">{formatPercentage(summary.overallHealth)}</p>
              </div>
              <TrendingUp className="w-8 h-8 text-indigo-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Active Components</p>
                <p className="text-2xl font-bold text-green-600">{summary.activeComponents}/{summary.totalComponents}</p>
              </div>
              <Layers className="w-8 h-8 text-green-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Coordination Efficiency</p>
                <p className="text-2xl font-bold text-blue-600">{formatPercentage(summary.coordinationEfficiency)}</p>
              </div>
              <Network className="w-8 h-8 text-blue-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Active Optimizations</p>
                <p className="text-2xl font-bold text-purple-600">{summary.activeOptimizations}</p>
              </div>
              <Zap className="w-8 h-8 text-purple-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Unresolved Alerts</p>
                <p className={`text-2xl font-bold ${summary.unresolvedAlerts > 0 ? 'text-red-600' : 'text-green-600'}`}>
                  {summary.unresolvedAlerts}
                </p>
              </div>
              <AlertTriangle className="w-8 h-8 text-red-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Coordination Status</p>
                <p className={`text-sm font-bold ${isCoordinating ? 'text-green-600' : 'text-gray-600'}`}>
                  {isCoordinating ? 'ACTIVE' : 'INACTIVE'}
                </p>
              </div>
              <Activity className={`w-8 h-8 ${isCoordinating ? 'text-green-500 animate-pulse' : 'text-gray-500'}`} />
            </div>
          </div>
        </div>

        {/* Status Banner */}
        <div className={`rounded-lg p-4 mb-6 ${
          isCoordinating ? 'bg-indigo-50 border border-indigo-200' : 'bg-gray-50 border border-gray-200'
        }`}>
          <div className="flex items-center gap-2">
            {isCoordinating ? (
              <>
                <Network className="w-5 h-5 text-indigo-600 animate-pulse" />
                <span className="font-medium text-indigo-800">System coordination is active</span>
                <span className="text-indigo-600">- Continuously optimizing performance across all components</span>
              </>
            ) : (
              <>
                <AlertTriangle className="w-5 h-5 text-gray-600" />
                <span className="font-medium text-gray-800">System coordination is inactive</span>
                <span className="text-gray-600">- Click "Start Coordination" to begin intelligent system optimization</span>
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
            { id: 'components', label: 'Components', icon: Layers },
            { id: 'coordinations', label: 'Coordinations', icon: Network },
            { id: 'tasks', label: 'Optimization Tasks', icon: Zap },
            { id: 'alerts', label: 'Alerts', icon: AlertTriangle }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setSelectedTab(tab.id as any)}
              className={`flex items-center gap-2 py-2 px-1 border-b-2 font-medium text-sm ${
                selectedTab === tab.id
                  ? 'border-indigo-500 text-indigo-600'
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
                <TrendingUp className="w-5 h-5 text-indigo-500" />
                System Health Trends
              </h3>
              <div className="space-y-4">
                {healthHistory.length > 0 && (
                  <>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600">Overall Score</span>
                      <span className="font-medium">{formatPercentage(healthHistory[healthHistory.length - 1].overallScore)}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600">Component Health</span>
                      <span className="font-medium">{formatPercentage(healthHistory[healthHistory.length - 1].componentHealth)}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600">Coordination Efficiency</span>
                      <span className="font-medium">{formatPercentage(healthHistory[healthHistory.length - 1].coordinationEfficiency)}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600">User Experience</span>
                      <span className="font-medium">{formatPercentage(healthHistory[healthHistory.length - 1].userExperience)}</span>
                    </div>
                  </>
                )}
              </div>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-green-500" />
                System Status
              </h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Coordination Status</span>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    isCoordinating ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                  }`}>
                    {isCoordinating ? 'Active' : 'Inactive'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Active Components</span>
                  <span className="font-medium">{summary.activeComponents} / {summary.totalComponents}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Running Optimizations</span>
                  <span className="font-medium">{summary.activeOptimizations}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Pending Alerts</span>
                  <span className={`font-medium ${summary.unresolvedAlerts > 0 ? 'text-red-600' : 'text-green-600'}`}>
                    {summary.unresolvedAlerts}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Component Status Grid */}
          <div className="bg-white rounded-lg shadow-md">
            <div className="p-6 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900">Component Status Overview</h3>
              <p className="text-sm text-gray-600 mt-1">Real-time status and performance of all system components</p>
            </div>
            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {components.map(component => (
                  <div key={component.id} className="p-4 border border-gray-200 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        {getTypeIcon(component.type)}
                        <h4 className="font-medium text-gray-900 text-sm">{component.name}</h4>
                      </div>
                      <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(component.status)}`}>
                        {component.status}
                      </span>
                    </div>
                    <div className="space-y-1 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Performance:</span>
                        <span className="font-medium">{formatPercentage(component.performanceScore)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">CPU:</span>
                        <span className="font-medium">{formatPercentage(component.metrics.cpuUsage)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Memory:</span>
                        <span className="font-medium">{formatPercentage(component.metrics.memoryUsage)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Response:</span>
                        <span className="font-medium">{formatTime(component.metrics.responseTime)}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {selectedTab === 'components' && (
        <div className="bg-white rounded-lg shadow-md">
          <div className="p-6 border-b border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900">System Components</h3>
            <p className="text-sm text-gray-600 mt-1">
              Detailed view and configuration of all system components
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
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Performance
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Metrics
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Configuration
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {components.map(component => (
                  <tr key={component.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center gap-3">
                        {getTypeIcon(component.type)}
                        <div>
                          <div className="text-sm font-medium text-gray-900">{component.name}</div>
                          <div className="text-sm text-gray-500 capitalize">{component.type}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(component.status)}`}>
                        {component.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">{formatPercentage(component.performanceScore)}</div>
                      <div className="text-sm text-gray-500">Last update: {component.lastUpdate.toLocaleTimeString()}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      <div>CPU: {formatPercentage(component.metrics.cpuUsage)}</div>
                      <div>Memory: {formatPercentage(component.metrics.memoryUsage)}</div>
                      <div>Response: {formatTime(component.metrics.responseTime)}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      <div>Priority: {component.configuration.priority}</div>
                      <div>Auto-optimize: {component.configuration.autoOptimize ? 'Yes' : 'No'}</div>
                      <div>Dependencies: {component.dependencies.length}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <button
                        onClick={() => toggleComponent(component.id)}
                        className={`px-3 py-1 rounded text-xs ${
                          component.configuration.enabled
                            ? 'bg-red-100 text-red-800 hover:bg-red-200'
                            : 'bg-green-100 text-green-800 hover:bg-green-200'
                        }`}
                      >
                        {component.configuration.enabled ? 'Disable' : 'Enable'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {selectedTab === 'coordinations' && (
        <div className="space-y-6">
          {coordinations.map(coordination => (
            <div key={coordination.id} className="bg-white rounded-lg shadow-md p-6">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h4 className="text-lg font-semibold text-gray-900">{coordination.name}</h4>
                  <p className="text-sm text-gray-600 mt-1">{coordination.description}</p>
                </div>
                <div className="flex items-center gap-3">
                  <span className={`px-3 py-1 text-sm font-medium rounded-full ${
                    coordination.status === 'active' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                  }`}>
                    {coordination.status}
                  </span>
                  <span className="text-sm font-medium text-gray-900">
                    {formatPercentage(coordination.effectiveness)} effective
                  </span>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <p className="text-sm font-medium text-gray-700 mb-2">Type</p>
                  <p className="text-sm text-gray-600 capitalize">{coordination.coordinationType.replace('-', ' ')}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-700 mb-2">Components</p>
                  <div className="flex flex-wrap gap-1">
                    {coordination.components.map(componentId => {
                      const component = components.find(c => c.id === componentId);
                      return (
                        <span key={componentId} className="px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded">
                          {component?.name.split(' ')[0] || componentId}
                        </span>
                      );
                    })}
                  </div>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-700 mb-2">Last Execution</p>
                  <p className="text-sm text-gray-600">{coordination.lastExecution.toLocaleString()}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {selectedTab === 'tasks' && (
        <div className="bg-white rounded-lg shadow-md">
          <div className="p-6 border-b border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900">Optimization Tasks</h3>
            <p className="text-sm text-gray-600 mt-1">
              Active and completed performance optimization tasks
            </p>
          </div>
          <div className="divide-y divide-gray-200">
            {optimizationTasks.map(task => (
              <div key={task.id} className="p-6">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h4 className="font-medium text-gray-900">{task.name}</h4>
                    <p className="text-sm text-gray-600">
                      Priority: {task.priority}/10 • Expected impact: {formatPercentage(task.impact)}
                    </p>
                  </div>
                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                    task.status === 'completed' ? 'bg-green-100 text-green-800' :
                    task.status === 'running' ? 'bg-blue-100 text-blue-800' :
                    task.status === 'failed' ? 'bg-red-100 text-red-800' :
                    'bg-gray-100 text-gray-800'
                  }`}>
                    {task.status}
                  </span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-3">
                  <div>
                    <p className="text-sm text-gray-500">Type</p>
                    <p className="font-medium capitalize">{task.type}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Progress</p>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${task.progress}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium">{task.progress}%</span>
                    </div>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Estimated Completion</p>
                    <p className="font-medium">{task.estimatedCompletion.toLocaleString()}</p>
                  </div>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-700 mb-1">Target Components:</p>
                  <div className="flex flex-wrap gap-2">
                    {task.targetComponents.map(componentId => {
                      const component = components.find(c => c.id === componentId);
                      return (
                        <span key={componentId} className="px-2 py-1 text-xs bg-purple-100 text-purple-800 rounded">
                          {component?.name.split(' ')[0] || componentId}
                        </span>
                      );
                    })}
                  </div>
                </div>
              </div>
            ))}
            {optimizationTasks.length === 0 && (
              <div className="text-center py-12">
                <Zap className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500">No optimization tasks</p>
                <p className="text-sm text-gray-400">Tasks will appear here when optimizations are scheduled</p>
              </div>
            )}
          </div>
        </div>
      )}

      {selectedTab === 'alerts' && (
        <div className="bg-white rounded-lg shadow-md">
          <div className="p-6 border-b border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900">System Alerts</h3>
            <p className="text-sm text-gray-600 mt-1">
              Performance alerts and system notifications
            </p>
          </div>
          <div className="divide-y divide-gray-200">
            {alerts.length > 0 ? (
              alerts.slice().reverse().map(alert => (
                <div key={alert.id} className={`p-6 ${alert.resolved ? 'bg-gray-50' : ''}`}>
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3">
                      {getAlertIcon(alert.severity)}
                      <div>
                        <p className={`font-medium ${alert.resolved ? 'text-gray-600' : 'text-gray-900'}`}>
                          {alert.message}
                        </p>
                        <div className="flex items-center gap-4 mt-1 text-sm text-gray-500">
                          <span>{alert.timestamp.toLocaleString()}</span>
                          <span>Component: {components.find(c => c.id === alert.component)?.name || alert.component}</span>
                          {alert.autoResolvable && (
                            <span className="px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded">
                              Auto-resolvable
                            </span>
                          )}
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
                <p className="text-sm text-gray-400">All systems are operating normally</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default SystemCoordinationDashboard;

