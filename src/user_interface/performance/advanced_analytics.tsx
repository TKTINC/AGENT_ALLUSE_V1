import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Brain, TrendingUp, Target, Zap, BarChart3, LineChart, PieChart, Activity, AlertTriangle, CheckCircle, Clock, Cpu } from 'lucide-react';
import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, PieChart as RechartsPieChart, Cell, BarChart as RechartsBarChart, Bar } from 'recharts';

// Advanced analytics interfaces and types
interface PerformanceDataPoint {
  timestamp: Date;
  renderTime: number;
  memoryUsage: number;
  networkLatency: number;
  bundleSize: number;
  userInteractions: number;
  errorRate: number;
  cacheHitRate: number;
  cpuUsage: number;
}

interface PredictiveModel {
  id: string;
  name: string;
  type: 'linear' | 'polynomial' | 'neural' | 'ensemble';
  accuracy: number;
  lastTrained: Date;
  predictions: PredictionResult[];
  features: string[];
  hyperparameters: Record<string, any>;
}

interface PredictionResult {
  metric: string;
  currentValue: number;
  predictedValue: number;
  confidence: number;
  timeHorizon: number; // hours
  trend: 'improving' | 'degrading' | 'stable';
  recommendations: string[];
}

interface PerformanceAnomaly {
  id: string;
  timestamp: Date;
  metric: string;
  value: number;
  expectedValue: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  possibleCauses: string[];
  recommendations: string[];
  resolved: boolean;
}

interface UserBehaviorPattern {
  id: string;
  pattern: string;
  frequency: number;
  performanceImpact: number;
  userSegment: string;
  timeOfDay: string;
  components: string[];
  optimizationOpportunity: number;
}

interface PerformanceInsight {
  id: string;
  type: 'trend' | 'anomaly' | 'optimization' | 'prediction';
  title: string;
  description: string;
  impact: 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
  actionable: boolean;
  recommendations: string[];
  metrics: Record<string, number>;
  timestamp: Date;
}

interface ABTestResult {
  id: string;
  name: string;
  variant: 'A' | 'B';
  metric: string;
  improvement: number;
  significance: number;
  sampleSize: number;
  duration: number;
  status: 'running' | 'completed' | 'paused';
}

// Advanced analytics and predictive optimization engine
class AdvancedAnalyticsEngine {
  private dataPoints: PerformanceDataPoint[] = [];
  private models: Map<string, PredictiveModel> = new Map();
  private anomalies: PerformanceAnomaly[] = [];
  private userPatterns: UserBehaviorPattern[] = [];
  private insights: PerformanceInsight[] = [];
  private abTests: ABTestResult[] = [];
  private isAnalyzing: boolean = false;

  constructor() {
    this.initializePredictiveModels();
    this.generateSampleData();
  }

  private initializePredictiveModels(): void {
    const models: PredictiveModel[] = [
      {
        id: 'render-time-predictor',
        name: 'Render Time Predictor',
        type: 'neural',
        accuracy: 0.89,
        lastTrained: new Date(Date.now() - 24 * 60 * 60 * 1000),
        predictions: [],
        features: ['userInteractions', 'memoryUsage', 'componentCount', 'timeOfDay'],
        hyperparameters: { learningRate: 0.001, epochs: 100, hiddenLayers: [64, 32] }
      },
      {
        id: 'memory-usage-predictor',
        name: 'Memory Usage Predictor',
        type: 'ensemble',
        accuracy: 0.92,
        lastTrained: new Date(Date.now() - 12 * 60 * 60 * 1000),
        predictions: [],
        features: ['renderTime', 'userInteractions', 'cacheHitRate', 'componentComplexity'],
        hyperparameters: { estimators: 100, maxDepth: 10, minSamplesSplit: 5 }
      },
      {
        id: 'network-latency-predictor',
        name: 'Network Latency Predictor',
        type: 'linear',
        accuracy: 0.76,
        lastTrained: new Date(Date.now() - 6 * 60 * 60 * 1000),
        predictions: [],
        features: ['requestSize', 'serverLoad', 'timeOfDay', 'userLocation'],
        hyperparameters: { regularization: 0.01, polynomial: 2 }
      },
      {
        id: 'user-experience-predictor',
        name: 'User Experience Predictor',
        type: 'polynomial',
        accuracy: 0.84,
        lastTrained: new Date(Date.now() - 18 * 60 * 60 * 1000),
        predictions: [],
        features: ['renderTime', 'networkLatency', 'errorRate', 'interactionDelay'],
        hyperparameters: { degree: 3, regularization: 0.05 }
      }
    ];

    models.forEach(model => this.models.set(model.id, model));
  }

  private generateSampleData(): void {
    const now = new Date();
    const hoursBack = 168; // 7 days

    for (let i = hoursBack; i >= 0; i--) {
      const timestamp = new Date(now.getTime() - i * 60 * 60 * 1000);
      const baseRenderTime = 80 + Math.sin(i / 24) * 20; // Daily pattern
      const baseMemory = 45 + Math.cos(i / 12) * 15; // Semi-daily pattern
      
      this.dataPoints.push({
        timestamp,
        renderTime: Math.max(20, baseRenderTime + (Math.random() - 0.5) * 30),
        memoryUsage: Math.max(10, baseMemory + (Math.random() - 0.5) * 20),
        networkLatency: 150 + (Math.random() - 0.5) * 100,
        bundleSize: 1.8 + (Math.random() - 0.5) * 0.4,
        userInteractions: Math.floor(Math.random() * 100),
        errorRate: Math.random() * 5,
        cacheHitRate: 70 + Math.random() * 25,
        cpuUsage: 30 + Math.random() * 40
      });
    }

    this.generateSampleAnomalies();
    this.generateSampleUserPatterns();
    this.generateSampleInsights();
    this.generateSampleABTests();
  }

  private generateSampleAnomalies(): void {
    const anomalies: PerformanceAnomaly[] = [
      {
        id: 'anomaly-1',
        timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
        metric: 'renderTime',
        value: 250,
        expectedValue: 85,
        severity: 'high',
        description: 'Render time spike detected in dashboard components',
        possibleCauses: ['Memory leak in chart rendering', 'Inefficient data processing', 'Browser resource contention'],
        recommendations: ['Implement component memoization', 'Optimize data structures', 'Add performance monitoring'],
        resolved: false
      },
      {
        id: 'anomaly-2',
        timestamp: new Date(Date.now() - 6 * 60 * 60 * 1000),
        metric: 'memoryUsage',
        value: 120,
        expectedValue: 50,
        severity: 'critical',
        description: 'Memory usage exceeded threshold in trading interface',
        possibleCauses: ['Event listener leaks', 'Uncleaned intervals', 'Large object retention'],
        recommendations: ['Audit event listeners', 'Implement cleanup patterns', 'Use WeakMap for caching'],
        resolved: true
      },
      {
        id: 'anomaly-3',
        timestamp: new Date(Date.now() - 12 * 60 * 60 * 1000),
        metric: 'networkLatency',
        value: 800,
        expectedValue: 180,
        severity: 'medium',
        description: 'Network latency increase during peak hours',
        possibleCauses: ['Server overload', 'CDN issues', 'Database query optimization needed'],
        recommendations: ['Implement request caching', 'Optimize API endpoints', 'Add request queuing'],
        resolved: true
      }
    ];

    this.anomalies = anomalies;
  }

  private generateSampleUserPatterns(): void {
    const patterns: UserBehaviorPattern[] = [
      {
        id: 'pattern-1',
        pattern: 'Heavy dashboard usage during market hours',
        frequency: 85,
        performanceImpact: 35,
        userSegment: 'Active Traders',
        timeOfDay: '09:00-16:00',
        components: ['TradingDashboard', 'MarketAnalysis', 'RealTimeCharts'],
        optimizationOpportunity: 70
      },
      {
        id: 'pattern-2',
        pattern: 'Portfolio analysis on mobile devices',
        frequency: 60,
        performanceImpact: 45,
        userSegment: 'Mobile Users',
        timeOfDay: '18:00-22:00',
        components: ['AccountVisualization', 'PortfolioCharts', 'TransactionHistory'],
        optimizationOpportunity: 80
      },
      {
        id: 'pattern-3',
        pattern: 'Batch report generation',
        frequency: 25,
        performanceImpact: 60,
        userSegment: 'Enterprise Users',
        timeOfDay: '06:00-08:00',
        components: ['ReportGenerator', 'DataExport', 'AnalyticsEngine'],
        optimizationOpportunity: 90
      }
    ];

    this.userPatterns = patterns;
  }

  private generateSampleInsights(): void {
    const insights: PerformanceInsight[] = [
      {
        id: 'insight-1',
        type: 'trend',
        title: 'Render Performance Improving',
        description: 'Component render times have improved by 15% over the past week due to recent optimizations',
        impact: 'medium',
        confidence: 0.92,
        actionable: false,
        recommendations: ['Continue monitoring', 'Document successful optimizations'],
        metrics: { improvement: 15, timeframe: 7 },
        timestamp: new Date(Date.now() - 1 * 60 * 60 * 1000)
      },
      {
        id: 'insight-2',
        type: 'optimization',
        title: 'Bundle Splitting Opportunity',
        description: 'Large vendor bundle is impacting initial load time. Code splitting could reduce load time by 40%',
        impact: 'high',
        confidence: 0.87,
        actionable: true,
        recommendations: ['Implement route-based code splitting', 'Lazy load non-critical components', 'Optimize vendor bundle'],
        metrics: { potentialImprovement: 40, effort: 3 },
        timestamp: new Date(Date.now() - 3 * 60 * 60 * 1000)
      },
      {
        id: 'insight-3',
        type: 'prediction',
        title: 'Memory Usage Trend Alert',
        description: 'Predictive model indicates memory usage will exceed 100MB within 24 hours if current trend continues',
        impact: 'high',
        confidence: 0.78,
        actionable: true,
        recommendations: ['Implement aggressive garbage collection', 'Review component lifecycle', 'Add memory monitoring'],
        metrics: { predictedValue: 105, timeframe: 24 },
        timestamp: new Date(Date.now() - 30 * 60 * 1000)
      }
    ];

    this.insights = insights;
  }

  private generateSampleABTests(): void {
    const tests: ABTestResult[] = [
      {
        id: 'test-1',
        name: 'Lazy Loading vs Eager Loading',
        variant: 'A',
        metric: 'initialLoadTime',
        improvement: 23.5,
        significance: 0.95,
        sampleSize: 1000,
        duration: 7,
        status: 'completed'
      },
      {
        id: 'test-2',
        name: 'Memoization Strategy',
        variant: 'B',
        metric: 'renderTime',
        improvement: 18.2,
        significance: 0.89,
        sampleSize: 750,
        duration: 5,
        status: 'running'
      },
      {
        id: 'test-3',
        name: 'Caching Strategy',
        variant: 'A',
        metric: 'networkLatency',
        improvement: 35.7,
        significance: 0.98,
        sampleSize: 1200,
        duration: 10,
        status: 'completed'
      }
    ];

    this.abTests = tests;
  }

  startAnalysis(): void {
    this.isAnalyzing = true;
    this.runAnalysisCycle();
  }

  stopAnalysis(): void {
    this.isAnalyzing = false;
  }

  private async runAnalysisCycle(): Promise<void> {
    if (!this.isAnalyzing) return;

    try {
      // Generate new data point
      this.addNewDataPoint();
      
      // Run predictive models
      await this.runPredictiveModels();
      
      // Detect anomalies
      this.detectAnomalies();
      
      // Analyze user patterns
      this.analyzeUserPatterns();
      
      // Generate insights
      this.generateInsights();
      
      // Schedule next cycle
      setTimeout(() => this.runAnalysisCycle(), 30000); // Every 30 seconds
    } catch (error) {
      console.error('Analysis cycle error:', error);
      setTimeout(() => this.runAnalysisCycle(), 60000); // Retry in 1 minute
    }
  }

  private addNewDataPoint(): void {
    const latest = this.dataPoints[this.dataPoints.length - 1];
    const now = new Date();
    
    // Simulate realistic data evolution
    const newPoint: PerformanceDataPoint = {
      timestamp: now,
      renderTime: Math.max(20, latest.renderTime + (Math.random() - 0.5) * 10),
      memoryUsage: Math.max(10, latest.memoryUsage + (Math.random() - 0.5) * 5),
      networkLatency: Math.max(50, latest.networkLatency + (Math.random() - 0.5) * 20),
      bundleSize: Math.max(1.0, latest.bundleSize + (Math.random() - 0.5) * 0.1),
      userInteractions: Math.floor(Math.random() * 100),
      errorRate: Math.max(0, latest.errorRate + (Math.random() - 0.5) * 1),
      cacheHitRate: Math.min(100, Math.max(0, latest.cacheHitRate + (Math.random() - 0.5) * 5)),
      cpuUsage: Math.min(100, Math.max(0, latest.cpuUsage + (Math.random() - 0.5) * 10))
    };

    this.dataPoints.push(newPoint);
    
    // Keep only last 1000 data points
    if (this.dataPoints.length > 1000) {
      this.dataPoints = this.dataPoints.slice(-1000);
    }
  }

  private async runPredictiveModels(): Promise<void> {
    for (const model of this.models.values()) {
      const predictions = await this.generatePredictions(model);
      model.predictions = predictions;
    }
  }

  private async generatePredictions(model: PredictiveModel): Promise<PredictionResult[]> {
    const recent = this.dataPoints.slice(-24); // Last 24 hours
    const predictions: PredictionResult[] = [];

    // Simulate model predictions based on type
    switch (model.type) {
      case 'neural':
        predictions.push(...this.generateNeuralPredictions(model, recent));
        break;
      case 'ensemble':
        predictions.push(...this.generateEnsemblePredictions(model, recent));
        break;
      case 'linear':
        predictions.push(...this.generateLinearPredictions(model, recent));
        break;
      case 'polynomial':
        predictions.push(...this.generatePolynomialPredictions(model, recent));
        break;
    }

    return predictions;
  }

  private generateNeuralPredictions(model: PredictiveModel, data: PerformanceDataPoint[]): PredictionResult[] {
    const latest = data[data.length - 1];
    
    return [{
      metric: 'renderTime',
      currentValue: latest.renderTime,
      predictedValue: latest.renderTime * (0.95 + Math.random() * 0.1),
      confidence: model.accuracy,
      timeHorizon: 6,
      trend: Math.random() > 0.5 ? 'improving' : 'stable',
      recommendations: ['Monitor component complexity', 'Consider memoization']
    }];
  }

  private generateEnsemblePredictions(model: PredictiveModel, data: PerformanceDataPoint[]): PredictionResult[] {
    const latest = data[data.length - 1];
    
    return [{
      metric: 'memoryUsage',
      currentValue: latest.memoryUsage,
      predictedValue: latest.memoryUsage * (1.02 + Math.random() * 0.06),
      confidence: model.accuracy,
      timeHorizon: 12,
      trend: 'degrading',
      recommendations: ['Implement garbage collection', 'Review memory leaks']
    }];
  }

  private generateLinearPredictions(model: PredictiveModel, data: PerformanceDataPoint[]): PredictionResult[] {
    const latest = data[data.length - 1];
    
    return [{
      metric: 'networkLatency',
      currentValue: latest.networkLatency,
      predictedValue: latest.networkLatency * (0.98 + Math.random() * 0.04),
      confidence: model.accuracy,
      timeHorizon: 3,
      trend: 'improving',
      recommendations: ['Optimize API calls', 'Implement caching']
    }];
  }

  private generatePolynomialPredictions(model: PredictiveModel, data: PerformanceDataPoint[]): PredictionResult[] {
    const latest = data[data.length - 1];
    
    return [{
      metric: 'userExperience',
      currentValue: 85,
      predictedValue: 87 + Math.random() * 5,
      confidence: model.accuracy,
      timeHorizon: 24,
      trend: 'improving',
      recommendations: ['Continue current optimizations', 'Monitor user feedback']
    }];
  }

  private detectAnomalies(): void {
    const latest = this.dataPoints[this.dataPoints.length - 1];
    const recent = this.dataPoints.slice(-24);
    
    // Calculate moving averages and standard deviations
    const avgRenderTime = recent.reduce((sum, p) => sum + p.renderTime, 0) / recent.length;
    const stdRenderTime = Math.sqrt(recent.reduce((sum, p) => sum + Math.pow(p.renderTime - avgRenderTime, 2), 0) / recent.length);
    
    // Detect anomalies (values beyond 2 standard deviations)
    if (Math.abs(latest.renderTime - avgRenderTime) > 2 * stdRenderTime) {
      const anomaly: PerformanceAnomaly = {
        id: `anomaly-${Date.now()}`,
        timestamp: latest.timestamp,
        metric: 'renderTime',
        value: latest.renderTime,
        expectedValue: avgRenderTime,
        severity: latest.renderTime > avgRenderTime + 2 * stdRenderTime ? 'high' : 'medium',
        description: `Render time anomaly detected: ${latest.renderTime.toFixed(2)}ms vs expected ${avgRenderTime.toFixed(2)}ms`,
        possibleCauses: ['Component complexity increase', 'Memory pressure', 'Browser resource contention'],
        recommendations: ['Review recent changes', 'Check memory usage', 'Optimize rendering'],
        resolved: false
      };
      
      this.anomalies.push(anomaly);
      
      // Keep only last 50 anomalies
      if (this.anomalies.length > 50) {
        this.anomalies = this.anomalies.slice(-50);
      }
    }
  }

  private analyzeUserPatterns(): void {
    // Simulate user pattern analysis
    const hour = new Date().getHours();
    
    this.userPatterns.forEach(pattern => {
      // Update pattern frequency based on time of day
      if (pattern.timeOfDay.includes(hour.toString().padStart(2, '0'))) {
        pattern.frequency = Math.min(100, pattern.frequency + Math.random() * 5);
      } else {
        pattern.frequency = Math.max(0, pattern.frequency - Math.random() * 2);
      }
    });
  }

  private generateInsights(): void {
    const recent = this.dataPoints.slice(-24);
    const older = this.dataPoints.slice(-48, -24);
    
    if (recent.length === 0 || older.length === 0) return;
    
    // Calculate trends
    const recentAvgRender = recent.reduce((sum, p) => sum + p.renderTime, 0) / recent.length;
    const olderAvgRender = older.reduce((sum, p) => sum + p.renderTime, 0) / older.length;
    const renderTrend = ((recentAvgRender - olderAvgRender) / olderAvgRender) * 100;
    
    if (Math.abs(renderTrend) > 5) {
      const insight: PerformanceInsight = {
        id: `insight-${Date.now()}`,
        type: 'trend',
        title: renderTrend > 0 ? 'Render Performance Degrading' : 'Render Performance Improving',
        description: `Render time has ${renderTrend > 0 ? 'increased' : 'decreased'} by ${Math.abs(renderTrend).toFixed(1)}% in the last 24 hours`,
        impact: Math.abs(renderTrend) > 15 ? 'high' : 'medium',
        confidence: 0.85,
        actionable: renderTrend > 0,
        recommendations: renderTrend > 0 
          ? ['Investigate recent changes', 'Review component performance', 'Check for memory leaks']
          : ['Document successful optimizations', 'Continue monitoring'],
        metrics: { trend: renderTrend, timeframe: 24 },
        timestamp: new Date()
      };
      
      this.insights.push(insight);
      
      // Keep only last 20 insights
      if (this.insights.length > 20) {
        this.insights = this.insights.slice(-20);
      }
    }
  }

  getDataPoints(): PerformanceDataPoint[] {
    return [...this.dataPoints];
  }

  getModels(): PredictiveModel[] {
    return Array.from(this.models.values());
  }

  getAnomalies(): PerformanceAnomaly[] {
    return [...this.anomalies];
  }

  getUserPatterns(): UserBehaviorPattern[] {
    return [...this.userPatterns];
  }

  getInsights(): PerformanceInsight[] {
    return [...this.insights];
  }

  getABTests(): ABTestResult[] {
    return [...this.abTests];
  }

  resolveAnomaly(anomalyId: string): void {
    const anomaly = this.anomalies.find(a => a.id === anomalyId);
    if (anomaly) {
      anomaly.resolved = true;
    }
  }

  getAnalyticsSummary(): {
    totalDataPoints: number;
    activeModels: number;
    unresolvedAnomalies: number;
    actionableInsights: number;
    averageModelAccuracy: number;
    trendsDetected: number;
  } {
    const unresolvedAnomalies = this.anomalies.filter(a => !a.resolved).length;
    const actionableInsights = this.insights.filter(i => i.actionable).length;
    const averageModelAccuracy = Array.from(this.models.values())
      .reduce((sum, m) => sum + m.accuracy, 0) / this.models.size;
    const trendsDetected = this.insights.filter(i => i.type === 'trend').length;

    return {
      totalDataPoints: this.dataPoints.length,
      activeModels: this.models.size,
      unresolvedAnomalies,
      actionableInsights,
      averageModelAccuracy,
      trendsDetected
    };
  }

  isAnalysisRunning(): boolean {
    return this.isAnalyzing;
  }
}

// Advanced analytics dashboard component
const AdvancedAnalyticsDashboard: React.FC = () => {
  const [engine] = useState(() => new AdvancedAnalyticsEngine());
  const [dataPoints, setDataPoints] = useState<PerformanceDataPoint[]>([]);
  const [models, setModels] = useState<PredictiveModel[]>([]);
  const [anomalies, setAnomalies] = useState<PerformanceAnomaly[]>([]);
  const [userPatterns, setUserPatterns] = useState<UserBehaviorPattern[]>([]);
  const [insights, setInsights] = useState<PerformanceInsight[]>([]);
  const [abTests, setABTests] = useState<ABTestResult[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedTab, setSelectedTab] = useState<'overview' | 'predictions' | 'anomalies' | 'patterns' | 'insights' | 'abtests'>('overview');

  useEffect(() => {
    setDataPoints(engine.getDataPoints());
    setModels(engine.getModels());
    setAnomalies(engine.getAnomalies());
    setUserPatterns(engine.getUserPatterns());
    setInsights(engine.getInsights());
    setABTests(engine.getABTests());
  }, [engine]);

  const startAnalysis = useCallback(() => {
    engine.startAnalysis();
    setIsAnalyzing(true);
    
    const interval = setInterval(() => {
      setDataPoints(engine.getDataPoints());
      setModels(engine.getModels());
      setAnomalies(engine.getAnomalies());
      setUserPatterns(engine.getUserPatterns());
      setInsights(engine.getInsights());
      setABTests(engine.getABTests());
    }, 5000);

    return () => clearInterval(interval);
  }, [engine]);

  const stopAnalysis = useCallback(() => {
    engine.stopAnalysis();
    setIsAnalyzing(false);
  }, [engine]);

  const resolveAnomaly = useCallback((anomalyId: string) => {
    engine.resolveAnomaly(anomalyId);
    setAnomalies(engine.getAnomalies());
  }, [engine]);

  const summary = engine.getAnalyticsSummary();

  // Prepare chart data
  const chartData = useMemo(() => {
    return dataPoints.slice(-24).map(point => ({
      time: point.timestamp.toLocaleTimeString(),
      renderTime: point.renderTime,
      memoryUsage: point.memoryUsage,
      networkLatency: point.networkLatency,
      cpuUsage: point.cpuUsage
    }));
  }, [dataPoints]);

  const anomalyData = useMemo(() => {
    const severityCounts = anomalies.reduce((acc, anomaly) => {
      acc[anomaly.severity] = (acc[anomaly.severity] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return Object.entries(severityCounts).map(([severity, count]) => ({
      name: severity,
      value: count,
      color: severity === 'critical' ? '#ef4444' : 
             severity === 'high' ? '#f97316' :
             severity === 'medium' ? '#eab308' : '#22c55e'
    }));
  }, [anomalies]);

  const formatPercentage = (value: number): string => `${value.toFixed(1)}%`;
  const formatTime = (ms: number): string => `${ms.toFixed(1)}ms`;

  const getImpactColor = (impact: string): string => {
    switch (impact) {
      case 'critical': return 'text-red-600 bg-red-100';
      case 'high': return 'text-orange-600 bg-orange-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'low': return 'text-green-600 bg-green-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'improving': return <TrendingUp className="w-4 h-4 text-green-500" />;
      case 'degrading': return <TrendingUp className="w-4 h-4 text-red-500 transform rotate-180" />;
      case 'stable': return <Target className="w-4 h-4 text-blue-500" />;
      default: return <Activity className="w-4 h-4 text-gray-500" />;
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
              <Brain className="w-8 h-8 text-purple-600" />
              Advanced Analytics & Predictive Optimization
            </h1>
            <p className="text-gray-600 mt-2">
              AI-powered performance analytics with predictive insights and optimization recommendations
            </p>
          </div>
          <div className="flex gap-3">
            <button
              onClick={isAnalyzing ? stopAnalysis : startAnalysis}
              className={`px-6 py-2 rounded-lg font-medium flex items-center gap-2 ${
                isAnalyzing 
                  ? 'bg-red-600 hover:bg-red-700 text-white' 
                  : 'bg-purple-600 hover:bg-purple-700 text-white'
              }`}
            >
              {isAnalyzing ? (
                <>
                  <Activity className="w-4 h-4 animate-pulse" />
                  Stop Analysis
                </>
              ) : (
                <>
                  <Brain className="w-4 h-4" />
                  Start Analysis
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
                <p className="text-sm font-medium text-gray-600">Data Points</p>
                <p className="text-2xl font-bold text-gray-900">{summary.totalDataPoints}</p>
              </div>
              <BarChart3 className="w-8 h-8 text-blue-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Active Models</p>
                <p className="text-2xl font-bold text-purple-600">{summary.activeModels}</p>
              </div>
              <Brain className="w-8 h-8 text-purple-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Model Accuracy</p>
                <p className="text-2xl font-bold text-green-600">{formatPercentage(summary.averageModelAccuracy * 100)}</p>
              </div>
              <Target className="w-8 h-8 text-green-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Anomalies</p>
                <p className={`text-2xl font-bold ${summary.unresolvedAnomalies > 0 ? 'text-red-600' : 'text-green-600'}`}>
                  {summary.unresolvedAnomalies}
                </p>
              </div>
              <AlertTriangle className="w-8 h-8 text-red-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Actionable Insights</p>
                <p className="text-2xl font-bold text-blue-600">{summary.actionableInsights}</p>
              </div>
              <Zap className="w-8 h-8 text-blue-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Trends Detected</p>
                <p className="text-2xl font-bold text-orange-600">{summary.trendsDetected}</p>
              </div>
              <TrendingUp className="w-8 h-8 text-orange-500" />
            </div>
          </div>
        </div>

        {/* Status Banner */}
        <div className={`rounded-lg p-4 mb-6 ${
          isAnalyzing ? 'bg-purple-50 border border-purple-200' : 'bg-gray-50 border border-gray-200'
        }`}>
          <div className="flex items-center gap-2">
            {isAnalyzing ? (
              <>
                <Brain className="w-5 h-5 text-purple-600 animate-pulse" />
                <span className="font-medium text-purple-800">Advanced analytics engine is running</span>
                <span className="text-purple-600">- Continuously analyzing performance and generating predictions</span>
              </>
            ) : (
              <>
                <AlertTriangle className="w-5 h-5 text-gray-600" />
                <span className="font-medium text-gray-800">Analytics engine is stopped</span>
                <span className="text-gray-600">- Click "Start Analysis" to begin predictive analytics</span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'overview', label: 'Overview', icon: BarChart3 },
            { id: 'predictions', label: 'Predictions', icon: Brain },
            { id: 'anomalies', label: 'Anomalies', icon: AlertTriangle },
            { id: 'patterns', label: 'User Patterns', icon: Activity },
            { id: 'insights', label: 'Insights', icon: Zap },
            { id: 'abtests', label: 'A/B Tests', icon: Target }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setSelectedTab(tab.id as any)}
              className={`flex items-center gap-2 py-2 px-1 border-b-2 font-medium text-sm ${
                selectedTab === tab.id
                  ? 'border-purple-500 text-purple-600'
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
                <LineChart className="w-5 h-5 text-blue-500" />
                Performance Trends (24h)
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <RechartsLineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="renderTime" stroke="#3b82f6" strokeWidth={2} />
                  <Line type="monotone" dataKey="memoryUsage" stroke="#10b981" strokeWidth={2} />
                  <Line type="monotone" dataKey="networkLatency" stroke="#f59e0b" strokeWidth={2} />
                </RechartsLineChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <PieChart className="w-5 h-5 text-red-500" />
                Anomaly Distribution
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <RechartsPieChart>
                  <Pie
                    data={anomalyData}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, value }) => `${name}: ${value}`}
                  >
                    {anomalyData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </RechartsPieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Recent Insights */}
          <div className="bg-white rounded-lg shadow-md">
            <div className="p-6 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900">Recent Insights</h3>
              <p className="text-sm text-gray-600 mt-1">Latest AI-generated performance insights and recommendations</p>
            </div>
            <div className="p-6">
              {insights.length > 0 ? (
                <div className="space-y-4">
                  {insights.slice(-3).reverse().map(insight => (
                    <div key={insight.id} className="p-4 border border-gray-200 rounded-lg">
                      <div className="flex items-start justify-between mb-2">
                        <h4 className="font-medium text-gray-900">{insight.title}</h4>
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${getImpactColor(insight.impact)}`}>
                          {insight.impact} impact
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 mb-3">{insight.description}</p>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2 text-sm text-gray-500">
                          <span>Confidence: {formatPercentage(insight.confidence * 100)}</span>
                          <span>•</span>
                          <span>{insight.timestamp.toLocaleString()}</span>
                        </div>
                        {insight.actionable && (
                          <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">
                            Actionable
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <Brain className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-500">No insights generated yet</p>
                  <p className="text-sm text-gray-400">Start the analytics engine to generate AI insights</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {selectedTab === 'predictions' && (
        <div className="space-y-6">
          {models.map(model => (
            <div key={model.id} className="bg-white rounded-lg shadow-md">
              <div className="p-6 border-b border-gray-200">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">{model.name}</h3>
                    <p className="text-sm text-gray-600">
                      {model.type} model • Accuracy: {formatPercentage(model.accuracy * 100)} • 
                      Last trained: {model.lastTrained.toLocaleDateString()}
                    </p>
                  </div>
                  <span className={`px-3 py-1 text-sm font-medium rounded-full ${
                    model.accuracy > 0.9 ? 'bg-green-100 text-green-800' :
                    model.accuracy > 0.8 ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {formatPercentage(model.accuracy * 100)} accuracy
                  </span>
                </div>
              </div>
              <div className="p-6">
                {model.predictions.length > 0 ? (
                  <div className="space-y-4">
                    {model.predictions.map((prediction, index) => (
                      <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                        <div className="flex items-center gap-3">
                          {getTrendIcon(prediction.trend)}
                          <div>
                            <p className="font-medium text-gray-900 capitalize">{prediction.metric}</p>
                            <p className="text-sm text-gray-600">
                              Current: {prediction.currentValue.toFixed(2)} → 
                              Predicted: {prediction.predictedValue.toFixed(2)} 
                              ({prediction.timeHorizon}h horizon)
                            </p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="font-medium text-gray-900">
                            {formatPercentage(prediction.confidence * 100)} confidence
                          </p>
                          <p className={`text-sm ${
                            prediction.trend === 'improving' ? 'text-green-600' :
                            prediction.trend === 'degrading' ? 'text-red-600' :
                            'text-blue-600'
                          }`}>
                            {prediction.trend}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <Brain className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-500">No predictions available</p>
                    <p className="text-sm text-gray-400">Model is generating predictions...</p>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {selectedTab === 'anomalies' && (
        <div className="bg-white rounded-lg shadow-md">
          <div className="p-6 border-b border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900">Performance Anomalies</h3>
            <p className="text-sm text-gray-600 mt-1">
              Detected anomalies in performance metrics with AI-powered analysis
            </p>
          </div>
          <div className="divide-y divide-gray-200">
            {anomalies.length > 0 ? (
              anomalies.slice().reverse().map(anomaly => (
                <div key={anomaly.id} className={`p-6 ${anomaly.resolved ? 'bg-gray-50' : ''}`}>
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3">
                      <AlertTriangle className={`w-5 h-5 mt-1 ${
                        anomaly.severity === 'critical' ? 'text-red-500' :
                        anomaly.severity === 'high' ? 'text-orange-500' :
                        anomaly.severity === 'medium' ? 'text-yellow-500' :
                        'text-green-500'
                      }`} />
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <h4 className={`font-medium ${anomaly.resolved ? 'text-gray-600' : 'text-gray-900'}`}>
                            {anomaly.description}
                          </h4>
                          <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                            anomaly.severity === 'critical' ? 'bg-red-100 text-red-800' :
                            anomaly.severity === 'high' ? 'bg-orange-100 text-orange-800' :
                            anomaly.severity === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-green-100 text-green-800'
                          }`}>
                            {anomaly.severity}
                          </span>
                        </div>
                        <div className="text-sm text-gray-600 mb-3">
                          <p>Value: {anomaly.value.toFixed(2)} (Expected: {anomaly.expectedValue.toFixed(2)})</p>
                          <p>Detected: {anomaly.timestamp.toLocaleString()}</p>
                        </div>
                        <div className="space-y-2">
                          <div>
                            <p className="text-sm font-medium text-gray-700">Possible Causes:</p>
                            <ul className="text-sm text-gray-600 list-disc list-inside">
                              {anomaly.possibleCauses.map((cause, index) => (
                                <li key={index}>{cause}</li>
                              ))}
                            </ul>
                          </div>
                          <div>
                            <p className="text-sm font-medium text-gray-700">Recommendations:</p>
                            <ul className="text-sm text-gray-600 list-disc list-inside">
                              {anomaly.recommendations.map((rec, index) => (
                                <li key={index}>{rec}</li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </div>
                    </div>
                    {!anomaly.resolved && (
                      <button
                        onClick={() => resolveAnomaly(anomaly.id)}
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
                <p className="text-gray-500">No anomalies detected</p>
                <p className="text-sm text-gray-400">All performance metrics are within normal ranges</p>
              </div>
            )}
          </div>
        </div>
      )}

      {selectedTab === 'patterns' && (
        <div className="bg-white rounded-lg shadow-md">
          <div className="p-6 border-b border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900">User Behavior Patterns</h3>
            <p className="text-sm text-gray-600 mt-1">
              AI-identified user behavior patterns and their performance impact
            </p>
          </div>
          <div className="p-6">
            <div className="space-y-6">
              {userPatterns.map(pattern => (
                <div key={pattern.id} className="p-4 border border-gray-200 rounded-lg">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h4 className="font-medium text-gray-900">{pattern.pattern}</h4>
                      <p className="text-sm text-gray-600">
                        {pattern.userSegment} • {pattern.timeOfDay}
                      </p>
                    </div>
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                      pattern.optimizationOpportunity > 80 ? 'bg-green-100 text-green-800' :
                      pattern.optimizationOpportunity > 60 ? 'bg-yellow-100 text-yellow-800' :
                      'bg-red-100 text-red-800'
                    }`}>
                      {pattern.optimizationOpportunity}% optimization opportunity
                    </span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-3">
                    <div>
                      <p className="text-sm text-gray-500">Frequency</p>
                      <p className="font-medium">{formatPercentage(pattern.frequency)}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Performance Impact</p>
                      <p className="font-medium">{formatPercentage(pattern.performanceImpact)}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Components</p>
                      <p className="font-medium">{pattern.components.length} components</p>
                    </div>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-700 mb-1">Affected Components:</p>
                    <div className="flex flex-wrap gap-2">
                      {pattern.components.map(component => (
                        <span key={component} className="px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded">
                          {component}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {selectedTab === 'insights' && (
        <div className="space-y-6">
          {insights.map(insight => (
            <div key={insight.id} className="bg-white rounded-lg shadow-md p-6">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-start gap-3">
                  <div className={`p-2 rounded-lg ${
                    insight.type === 'trend' ? 'bg-blue-100' :
                    insight.type === 'anomaly' ? 'bg-red-100' :
                    insight.type === 'optimization' ? 'bg-green-100' :
                    'bg-purple-100'
                  }`}>
                    {insight.type === 'trend' && <TrendingUp className="w-5 h-5 text-blue-600" />}
                    {insight.type === 'anomaly' && <AlertTriangle className="w-5 h-5 text-red-600" />}
                    {insight.type === 'optimization' && <Zap className="w-5 h-5 text-green-600" />}
                    {insight.type === 'prediction' && <Brain className="w-5 h-5 text-purple-600" />}
                  </div>
                  <div className="flex-1">
                    <h4 className="font-medium text-gray-900">{insight.title}</h4>
                    <p className="text-sm text-gray-600 mt-1">{insight.description}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${getImpactColor(insight.impact)}`}>
                    {insight.impact}
                  </span>
                  {insight.actionable && (
                    <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">
                      Actionable
                    </span>
                  )}
                </div>
              </div>
              <div className="space-y-3">
                <div className="flex items-center gap-4 text-sm text-gray-500">
                  <span>Confidence: {formatPercentage(insight.confidence * 100)}</span>
                  <span>•</span>
                  <span>{insight.timestamp.toLocaleString()}</span>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-700 mb-1">Recommendations:</p>
                  <ul className="text-sm text-gray-600 list-disc list-inside space-y-1">
                    {insight.recommendations.map((rec, index) => (
                      <li key={index}>{rec}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          ))}
          {insights.length === 0 && (
            <div className="bg-white rounded-lg shadow-md p-12 text-center">
              <Zap className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">No insights available</p>
              <p className="text-sm text-gray-400">Start the analytics engine to generate AI insights</p>
            </div>
          )}
        </div>
      )}

      {selectedTab === 'abtests' && (
        <div className="bg-white rounded-lg shadow-md">
          <div className="p-6 border-b border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900">A/B Test Results</h3>
            <p className="text-sm text-gray-600 mt-1">
              Performance optimization A/B test results and statistical significance
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Test Name
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Variant
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Metric
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Improvement
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Significance
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {abTests.map(test => (
                  <tr key={test.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">{test.name}</div>
                      <div className="text-sm text-gray-500">
                        {test.sampleSize} users • {test.duration} days
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                        test.variant === 'A' ? 'bg-blue-100 text-blue-800' : 'bg-green-100 text-green-800'
                      }`}>
                        Variant {test.variant}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 capitalize">
                      {test.metric.replace(/([A-Z])/g, ' $1').trim()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`text-sm font-medium ${
                        test.improvement > 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {test.improvement > 0 ? '+' : ''}{formatPercentage(test.improvement)}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`text-sm font-medium ${
                        test.significance > 0.95 ? 'text-green-600' : 
                        test.significance > 0.9 ? 'text-yellow-600' : 'text-red-600'
                      }`}>
                        {formatPercentage(test.significance * 100)}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                        test.status === 'completed' ? 'bg-green-100 text-green-800' :
                        test.status === 'running' ? 'bg-blue-100 text-blue-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {test.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {abTests.length === 0 && (
              <div className="text-center py-12">
                <Target className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500">No A/B tests available</p>
                <p className="text-sm text-gray-400">Performance optimization tests will appear here</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default AdvancedAnalyticsDashboard;

