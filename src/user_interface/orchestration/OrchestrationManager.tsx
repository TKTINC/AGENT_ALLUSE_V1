import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { 
  Command, 
  Play, 
  Pause, 
  Square, 
  SkipForward, 
  SkipBack, 
  Settings, 
  Activity, 
  Clock, 
  AlertTriangle, 
  CheckCircle, 
  RefreshCw,
  Zap,
  Network,
  Database,
  Shield,
  Eye,
  Bell,
  Filter,
  Search,
  Download,
  Upload,
  Sync,
  GitBranch,
  Target,
  Layers,
  Calendar,
  BarChart3,
  TrendingUp,
  Users,
  Workflow,
  Timer,
  Cpu,
  HardDrive,
  Wifi,
  Server,
  Globe,
  Lock
} from 'lucide-react';

// Types for orchestration manager
interface OrchestrationWorkflow {
  id: string;
  name: string;
  description: string;
  status: 'running' | 'paused' | 'completed' | 'failed' | 'scheduled';
  priority: 'low' | 'medium' | 'high' | 'critical';
  steps: WorkflowStep[];
  schedule?: {
    type: 'immediate' | 'scheduled' | 'recurring';
    startTime?: Date;
    interval?: number;
    cron?: string;
  };
  dependencies: string[];
  createdAt: Date;
  lastRun?: Date;
  nextRun?: Date;
  metrics: {
    totalRuns: number;
    successRate: number;
    averageDuration: number;
    lastDuration?: number;
  };
  configuration: Record<string, any>;
}

interface WorkflowStep {
  id: string;
  name: string;
  type: 'action' | 'condition' | 'parallel' | 'sequential' | 'delay';
  workstream: string;
  action: string;
  parameters: Record<string, any>;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  duration?: number;
  retryCount: number;
  maxRetries: number;
  timeout: number;
  onSuccess?: string[];
  onFailure?: string[];
}

interface OrchestrationMetrics {
  totalWorkflows: number;
  activeWorkflows: number;
  completedToday: number;
  failedToday: number;
  averageExecutionTime: number;
  systemLoad: number;
  queuedJobs: number;
  resourceUtilization: {
    cpu: number;
    memory: number;
    network: number;
    storage: number;
  };
}

interface ExecutionLog {
  id: string;
  workflowId: string;
  workflowName: string;
  timestamp: Date;
  status: 'started' | 'completed' | 'failed' | 'cancelled';
  duration?: number;
  steps: {
    stepId: string;
    stepName: string;
    status: 'completed' | 'failed' | 'skipped';
    duration: number;
    error?: string;
  }[];
  error?: string;
  triggeredBy: 'manual' | 'scheduled' | 'event' | 'dependency';
}

interface ResourceMonitor {
  workstream: string;
  resources: {
    cpu: number;
    memory: number;
    network: number;
    storage: number;
    connections: number;
  };
  limits: {
    cpu: number;
    memory: number;
    network: number;
    storage: number;
    connections: number;
  };
  status: 'healthy' | 'warning' | 'critical';
  lastUpdate: Date;
}

// Mock data for development
const mockWorkflows: OrchestrationWorkflow[] = [
  {
    id: 'wf_001',
    name: 'Daily Market Analysis',
    description: 'Comprehensive daily market analysis across all workstreams',
    status: 'running',
    priority: 'high',
    steps: [
      {
        id: 'step_001',
        name: 'Collect Market Data',
        type: 'action',
        workstream: 'ws3',
        action: 'collect_market_data',
        parameters: { symbols: ['SPY', 'QQQ', 'IWM'], timeframe: '1d' },
        status: 'completed',
        duration: 45,
        retryCount: 0,
        maxRetries: 3,
        timeout: 120,
        onSuccess: ['step_002'],
        onFailure: ['step_error']
      },
      {
        id: 'step_002',
        name: 'Analyze Patterns',
        type: 'action',
        workstream: 'ws5',
        action: 'analyze_patterns',
        parameters: { depth: 'full', confidence_threshold: 0.8 },
        status: 'running',
        retryCount: 0,
        maxRetries: 2,
        timeout: 300,
        onSuccess: ['step_003'],
        onFailure: ['step_retry']
      },
      {
        id: 'step_003',
        name: 'Update Protocols',
        type: 'action',
        workstream: 'ws2',
        action: 'update_protocols',
        parameters: { strategy: 'adaptive' },
        status: 'pending',
        retryCount: 0,
        maxRetries: 1,
        timeout: 60
      }
    ],
    schedule: {
      type: 'recurring',
      startTime: new Date(),
      cron: '0 0 9 * * 1-5'
    },
    dependencies: [],
    createdAt: new Date(Date.now() - 86400000),
    lastRun: new Date(Date.now() - 3600000),
    nextRun: new Date(Date.now() + 82800000),
    metrics: {
      totalRuns: 156,
      successRate: 94.2,
      averageDuration: 420,
      lastDuration: 387
    },
    configuration: {
      notifications: true,
      retryPolicy: 'exponential',
      timeout: 1800
    }
  },
  {
    id: 'wf_002',
    name: 'Risk Assessment Pipeline',
    description: 'Continuous risk assessment and mitigation workflow',
    status: 'scheduled',
    priority: 'critical',
    steps: [
      {
        id: 'step_004',
        name: 'Calculate VaR',
        type: 'action',
        workstream: 'ws2',
        action: 'calculate_var',
        parameters: { confidence: 0.95, horizon: 1 },
        status: 'pending',
        retryCount: 0,
        maxRetries: 2,
        timeout: 180
      },
      {
        id: 'step_005',
        name: 'Stress Test',
        type: 'parallel',
        workstream: 'ws4',
        action: 'stress_test',
        parameters: { scenarios: ['market_crash', 'volatility_spike', 'liquidity_crisis'] },
        status: 'pending',
        retryCount: 0,
        maxRetries: 1,
        timeout: 600
      }
    ],
    schedule: {
      type: 'recurring',
      interval: 3600
    },
    dependencies: ['wf_001'],
    createdAt: new Date(Date.now() - 172800000),
    nextRun: new Date(Date.now() + 1800000),
    metrics: {
      totalRuns: 89,
      successRate: 98.9,
      averageDuration: 234,
      lastDuration: 198
    },
    configuration: {
      notifications: true,
      alertThreshold: 0.8
    }
  },
  {
    id: 'wf_003',
    name: 'Learning Model Update',
    description: 'Automated learning model training and deployment',
    status: 'completed',
    priority: 'medium',
    steps: [
      {
        id: 'step_006',
        name: 'Prepare Training Data',
        type: 'action',
        workstream: 'ws5',
        action: 'prepare_data',
        parameters: { lookback: 30, features: 'all' },
        status: 'completed',
        duration: 156,
        retryCount: 0,
        maxRetries: 2,
        timeout: 300
      },
      {
        id: 'step_007',
        name: 'Train Model',
        type: 'action',
        workstream: 'ws5',
        action: 'train_model',
        parameters: { algorithm: 'ensemble', validation: 'cross' },
        status: 'completed',
        duration: 1245,
        retryCount: 0,
        maxRetries: 1,
        timeout: 3600
      },
      {
        id: 'step_008',
        name: 'Deploy Model',
        type: 'sequential',
        workstream: 'ws5',
        action: 'deploy_model',
        parameters: { environment: 'production', rollback: true },
        status: 'completed',
        duration: 67,
        retryCount: 0,
        maxRetries: 2,
        timeout: 180
      }
    ],
    schedule: {
      type: 'scheduled',
      startTime: new Date(Date.now() - 7200000)
    },
    dependencies: [],
    createdAt: new Date(Date.now() - 259200000),
    lastRun: new Date(Date.now() - 7200000),
    metrics: {
      totalRuns: 23,
      successRate: 87.0,
      averageDuration: 1468,
      lastDuration: 1468
    },
    configuration: {
      notifications: true,
      backup: true
    }
  }
];

const mockResourceMonitors: ResourceMonitor[] = [
  {
    workstream: 'ws1',
    resources: { cpu: 45, memory: 67, network: 23, storage: 34, connections: 156 },
    limits: { cpu: 80, memory: 85, network: 70, storage: 90, connections: 500 },
    status: 'healthy',
    lastUpdate: new Date()
  },
  {
    workstream: 'ws2',
    resources: { cpu: 32, memory: 54, network: 18, storage: 28, connections: 89 },
    limits: { cpu: 80, memory: 85, network: 70, storage: 90, connections: 300 },
    status: 'healthy',
    lastUpdate: new Date()
  },
  {
    workstream: 'ws3',
    resources: { cpu: 78, memory: 82, network: 65, storage: 45, connections: 234 },
    limits: { cpu: 80, memory: 85, network: 70, storage: 90, connections: 400 },
    status: 'warning',
    lastUpdate: new Date()
  },
  {
    workstream: 'ws4',
    resources: { cpu: 56, memory: 71, network: 43, storage: 38, connections: 178 },
    limits: { cpu: 80, memory: 85, network: 70, storage: 90, connections: 350 },
    status: 'healthy',
    lastUpdate: new Date()
  },
  {
    workstream: 'ws5',
    resources: { cpu: 89, memory: 91, network: 34, storage: 67, connections: 123 },
    limits: { cpu: 90, memory: 95, network: 70, storage: 90, connections: 200 },
    status: 'critical',
    lastUpdate: new Date()
  }
];

const OrchestrationManager: React.FC = () => {
  const [workflows, setWorkflows] = useState<OrchestrationWorkflow[]>(mockWorkflows);
  const [resourceMonitors, setResourceMonitors] = useState<ResourceMonitor[]>(mockResourceMonitors);
  const [executionLogs, setExecutionLogs] = useState<ExecutionLog[]>([]);
  const [selectedTab, setSelectedTab] = useState<'dashboard' | 'workflows' | 'resources' | 'logs'>('dashboard');
  const [selectedWorkflow, setSelectedWorkflow] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');

  // Calculate orchestration metrics
  const orchestrationMetrics = useMemo((): OrchestrationMetrics => {
    const totalWorkflows = workflows.length;
    const activeWorkflows = workflows.filter(wf => wf.status === 'running').length;
    const completedToday = workflows.filter(wf => 
      wf.lastRun && wf.lastRun.toDateString() === new Date().toDateString() && wf.status === 'completed'
    ).length;
    const failedToday = workflows.filter(wf => 
      wf.lastRun && wf.lastRun.toDateString() === new Date().toDateString() && wf.status === 'failed'
    ).length;
    const averageExecutionTime = workflows.reduce((sum, wf) => sum + wf.metrics.averageDuration, 0) / workflows.length;
    const systemLoad = resourceMonitors.reduce((sum, rm) => sum + rm.resources.cpu, 0) / resourceMonitors.length;
    const queuedJobs = workflows.filter(wf => wf.status === 'scheduled').length;
    
    const resourceUtilization = {
      cpu: resourceMonitors.reduce((sum, rm) => sum + rm.resources.cpu, 0) / resourceMonitors.length,
      memory: resourceMonitors.reduce((sum, rm) => sum + rm.resources.memory, 0) / resourceMonitors.length,
      network: resourceMonitors.reduce((sum, rm) => sum + rm.resources.network, 0) / resourceMonitors.length,
      storage: resourceMonitors.reduce((sum, rm) => sum + rm.resources.storage, 0) / resourceMonitors.length
    };

    return {
      totalWorkflows,
      activeWorkflows,
      completedToday,
      failedToday,
      averageExecutionTime: Math.round(averageExecutionTime),
      systemLoad: Math.round(systemLoad),
      queuedJobs,
      resourceUtilization: {
        cpu: Math.round(resourceUtilization.cpu),
        memory: Math.round(resourceUtilization.memory),
        network: Math.round(resourceUtilization.network),
        storage: Math.round(resourceUtilization.storage)
      }
    };
  }, [workflows, resourceMonitors]);

  // Simulate real-time updates
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      // Update workflow statuses
      setWorkflows(prev => prev.map(wf => {
        if (wf.status === 'running') {
          // Simulate step progression
          const runningStepIndex = wf.steps.findIndex(step => step.status === 'running');
          if (runningStepIndex !== -1 && Math.random() < 0.3) {
            const updatedSteps = [...wf.steps];
            updatedSteps[runningStepIndex] = {
              ...updatedSteps[runningStepIndex],
              status: 'completed',
              duration: Math.floor(Math.random() * 200) + 50
            };
            
            // Start next step if available
            if (runningStepIndex + 1 < updatedSteps.length) {
              updatedSteps[runningStepIndex + 1] = {
                ...updatedSteps[runningStepIndex + 1],
                status: 'running'
              };
            } else {
              // Workflow completed
              return {
                ...wf,
                status: 'completed' as const,
                steps: updatedSteps,
                lastRun: new Date(),
                metrics: {
                  ...wf.metrics,
                  totalRuns: wf.metrics.totalRuns + 1,
                  lastDuration: updatedSteps.reduce((sum, step) => sum + (step.duration || 0), 0)
                }
              };
            }
            
            return { ...wf, steps: updatedSteps };
          }
        }
        return wf;
      }));

      // Update resource monitors
      setResourceMonitors(prev => prev.map(rm => ({
        ...rm,
        resources: {
          cpu: Math.max(0, Math.min(100, rm.resources.cpu + (Math.random() - 0.5) * 10)),
          memory: Math.max(0, Math.min(100, rm.resources.memory + (Math.random() - 0.5) * 8)),
          network: Math.max(0, Math.min(100, rm.resources.network + (Math.random() - 0.5) * 15)),
          storage: Math.max(0, Math.min(100, rm.resources.storage + (Math.random() - 0.5) * 5)),
          connections: Math.max(0, rm.resources.connections + Math.floor((Math.random() - 0.5) * 20))
        },
        status: rm.resources.cpu > 85 || rm.resources.memory > 90 ? 'critical' :
                rm.resources.cpu > 70 || rm.resources.memory > 80 ? 'warning' : 'healthy',
        lastUpdate: new Date()
      })));

      // Generate execution logs
      if (Math.random() < 0.2) {
        const workflow = workflows[Math.floor(Math.random() * workflows.length)];
        const statuses = ['started', 'completed', 'failed'] as const;
        const triggers = ['manual', 'scheduled', 'event', 'dependency'] as const;
        
        const newLog: ExecutionLog = {
          id: Date.now().toString(),
          workflowId: workflow.id,
          workflowName: workflow.name,
          timestamp: new Date(),
          status: statuses[Math.floor(Math.random() * statuses.length)],
          duration: Math.floor(Math.random() * 600) + 60,
          steps: workflow.steps.map(step => ({
            stepId: step.id,
            stepName: step.name,
            status: Math.random() > 0.1 ? 'completed' : 'failed',
            duration: Math.floor(Math.random() * 120) + 10
          })),
          triggeredBy: triggers[Math.floor(Math.random() * triggers.length)]
        };

        setExecutionLogs(prev => [newLog, ...prev.slice(0, 49)]);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [autoRefresh, workflows]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'text-blue-600 bg-blue-100';
      case 'completed': return 'text-green-600 bg-green-100';
      case 'failed': return 'text-red-600 bg-red-100';
      case 'paused': return 'text-yellow-600 bg-yellow-100';
      case 'scheduled': return 'text-purple-600 bg-purple-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'text-red-600 bg-red-100';
      case 'high': return 'text-orange-600 bg-orange-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'low': return 'text-green-600 bg-green-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getResourceStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-600 bg-green-100';
      case 'warning': return 'text-yellow-600 bg-yellow-100';
      case 'critical': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getStepStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'running': return <Play className="w-4 h-4 text-blue-500" />;
      case 'failed': return <AlertTriangle className="w-4 h-4 text-red-500" />;
      case 'pending': return <Clock className="w-4 h-4 text-gray-500" />;
      case 'skipped': return <SkipForward className="w-4 h-4 text-yellow-500" />;
      default: return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  const startWorkflow = useCallback((workflowId: string) => {
    setWorkflows(prev => prev.map(wf => 
      wf.id === workflowId 
        ? { 
            ...wf, 
            status: 'running' as const,
            steps: wf.steps.map((step, index) => ({
              ...step,
              status: index === 0 ? 'running' as const : 'pending' as const
            }))
          }
        : wf
    ));
  }, []);

  const pauseWorkflow = useCallback((workflowId: string) => {
    setWorkflows(prev => prev.map(wf => 
      wf.id === workflowId ? { ...wf, status: 'paused' as const } : wf
    ));
  }, []);

  const stopWorkflow = useCallback((workflowId: string) => {
    setWorkflows(prev => prev.map(wf => 
      wf.id === workflowId 
        ? { 
            ...wf, 
            status: 'completed' as const,
            steps: wf.steps.map(step => ({
              ...step,
              status: step.status === 'running' ? 'completed' as const : step.status
            }))
          }
        : wf
    ));
  }, []);

  const filteredWorkflows = useMemo(() => {
    return workflows.filter(wf => {
      const matchesSearch = wf.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           wf.description.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesStatus = statusFilter === 'all' || wf.status === statusFilter;
      return matchesSearch && matchesStatus;
    });
  }, [workflows, searchTerm, statusFilter]);

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
                <Command className="w-8 h-8 text-blue-600" />
                Orchestration Manager
              </h1>
              <p className="text-gray-600 mt-1">
                Advanced workflow orchestration and automation across ALL-USE workstreams
              </p>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium text-gray-700">Auto Refresh</label>
                <button
                  onClick={() => setAutoRefresh(!autoRefresh)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    autoRefresh ? 'bg-blue-600' : 'bg-gray-200'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      autoRefresh ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
              <button className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                <Workflow className="w-4 h-4" />
                New Workflow
              </button>
            </div>
          </div>
        </div>

        {/* Orchestration Overview Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-8 gap-4">
          <div className="bg-white rounded-lg shadow-sm border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Workflows</p>
                <p className="text-2xl font-bold text-gray-900">{orchestrationMetrics.totalWorkflows}</p>
              </div>
              <Workflow className="w-8 h-8 text-blue-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Active</p>
                <p className="text-2xl font-bold text-blue-600">{orchestrationMetrics.activeWorkflows}</p>
              </div>
              <Play className="w-8 h-8 text-blue-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Completed Today</p>
                <p className="text-2xl font-bold text-green-600">{orchestrationMetrics.completedToday}</p>
              </div>
              <CheckCircle className="w-8 h-8 text-green-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Failed Today</p>
                <p className="text-2xl font-bold text-red-600">{orchestrationMetrics.failedToday}</p>
              </div>
              <AlertTriangle className="w-8 h-8 text-red-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Avg Duration</p>
                <p className="text-2xl font-bold text-gray-900">{orchestrationMetrics.averageExecutionTime}s</p>
              </div>
              <Timer className="w-8 h-8 text-yellow-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">System Load</p>
                <p className="text-2xl font-bold text-orange-600">{orchestrationMetrics.systemLoad}%</p>
              </div>
              <Cpu className="w-8 h-8 text-orange-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Queued Jobs</p>
                <p className="text-2xl font-bold text-purple-600">{orchestrationMetrics.queuedJobs}</p>
              </div>
              <Calendar className="w-8 h-8 text-purple-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Memory Usage</p>
                <p className="text-2xl font-bold text-indigo-600">{orchestrationMetrics.resourceUtilization.memory}%</p>
              </div>
              <HardDrive className="w-8 h-8 text-indigo-500" />
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="bg-white rounded-lg shadow-sm border">
          <div className="border-b">
            <nav className="flex space-x-8 px-6">
              {[
                { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
                { id: 'workflows', label: 'Workflows', icon: Workflow },
                { id: 'resources', label: 'Resources', icon: Server },
                { id: 'logs', label: 'Execution Logs', icon: Eye }
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setSelectedTab(tab.id as any)}
                  className={`flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm ${
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

          <div className="p-6">
            {selectedTab === 'dashboard' && (
              <div className="space-y-6">
                <h3 className="text-lg font-semibold text-gray-900">System Overview</h3>
                
                {/* Resource Utilization Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-700">CPU Usage</span>
                      <Cpu className="w-4 h-4 text-gray-500" />
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full" 
                        style={{ width: `${orchestrationMetrics.resourceUtilization.cpu}%` }}
                      />
                    </div>
                    <p className="text-sm text-gray-600 mt-1">{orchestrationMetrics.resourceUtilization.cpu}%</p>
                  </div>

                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-700">Memory Usage</span>
                      <HardDrive className="w-4 h-4 text-gray-500" />
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-green-600 h-2 rounded-full" 
                        style={{ width: `${orchestrationMetrics.resourceUtilization.memory}%` }}
                      />
                    </div>
                    <p className="text-sm text-gray-600 mt-1">{orchestrationMetrics.resourceUtilization.memory}%</p>
                  </div>

                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-700">Network Usage</span>
                      <Wifi className="w-4 h-4 text-gray-500" />
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-yellow-600 h-2 rounded-full" 
                        style={{ width: `${orchestrationMetrics.resourceUtilization.network}%` }}
                      />
                    </div>
                    <p className="text-sm text-gray-600 mt-1">{orchestrationMetrics.resourceUtilization.network}%</p>
                  </div>

                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-700">Storage Usage</span>
                      <Database className="w-4 h-4 text-gray-500" />
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-purple-600 h-2 rounded-full" 
                        style={{ width: `${orchestrationMetrics.resourceUtilization.storage}%` }}
                      />
                    </div>
                    <p className="text-sm text-gray-600 mt-1">{orchestrationMetrics.resourceUtilization.storage}%</p>
                  </div>
                </div>

                {/* Active Workflows Summary */}
                <div>
                  <h4 className="text-md font-medium text-gray-900 mb-3">Active Workflows</h4>
                  <div className="space-y-3">
                    {workflows.filter(wf => wf.status === 'running').map((workflow) => (
                      <div key={workflow.id} className="border rounded-lg p-4">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(workflow.status)}`}>
                              {workflow.status.toUpperCase()}
                            </span>
                            <h5 className="font-medium text-gray-900">{workflow.name}</h5>
                          </div>
                          <div className="text-sm text-gray-500">
                            {workflow.steps.filter(s => s.status === 'completed').length}/{workflow.steps.length} steps
                          </div>
                        </div>
                        <div className="mt-2">
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-blue-600 h-2 rounded-full" 
                              style={{ 
                                width: `${(workflow.steps.filter(s => s.status === 'completed').length / workflow.steps.length) * 100}%` 
                              }}
                            />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {selectedTab === 'workflows' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900">Workflow Management</h3>
                  <div className="flex items-center gap-4">
                    <div className="relative">
                      <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                      <input
                        type="text"
                        placeholder="Search workflows..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                    <select
                      value={statusFilter}
                      onChange={(e) => setStatusFilter(e.target.value)}
                      className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="all">All Status</option>
                      <option value="running">Running</option>
                      <option value="completed">Completed</option>
                      <option value="failed">Failed</option>
                      <option value="paused">Paused</option>
                      <option value="scheduled">Scheduled</option>
                    </select>
                  </div>
                </div>

                <div className="space-y-4">
                  {filteredWorkflows.map((workflow) => (
                    <div key={workflow.id} className="border rounded-lg p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <h4 className="font-medium text-gray-900">{workflow.name}</h4>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(workflow.status)}`}>
                            {workflow.status.toUpperCase()}
                          </span>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getPriorityColor(workflow.priority)}`}>
                            {workflow.priority.toUpperCase()}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => startWorkflow(workflow.id)}
                            disabled={workflow.status === 'running'}
                            className="px-3 py-1 bg-green-100 text-green-700 text-xs rounded hover:bg-green-200 disabled:opacity-50"
                          >
                            <Play className="w-3 h-3" />
                          </button>
                          <button
                            onClick={() => pauseWorkflow(workflow.id)}
                            disabled={workflow.status !== 'running'}
                            className="px-3 py-1 bg-yellow-100 text-yellow-700 text-xs rounded hover:bg-yellow-200 disabled:opacity-50"
                          >
                            <Pause className="w-3 h-3" />
                          </button>
                          <button
                            onClick={() => stopWorkflow(workflow.id)}
                            disabled={workflow.status !== 'running'}
                            className="px-3 py-1 bg-red-100 text-red-700 text-xs rounded hover:bg-red-200 disabled:opacity-50"
                          >
                            <Square className="w-3 h-3" />
                          </button>
                        </div>
                      </div>

                      <p className="text-sm text-gray-600 mb-3">{workflow.description}</p>

                      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm mb-3">
                        <div>
                          <p className="text-xs text-gray-500">Total Runs</p>
                          <p className="font-medium">{workflow.metrics.totalRuns}</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-500">Success Rate</p>
                          <p className="font-medium text-green-600">{workflow.metrics.successRate}%</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-500">Avg Duration</p>
                          <p className="font-medium">{workflow.metrics.averageDuration}s</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-500">Next Run</p>
                          <p className="font-medium">
                            {workflow.nextRun ? workflow.nextRun.toLocaleTimeString() : 'N/A'}
                          </p>
                        </div>
                      </div>

                      {selectedWorkflow === workflow.id && (
                        <div className="mt-4 pt-4 border-t">
                          <h5 className="font-medium text-gray-900 mb-3">Workflow Steps</h5>
                          <div className="space-y-2">
                            {workflow.steps.map((step, index) => (
                              <div key={step.id} className="flex items-center gap-3 p-2 bg-gray-50 rounded">
                                {getStepStatusIcon(step.status)}
                                <div className="flex-1">
                                  <p className="text-sm font-medium text-gray-900">{step.name}</p>
                                  <p className="text-xs text-gray-500">
                                    {step.workstream.toUpperCase()}: {step.action}
                                    {step.duration && ` (${step.duration}s)`}
                                  </p>
                                </div>
                                <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(step.status)}`}>
                                  {step.status.toUpperCase()}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      <button
                        onClick={() => setSelectedWorkflow(
                          selectedWorkflow === workflow.id ? null : workflow.id
                        )}
                        className="mt-3 text-sm text-blue-600 hover:text-blue-800"
                      >
                        {selectedWorkflow === workflow.id ? 'Hide Details' : 'Show Details'}
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {selectedTab === 'resources' && (
              <div className="space-y-6">
                <h3 className="text-lg font-semibold text-gray-900">Resource Monitoring</h3>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {resourceMonitors.map((monitor) => (
                    <div key={monitor.workstream} className="border rounded-lg p-4">
                      <div className="flex items-center justify-between mb-4">
                        <div>
                          <h4 className="font-medium text-gray-900">
                            {monitor.workstream.toUpperCase()} Resources
                          </h4>
                          <p className="text-sm text-gray-500">
                            Last updated: {monitor.lastUpdate.toLocaleTimeString()}
                          </p>
                        </div>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getResourceStatusColor(monitor.status)}`}>
                          {monitor.status.toUpperCase()}
                        </span>
                      </div>

                      <div className="space-y-3">
                        {Object.entries(monitor.resources).map(([resource, value]) => (
                          <div key={resource}>
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-sm font-medium text-gray-700 capitalize">
                                {resource}
                              </span>
                              <span className="text-sm text-gray-600">
                                {typeof value === 'number' && resource !== 'connections' 
                                  ? `${value}%` 
                                  : value.toLocaleString()
                                }
                                {resource !== 'connections' && (
                                  <span className="text-gray-400">
                                    /{monitor.limits[resource as keyof typeof monitor.limits]}
                                    {typeof monitor.limits[resource as keyof typeof monitor.limits] === 'number' && 
                                     monitor.limits[resource as keyof typeof monitor.limits] <= 100 ? '%' : ''}
                                  </span>
                                )}
                              </span>
                            </div>
                            {resource !== 'connections' && (
                              <div className="w-full bg-gray-200 rounded-full h-2">
                                <div 
                                  className={`h-2 rounded-full ${
                                    value > 85 ? 'bg-red-600' :
                                    value > 70 ? 'bg-yellow-600' :
                                    'bg-green-600'
                                  }`}
                                  style={{ width: `${Math.min(100, value)}%` }}
                                />
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {selectedTab === 'logs' && (
              <div className="space-y-6">
                <h3 className="text-lg font-semibold text-gray-900">Execution Logs</h3>
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {executionLogs.length === 0 ? (
                    <p className="text-gray-500 text-center py-8">No execution logs available</p>
                  ) : (
                    executionLogs.map((log) => (
                      <div key={log.id} className="border rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-3">
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(log.status)}`}>
                              {log.status.toUpperCase()}
                            </span>
                            <h4 className="font-medium text-gray-900">{log.workflowName}</h4>
                          </div>
                          <div className="text-sm text-gray-500">
                            {log.timestamp.toLocaleTimeString()}
                          </div>
                        </div>
                        
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                          <div>
                            <p className="text-xs text-gray-500">Duration</p>
                            <p className="font-medium">{log.duration ? `${log.duration}s` : 'N/A'}</p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-500">Triggered By</p>
                            <p className="font-medium capitalize">{log.triggeredBy}</p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-500">Steps</p>
                            <p className="font-medium">
                              {log.steps.filter(s => s.status === 'completed').length}/{log.steps.length}
                            </p>
                          </div>
                        </div>

                        {log.error && (
                          <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded">
                            <p className="text-sm text-red-800">{log.error}</p>
                          </div>
                        )}
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default OrchestrationManager;

