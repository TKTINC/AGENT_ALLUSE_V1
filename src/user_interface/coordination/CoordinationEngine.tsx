import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { 
  Cpu, 
  Network, 
  Zap, 
  Settings, 
  Activity, 
  Clock, 
  AlertTriangle, 
  CheckCircle, 
  RefreshCw,
  Play,
  Pause,
  Square,
  BarChart3,
  TrendingUp,
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
  Layers
} from 'lucide-react';

// Types for coordination engine
interface CoordinationRule {
  id: string;
  name: string;
  description: string;
  trigger: {
    workstream: string;
    event: string;
    condition: string;
  };
  actions: {
    workstream: string;
    action: string;
    parameters: Record<string, any>;
  }[];
  priority: 'low' | 'medium' | 'high' | 'critical';
  enabled: boolean;
  lastTriggered?: Date;
  executionCount: number;
}

interface WorkstreamCoordination {
  workstreamId: string;
  name: string;
  coordinationLevel: 'autonomous' | 'supervised' | 'manual';
  dependencies: string[];
  coordinationRules: string[];
  status: 'coordinated' | 'uncoordinated' | 'conflict' | 'maintenance';
  lastSync: Date;
  syncFrequency: number; // seconds
  metrics: {
    coordinationScore: number;
    conflictCount: number;
    syncLatency: number;
    ruleExecutions: number;
  };
}

interface CoordinationEvent {
  id: string;
  timestamp: Date;
  type: 'sync' | 'conflict' | 'resolution' | 'rule_execution' | 'error';
  workstreams: string[];
  rule?: string;
  description: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  resolved: boolean;
  resolution?: string;
}

interface SyncOperation {
  id: string;
  name: string;
  sourceWorkstream: string;
  targetWorkstreams: string[];
  dataType: string;
  frequency: 'real-time' | 'periodic' | 'event-driven';
  interval?: number;
  status: 'active' | 'paused' | 'error' | 'completed';
  lastSync: Date;
  nextSync?: Date;
  metrics: {
    successRate: number;
    averageLatency: number;
    dataVolume: number;
    errorCount: number;
  };
}

// Mock data for development
const mockCoordinationRules: CoordinationRule[] = [
  {
    id: 'rule_001',
    name: 'Market Data Sync',
    description: 'Synchronize market data across WS3 and WS4 when new data arrives',
    trigger: { workstream: 'ws3', event: 'market_data_update', condition: 'volume > 1000' },
    actions: [
      { workstream: 'ws4', action: 'update_positions', parameters: { priority: 'high' } },
      { workstream: 'ws5', action: 'analyze_patterns', parameters: { depth: 'full' } }
    ],
    priority: 'high',
    enabled: true,
    lastTriggered: new Date(Date.now() - 300000),
    executionCount: 1247
  },
  {
    id: 'rule_002',
    name: 'Risk Threshold Alert',
    description: 'Alert all workstreams when risk threshold is exceeded',
    trigger: { workstream: 'ws2', event: 'risk_threshold_exceeded', condition: 'risk_level > 0.8' },
    actions: [
      { workstream: 'ws1', action: 'send_alert', parameters: { urgency: 'critical' } },
      { workstream: 'ws4', action: 'pause_trading', parameters: { duration: 300 } },
      { workstream: 'ws5', action: 'emergency_analysis', parameters: { mode: 'risk_assessment' } }
    ],
    priority: 'critical',
    enabled: true,
    lastTriggered: new Date(Date.now() - 1800000),
    executionCount: 23
  },
  {
    id: 'rule_003',
    name: 'Learning Model Update',
    description: 'Coordinate learning model updates across all workstreams',
    trigger: { workstream: 'ws5', event: 'model_updated', condition: 'accuracy_improvement > 0.05' },
    actions: [
      { workstream: 'ws1', action: 'update_agent_knowledge', parameters: { version: 'latest' } },
      { workstream: 'ws2', action: 'recalibrate_protocols', parameters: { model_version: 'latest' } },
      { workstream: 'ws3', action: 'update_intelligence', parameters: { model_version: 'latest' } },
      { workstream: 'ws4', action: 'update_strategies', parameters: { model_version: 'latest' } }
    ],
    priority: 'medium',
    enabled: true,
    lastTriggered: new Date(Date.now() - 7200000),
    executionCount: 156
  }
];

const mockWorkstreamCoordination: WorkstreamCoordination[] = [
  {
    workstreamId: 'ws1',
    name: 'Agent Foundation',
    coordinationLevel: 'autonomous',
    dependencies: [],
    coordinationRules: ['rule_002', 'rule_003'],
    status: 'coordinated',
    lastSync: new Date(),
    syncFrequency: 30,
    metrics: { coordinationScore: 98.5, conflictCount: 0, syncLatency: 45, ruleExecutions: 234 }
  },
  {
    workstreamId: 'ws2',
    name: 'Protocol Systems',
    coordinationLevel: 'supervised',
    dependencies: ['ws1'],
    coordinationRules: ['rule_002', 'rule_003'],
    status: 'coordinated',
    lastSync: new Date(),
    syncFrequency: 15,
    metrics: { coordinationScore: 96.2, conflictCount: 1, syncLatency: 32, ruleExecutions: 189 }
  },
  {
    workstreamId: 'ws3',
    name: 'Market Intelligence',
    coordinationLevel: 'autonomous',
    dependencies: ['ws1', 'ws2'],
    coordinationRules: ['rule_001', 'rule_003'],
    status: 'coordinated',
    lastSync: new Date(),
    syncFrequency: 5,
    metrics: { coordinationScore: 94.8, conflictCount: 2, syncLatency: 78, ruleExecutions: 1456 }
  },
  {
    workstreamId: 'ws4',
    name: 'Market Integration',
    coordinationLevel: 'autonomous',
    dependencies: ['ws1', 'ws2', 'ws3'],
    coordinationRules: ['rule_001', 'rule_002', 'rule_003'],
    status: 'coordinated',
    lastSync: new Date(),
    syncFrequency: 3,
    metrics: { coordinationScore: 97.1, conflictCount: 0, syncLatency: 28, ruleExecutions: 2341 }
  },
  {
    workstreamId: 'ws5',
    name: 'Learning Systems',
    coordinationLevel: 'supervised',
    dependencies: ['ws1', 'ws2', 'ws3', 'ws4'],
    coordinationRules: ['rule_003'],
    status: 'coordinated',
    lastSync: new Date(),
    syncFrequency: 60,
    metrics: { coordinationScore: 92.3, conflictCount: 3, syncLatency: 156, ruleExecutions: 167 }
  }
];

const mockSyncOperations: SyncOperation[] = [
  {
    id: 'sync_001',
    name: 'Market Data Sync',
    sourceWorkstream: 'ws3',
    targetWorkstreams: ['ws4', 'ws5'],
    dataType: 'market_data',
    frequency: 'real-time',
    status: 'active',
    lastSync: new Date(),
    metrics: { successRate: 99.8, averageLatency: 25, dataVolume: 15420, errorCount: 2 }
  },
  {
    id: 'sync_002',
    name: 'Protocol Updates',
    sourceWorkstream: 'ws2',
    targetWorkstreams: ['ws3', 'ws4', 'ws5'],
    dataType: 'protocol_config',
    frequency: 'event-driven',
    status: 'active',
    lastSync: new Date(Date.now() - 1800000),
    metrics: { successRate: 100, averageLatency: 45, dataVolume: 234, errorCount: 0 }
  },
  {
    id: 'sync_003',
    name: 'Learning Models',
    sourceWorkstream: 'ws5',
    targetWorkstreams: ['ws1', 'ws2', 'ws3', 'ws4'],
    dataType: 'ml_models',
    frequency: 'periodic',
    interval: 3600,
    status: 'active',
    lastSync: new Date(Date.now() - 3600000),
    nextSync: new Date(Date.now() + 1800000),
    metrics: { successRate: 95.2, averageLatency: 234, dataVolume: 5670, errorCount: 12 }
  }
];

const CoordinationEngine: React.FC = () => {
  const [coordinationRules, setCoordinationRules] = useState<CoordinationRule[]>(mockCoordinationRules);
  const [workstreamCoordination, setWorkstreamCoordination] = useState<WorkstreamCoordination[]>(mockWorkstreamCoordination);
  const [syncOperations, setSyncOperations] = useState<SyncOperation[]>(mockSyncOperations);
  const [coordinationEvents, setCoordinationEvents] = useState<CoordinationEvent[]>([]);
  const [selectedTab, setSelectedTab] = useState<'overview' | 'rules' | 'sync' | 'events'>('overview');
  const [autoCoordination, setAutoCoordination] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterPriority, setFilterPriority] = useState<string>('all');

  // Calculate coordination metrics
  const coordinationMetrics = useMemo(() => {
    const totalScore = workstreamCoordination.reduce((sum, ws) => sum + ws.metrics.coordinationScore, 0);
    const averageScore = totalScore / workstreamCoordination.length;
    const totalConflicts = workstreamCoordination.reduce((sum, ws) => sum + ws.metrics.conflictCount, 0);
    const totalRuleExecutions = workstreamCoordination.reduce((sum, ws) => sum + ws.metrics.ruleExecutions, 0);
    const averageLatency = workstreamCoordination.reduce((sum, ws) => sum + ws.metrics.syncLatency, 0) / workstreamCoordination.length;
    const activeRules = coordinationRules.filter(rule => rule.enabled).length;
    const activeSyncs = syncOperations.filter(sync => sync.status === 'active').length;

    return {
      averageScore: Math.round(averageScore * 10) / 10,
      totalConflicts,
      totalRuleExecutions,
      averageLatency: Math.round(averageLatency),
      activeRules,
      activeSyncs,
      coordinatedWorkstreams: workstreamCoordination.filter(ws => ws.status === 'coordinated').length
    };
  }, [workstreamCoordination, coordinationRules, syncOperations]);

  // Simulate real-time coordination updates
  useEffect(() => {
    if (!autoCoordination) return;

    const interval = setInterval(() => {
      // Update workstream coordination metrics
      setWorkstreamCoordination(prev => prev.map(ws => ({
        ...ws,
        lastSync: new Date(),
        metrics: {
          ...ws.metrics,
          coordinationScore: Math.max(85, Math.min(100, ws.metrics.coordinationScore + (Math.random() - 0.5) * 2)),
          syncLatency: Math.max(10, ws.metrics.syncLatency + (Math.random() - 0.5) * 10),
          ruleExecutions: ws.metrics.ruleExecutions + Math.floor(Math.random() * 5)
        }
      })));

      // Update sync operations
      setSyncOperations(prev => prev.map(sync => ({
        ...sync,
        lastSync: sync.frequency === 'real-time' ? new Date() : sync.lastSync,
        metrics: {
          ...sync.metrics,
          dataVolume: sync.metrics.dataVolume + Math.floor(Math.random() * 100),
          averageLatency: Math.max(10, sync.metrics.averageLatency + (Math.random() - 0.5) * 5)
        }
      })));

      // Generate coordination events
      if (Math.random() < 0.4) {
        const eventTypes = ['sync', 'conflict', 'resolution', 'rule_execution'] as const;
        const severities = ['info', 'warning', 'error'] as const;
        const workstreams = ['ws1', 'ws2', 'ws3', 'ws4', 'ws5'];
        const descriptions = [
          'Workstream synchronization completed successfully',
          'Coordination rule executed',
          'Data consistency check passed',
          'Cross-workstream communication established',
          'Conflict resolution applied'
        ];

        const newEvent: CoordinationEvent = {
          id: Date.now().toString(),
          timestamp: new Date(),
          type: eventTypes[Math.floor(Math.random() * eventTypes.length)],
          workstreams: [workstreams[Math.floor(Math.random() * workstreams.length)]],
          description: descriptions[Math.floor(Math.random() * descriptions.length)],
          severity: severities[Math.floor(Math.random() * severities.length)],
          resolved: Math.random() > 0.3
        };

        setCoordinationEvents(prev => [newEvent, ...prev.slice(0, 49)]);
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [autoCoordination]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'coordinated': return 'text-green-600 bg-green-100';
      case 'uncoordinated': return 'text-yellow-600 bg-yellow-100';
      case 'conflict': return 'text-red-600 bg-red-100';
      case 'maintenance': return 'text-blue-600 bg-blue-100';
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

  const getCoordinationLevelColor = (level: string) => {
    switch (level) {
      case 'autonomous': return 'text-green-600 bg-green-100';
      case 'supervised': return 'text-blue-600 bg-blue-100';
      case 'manual': return 'text-gray-600 bg-gray-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const toggleRule = useCallback((ruleId: string) => {
    setCoordinationRules(prev => prev.map(rule => 
      rule.id === ruleId ? { ...rule, enabled: !rule.enabled } : rule
    ));
  }, []);

  const executeRule = useCallback((ruleId: string) => {
    setCoordinationRules(prev => prev.map(rule => 
      rule.id === ruleId 
        ? { ...rule, lastTriggered: new Date(), executionCount: rule.executionCount + 1 }
        : rule
    ));

    const newEvent: CoordinationEvent = {
      id: Date.now().toString(),
      timestamp: new Date(),
      type: 'rule_execution',
      workstreams: ['ws1', 'ws2', 'ws3', 'ws4', 'ws5'],
      rule: ruleId,
      description: `Manual execution of coordination rule ${ruleId}`,
      severity: 'info',
      resolved: true
    };
    setCoordinationEvents(prev => [newEvent, ...prev.slice(0, 49)]);
  }, []);

  const filteredRules = useMemo(() => {
    return coordinationRules.filter(rule => {
      const matchesSearch = rule.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           rule.description.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesPriority = filterPriority === 'all' || rule.priority === filterPriority;
      return matchesSearch && matchesPriority;
    });
  }, [coordinationRules, searchTerm, filterPriority]);

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
                <Cpu className="w-8 h-8 text-blue-600" />
                Coordination Engine
              </h1>
              <p className="text-gray-600 mt-1">
                Advanced coordination and synchronization across ALL-USE workstreams
              </p>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium text-gray-700">Auto Coordination</label>
                <button
                  onClick={() => setAutoCoordination(!autoCoordination)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    autoCoordination ? 'bg-blue-600' : 'bg-gray-200'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      autoCoordination ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
              <button className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                <Sync className="w-4 h-4" />
                Force Sync
              </button>
            </div>
          </div>
        </div>

        {/* Coordination Overview Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-7 gap-4">
          <div className="bg-white rounded-lg shadow-sm border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Coordination Score</p>
                <p className="text-2xl font-bold text-green-600">{coordinationMetrics.averageScore}%</p>
              </div>
              <Target className="w-8 h-8 text-green-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Coordinated WS</p>
                <p className="text-2xl font-bold text-gray-900">{coordinationMetrics.coordinatedWorkstreams}/5</p>
              </div>
              <Layers className="w-8 h-8 text-blue-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Active Rules</p>
                <p className="text-2xl font-bold text-gray-900">{coordinationMetrics.activeRules}</p>
              </div>
              <GitBranch className="w-8 h-8 text-purple-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Active Syncs</p>
                <p className="text-2xl font-bold text-gray-900">{coordinationMetrics.activeSyncs}</p>
              </div>
              <Sync className="w-8 h-8 text-green-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Rule Executions</p>
                <p className="text-2xl font-bold text-gray-900">{coordinationMetrics.totalRuleExecutions.toLocaleString()}</p>
              </div>
              <Activity className="w-8 h-8 text-orange-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Avg Latency</p>
                <p className="text-2xl font-bold text-gray-900">{coordinationMetrics.averageLatency}ms</p>
              </div>
              <Clock className="w-8 h-8 text-yellow-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Conflicts</p>
                <p className="text-2xl font-bold text-red-600">{coordinationMetrics.totalConflicts}</p>
              </div>
              <AlertTriangle className="w-8 h-8 text-red-500" />
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="bg-white rounded-lg shadow-sm border">
          <div className="border-b">
            <nav className="flex space-x-8 px-6">
              {[
                { id: 'overview', label: 'Overview', icon: BarChart3 },
                { id: 'rules', label: 'Coordination Rules', icon: GitBranch },
                { id: 'sync', label: 'Sync Operations', icon: Sync },
                { id: 'events', label: 'Events', icon: Bell }
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
            {selectedTab === 'overview' && (
              <div className="space-y-6">
                <h3 className="text-lg font-semibold text-gray-900">Workstream Coordination Status</h3>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {workstreamCoordination.map((ws) => (
                    <div key={ws.workstreamId} className="border rounded-lg p-4">
                      <div className="flex items-center justify-between mb-4">
                        <div>
                          <h4 className="font-medium text-gray-900">{ws.name}</h4>
                          <div className="flex items-center gap-2 mt-1">
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(ws.status)}`}>
                              {ws.status.toUpperCase()}
                            </span>
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${getCoordinationLevelColor(ws.coordinationLevel)}`}>
                              {ws.coordinationLevel.toUpperCase()}
                            </span>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-lg font-bold text-green-600">{ws.metrics.coordinationScore}%</p>
                          <p className="text-xs text-gray-500">Score</p>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4 mb-4">
                        <div>
                          <p className="text-xs text-gray-500">Sync Latency</p>
                          <p className="font-medium">{ws.metrics.syncLatency}ms</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-500">Rule Executions</p>
                          <p className="font-medium">{ws.metrics.ruleExecutions}</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-500">Conflicts</p>
                          <p className="font-medium text-red-600">{ws.metrics.conflictCount}</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-500">Sync Frequency</p>
                          <p className="font-medium">{ws.syncFrequency}s</p>
                        </div>
                      </div>

                      {ws.dependencies.length > 0 && (
                        <div>
                          <p className="text-xs text-gray-500 mb-1">Dependencies</p>
                          <div className="flex gap-1">
                            {ws.dependencies.map(dep => (
                              <span key={dep} className="px-2 py-1 bg-gray-100 text-xs rounded">
                                {dep.toUpperCase()}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {selectedTab === 'rules' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900">Coordination Rules</h3>
                  <div className="flex items-center gap-4">
                    <div className="relative">
                      <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                      <input
                        type="text"
                        placeholder="Search rules..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                    <select
                      value={filterPriority}
                      onChange={(e) => setFilterPriority(e.target.value)}
                      className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="all">All Priorities</option>
                      <option value="critical">Critical</option>
                      <option value="high">High</option>
                      <option value="medium">Medium</option>
                      <option value="low">Low</option>
                    </select>
                  </div>
                </div>

                <div className="space-y-4">
                  {filteredRules.map((rule) => (
                    <div key={rule.id} className="border rounded-lg p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <h4 className="font-medium text-gray-900">{rule.name}</h4>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getPriorityColor(rule.priority)}`}>
                            {rule.priority.toUpperCase()}
                          </span>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                            rule.enabled ? 'text-green-600 bg-green-100' : 'text-gray-600 bg-gray-100'
                          }`}>
                            {rule.enabled ? 'ENABLED' : 'DISABLED'}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => toggleRule(rule.id)}
                            className={`px-3 py-1 text-xs rounded ${
                              rule.enabled 
                                ? 'bg-red-100 text-red-700 hover:bg-red-200' 
                                : 'bg-green-100 text-green-700 hover:bg-green-200'
                            }`}
                          >
                            {rule.enabled ? 'Disable' : 'Enable'}
                          </button>
                          <button
                            onClick={() => executeRule(rule.id)}
                            className="px-3 py-1 bg-blue-100 text-blue-700 text-xs rounded hover:bg-blue-200"
                          >
                            Execute
                          </button>
                        </div>
                      </div>

                      <p className="text-sm text-gray-600 mb-3">{rule.description}</p>

                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                        <div>
                          <p className="font-medium text-gray-700 mb-1">Trigger</p>
                          <p className="text-gray-600">
                            {rule.trigger.workstream.toUpperCase()}: {rule.trigger.event}
                          </p>
                          <p className="text-xs text-gray-500">{rule.trigger.condition}</p>
                        </div>
                        <div>
                          <p className="font-medium text-gray-700 mb-1">Actions ({rule.actions.length})</p>
                          {rule.actions.slice(0, 2).map((action, index) => (
                            <p key={index} className="text-gray-600 text-xs">
                              {action.workstream.toUpperCase()}: {action.action}
                            </p>
                          ))}
                          {rule.actions.length > 2 && (
                            <p className="text-xs text-gray-500">+{rule.actions.length - 2} more</p>
                          )}
                        </div>
                        <div>
                          <p className="font-medium text-gray-700 mb-1">Statistics</p>
                          <p className="text-gray-600 text-xs">
                            Executions: {rule.executionCount.toLocaleString()}
                          </p>
                          {rule.lastTriggered && (
                            <p className="text-gray-600 text-xs">
                              Last: {rule.lastTriggered.toLocaleTimeString()}
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {selectedTab === 'sync' && (
              <div className="space-y-6">
                <h3 className="text-lg font-semibold text-gray-900">Synchronization Operations</h3>
                <div className="space-y-4">
                  {syncOperations.map((sync) => (
                    <div key={sync.id} className="border rounded-lg p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <h4 className="font-medium text-gray-900">{sync.name}</h4>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                            sync.status === 'active' ? 'text-green-600 bg-green-100' :
                            sync.status === 'paused' ? 'text-yellow-600 bg-yellow-100' :
                            sync.status === 'error' ? 'text-red-600 bg-red-100' :
                            'text-gray-600 bg-gray-100'
                          }`}>
                            {sync.status.toUpperCase()}
                          </span>
                          <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded font-medium">
                            {sync.frequency.toUpperCase()}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <button className="px-3 py-1 bg-blue-100 text-blue-700 text-xs rounded hover:bg-blue-200">
                            <Play className="w-3 h-3" />
                          </button>
                          <button className="px-3 py-1 bg-yellow-100 text-yellow-700 text-xs rounded hover:bg-yellow-200">
                            <Pause className="w-3 h-3" />
                          </button>
                          <button className="px-3 py-1 bg-red-100 text-red-700 text-xs rounded hover:bg-red-200">
                            <Square className="w-3 h-3" />
                          </button>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <p className="font-medium text-gray-700 mb-1">Source</p>
                          <p className="text-gray-600">{sync.sourceWorkstream.toUpperCase()}</p>
                        </div>
                        <div>
                          <p className="font-medium text-gray-700 mb-1">Targets</p>
                          <div className="flex gap-1">
                            {sync.targetWorkstreams.map(target => (
                              <span key={target} className="px-1 py-0.5 bg-gray-100 text-xs rounded">
                                {target.toUpperCase()}
                              </span>
                            ))}
                          </div>
                        </div>
                        <div>
                          <p className="font-medium text-gray-700 mb-1">Data Type</p>
                          <p className="text-gray-600">{sync.dataType}</p>
                        </div>
                        <div>
                          <p className="font-medium text-gray-700 mb-1">Last Sync</p>
                          <p className="text-gray-600">{sync.lastSync.toLocaleTimeString()}</p>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-3 pt-3 border-t text-sm">
                        <div>
                          <p className="text-xs text-gray-500">Success Rate</p>
                          <p className="font-medium text-green-600">{sync.metrics.successRate}%</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-500">Avg Latency</p>
                          <p className="font-medium">{sync.metrics.averageLatency}ms</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-500">Data Volume</p>
                          <p className="font-medium">{sync.metrics.dataVolume.toLocaleString()}</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-500">Errors</p>
                          <p className="font-medium text-red-600">{sync.metrics.errorCount}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {selectedTab === 'events' && (
              <div className="space-y-6">
                <h3 className="text-lg font-semibold text-gray-900">Coordination Events</h3>
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {coordinationEvents.length === 0 ? (
                    <p className="text-gray-500 text-center py-8">No recent coordination events</p>
                  ) : (
                    coordinationEvents.map((event) => (
                      <div key={event.id} className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
                        <div className={`w-2 h-2 rounded-full mt-2 ${
                          event.severity === 'critical' ? 'bg-red-500' :
                          event.severity === 'error' ? 'bg-red-400' :
                          event.severity === 'warning' ? 'bg-yellow-500' :
                          'bg-blue-500'
                        }`} />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className={`px-2 py-1 rounded text-xs font-medium ${
                              event.type === 'sync' ? 'bg-blue-100 text-blue-800' :
                              event.type === 'conflict' ? 'bg-red-100 text-red-800' :
                              event.type === 'resolution' ? 'bg-green-100 text-green-800' :
                              event.type === 'rule_execution' ? 'bg-purple-100 text-purple-800' :
                              'bg-gray-100 text-gray-800'
                            }`}>
                              {event.type.replace('_', ' ').toUpperCase()}
                            </span>
                            {event.resolved && (
                              <CheckCircle className="w-4 h-4 text-green-500" />
                            )}
                          </div>
                          <p className="text-sm font-medium text-gray-900">{event.description}</p>
                          <div className="flex items-center gap-2 mt-1">
                            <div className="flex gap-1">
                              {event.workstreams.map(ws => (
                                <span key={ws} className="text-xs text-gray-500">{ws.toUpperCase()}</span>
                              ))}
                            </div>
                            <span className="text-xs text-gray-400">•</span>
                            <span className="text-xs text-gray-500">
                              {event.timestamp.toLocaleTimeString()}
                            </span>
                          </div>
                        </div>
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

export default CoordinationEngine;

