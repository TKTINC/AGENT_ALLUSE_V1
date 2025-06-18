import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { 
  Activity, 
  Database, 
  Brain, 
  TrendingUp, 
  BarChart3, 
  Users, 
  Settings, 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  Zap,
  Network,
  Cpu,
  HardDrive,
  Wifi,
  Shield,
  Eye,
  RefreshCw
} from 'lucide-react';

// Types for system integration
interface WorkstreamStatus {
  id: string;
  name: string;
  status: 'active' | 'idle' | 'error' | 'maintenance';
  health: number;
  lastUpdate: Date;
  metrics: {
    requests: number;
    latency: number;
    errors: number;
    uptime: number;
  };
  dependencies: string[];
  version: string;
}

interface IntegrationMetrics {
  totalRequests: number;
  averageLatency: number;
  errorRate: number;
  throughput: number;
  activeConnections: number;
  systemHealth: number;
}

interface DataFlow {
  source: string;
  target: string;
  type: 'real-time' | 'batch' | 'event';
  volume: number;
  latency: number;
  status: 'active' | 'paused' | 'error';
}

interface SystemEvent {
  id: string;
  timestamp: Date;
  type: 'info' | 'warning' | 'error' | 'success';
  source: string;
  message: string;
  details?: any;
}

// Mock data for development
const mockWorkstreams: WorkstreamStatus[] = [
  {
    id: 'ws1',
    name: 'Agent Foundation',
    status: 'active',
    health: 98.5,
    lastUpdate: new Date(),
    metrics: { requests: 15420, latency: 45, errors: 2, uptime: 99.8 },
    dependencies: [],
    version: '2.1.0'
  },
  {
    id: 'ws2',
    name: 'Protocol Systems',
    status: 'active',
    health: 97.2,
    lastUpdate: new Date(),
    metrics: { requests: 8930, latency: 32, errors: 1, uptime: 99.9 },
    dependencies: ['ws1'],
    version: '1.8.3'
  },
  {
    id: 'ws3',
    name: 'Market Intelligence',
    status: 'active',
    health: 96.8,
    lastUpdate: new Date(),
    metrics: { requests: 22150, latency: 78, errors: 5, uptime: 99.7 },
    dependencies: ['ws1', 'ws2'],
    version: '3.2.1'
  },
  {
    id: 'ws4',
    name: 'Market Integration',
    status: 'active',
    health: 99.1,
    lastUpdate: new Date(),
    metrics: { requests: 45680, latency: 28, errors: 1, uptime: 99.95 },
    dependencies: ['ws1', 'ws2', 'ws3'],
    version: '4.1.2'
  },
  {
    id: 'ws5',
    name: 'Learning Systems',
    status: 'active',
    health: 95.3,
    lastUpdate: new Date(),
    metrics: { requests: 12340, latency: 156, errors: 8, uptime: 99.2 },
    dependencies: ['ws1', 'ws2', 'ws3', 'ws4'],
    version: '1.5.7'
  }
];

const mockDataFlows: DataFlow[] = [
  { source: 'ws1', target: 'ws2', type: 'real-time', volume: 1250, latency: 15, status: 'active' },
  { source: 'ws2', target: 'ws3', type: 'event', volume: 890, latency: 22, status: 'active' },
  { source: 'ws3', target: 'ws4', type: 'real-time', volume: 2340, latency: 18, status: 'active' },
  { source: 'ws4', target: 'ws5', type: 'batch', volume: 567, latency: 45, status: 'active' },
  { source: 'ws1', target: 'ws5', type: 'event', volume: 234, latency: 35, status: 'active' }
];

const SystemIntegrationHub: React.FC = () => {
  const [workstreams, setWorkstreams] = useState<WorkstreamStatus[]>(mockWorkstreams);
  const [dataFlows, setDataFlows] = useState<DataFlow[]>(mockDataFlows);
  const [events, setEvents] = useState<SystemEvent[]>([]);
  const [selectedWorkstream, setSelectedWorkstream] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(5000);

  // Calculate integration metrics
  const integrationMetrics = useMemo((): IntegrationMetrics => {
    const totalRequests = workstreams.reduce((sum, ws) => sum + ws.metrics.requests, 0);
    const averageLatency = workstreams.reduce((sum, ws) => sum + ws.metrics.latency, 0) / workstreams.length;
    const totalErrors = workstreams.reduce((sum, ws) => sum + ws.metrics.errors, 0);
    const errorRate = (totalErrors / totalRequests) * 100;
    const throughput = totalRequests / 3600; // requests per second (assuming 1 hour window)
    const activeConnections = dataFlows.filter(df => df.status === 'active').length;
    const systemHealth = workstreams.reduce((sum, ws) => sum + ws.health, 0) / workstreams.length;

    return {
      totalRequests,
      averageLatency: Math.round(averageLatency),
      errorRate: Math.round(errorRate * 100) / 100,
      throughput: Math.round(throughput),
      activeConnections,
      systemHealth: Math.round(systemHealth * 10) / 10
    };
  }, [workstreams, dataFlows]);

  // Simulate real-time updates
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      setWorkstreams(prev => prev.map(ws => ({
        ...ws,
        lastUpdate: new Date(),
        health: Math.max(90, Math.min(100, ws.health + (Math.random() - 0.5) * 2)),
        metrics: {
          ...ws.metrics,
          requests: ws.metrics.requests + Math.floor(Math.random() * 100),
          latency: Math.max(10, ws.metrics.latency + (Math.random() - 0.5) * 10),
          errors: ws.metrics.errors + (Math.random() < 0.01 ? 1 : 0)
        }
      })));

      // Add random events
      if (Math.random() < 0.3) {
        const eventTypes = ['info', 'warning', 'error', 'success'] as const;
        const sources = ['ws1', 'ws2', 'ws3', 'ws4', 'ws5'];
        const messages = [
          'System health check completed',
          'Data synchronization in progress',
          'Performance optimization applied',
          'Cache refresh completed',
          'Connection pool optimized'
        ];

        const newEvent: SystemEvent = {
          id: Date.now().toString(),
          timestamp: new Date(),
          type: eventTypes[Math.floor(Math.random() * eventTypes.length)],
          source: sources[Math.floor(Math.random() * sources.length)],
          message: messages[Math.floor(Math.random() * messages.length)]
        };

        setEvents(prev => [newEvent, ...prev.slice(0, 49)]);
      }
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-600 bg-green-100';
      case 'idle': return 'text-yellow-600 bg-yellow-100';
      case 'error': return 'text-red-600 bg-red-100';
      case 'maintenance': return 'text-blue-600 bg-blue-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getHealthColor = (health: number) => {
    if (health >= 95) return 'text-green-600';
    if (health >= 85) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'info': return <Eye className="w-4 h-4 text-blue-500" />;
      case 'warning': return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
      case 'error': return <AlertTriangle className="w-4 h-4 text-red-500" />;
      case 'success': return <CheckCircle className="w-4 h-4 text-green-500" />;
      default: return <Activity className="w-4 h-4 text-gray-500" />;
    }
  };

  const restartWorkstream = useCallback((workstreamId: string) => {
    setWorkstreams(prev => prev.map(ws => 
      ws.id === workstreamId 
        ? { ...ws, status: 'active' as const, health: 100, metrics: { ...ws.metrics, errors: 0 } }
        : ws
    ));
    
    const newEvent: SystemEvent = {
      id: Date.now().toString(),
      timestamp: new Date(),
      type: 'success',
      source: workstreamId,
      message: 'Workstream restarted successfully'
    };
    setEvents(prev => [newEvent, ...prev.slice(0, 49)]);
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
                <Network className="w-8 h-8 text-blue-600" />
                System Integration Hub
              </h1>
              <p className="text-gray-600 mt-1">
                Comprehensive monitoring and coordination of ALL-USE workstreams
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
              <button
                onClick={() => window.location.reload()}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                Refresh
              </button>
            </div>
          </div>
        </div>

        {/* System Overview Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
          <div className="bg-white rounded-lg shadow-sm border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">System Health</p>
                <p className={`text-2xl font-bold ${getHealthColor(integrationMetrics.systemHealth)}`}>
                  {integrationMetrics.systemHealth}%
                </p>
              </div>
              <Shield className="w-8 h-8 text-green-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Requests</p>
                <p className="text-2xl font-bold text-gray-900">
                  {integrationMetrics.totalRequests.toLocaleString()}
                </p>
              </div>
              <Activity className="w-8 h-8 text-blue-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Avg Latency</p>
                <p className="text-2xl font-bold text-gray-900">
                  {integrationMetrics.averageLatency}ms
                </p>
              </div>
              <Clock className="w-8 h-8 text-yellow-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Error Rate</p>
                <p className="text-2xl font-bold text-red-600">
                  {integrationMetrics.errorRate}%
                </p>
              </div>
              <AlertTriangle className="w-8 h-8 text-red-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Throughput</p>
                <p className="text-2xl font-bold text-gray-900">
                  {integrationMetrics.throughput}/s
                </p>
              </div>
              <Zap className="w-8 h-8 text-purple-500" />
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Connections</p>
                <p className="text-2xl font-bold text-gray-900">
                  {integrationMetrics.activeConnections}
                </p>
              </div>
              <Wifi className="w-8 h-8 text-green-500" />
            </div>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Workstreams Status */}
          <div className="lg:col-span-2 bg-white rounded-lg shadow-sm border">
            <div className="p-6 border-b">
              <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                <Database className="w-5 h-5 text-blue-600" />
                Workstream Status
              </h2>
            </div>
            <div className="p-6">
              <div className="space-y-4">
                {workstreams.map((workstream) => (
                  <div
                    key={workstream.id}
                    className={`border rounded-lg p-4 cursor-pointer transition-all ${
                      selectedWorkstream === workstream.id 
                        ? 'border-blue-500 bg-blue-50' 
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => setSelectedWorkstream(
                      selectedWorkstream === workstream.id ? null : workstream.id
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(workstream.status)}`}>
                          {workstream.status.toUpperCase()}
                        </div>
                        <div>
                          <h3 className="font-medium text-gray-900">{workstream.name}</h3>
                          <p className="text-sm text-gray-500">v{workstream.version}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className={`text-lg font-bold ${getHealthColor(workstream.health)}`}>
                          {workstream.health}%
                        </p>
                        <p className="text-xs text-gray-500">
                          {workstream.lastUpdate.toLocaleTimeString()}
                        </p>
                      </div>
                    </div>

                    {selectedWorkstream === workstream.id && (
                      <div className="mt-4 pt-4 border-t border-gray-200">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div>
                            <p className="text-xs text-gray-500">Requests</p>
                            <p className="font-medium">{workstream.metrics.requests.toLocaleString()}</p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-500">Latency</p>
                            <p className="font-medium">{workstream.metrics.latency}ms</p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-500">Errors</p>
                            <p className="font-medium text-red-600">{workstream.metrics.errors}</p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-500">Uptime</p>
                            <p className="font-medium text-green-600">{workstream.metrics.uptime}%</p>
                          </div>
                        </div>
                        
                        {workstream.dependencies.length > 0 && (
                          <div className="mt-3">
                            <p className="text-xs text-gray-500 mb-1">Dependencies</p>
                            <div className="flex gap-2">
                              {workstream.dependencies.map(dep => (
                                <span key={dep} className="px-2 py-1 bg-gray-100 text-xs rounded">
                                  {dep.toUpperCase()}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        <div className="mt-3 flex gap-2">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              restartWorkstream(workstream.id);
                            }}
                            className="px-3 py-1 bg-blue-600 text-white text-xs rounded hover:bg-blue-700 transition-colors"
                          >
                            Restart
                          </button>
                          <button
                            onClick={(e) => e.stopPropagation()}
                            className="px-3 py-1 bg-gray-600 text-white text-xs rounded hover:bg-gray-700 transition-colors"
                          >
                            View Logs
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* System Events */}
          <div className="bg-white rounded-lg shadow-sm border">
            <div className="p-6 border-b">
              <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                <Activity className="w-5 h-5 text-blue-600" />
                System Events
              </h2>
            </div>
            <div className="p-6">
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {events.length === 0 ? (
                  <p className="text-gray-500 text-center py-8">No recent events</p>
                ) : (
                  events.map((event) => (
                    <div key={event.id} className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
                      {getEventIcon(event.type)}
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900">{event.message}</p>
                        <div className="flex items-center gap-2 mt-1">
                          <span className="text-xs text-gray-500">{event.source.toUpperCase()}</span>
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
          </div>
        </div>

        {/* Data Flow Visualization */}
        <div className="bg-white rounded-lg shadow-sm border">
          <div className="p-6 border-b">
            <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-blue-600" />
              Data Flow Monitoring
            </h2>
          </div>
          <div className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {dataFlows.map((flow, index) => (
                <div key={index} className="border rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded font-medium">
                        {flow.source.toUpperCase()}
                      </span>
                      <span className="text-gray-400">→</span>
                      <span className="px-2 py-1 bg-green-100 text-green-800 text-xs rounded font-medium">
                        {flow.target.toUpperCase()}
                      </span>
                    </div>
                    <div className={`w-2 h-2 rounded-full ${
                      flow.status === 'active' ? 'bg-green-500' : 
                      flow.status === 'paused' ? 'bg-yellow-500' : 'bg-red-500'
                    }`} />
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Type</span>
                      <span className="font-medium capitalize">{flow.type}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Volume</span>
                      <span className="font-medium">{flow.volume.toLocaleString()}/hr</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Latency</span>
                      <span className="font-medium">{flow.latency}ms</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemIntegrationHub;

