import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { 
  Users, 
  Shield, 
  Key, 
  Database, 
  Server, 
  Monitor, 
  BarChart3, 
  Settings, 
  Lock, 
  Unlock,
  UserCheck,
  UserX,
  Activity,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Download,
  Upload,
  FileText,
  Search,
  Filter,
  MoreHorizontal,
  Edit3,
  Trash2,
  Plus,
  Eye,
  EyeOff,
  RefreshCw,
  Bell,
  Mail,
  Phone,
  Globe,
  Calendar,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Percent,
  Target,
  Zap,
  Cpu,
  HardDrive,
  Wifi,
  WifiOff
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// Enterprise Types
export interface User {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  role: 'admin' | 'manager' | 'trader' | 'analyst' | 'viewer';
  department: string;
  status: 'active' | 'inactive' | 'suspended' | 'pending';
  lastLogin: number;
  createdAt: number;
  permissions: Permission[];
  preferences: Record<string, any>;
  avatar?: string;
  phone?: string;
  timezone: string;
  language: string;
}

export interface Permission {
  id: string;
  name: string;
  description: string;
  category: 'trading' | 'analytics' | 'admin' | 'data' | 'system';
  level: 'read' | 'write' | 'admin';
}

export interface Role {
  id: string;
  name: string;
  description: string;
  permissions: Permission[];
  isSystem: boolean;
  userCount: number;
}

export interface AuditLog {
  id: string;
  userId: string;
  userName: string;
  action: string;
  resource: string;
  details: Record<string, any>;
  timestamp: number;
  ipAddress: string;
  userAgent: string;
  success: boolean;
}

export interface SystemMetrics {
  cpu: {
    usage: number;
    cores: number;
    temperature: number;
  };
  memory: {
    used: number;
    total: number;
    percentage: number;
  };
  disk: {
    used: number;
    total: number;
    percentage: number;
  };
  network: {
    bytesIn: number;
    bytesOut: number;
    packetsIn: number;
    packetsOut: number;
  };
  uptime: number;
  timestamp: number;
}

export interface ComplianceReport {
  id: string;
  type: 'sox' | 'gdpr' | 'finra' | 'mifid' | 'custom';
  name: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  startTime: number;
  endTime?: number;
  findings: ComplianceFinding[];
  generatedBy: string;
}

export interface ComplianceFinding {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  category: string;
  description: string;
  recommendation: string;
  affectedUsers: string[];
  affectedSystems: string[];
  timestamp: number;
}

// Enterprise Dashboard Component
export const EnterpriseDashboard: React.FC<{
  className?: string;
}> = ({ className = '' }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'users' | 'security' | 'compliance' | 'system' | 'audit'>('overview');
  const [users, setUsers] = useState<User[]>([]);
  const [roles, setRoles] = useState<Role[]>([]);
  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([]);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [complianceReports, setComplianceReports] = useState<ComplianceReport[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Simulate data loading
  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      
      // Simulate API calls
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Mock data
      setUsers([
        {
          id: '1',
          email: 'admin@alluse.com',
          firstName: 'System',
          lastName: 'Administrator',
          role: 'admin',
          department: 'IT',
          status: 'active',
          lastLogin: Date.now() - 3600000,
          createdAt: Date.now() - 86400000 * 30,
          permissions: [],
          timezone: 'UTC',
          language: 'en-US'
        },
        {
          id: '2',
          email: 'trader@alluse.com',
          firstName: 'John',
          lastName: 'Trader',
          role: 'trader',
          department: 'Trading',
          status: 'active',
          lastLogin: Date.now() - 1800000,
          createdAt: Date.now() - 86400000 * 15,
          permissions: [],
          timezone: 'America/New_York',
          language: 'en-US'
        }
      ]);

      setRoles([
        {
          id: 'admin',
          name: 'Administrator',
          description: 'Full system access',
          permissions: [],
          isSystem: true,
          userCount: 1
        },
        {
          id: 'trader',
          name: 'Trader',
          description: 'Trading and portfolio management',
          permissions: [],
          isSystem: true,
          userCount: 5
        }
      ]);

      setSystemMetrics({
        cpu: { usage: 45, cores: 8, temperature: 65 },
        memory: { used: 8.2, total: 16, percentage: 51.25 },
        disk: { used: 120, total: 500, percentage: 24 },
        network: { bytesIn: 1024000, bytesOut: 512000, packetsIn: 1500, packetsOut: 1200 },
        uptime: 86400 * 7,
        timestamp: Date.now()
      });

      setComplianceReports([
        {
          id: '1',
          type: 'sox',
          name: 'SOX Compliance Check',
          description: 'Sarbanes-Oxley compliance verification',
          status: 'completed',
          progress: 100,
          startTime: Date.now() - 3600000,
          endTime: Date.now() - 1800000,
          findings: [],
          generatedBy: 'System'
        }
      ]);

      setIsLoading(false);
    };

    loadData();
  }, []);

  const formatBytes = useCallback((bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }, []);

  const formatUptime = useCallback((seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${days}d ${hours}h ${minutes}m`;
  }, []);

  if (isLoading) {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 ${className}`}>
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-lg ${className}`}>
      {/* Header */}
      <div className="border-b border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Shield className="w-6 h-6 text-blue-500" />
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
              Enterprise Administration
            </h2>
          </div>

          <div className="flex items-center gap-3">
            <button className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors">
              <Plus className="w-4 h-4" />
              Add User
            </button>
            
            <button className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200">
              <Settings className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex gap-1 mt-4">
          {([
            { id: 'overview', label: 'Overview', icon: Monitor },
            { id: 'users', label: 'Users', icon: Users },
            { id: 'security', label: 'Security', icon: Shield },
            { id: 'compliance', label: 'Compliance', icon: FileText },
            { id: 'system', label: 'System', icon: Server },
            { id: 'audit', label: 'Audit', icon: Activity }
          ] as const).map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                activeTab === id
                  ? 'bg-blue-500 text-white'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              <Icon className="w-4 h-4" />
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="p-6">
        <AnimatePresence mode="wait">
          {activeTab === 'overview' && (
            <OverviewPanel 
              key="overview"
              users={users}
              systemMetrics={systemMetrics}
              complianceReports={complianceReports}
            />
          )}
          {activeTab === 'users' && (
            <UsersPanel 
              key="users"
              users={users}
              roles={roles}
              onUpdateUser={(user) => {
                setUsers(prev => prev.map(u => u.id === user.id ? user : u));
              }}
            />
          )}
          {activeTab === 'security' && (
            <SecurityPanel 
              key="security"
              users={users}
              roles={roles}
            />
          )}
          {activeTab === 'compliance' && (
            <CompliancePanel 
              key="compliance"
              reports={complianceReports}
            />
          )}
          {activeTab === 'system' && (
            <SystemPanel 
              key="system"
              metrics={systemMetrics}
              formatBytes={formatBytes}
              formatUptime={formatUptime}
            />
          )}
          {activeTab === 'audit' && (
            <AuditPanel 
              key="audit"
              logs={auditLogs}
            />
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

// Overview Panel
const OverviewPanel: React.FC<{
  users: User[];
  systemMetrics: SystemMetrics | null;
  complianceReports: ComplianceReport[];
}> = ({ users, systemMetrics, complianceReports }) => {
  const stats = useMemo(() => {
    const activeUsers = users.filter(u => u.status === 'active').length;
    const pendingUsers = users.filter(u => u.status === 'pending').length;
    const suspendedUsers = users.filter(u => u.status === 'suspended').length;
    const completedReports = complianceReports.filter(r => r.status === 'completed').length;
    
    return {
      totalUsers: users.length,
      activeUsers,
      pendingUsers,
      suspendedUsers,
      completedReports,
      systemHealth: systemMetrics ? 
        (systemMetrics.cpu.usage < 80 && systemMetrics.memory.percentage < 80 ? 'healthy' : 'warning') : 'unknown'
    };
  }, [users, complianceReports, systemMetrics]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
      className="space-y-6"
    >
      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-blue-600 dark:text-blue-400">Total Users</p>
              <p className="text-2xl font-semibold text-blue-900 dark:text-blue-100">{stats.totalUsers}</p>
            </div>
            <Users className="w-8 h-8 text-blue-500" />
          </div>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-green-600 dark:text-green-400">Active Users</p>
              <p className="text-2xl font-semibold text-green-900 dark:text-green-100">{stats.activeUsers}</p>
            </div>
            <UserCheck className="w-8 h-8 text-green-500" />
          </div>
        </div>

        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-yellow-600 dark:text-yellow-400">System Health</p>
              <p className="text-2xl font-semibold text-yellow-900 dark:text-yellow-100 capitalize">{stats.systemHealth}</p>
            </div>
            <Activity className="w-8 h-8 text-yellow-500" />
          </div>
        </div>

        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-purple-600 dark:text-purple-400">Compliance Reports</p>
              <p className="text-2xl font-semibold text-purple-900 dark:text-purple-100">{stats.completedReports}</p>
            </div>
            <FileText className="w-8 h-8 text-purple-500" />
          </div>
        </div>
      </div>

      {/* System Status */}
      {systemMetrics && (
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">System Status</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600 dark:text-gray-400">CPU Usage</span>
                <span className="text-sm font-medium text-gray-900 dark:text-white">{systemMetrics.cpu.usage}%</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                <div 
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${systemMetrics.cpu.usage}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600 dark:text-gray-400">Memory Usage</span>
                <span className="text-sm font-medium text-gray-900 dark:text-white">{systemMetrics.memory.percentage.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                <div 
                  className="bg-green-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${systemMetrics.memory.percentage}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600 dark:text-gray-400">Disk Usage</span>
                <span className="text-sm font-medium text-gray-900 dark:text-white">{systemMetrics.disk.percentage}%</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                <div 
                  className="bg-yellow-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${systemMetrics.disk.percentage}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Recent Activity */}
      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Recent Activity</h3>
        <div className="space-y-3">
          <div className="flex items-center gap-3 p-3 bg-white dark:bg-gray-800 rounded-lg">
            <UserCheck className="w-5 h-5 text-green-500" />
            <div className="flex-1">
              <p className="text-sm text-gray-900 dark:text-white">New user registered</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">2 minutes ago</p>
            </div>
          </div>
          
          <div className="flex items-center gap-3 p-3 bg-white dark:bg-gray-800 rounded-lg">
            <Shield className="w-5 h-5 text-blue-500" />
            <div className="flex-1">
              <p className="text-sm text-gray-900 dark:text-white">Security scan completed</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">15 minutes ago</p>
            </div>
          </div>
          
          <div className="flex items-center gap-3 p-3 bg-white dark:bg-gray-800 rounded-lg">
            <FileText className="w-5 h-5 text-purple-500" />
            <div className="flex-1">
              <p className="text-sm text-gray-900 dark:text-white">Compliance report generated</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">1 hour ago</p>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

// Users Panel
const UsersPanel: React.FC<{
  users: User[];
  roles: Role[];
  onUpdateUser: (user: User) => void;
}> = ({ users, roles, onUpdateUser }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterRole, setFilterRole] = useState<string>('all');
  const [filterStatus, setFilterStatus] = useState<string>('all');

  const filteredUsers = useMemo(() => {
    return users.filter(user => {
      const matchesSearch = user.firstName.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           user.lastName.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           user.email.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesRole = filterRole === 'all' || user.role === filterRole;
      const matchesStatus = filterStatus === 'all' || user.status === filterStatus;
      
      return matchesSearch && matchesRole && matchesStatus;
    });
  }, [users, searchTerm, filterRole, filterStatus]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'inactive': return <XCircle className="w-4 h-4 text-gray-500" />;
      case 'suspended': return <XCircle className="w-4 h-4 text-red-500" />;
      case 'pending': return <Clock className="w-4 h-4 text-yellow-500" />;
      default: return <XCircle className="w-4 h-4 text-gray-500" />;
    }
  };

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'admin': return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400';
      case 'manager': return 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400';
      case 'trader': return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400';
      case 'analyst': return 'bg-purple-100 text-purple-800 dark:bg-purple-900/20 dark:text-purple-400';
      case 'viewer': return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400';
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
    >
      {/* Controls */}
      <div className="flex items-center gap-4 mb-6">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search users..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          />
        </div>

        <select
          value={filterRole}
          onChange={(e) => setFilterRole(e.target.value)}
          className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
        >
          <option value="all">All Roles</option>
          {roles.map(role => (
            <option key={role.id} value={role.id}>{role.name}</option>
          ))}
        </select>

        <select
          value={filterStatus}
          onChange={(e) => setFilterStatus(e.target.value)}
          className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
        >
          <option value="all">All Status</option>
          <option value="active">Active</option>
          <option value="inactive">Inactive</option>
          <option value="suspended">Suspended</option>
          <option value="pending">Pending</option>
        </select>
      </div>

      {/* Users Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <th className="text-left py-3 text-gray-600 dark:text-gray-400">User</th>
              <th className="text-left py-3 text-gray-600 dark:text-gray-400">Role</th>
              <th className="text-left py-3 text-gray-600 dark:text-gray-400">Department</th>
              <th className="text-left py-3 text-gray-600 dark:text-gray-400">Status</th>
              <th className="text-left py-3 text-gray-600 dark:text-gray-400">Last Login</th>
              <th className="text-center py-3 text-gray-600 dark:text-gray-400">Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredUsers.map((user) => (
              <motion.tr
                key={user.id}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="border-b border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                <td className="py-4">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white text-sm font-medium">
                      {user.firstName[0]}{user.lastName[0]}
                    </div>
                    <div>
                      <p className="font-medium text-gray-900 dark:text-white">
                        {user.firstName} {user.lastName}
                      </p>
                      <p className="text-gray-500 dark:text-gray-400">{user.email}</p>
                    </div>
                  </div>
                </td>
                <td className="py-4">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRoleColor(user.role)}`}>
                    {user.role}
                  </span>
                </td>
                <td className="py-4 text-gray-900 dark:text-white">{user.department}</td>
                <td className="py-4">
                  <div className="flex items-center gap-2">
                    {getStatusIcon(user.status)}
                    <span className="capitalize">{user.status}</span>
                  </div>
                </td>
                <td className="py-4 text-gray-600 dark:text-gray-400">
                  {new Date(user.lastLogin).toLocaleDateString()}
                </td>
                <td className="py-4">
                  <div className="flex items-center justify-center gap-1">
                    <button
                      className="p-1 text-blue-500 hover:text-blue-700"
                      title="Edit User"
                    >
                      <Edit3 className="w-4 h-4" />
                    </button>
                    <button
                      className="p-1 text-gray-500 hover:text-gray-700"
                      title="View Details"
                    >
                      <Eye className="w-4 h-4" />
                    </button>
                    <button
                      className="p-1 text-red-500 hover:text-red-700"
                      title="Suspend User"
                    >
                      <UserX className="w-4 h-4" />
                    </button>
                  </div>
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>

        {filteredUsers.length === 0 && (
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            No users found
          </div>
        )}
      </div>
    </motion.div>
  );
};

// Additional panels would be implemented similarly...
const SecurityPanel: React.FC<any> = () => <div>Security Panel</div>;
const CompliancePanel: React.FC<any> = () => <div>Compliance Panel</div>;
const SystemPanel: React.FC<any> = () => <div>System Panel</div>;
const AuditPanel: React.FC<any> = () => <div>Audit Panel</div>;

// Export all components
export {
  OverviewPanel,
  UsersPanel,
  SecurityPanel,
  CompliancePanel,
  SystemPanel,
  AuditPanel
};

