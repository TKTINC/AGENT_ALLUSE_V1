import React, { useState, useEffect } from 'react';
import { 
  Shield, 
  AlertTriangle, 
  CheckCircle, 
  XCircle, 
  Clock,
  TrendingUp,
  TrendingDown,
  Activity,
  Settings,
  RefreshCw,
  Download,
  Eye,
  BarChart3,
  Target,
  Zap
} from 'lucide-react';
import { Button, Card, Modal, Tabs, ProgressBar } from '../advanced/UIComponents';
import { 
  useProtocolCompliance, 
  useAutomatedStrategies,
  ProtocolCompliance,
  AutomatedStrategy
} from '../../lib/ws2-integration';
import { formatCurrency, formatPercentage, formatDate } from '../../utils/chartUtils';

// Protocol Compliance Monitor
interface ProtocolComplianceMonitorProps {
  accountId: string;
  onViolationAction: (violation: any, action: string) => void;
}

export const ProtocolComplianceMonitor: React.FC<ProtocolComplianceMonitorProps> = ({
  accountId,
  onViolationAction
}) => {
  const { compliance, loading, error, refreshCompliance } = useProtocolCompliance(accountId);
  const [selectedViolation, setSelectedViolation] = useState<any>(null);

  const getComplianceColor = (score: number) => {
    if (score >= 90) return 'text-green-600 bg-green-100';
    if (score >= 70) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getClassificationColor = (classification: string) => {
    switch (classification) {
      case 'Green': return 'text-green-600 bg-green-100';
      case 'Yellow': return 'text-yellow-600 bg-yellow-100';
      case 'Red': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'High': return <XCircle className="w-5 h-5 text-red-600" />;
      case 'Medium': return <AlertTriangle className="w-5 h-5 text-yellow-600" />;
      case 'Low': return <Clock className="w-5 h-5 text-blue-600" />;
      default: return <CheckCircle className="w-5 h-5 text-gray-600" />;
    }
  };

  if (loading) {
    return (
      <Card className="p-6">
        <div className="text-center py-8 text-gray-500">Loading compliance data...</div>
      </Card>
    );
  }

  if (error || !compliance) {
    return (
      <Card className="p-6">
        <div className="text-center py-8 text-red-500">
          Error loading compliance data: {error}
        </div>
      </Card>
    );
  }

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-gray-900">Protocol Compliance</h3>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            icon={<RefreshCw className="w-4 h-4" />}
            onClick={refreshCompliance}
          >
            Refresh
          </Button>
          <Button
            variant="outline"
            size="sm"
            icon={<Download className="w-4 h-4" />}
          >
            Export Report
          </Button>
        </div>
      </div>

      {/* Compliance Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <Card className="p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Compliance Score</span>
            <Shield className="w-5 h-5 text-blue-600" />
          </div>
          <div className="text-2xl font-bold text-gray-900">
            {compliance.complianceScore.toFixed(1)}%
          </div>
          <div className={`text-sm px-2 py-1 rounded mt-1 ${
            getComplianceColor(compliance.complianceScore)
          }`}>
            {compliance.complianceScore >= 90 ? 'Excellent' :
             compliance.complianceScore >= 70 ? 'Good' : 'Needs Improvement'}
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Week Classification</span>
            <Target className="w-5 h-5 text-purple-600" />
          </div>
          <div className={`text-lg font-bold px-3 py-1 rounded-full text-center ${
            getClassificationColor(compliance.weekClassification)
          }`}>
            {compliance.weekClassification}
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Active Violations</span>
            <AlertTriangle className="w-5 h-5 text-orange-600" />
          </div>
          <div className="text-2xl font-bold text-gray-900">
            {compliance.violations.length}
          </div>
          <div className="text-sm text-gray-600">
            {compliance.violations.filter(v => v.severity === 'High').length} High Priority
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Sharpe Ratio</span>
            <TrendingUp className="w-5 h-5 text-green-600" />
          </div>
          <div className="text-2xl font-bold text-gray-900">
            {compliance.metrics.sharpeRatio.toFixed(2)}
          </div>
          <div className="text-sm text-gray-600">Risk-Adjusted Return</div>
        </Card>
      </div>

      {/* Performance Metrics */}
      <Card className="p-6 mb-6">
        <h4 className="text-lg font-semibold mb-4">Performance Metrics</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {formatPercentage(compliance.metrics.riskAdjustedReturn)}
            </div>
            <div className="text-sm text-gray-600">Risk-Adjusted Return</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-600">
              {formatPercentage(compliance.metrics.maxDrawdown)}
            </div>
            <div className="text-sm text-gray-600">Max Drawdown</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {compliance.metrics.sharpeRatio.toFixed(2)}
            </div>
            <div className="text-sm text-gray-600">Sharpe Ratio</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {formatPercentage(compliance.metrics.protocolAdherence)}
            </div>
            <div className="text-sm text-gray-600">Protocol Adherence</div>
          </div>
        </div>
      </Card>

      {/* Violations List */}
      {compliance.violations.length > 0 && (
        <Card className="p-6">
          <h4 className="text-lg font-semibold mb-4">Compliance Violations</h4>
          <div className="space-y-3">
            {compliance.violations.map((violation, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-3">
                    {getSeverityIcon(violation.severity)}
                    <span className="font-medium">{violation.type}</span>
                    <span className={`px-2 py-1 text-xs rounded ${
                      violation.severity === 'High' ? 'bg-red-100 text-red-700' :
                      violation.severity === 'Medium' ? 'bg-yellow-100 text-yellow-700' :
                      'bg-blue-100 text-blue-700'
                    }`}>
                      {violation.severity}
                    </span>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setSelectedViolation(violation)}
                  >
                    Details
                  </Button>
                </div>
                <p className="text-sm text-gray-700 mb-2">{violation.description}</p>
                <p className="text-sm text-blue-600">{violation.recommendation}</p>
              </div>
            ))}
          </div>
        </Card>
      )}

      {compliance.violations.length === 0 && (
        <Card className="p-6">
          <div className="text-center py-8">
            <CheckCircle className="w-16 h-16 text-green-600 mx-auto mb-4" />
            <h4 className="text-lg font-semibold text-gray-900 mb-2">All Clear!</h4>
            <p className="text-gray-600">No compliance violations detected. Your account is operating within all protocol guidelines.</p>
          </div>
        </Card>
      )}

      {/* Violation Details Modal */}
      {selectedViolation && (
        <Modal
          isOpen={!!selectedViolation}
          onClose={() => setSelectedViolation(null)}
          title={`Violation Details: ${selectedViolation.type}`}
          size="lg"
        >
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              {getSeverityIcon(selectedViolation.severity)}
              <span className={`px-3 py-1 text-sm font-medium rounded ${
                selectedViolation.severity === 'High' ? 'bg-red-100 text-red-700' :
                selectedViolation.severity === 'Medium' ? 'bg-yellow-100 text-yellow-700' :
                'bg-blue-100 text-blue-700'
              }`}>
                {selectedViolation.severity} Severity
              </span>
            </div>

            <div>
              <h4 className="font-medium mb-2">Description</h4>
              <p className="text-gray-700">{selectedViolation.description}</p>
            </div>

            <div>
              <h4 className="font-medium mb-2">Recommendation</h4>
              <p className="text-blue-600">{selectedViolation.recommendation}</p>
            </div>

            <div className="flex gap-3 pt-4">
              <Button
                variant="outline"
                fullWidth
                onClick={() => setSelectedViolation(null)}
              >
                Close
              </Button>
              <Button
                variant="primary"
                fullWidth
                onClick={() => {
                  onViolationAction(selectedViolation, 'resolve');
                  setSelectedViolation(null);
                }}
              >
                Take Action
              </Button>
            </div>
          </div>
        </Modal>
      )}
    </Card>
  );
};

// Automated Strategies Manager
interface AutomatedStrategiesManagerProps {
  onStrategyUpdate: (strategyId: string, parameters: any) => void;
}

export const AutomatedStrategiesManager: React.FC<AutomatedStrategiesManagerProps> = ({
  onStrategyUpdate
}) => {
  const { strategies, loading, error, updateStrategy, toggleStrategy, refreshStrategies } = useAutomatedStrategies();
  const [selectedStrategy, setSelectedStrategy] = useState<AutomatedStrategy | null>(null);
  const [showParametersModal, setShowParametersModal] = useState(false);

  const getPerformanceColor = (value: number, isPositive: boolean = true) => {
    if (isPositive) {
      return value > 0 ? 'text-green-600' : 'text-red-600';
    } else {
      return value < 0.1 ? 'text-green-600' : value < 0.2 ? 'text-yellow-600' : 'text-red-600';
    }
  };

  if (loading) {
    return (
      <Card className="p-6">
        <div className="text-center py-8 text-gray-500">Loading automated strategies...</div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="p-6">
        <div className="text-center py-8 text-red-500">
          Error loading strategies: {error}
        </div>
      </Card>
    );
  }

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-gray-900">Automated Strategies</h3>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            icon={<RefreshCw className="w-4 h-4" />}
            onClick={refreshStrategies}
          >
            Refresh
          </Button>
          <Button
            variant="primary"
            size="sm"
            icon={<Zap className="w-4 h-4" />}
          >
            New Strategy
          </Button>
        </div>
      </div>

      <div className="space-y-6">
        {strategies.map(strategy => (
          <Card key={strategy.id} className="p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className={`w-3 h-3 rounded-full ${
                  strategy.isActive ? 'bg-green-500' : 'bg-gray-400'
                }`} />
                <h4 className="text-lg font-semibold">{strategy.name}</h4>
                <span className={`px-2 py-1 text-xs rounded ${
                  strategy.isActive ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-700'
                }`}>
                  {strategy.isActive ? 'Active' : 'Inactive'}
                </span>
              </div>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  icon={<Settings className="w-4 h-4" />}
                  onClick={() => {
                    setSelectedStrategy(strategy);
                    setShowParametersModal(true);
                  }}
                >
                  Configure
                </Button>
                <Button
                  variant={strategy.isActive ? "destructive" : "primary"}
                  size="sm"
                  icon={strategy.isActive ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  onClick={() => toggleStrategy(strategy.id, !strategy.isActive)}
                >
                  {strategy.isActive ? 'Pause' : 'Start'}
                </Button>
              </div>
            </div>

            <p className="text-gray-600 mb-4">{strategy.description}</p>

            {/* Performance Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-4">
              <div className="text-center">
                <div className={`text-lg font-bold ${getPerformanceColor(strategy.performance.totalReturn)}`}>
                  {formatPercentage(strategy.performance.totalReturn)}
                </div>
                <div className="text-sm text-gray-600">Total Return</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-blue-600">
                  {formatPercentage(strategy.performance.winRate)}
                </div>
                <div className="text-sm text-gray-600">Win Rate</div>
              </div>
              <div className="text-center">
                <div className={`text-lg font-bold ${getPerformanceColor(strategy.performance.averageReturn)}`}>
                  {formatPercentage(strategy.performance.averageReturn)}
                </div>
                <div className="text-sm text-gray-600">Avg Return</div>
              </div>
              <div className="text-center">
                <div className={`text-lg font-bold ${getPerformanceColor(strategy.performance.maxDrawdown, false)}`}>
                  {formatPercentage(strategy.performance.maxDrawdown)}
                </div>
                <div className="text-sm text-gray-600">Max Drawdown</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-purple-600">
                  {strategy.performance.sharpeRatio.toFixed(2)}
                </div>
                <div className="text-sm text-gray-600">Sharpe Ratio</div>
              </div>
            </div>

            {/* Execution Schedule */}
            <div className="flex items-center justify-between text-sm text-gray-600">
              <div>
                Last executed: {formatDate(strategy.lastExecution, 'medium')} at {new Date(strategy.lastExecution).toLocaleTimeString()}
              </div>
              <div>
                Next execution: {formatDate(strategy.nextExecution, 'medium')} at {new Date(strategy.nextExecution).toLocaleTimeString()}
              </div>
            </div>
          </Card>
        ))}
      </div>

      {strategies.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          No automated strategies configured. Create your first strategy to get started.
        </div>
      )}

      {/* Strategy Parameters Modal */}
      {selectedStrategy && showParametersModal && (
        <Modal
          isOpen={showParametersModal}
          onClose={() => {
            setShowParametersModal(false);
            setSelectedStrategy(null);
          }}
          title={`Configure: ${selectedStrategy.name}`}
          size="lg"
        >
          <div className="space-y-4">
            <p className="text-gray-600">{selectedStrategy.description}</p>

            <div>
              <h4 className="font-medium mb-3">Strategy Parameters</h4>
              <div className="space-y-3">
                {Object.entries(selectedStrategy.parameters).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between">
                    <label className="text-sm font-medium text-gray-700 capitalize">
                      {key.replace(/([A-Z])/g, ' $1').trim()}:
                    </label>
                    <input
                      type={typeof value === 'number' ? 'number' : 'text'}
                      defaultValue={value.toString()}
                      className="w-32 px-3 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                ))}
              </div>
            </div>

            <div className="flex gap-3 pt-4">
              <Button
                variant="outline"
                fullWidth
                onClick={() => {
                  setShowParametersModal(false);
                  setSelectedStrategy(null);
                }}
              >
                Cancel
              </Button>
              <Button
                variant="primary"
                fullWidth
                onClick={() => {
                  // In a real implementation, collect the form values
                  onStrategyUpdate(selectedStrategy.id, selectedStrategy.parameters);
                  setShowParametersModal(false);
                  setSelectedStrategy(null);
                }}
              >
                Update Parameters
              </Button>
            </div>
          </div>
        </Modal>
      )}
    </Card>
  );
};

