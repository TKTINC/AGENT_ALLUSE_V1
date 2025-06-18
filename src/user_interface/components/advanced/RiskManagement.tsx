import React, { useState, useEffect, useMemo } from 'react';
import { 
  Shield, 
  AlertTriangle, 
  TrendingDown, 
  Activity, 
  Calculator, 
  Target,
  BarChart3,
  PieChart,
  Zap,
  Settings,
  RefreshCw,
  Download,
  Eye,
  EyeOff
} from 'lucide-react';
import { Button, Card, Modal, ProgressBar, Tabs } from '../advanced/UIComponents';
import { 
  formatCurrency, 
  formatPercentage, 
  calculateSharpeRatio, 
  calculateMaxDrawdown, 
  calculateVolatility,
  calculateVaR,
  calculateBeta
} from '../../utils/chartUtils';

// Risk Management Tools
interface RiskMetrics {
  portfolioValue: number;
  dailyVaR: number;
  weeklyVaR: number;
  monthlyVaR: number;
  maxDrawdown: number;
  sharpeRatio: number;
  beta: number;
  volatility: number;
  correlationMatrix: Record<string, Record<string, number>>;
  concentrationRisk: {
    topPositions: Array<{
      symbol: string;
      allocation: number;
      risk: 'Low' | 'Medium' | 'High';
    }>;
    sectorConcentration: Record<string, number>;
  };
}

interface StressTestScenario {
  id: string;
  name: string;
  description: string;
  marketShock: number;
  volatilityIncrease: number;
  correlationIncrease: number;
  estimatedLoss: number;
  probability: number;
}

interface RiskManagementProps {
  riskMetrics: RiskMetrics;
  stressTestScenarios: StressTestScenario[];
  onRunStressTest: (scenarioId: string) => void;
  onUpdateRiskLimits: (limits: RiskLimits) => void;
}

interface RiskLimits {
  maxPortfolioVaR: number;
  maxPositionSize: number;
  maxSectorConcentration: number;
  minLiquidity: number;
  maxLeverage: number;
}

export const RiskManagement: React.FC<RiskManagementProps> = ({
  riskMetrics,
  stressTestScenarios,
  onRunStressTest,
  onUpdateRiskLimits
}) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedScenario, setSelectedScenario] = useState<StressTestScenario | null>(null);
  const [showRiskLimits, setShowRiskLimits] = useState(false);
  const [riskLimits, setRiskLimits] = useState<RiskLimits>({
    maxPortfolioVaR: 0.05,
    maxPositionSize: 0.1,
    maxSectorConcentration: 0.3,
    minLiquidity: 0.2,
    maxLeverage: 2.0
  });

  const getRiskLevel = (value: number, thresholds: { low: number; high: number }) => {
    if (value <= thresholds.low) return 'Low';
    if (value <= thresholds.high) return 'Medium';
    return 'High';
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'Low': return 'text-green-600 bg-green-100';
      case 'Medium': return 'text-yellow-600 bg-yellow-100';
      case 'High': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const tabs = [
    { id: 'overview', label: 'Risk Overview', icon: <Shield className="w-4 h-4" /> },
    { id: 'var', label: 'Value at Risk', icon: <TrendingDown className="w-4 h-4" /> },
    { id: 'stress', label: 'Stress Testing', icon: <AlertTriangle className="w-4 h-4" /> },
    { id: 'concentration', label: 'Concentration Risk', icon: <PieChart className="w-4 h-4" /> },
    { id: 'correlation', label: 'Correlation Analysis', icon: <Activity className="w-4 h-4" /> }
  ];

  const renderOverview = () => (
    <div className="space-y-6">
      {/* Risk Score Dashboard */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Portfolio Risk Score</span>
            <Shield className="w-5 h-5 text-blue-600" />
          </div>
          <div className="text-2xl font-bold text-gray-900">7.2/10</div>
          <div className="text-sm text-yellow-600">Medium Risk</div>
          <ProgressBar value={72} max={100} color="yellow" size="sm" className="mt-2" />
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Daily VaR (95%)</span>
            <TrendingDown className="w-5 h-5 text-red-600" />
          </div>
          <div className="text-2xl font-bold text-gray-900">
            {formatCurrency(riskMetrics.dailyVaR)}
          </div>
          <div className="text-sm text-gray-600">
            {formatPercentage(riskMetrics.dailyVaR / riskMetrics.portfolioValue)} of portfolio
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Max Drawdown</span>
            <AlertTriangle className="w-5 h-5 text-orange-600" />
          </div>
          <div className="text-2xl font-bold text-gray-900">
            {formatPercentage(riskMetrics.maxDrawdown)}
          </div>
          <div className={`text-sm px-2 py-1 rounded ${
            getRiskColor(getRiskLevel(riskMetrics.maxDrawdown, { low: 0.05, high: 0.15 }))
          }`}>
            {getRiskLevel(riskMetrics.maxDrawdown, { low: 0.05, high: 0.15 })} Risk
          </div>
        </Card>
      </div>

      {/* Key Risk Metrics */}
      <Card className="p-6">
        <h4 className="text-lg font-semibold mb-4">Key Risk Metrics</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">
              {riskMetrics.sharpeRatio.toFixed(2)}
            </div>
            <div className="text-sm text-gray-600">Sharpe Ratio</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">
              {riskMetrics.beta.toFixed(2)}
            </div>
            <div className="text-sm text-gray-600">Beta</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">
              {formatPercentage(riskMetrics.volatility)}
            </div>
            <div className="text-sm text-gray-600">Volatility</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">
              {riskMetrics.concentrationRisk.topPositions.length}
            </div>
            <div className="text-sm text-gray-600">Top Positions</div>
          </div>
        </div>
      </Card>

      {/* Risk Alerts */}
      <Card className="p-6">
        <h4 className="text-lg font-semibold mb-4">Risk Alerts</h4>
        <div className="space-y-3">
          {riskMetrics.dailyVaR / riskMetrics.portfolioValue > riskLimits.maxPortfolioVaR && (
            <div className="flex items-center gap-3 p-3 bg-red-50 border border-red-200 rounded-lg">
              <AlertTriangle className="w-5 h-5 text-red-600" />
              <div>
                <div className="font-medium text-red-800">High Portfolio VaR</div>
                <div className="text-sm text-red-600">
                  Daily VaR exceeds {formatPercentage(riskLimits.maxPortfolioVaR)} limit
                </div>
              </div>
            </div>
          )}
          
          {riskMetrics.concentrationRisk.topPositions.some(p => p.allocation > riskLimits.maxPositionSize) && (
            <div className="flex items-center gap-3 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
              <AlertTriangle className="w-5 h-5 text-yellow-600" />
              <div>
                <div className="font-medium text-yellow-800">Position Concentration Risk</div>
                <div className="text-sm text-yellow-600">
                  Some positions exceed {formatPercentage(riskLimits.maxPositionSize)} allocation limit
                </div>
              </div>
            </div>
          )}

          {Object.values(riskMetrics.concentrationRisk.sectorConcentration).some(c => c > riskLimits.maxSectorConcentration) && (
            <div className="flex items-center gap-3 p-3 bg-orange-50 border border-orange-200 rounded-lg">
              <AlertTriangle className="w-5 h-5 text-orange-600" />
              <div>
                <div className="font-medium text-orange-800">Sector Concentration Risk</div>
                <div className="text-sm text-orange-600">
                  Some sectors exceed {formatPercentage(riskLimits.maxSectorConcentration)} allocation limit
                </div>
              </div>
            </div>
          )}
        </div>
      </Card>
    </div>
  );

  const renderVaR = () => (
    <div className="space-y-6">
      <Card className="p-6">
        <h4 className="text-lg font-semibold mb-4">Value at Risk Analysis</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-3xl font-bold text-red-600 mb-2">
              {formatCurrency(riskMetrics.dailyVaR)}
            </div>
            <div className="text-sm text-gray-600 mb-1">Daily VaR (95%)</div>
            <div className="text-xs text-gray-500">
              Maximum expected loss in 1 day with 95% confidence
            </div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-orange-600 mb-2">
              {formatCurrency(riskMetrics.weeklyVaR)}
            </div>
            <div className="text-sm text-gray-600 mb-1">Weekly VaR (95%)</div>
            <div className="text-xs text-gray-500">
              Maximum expected loss in 1 week with 95% confidence
            </div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-yellow-600 mb-2">
              {formatCurrency(riskMetrics.monthlyVaR)}
            </div>
            <div className="text-sm text-gray-600 mb-1">Monthly VaR (95%)</div>
            <div className="text-xs text-gray-500">
              Maximum expected loss in 1 month with 95% confidence
            </div>
          </div>
        </div>
      </Card>

      <Card className="p-6">
        <h4 className="text-lg font-semibold mb-4">VaR Breakdown by Account</h4>
        <div className="space-y-4">
          {['Generation Account', 'Revenue Account', 'Compounding Account'].map((account, index) => {
            const varValue = [riskMetrics.dailyVaR * 0.4, riskMetrics.dailyVaR * 0.35, riskMetrics.dailyVaR * 0.25][index];
            const percentage = (varValue / riskMetrics.dailyVaR) * 100;
            
            return (
              <div key={account} className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className={`w-4 h-4 rounded-full ${
                    index === 0 ? 'bg-red-500' : index === 1 ? 'bg-green-500' : 'bg-yellow-500'
                  }`} />
                  <span className="font-medium">{account}</span>
                </div>
                <div className="text-right">
                  <div className="font-medium">{formatCurrency(varValue)}</div>
                  <div className="text-sm text-gray-500">{percentage.toFixed(1)}%</div>
                </div>
              </div>
            );
          })}
        </div>
      </Card>
    </div>
  );

  const renderStressTesting = () => (
    <div className="space-y-6">
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-lg font-semibold">Stress Test Scenarios</h4>
          <Button
            variant="primary"
            size="sm"
            icon={<RefreshCw className="w-4 h-4" />}
            onClick={() => stressTestScenarios.forEach(s => onRunStressTest(s.id))}
          >
            Run All Tests
          </Button>
        </div>
        
        <div className="space-y-4">
          {stressTestScenarios.map(scenario => (
            <div key={scenario.id} className="border border-gray-200 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h5 className="font-medium text-gray-900">{scenario.name}</h5>
                  <p className="text-sm text-gray-600">{scenario.description}</p>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setSelectedScenario(scenario)}
                >
                  View Details
                </Button>
              </div>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-gray-600">Market Shock:</span>
                  <div className="font-medium text-red-600">
                    {formatPercentage(scenario.marketShock)}
                  </div>
                </div>
                <div>
                  <span className="text-gray-600">Estimated Loss:</span>
                  <div className="font-medium text-red-600">
                    {formatCurrency(scenario.estimatedLoss)}
                  </div>
                </div>
                <div>
                  <span className="text-gray-600">Probability:</span>
                  <div className="font-medium">
                    {formatPercentage(scenario.probability)}
                  </div>
                </div>
                <div>
                  <span className="text-gray-600">Impact:</span>
                  <div className={`font-medium ${
                    Math.abs(scenario.estimatedLoss) / riskMetrics.portfolioValue > 0.1 
                      ? 'text-red-600' 
                      : Math.abs(scenario.estimatedLoss) / riskMetrics.portfolioValue > 0.05
                      ? 'text-yellow-600'
                      : 'text-green-600'
                  }`}>
                    {Math.abs(scenario.estimatedLoss) / riskMetrics.portfolioValue > 0.1 
                      ? 'High' 
                      : Math.abs(scenario.estimatedLoss) / riskMetrics.portfolioValue > 0.05
                      ? 'Medium'
                      : 'Low'
                    }
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );

  const renderConcentrationRisk = () => (
    <div className="space-y-6">
      <Card className="p-6">
        <h4 className="text-lg font-semibold mb-4">Position Concentration</h4>
        <div className="space-y-3">
          {riskMetrics.concentrationRisk.topPositions.map((position, index) => (
            <div key={position.symbol} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center gap-3">
                <span className="text-sm font-medium text-gray-500">#{index + 1}</span>
                <span className="font-medium">{position.symbol}</span>
                <span className={`px-2 py-1 text-xs rounded ${getRiskColor(position.risk)}`}>
                  {position.risk} Risk
                </span>
              </div>
              <div className="text-right">
                <div className="font-medium">{formatPercentage(position.allocation)}</div>
                <ProgressBar 
                  value={position.allocation * 100} 
                  max={100} 
                  size="sm" 
                  color={position.allocation > riskLimits.maxPositionSize ? 'red' : 'blue'}
                  className="w-20"
                />
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card className="p-6">
        <h4 className="text-lg font-semibold mb-4">Sector Concentration</h4>
        <div className="space-y-3">
          {Object.entries(riskMetrics.concentrationRisk.sectorConcentration).map(([sector, allocation]) => (
            <div key={sector} className="flex items-center justify-between">
              <span className="font-medium">{sector}</span>
              <div className="text-right">
                <div className="font-medium">{formatPercentage(allocation)}</div>
                <ProgressBar 
                  value={allocation * 100} 
                  max={100} 
                  size="sm" 
                  color={allocation > riskLimits.maxSectorConcentration ? 'red' : 'green'}
                  className="w-24"
                />
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );

  const renderCorrelationAnalysis = () => (
    <Card className="p-6">
      <h4 className="text-lg font-semibold mb-4">Asset Correlation Matrix</h4>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className="text-left p-2"></th>
              {Object.keys(riskMetrics.correlationMatrix).map(asset => (
                <th key={asset} className="text-center p-2 font-medium">{asset}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(riskMetrics.correlationMatrix).map(([asset1, correlations]) => (
              <tr key={asset1}>
                <td className="font-medium p-2">{asset1}</td>
                {Object.entries(correlations).map(([asset2, correlation]) => (
                  <td key={asset2} className="text-center p-2">
                    <span className={`px-2 py-1 rounded text-xs ${
                      Math.abs(correlation) > 0.7 
                        ? 'bg-red-100 text-red-700' 
                        : Math.abs(correlation) > 0.3
                        ? 'bg-yellow-100 text-yellow-700'
                        : 'bg-green-100 text-green-700'
                    }`}>
                      {correlation.toFixed(2)}
                    </span>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-4 text-sm text-gray-600">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-100 rounded"></div>
            <span>Low Correlation (&lt; 0.3)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-yellow-100 rounded"></div>
            <span>Medium Correlation (0.3 - 0.7)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-100 rounded"></div>
            <span>High Correlation (&gt; 0.7)</span>
          </div>
        </div>
      </div>
    </Card>
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold text-gray-900">Risk Management</h3>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            icon={<Settings className="w-4 h-4" />}
            onClick={() => setShowRiskLimits(true)}
          >
            Risk Limits
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

      <Tabs
        tabs={tabs}
        activeTab={activeTab}
        onChange={setActiveTab}
        variant="underline"
      />

      {activeTab === 'overview' && renderOverview()}
      {activeTab === 'var' && renderVaR()}
      {activeTab === 'stress' && renderStressTesting()}
      {activeTab === 'concentration' && renderConcentrationRisk()}
      {activeTab === 'correlation' && renderCorrelationAnalysis()}

      {/* Stress Test Details Modal */}
      {selectedScenario && (
        <Modal
          isOpen={!!selectedScenario}
          onClose={() => setSelectedScenario(null)}
          title={`Stress Test: ${selectedScenario.name}`}
          size="lg"
        >
          <div className="space-y-4">
            <p className="text-gray-600">{selectedScenario.description}</p>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <span className="text-sm text-gray-600">Market Shock:</span>
                <div className="text-lg font-bold text-red-600">
                  {formatPercentage(selectedScenario.marketShock)}
                </div>
              </div>
              <div>
                <span className="text-sm text-gray-600">Volatility Increase:</span>
                <div className="text-lg font-bold text-orange-600">
                  {formatPercentage(selectedScenario.volatilityIncrease)}
                </div>
              </div>
              <div>
                <span className="text-sm text-gray-600">Estimated Loss:</span>
                <div className="text-lg font-bold text-red-600">
                  {formatCurrency(selectedScenario.estimatedLoss)}
                </div>
              </div>
              <div>
                <span className="text-sm text-gray-600">Scenario Probability:</span>
                <div className="text-lg font-bold text-gray-900">
                  {formatPercentage(selectedScenario.probability)}
                </div>
              </div>
            </div>

            <div className="flex gap-3 pt-4">
              <Button
                variant="outline"
                fullWidth
                onClick={() => setSelectedScenario(null)}
              >
                Close
              </Button>
              <Button
                variant="primary"
                fullWidth
                onClick={() => {
                  onRunStressTest(selectedScenario.id);
                  setSelectedScenario(null);
                }}
              >
                Run This Test
              </Button>
            </div>
          </div>
        </Modal>
      )}

      {/* Risk Limits Modal */}
      <Modal
        isOpen={showRiskLimits}
        onClose={() => setShowRiskLimits(false)}
        title="Risk Limits Configuration"
        size="md"
      >
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Maximum Portfolio VaR (%)
            </label>
            <input
              type="number"
              step="0.1"
              min="0"
              max="20"
              value={(riskLimits.maxPortfolioVaR * 100).toFixed(1)}
              onChange={(e) => setRiskLimits(prev => ({
                ...prev,
                maxPortfolioVaR: parseFloat(e.target.value) / 100
              }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Maximum Position Size (%)
            </label>
            <input
              type="number"
              step="0.5"
              min="0"
              max="50"
              value={(riskLimits.maxPositionSize * 100).toFixed(1)}
              onChange={(e) => setRiskLimits(prev => ({
                ...prev,
                maxPositionSize: parseFloat(e.target.value) / 100
              }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Maximum Sector Concentration (%)
            </label>
            <input
              type="number"
              step="1"
              min="0"
              max="100"
              value={(riskLimits.maxSectorConcentration * 100).toFixed(0)}
              onChange={(e) => setRiskLimits(prev => ({
                ...prev,
                maxSectorConcentration: parseFloat(e.target.value) / 100
              }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="flex gap-3 pt-4">
            <Button
              variant="outline"
              fullWidth
              onClick={() => setShowRiskLimits(false)}
            >
              Cancel
            </Button>
            <Button
              variant="primary"
              fullWidth
              onClick={() => {
                onUpdateRiskLimits(riskLimits);
                setShowRiskLimits(false);
              }}
            >
              Update Limits
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
};

