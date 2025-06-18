import React, { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, Calendar, Target, Activity, AlertTriangle, CheckCircle, ArrowUp, ArrowDown, DollarSign } from 'lucide-react';

interface WeekData {
  week: number;
  classification: 'Green' | 'Yellow' | 'Red';
  confidence: number;
  returns: number;
  trades: number;
  protocolCompliance: number;
}

interface PerformanceMetrics {
  totalReturn: number;
  weeklyAverage: number;
  monthlyAverage: number;
  yearlyProjection: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  protocolCompliance: number;
}

interface AnalyticsProps {
  className?: string;
  timeframe?: 'week' | 'month' | 'quarter' | 'year';
  onTimeframeChange?: (timeframe: string) => void;
}

export const Analytics: React.FC<AnalyticsProps> = ({
  className = '',
  timeframe = 'month',
  onTimeframeChange
}) => {
  const [selectedMetric, setSelectedMetric] = useState<'returns' | 'compliance' | 'trades'>('returns');
  const [showDetails, setShowDetails] = useState(false);

  // Sample data - in real implementation, this would come from props or API
  const weekData: WeekData[] = [
    { week: 1, classification: 'Green', confidence: 87, returns: 2.3, trades: 5, protocolCompliance: 94 },
    { week: 2, classification: 'Yellow', confidence: 72, returns: 1.1, trades: 3, protocolCompliance: 89 },
    { week: 3, classification: 'Green', confidence: 91, returns: 2.8, trades: 6, protocolCompliance: 96 },
    { week: 4, classification: 'Red', confidence: 83, returns: -0.5, trades: 1, protocolCompliance: 98 },
    { week: 5, classification: 'Green', confidence: 89, returns: 2.1, trades: 4, protocolCompliance: 92 },
    { week: 6, classification: 'Yellow', confidence: 76, returns: 0.8, trades: 2, protocolCompliance: 87 },
    { week: 7, classification: 'Green', confidence: 93, returns: 3.2, trades: 7, protocolCompliance: 97 },
    { week: 8, classification: 'Green', confidence: 88, returns: 2.5, trades: 5, protocolCompliance: 95 },
  ];

  const performanceMetrics: PerformanceMetrics = {
    totalReturn: 14.3,
    weeklyAverage: 1.79,
    monthlyAverage: 7.15,
    yearlyProjection: 93.4,
    sharpeRatio: 2.34,
    maxDrawdown: -2.1,
    winRate: 87.5,
    protocolCompliance: 93.5
  };

  const getClassificationColor = (classification: string): string => {
    switch (classification) {
      case 'Green': return 'bg-green-500';
      case 'Yellow': return 'bg-yellow-500';
      case 'Red': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getClassificationTextColor = (classification: string): string => {
    switch (classification) {
      case 'Green': return 'text-green-600';
      case 'Yellow': return 'text-yellow-600';
      case 'Red': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const formatPercentage = (value: number): string => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`;
  };

  const formatCurrency = (amount: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const getMetricValue = (week: WeekData): number => {
    switch (selectedMetric) {
      case 'returns': return week.returns;
      case 'compliance': return week.protocolCompliance;
      case 'trades': return week.trades;
      default: return week.returns;
    }
  };

  const getMetricLabel = (): string => {
    switch (selectedMetric) {
      case 'returns': return 'Returns (%)';
      case 'compliance': return 'Compliance (%)';
      case 'trades': return 'Trades';
      default: return 'Returns (%)';
    }
  };

  const currentWeek = weekData[weekData.length - 1];
  const previousWeek = weekData[weekData.length - 2];

  return (
    <div className={`bg-gray-50 ${className}`}>
      {/* Header */}
      <div className="bg-white border-b border-gray-200 p-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <h3 className="text-xl font-semibold text-gray-900">Performance Analytics</h3>
          
          <div className="flex flex-col sm:flex-row gap-3">
            {/* Metric Selector */}
            <div className="flex bg-gray-100 rounded-lg p-1">
              {[
                { key: 'returns', label: 'Returns', icon: TrendingUp },
                { key: 'compliance', label: 'Compliance', icon: CheckCircle },
                { key: 'trades', label: 'Trades', icon: Activity }
              ].map(({ key, label, icon: Icon }) => (
                <button
                  key={key}
                  onClick={() => setSelectedMetric(key as any)}
                  className={`flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    selectedMetric === key
                      ? 'bg-white text-blue-600 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {label}
                </button>
              ))}
            </div>

            {/* Timeframe Selector */}
            <div className="flex bg-gray-100 rounded-lg p-1">
              {[
                { key: 'week', label: 'Week' },
                { key: 'month', label: 'Month' },
                { key: 'quarter', label: 'Quarter' },
                { key: 'year', label: 'Year' }
              ].map(({ key, label }) => (
                <button
                  key={key}
                  onClick={() => onTimeframeChange?.(key)}
                  className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    timeframe === key
                      ? 'bg-white text-blue-600 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="p-6 space-y-6">
        {/* Current Week Status */}
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-lg font-semibold text-gray-900">Current Week Status</h4>
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${getClassificationColor(currentWeek.classification)}`}></div>
              <span className={`font-semibold ${getClassificationTextColor(currentWeek.classification)}`}>
                {currentWeek.classification} Week
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Week {currentWeek.week}</div>
              <div className="text-2xl font-bold text-gray-900">{currentWeek.classification}</div>
              <div className="text-sm text-gray-600">{currentWeek.confidence}% confidence</div>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Returns</div>
              <div className={`text-2xl font-bold ${currentWeek.returns >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {formatPercentage(currentWeek.returns)}
              </div>
              <div className="text-sm text-gray-600 flex items-center gap-1">
                {currentWeek.returns > previousWeek.returns ? (
                  <ArrowUp className="w-3 h-3 text-green-600" />
                ) : (
                  <ArrowDown className="w-3 h-3 text-red-600" />
                )}
                vs last week
              </div>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Trades</div>
              <div className="text-2xl font-bold text-gray-900">{currentWeek.trades}</div>
              <div className="text-sm text-gray-600">executed</div>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Protocol Compliance</div>
              <div className={`text-2xl font-bold ${
                currentWeek.protocolCompliance >= 95 ? 'text-green-600' : 
                currentWeek.protocolCompliance >= 85 ? 'text-yellow-600' : 'text-red-600'
              }`}>
                {currentWeek.protocolCompliance}%
              </div>
              <div className="text-sm text-gray-600">adherence</div>
            </div>
          </div>
        </div>

        {/* Performance Overview */}
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-lg font-semibold text-gray-900">Performance Overview</h4>
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="text-blue-600 hover:text-blue-700 text-sm font-medium"
            >
              {showDetails ? 'Hide Details' : 'Show Details'}
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="text-center">
              <div className="text-sm text-gray-600 mb-1">Total Return</div>
              <div className={`text-2xl font-bold ${performanceMetrics.totalReturn >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {formatPercentage(performanceMetrics.totalReturn)}
              </div>
              <div className="text-sm text-gray-600">8 weeks</div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-600 mb-1">Weekly Average</div>
              <div className={`text-2xl font-bold ${performanceMetrics.weeklyAverage >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {formatPercentage(performanceMetrics.weeklyAverage)}
              </div>
              <div className="text-sm text-gray-600">per week</div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-600 mb-1">Monthly Average</div>
              <div className={`text-2xl font-bold ${performanceMetrics.monthlyAverage >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {formatPercentage(performanceMetrics.monthlyAverage)}
              </div>
              <div className="text-sm text-gray-600">per month</div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-600 mb-1">Yearly Projection</div>
              <div className={`text-2xl font-bold ${performanceMetrics.yearlyProjection >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {formatPercentage(performanceMetrics.yearlyProjection)}
              </div>
              <div className="text-sm text-gray-600">annualized</div>
            </div>
          </div>

          {showDetails && (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-4 border-t border-gray-100">
              <div className="text-center">
                <div className="text-sm text-gray-600 mb-1">Sharpe Ratio</div>
                <div className="text-lg font-semibold text-gray-900">{performanceMetrics.sharpeRatio}</div>
                <div className="text-sm text-gray-600">risk-adjusted</div>
              </div>
              <div className="text-center">
                <div className="text-sm text-gray-600 mb-1">Max Drawdown</div>
                <div className="text-lg font-semibold text-red-600">{formatPercentage(performanceMetrics.maxDrawdown)}</div>
                <div className="text-sm text-gray-600">worst period</div>
              </div>
              <div className="text-center">
                <div className="text-sm text-gray-600 mb-1">Win Rate</div>
                <div className="text-lg font-semibold text-green-600">{performanceMetrics.winRate}%</div>
                <div className="text-sm text-gray-600">profitable weeks</div>
              </div>
              <div className="text-center">
                <div className="text-sm text-gray-600 mb-1">Avg Compliance</div>
                <div className="text-lg font-semibold text-blue-600">{performanceMetrics.protocolCompliance}%</div>
                <div className="text-sm text-gray-600">protocol adherence</div>
              </div>
            </div>
          )}
        </div>

        {/* Week Classification History */}
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-lg font-semibold text-gray-900">Week Classification History</h4>
            <div className="text-sm text-gray-600">Showing {getMetricLabel()}</div>
          </div>

          <div className="space-y-3">
            {weekData.map((week, index) => (
              <div key={week.week} className="flex items-center gap-4 p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center gap-3 min-w-0 flex-1">
                  <div className="text-sm font-medium text-gray-900 w-16">
                    Week {week.week}
                  </div>
                  <div className="flex items-center gap-2">
                    <div className={`w-3 h-3 rounded-full ${getClassificationColor(week.classification)}`}></div>
                    <span className={`text-sm font-medium ${getClassificationTextColor(week.classification)}`}>
                      {week.classification}
                    </span>
                  </div>
                  <div className="text-sm text-gray-600">
                    {week.confidence}% confidence
                  </div>
                </div>

                <div className="flex items-center gap-6">
                  <div className="text-right">
                    <div className="text-sm text-gray-600">Returns</div>
                    <div className={`font-semibold ${week.returns >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {formatPercentage(week.returns)}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-gray-600">Trades</div>
                    <div className="font-semibold text-gray-900">{week.trades}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-gray-600">Compliance</div>
                    <div className={`font-semibold ${
                      week.protocolCompliance >= 95 ? 'text-green-600' : 
                      week.protocolCompliance >= 85 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {week.protocolCompliance}%
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Risk Metrics */}
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h4 className="text-lg font-semibold text-gray-900 mb-4">Risk Analysis</h4>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-green-50 border border-green-200 p-4 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle className="w-5 h-5 text-green-600" />
                <span className="font-semibold text-green-900">Low Risk Periods</span>
              </div>
              <div className="text-2xl font-bold text-green-600">
                {weekData.filter(w => w.classification === 'Green').length}
              </div>
              <div className="text-sm text-green-700">
                {((weekData.filter(w => w.classification === 'Green').length / weekData.length) * 100).toFixed(0)}% of weeks
              </div>
            </div>

            <div className="bg-yellow-50 border border-yellow-200 p-4 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="w-5 h-5 text-yellow-600" />
                <span className="font-semibold text-yellow-900">Medium Risk Periods</span>
              </div>
              <div className="text-2xl font-bold text-yellow-600">
                {weekData.filter(w => w.classification === 'Yellow').length}
              </div>
              <div className="text-sm text-yellow-700">
                {((weekData.filter(w => w.classification === 'Yellow').length / weekData.length) * 100).toFixed(0)}% of weeks
              </div>
            </div>

            <div className="bg-red-50 border border-red-200 p-4 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="w-5 h-5 text-red-600" />
                <span className="font-semibold text-red-900">High Risk Periods</span>
              </div>
              <div className="text-2xl font-bold text-red-600">
                {weekData.filter(w => w.classification === 'Red').length}
              </div>
              <div className="text-sm text-red-700">
                {((weekData.filter(w => w.classification === 'Red').length / weekData.length) * 100).toFixed(0)}% of weeks
              </div>
            </div>
          </div>

          <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="text-sm text-blue-800">
              <strong>Risk Management:</strong> The ALL-USE protocol automatically adjusts position sizing and strategy based on week classification. 
              Green weeks allow for more aggressive positioning, while Red weeks trigger defensive protocols.
            </div>
          </div>
        </div>

        {/* Protocol Insights */}
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h4 className="text-lg font-semibold text-gray-900 mb-4">Protocol Insights</h4>
          
          <div className="space-y-4">
            <div className="flex items-start gap-3 p-4 bg-green-50 border border-green-200 rounded-lg">
              <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
              <div>
                <div className="font-semibold text-green-900">Strong Protocol Adherence</div>
                <div className="text-sm text-green-700">
                  Average compliance of {performanceMetrics.protocolCompliance}% indicates excellent protocol execution. 
                  This consistency is key to long-term success.
                </div>
              </div>
            </div>

            <div className="flex items-start gap-3 p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <Target className="w-5 h-5 text-blue-600 mt-0.5" />
              <div>
                <div className="font-semibold text-blue-900">Optimal Week Classification</div>
                <div className="text-sm text-blue-700">
                  {((weekData.filter(w => w.classification === 'Green').length / weekData.length) * 100).toFixed(0)}% Green weeks 
                  shows effective market timing and risk assessment capabilities.
                </div>
              </div>
            </div>

            <div className="flex items-start gap-3 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
              <Activity className="w-5 h-5 text-yellow-600 mt-0.5" />
              <div>
                <div className="font-semibold text-yellow-900">Trading Frequency</div>
                <div className="text-sm text-yellow-700">
                  Average of {(weekData.reduce((sum, w) => sum + w.trades, 0) / weekData.length).toFixed(1)} trades per week 
                  maintains active management while avoiding overtrading.
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

