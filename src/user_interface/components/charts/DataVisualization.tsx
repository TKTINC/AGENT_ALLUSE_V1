import React, { useState, useEffect, useMemo } from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
  ScatterChart,
  Scatter,
  RadialBarChart,
  RadialBar
} from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Percent, Calendar, Filter, Download, Maximize2 } from 'lucide-react';

// Enhanced Portfolio Performance Chart with multiple timeframes and metrics
interface PortfolioData {
  date: string;
  totalValue: number;
  generationAccount: number;
  revenueAccount: number;
  compoundingAccount: number;
  dailyReturn: number;
  cumulativeReturn: number;
  weekClassification: 'Green' | 'Yellow' | 'Red';
}

interface PortfolioPerformanceChartProps {
  data: PortfolioData[];
  timeframe: 'week' | 'month' | 'quarter' | 'year' | 'all';
  metric: 'value' | 'returns' | 'allocation';
  height?: number;
}

export const PortfolioPerformanceChart: React.FC<PortfolioPerformanceChartProps> = ({
  data,
  timeframe,
  metric,
  height = 400
}) => {
  const [selectedAccounts, setSelectedAccounts] = useState({
    generation: true,
    revenue: true,
    compounding: true,
    total: true
  });

  const filteredData = useMemo(() => {
    const now = new Date();
    const filterDate = new Date();
    
    switch (timeframe) {
      case 'week':
        filterDate.setDate(now.getDate() - 7);
        break;
      case 'month':
        filterDate.setMonth(now.getMonth() - 1);
        break;
      case 'quarter':
        filterDate.setMonth(now.getMonth() - 3);
        break;
      case 'year':
        filterDate.setFullYear(now.getFullYear() - 1);
        break;
      default:
        return data;
    }
    
    return data.filter(item => new Date(item.date) >= filterDate);
  }, [data, timeframe]);

  const formatValue = (value: number) => {
    if (metric === 'returns') {
      return `${(value * 100).toFixed(2)}%`;
    }
    return `$${value.toLocaleString()}`;
  };

  const formatTooltip = (value: any, name: string) => {
    const formattedValue = typeof value === 'number' ? formatValue(value) : value;
    const nameMap: Record<string, string> = {
      totalValue: 'Total Portfolio',
      generationAccount: 'Generation Account',
      revenueAccount: 'Revenue Account',
      compoundingAccount: 'Compounding Account',
      dailyReturn: 'Daily Return',
      cumulativeReturn: 'Cumulative Return'
    };
    return [formattedValue, nameMap[name] || name];
  };

  const getLineColor = (account: string) => {
    const colors = {
      totalValue: '#3B82F6',
      generationAccount: '#EF4444',
      revenueAccount: '#10B981',
      compoundingAccount: '#F59E0B',
      dailyReturn: '#8B5CF6',
      cumulativeReturn: '#06B6D4'
    };
    return colors[account as keyof typeof colors] || '#6B7280';
  };

  if (metric === 'value') {
    return (
      <div className="w-full">
        <div className="mb-4 flex flex-wrap gap-2">
          {Object.entries(selectedAccounts).map(([key, selected]) => (
            <button
              key={key}
              onClick={() => setSelectedAccounts(prev => ({ ...prev, [key]: !selected }))}
              className={`px-3 py-1 rounded-full text-sm font-medium transition-colors ${
                selected
                  ? 'bg-blue-100 text-blue-700 border border-blue-300'
                  : 'bg-gray-100 text-gray-600 border border-gray-300'
              }`}
            >
              {key.charAt(0).toUpperCase() + key.slice(1)} Account
            </button>
          ))}
        </div>
        
        <ResponsiveContainer width="100%" height={height}>
          <LineChart data={filteredData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
            <XAxis 
              dataKey="date" 
              stroke="#6B7280"
              fontSize={12}
              tickFormatter={(value) => new Date(value).toLocaleDateString()}
            />
            <YAxis 
              stroke="#6B7280"
              fontSize={12}
              tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
            />
            <Tooltip 
              formatter={formatTooltip}
              labelFormatter={(value) => new Date(value).toLocaleDateString()}
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #E5E7EB',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
              }}
            />
            <Legend />
            
            {selectedAccounts.total && (
              <Line
                type="monotone"
                dataKey="totalValue"
                stroke={getLineColor('totalValue')}
                strokeWidth={3}
                dot={{ fill: getLineColor('totalValue'), strokeWidth: 2, r: 4 }}
                activeDot={{ r: 6, stroke: getLineColor('totalValue'), strokeWidth: 2 }}
                name="Total Portfolio"
              />
            )}
            {selectedAccounts.generation && (
              <Line
                type="monotone"
                dataKey="generationAccount"
                stroke={getLineColor('generationAccount')}
                strokeWidth={2}
                dot={{ fill: getLineColor('generationAccount'), strokeWidth: 2, r: 3 }}
                name="Generation Account"
              />
            )}
            {selectedAccounts.revenue && (
              <Line
                type="monotone"
                dataKey="revenueAccount"
                stroke={getLineColor('revenueAccount')}
                strokeWidth={2}
                dot={{ fill: getLineColor('revenueAccount'), strokeWidth: 2, r: 3 }}
                name="Revenue Account"
              />
            )}
            {selectedAccounts.compounding && (
              <Line
                type="monotone"
                dataKey="compoundingAccount"
                stroke={getLineColor('compoundingAccount')}
                strokeWidth={2}
                dot={{ fill: getLineColor('compoundingAccount'), strokeWidth: 2, r: 3 }}
                name="Compounding Account"
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  }

  if (metric === 'returns') {
    return (
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={filteredData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
          <XAxis 
            dataKey="date" 
            stroke="#6B7280"
            fontSize={12}
            tickFormatter={(value) => new Date(value).toLocaleDateString()}
          />
          <YAxis 
            stroke="#6B7280"
            fontSize={12}
            tickFormatter={(value) => `${(value * 100).toFixed(1)}%`}
          />
          <Tooltip 
            formatter={formatTooltip}
            labelFormatter={(value) => new Date(value).toLocaleDateString()}
            contentStyle={{
              backgroundColor: 'white',
              border: '1px solid #E5E7EB',
              borderRadius: '8px',
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
            }}
          />
          <Legend />
          
          <Bar
            dataKey="dailyReturn"
            fill="#8B5CF6"
            name="Daily Return"
            opacity={0.7}
          />
          <Line
            type="monotone"
            dataKey="cumulativeReturn"
            stroke="#06B6D4"
            strokeWidth={3}
            dot={{ fill: '#06B6D4', strokeWidth: 2, r: 4 }}
            name="Cumulative Return"
          />
        </ComposedChart>
      </ResponsiveContainer>
    );
  }

  // Allocation pie chart
  const allocationData = filteredData.length > 0 ? [
    { name: 'Generation Account', value: filteredData[filteredData.length - 1].generationAccount, color: '#EF4444' },
    { name: 'Revenue Account', value: filteredData[filteredData.length - 1].revenueAccount, color: '#10B981' },
    { name: 'Compounding Account', value: filteredData[filteredData.length - 1].compoundingAccount, color: '#F59E0B' }
  ] : [];

  return (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart>
        <Pie
          data={allocationData}
          cx="50%"
          cy="50%"
          labelLine={false}
          label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
          outerRadius={120}
          fill="#8884d8"
          dataKey="value"
        >
          {allocationData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Pie>
        <Tooltip formatter={(value) => [`$${Number(value).toLocaleString()}`, 'Value']} />
      </PieChart>
    </ResponsiveContainer>
  );
};

// Week Classification History Chart
interface WeekClassificationData {
  week: number;
  classification: 'Green' | 'Yellow' | 'Red';
  return: number;
  trades: number;
  compliance: number;
}

interface WeekClassificationChartProps {
  data: WeekClassificationData[];
  height?: number;
}

export const WeekClassificationChart: React.FC<WeekClassificationChartProps> = ({
  data,
  height = 300
}) => {
  const getClassificationColor = (classification: string) => {
    switch (classification) {
      case 'Green': return '#10B981';
      case 'Yellow': return '#F59E0B';
      case 'Red': return '#EF4444';
      default: return '#6B7280';
    }
  };

  const chartData = data.map(item => ({
    ...item,
    color: getClassificationColor(item.classification)
  }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
        <XAxis 
          dataKey="week" 
          stroke="#6B7280"
          fontSize={12}
          tickFormatter={(value) => `Week ${value}`}
        />
        <YAxis 
          stroke="#6B7280"
          fontSize={12}
          tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
        />
        <Tooltip 
          formatter={(value, name) => {
            if (name === 'return') return [`${(Number(value) * 100).toFixed(2)}%`, 'Return'];
            if (name === 'compliance') return [`${(Number(value) * 100).toFixed(1)}%`, 'Compliance'];
            return [value, name];
          }}
          labelFormatter={(value) => `Week ${value}`}
          contentStyle={{
            backgroundColor: 'white',
            border: '1px solid #E5E7EB',
            borderRadius: '8px',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
          }}
        />
        <Legend />
        
        <Bar dataKey="return" name="Weekly Return">
          {chartData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
};

// Risk Analysis Heat Map
interface RiskData {
  account: string;
  timeframe: string;
  riskLevel: number;
  volatility: number;
  maxDrawdown: number;
}

interface RiskAnalysisHeatMapProps {
  data: RiskData[];
  height?: number;
}

export const RiskAnalysisHeatMap: React.FC<RiskAnalysisHeatMapProps> = ({
  data,
  height = 300
}) => {
  const getRiskColor = (riskLevel: number) => {
    if (riskLevel <= 0.3) return '#10B981'; // Low risk - Green
    if (riskLevel <= 0.6) return '#F59E0B'; // Medium risk - Yellow
    return '#EF4444'; // High risk - Red
  };

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
        <XAxis 
          type="number" 
          dataKey="volatility" 
          name="Volatility"
          stroke="#6B7280"
          fontSize={12}
          tickFormatter={(value) => `${(value * 100).toFixed(1)}%`}
        />
        <YAxis 
          type="number" 
          dataKey="maxDrawdown" 
          name="Max Drawdown"
          stroke="#6B7280"
          fontSize={12}
          tickFormatter={(value) => `${(value * 100).toFixed(1)}%`}
        />
        <Tooltip 
          cursor={{ strokeDasharray: '3 3' }}
          formatter={(value, name) => [
            `${(Number(value) * 100).toFixed(2)}%`,
            name === 'volatility' ? 'Volatility' : 'Max Drawdown'
          ]}
          labelFormatter={() => ''}
          contentStyle={{
            backgroundColor: 'white',
            border: '1px solid #E5E7EB',
            borderRadius: '8px',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
          }}
        />
        <Scatter name="Risk Analysis" data={data} fill="#8884d8">
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={getRiskColor(entry.riskLevel)} />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
};

// Trading Opportunities Visualization
interface TradingOpportunity {
  symbol: string;
  probability: number;
  expectedReturn: number;
  riskLevel: 'Low' | 'Medium' | 'High';
  timeframe: string;
  strategy: string;
}

interface TradingOpportunitiesChartProps {
  data: TradingOpportunity[];
  height?: number;
}

export const TradingOpportunitiesChart: React.FC<TradingOpportunitiesChartProps> = ({
  data,
  height = 400
}) => {
  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'Low': return '#10B981';
      case 'Medium': return '#F59E0B';
      case 'High': return '#EF4444';
      default: return '#6B7280';
    }
  };

  const chartData = data.map(item => ({
    ...item,
    color: getRiskColor(item.riskLevel),
    size: item.probability * 1000 // Scale for bubble size
  }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
        <XAxis 
          type="number" 
          dataKey="probability" 
          name="Probability"
          stroke="#6B7280"
          fontSize={12}
          tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
        />
        <YAxis 
          type="number" 
          dataKey="expectedReturn" 
          name="Expected Return"
          stroke="#6B7280"
          fontSize={12}
          tickFormatter={(value) => `${(value * 100).toFixed(1)}%`}
        />
        <Tooltip 
          cursor={{ strokeDasharray: '3 3' }}
          formatter={(value, name) => {
            if (name === 'probability') return [`${(Number(value) * 100).toFixed(1)}%`, 'Probability'];
            if (name === 'expectedReturn') return [`${(Number(value) * 100).toFixed(2)}%`, 'Expected Return'];
            return [value, name];
          }}
          labelFormatter={() => ''}
          contentStyle={{
            backgroundColor: 'white',
            border: '1px solid #E5E7EB',
            borderRadius: '8px',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
          }}
        />
        <Scatter name="Trading Opportunities" data={chartData} fill="#8884d8">
          {chartData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
};

// Performance Metrics Dashboard
interface PerformanceMetrics {
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  averageWin: number;
  averageLoss: number;
  profitFactor: number;
  volatility: number;
}

interface PerformanceMetricsDashboardProps {
  metrics: PerformanceMetrics;
  height?: number;
}

export const PerformanceMetricsDashboard: React.FC<PerformanceMetricsDashboardProps> = ({
  metrics,
  height = 300
}) => {
  const radarData = [
    { metric: 'Total Return', value: Math.min(metrics.totalReturn * 100, 100), fullMark: 100 },
    { metric: 'Sharpe Ratio', value: Math.min(metrics.sharpeRatio * 20, 100), fullMark: 100 },
    { metric: 'Win Rate', value: metrics.winRate * 100, fullMark: 100 },
    { metric: 'Profit Factor', value: Math.min(metrics.profitFactor * 20, 100), fullMark: 100 },
    { metric: 'Low Volatility', value: Math.max(100 - (metrics.volatility * 100), 0), fullMark: 100 },
    { metric: 'Low Drawdown', value: Math.max(100 - (Math.abs(metrics.maxDrawdown) * 100), 0), fullMark: 100 }
  ];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Radar Chart */}
      <div>
        <h3 className="text-lg font-semibold mb-4">Performance Overview</h3>
        <ResponsiveContainer width="100%" height={height}>
          <RadialBarChart cx="50%" cy="50%" innerRadius="10%" outerRadius="80%" data={radarData}>
            <RadialBar
              minAngle={15}
              label={{ position: 'insideStart', fill: '#fff' }}
              background
              clockWise
              dataKey="value"
              fill="#3B82F6"
            />
            <Legend iconSize={10} layout="vertical" verticalAlign="middle" align="right" />
            <Tooltip formatter={(value) => [`${Number(value).toFixed(1)}%`, 'Score']} />
          </RadialBarChart>
        </ResponsiveContainer>
      </div>

      {/* Metrics Grid */}
      <div>
        <h3 className="text-lg font-semibold mb-4">Key Metrics</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-white p-4 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Total Return</span>
              <TrendingUp className="w-4 h-4 text-green-600" />
            </div>
            <div className="text-2xl font-bold text-gray-900">
              {(metrics.totalReturn * 100).toFixed(2)}%
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Sharpe Ratio</span>
              <DollarSign className="w-4 h-4 text-blue-600" />
            </div>
            <div className="text-2xl font-bold text-gray-900">
              {metrics.sharpeRatio.toFixed(2)}
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Max Drawdown</span>
              <TrendingDown className="w-4 h-4 text-red-600" />
            </div>
            <div className="text-2xl font-bold text-gray-900">
              {(metrics.maxDrawdown * 100).toFixed(2)}%
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Win Rate</span>
              <Percent className="w-4 h-4 text-green-600" />
            </div>
            <div className="text-2xl font-bold text-gray-900">
              {(metrics.winRate * 100).toFixed(1)}%
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Profit Factor</span>
              <TrendingUp className="w-4 h-4 text-blue-600" />
            </div>
            <div className="text-2xl font-bold text-gray-900">
              {metrics.profitFactor.toFixed(2)}
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Volatility</span>
              <TrendingDown className="w-4 h-4 text-yellow-600" />
            </div>
            <div className="text-2xl font-bold text-gray-900">
              {(metrics.volatility * 100).toFixed(2)}%
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Chart Controls Component
interface ChartControlsProps {
  timeframe: string;
  onTimeframeChange: (timeframe: string) => void;
  metric: string;
  onMetricChange: (metric: string) => void;
  onExport?: () => void;
  onFullscreen?: () => void;
}

export const ChartControls: React.FC<ChartControlsProps> = ({
  timeframe,
  onTimeframeChange,
  metric,
  onMetricChange,
  onExport,
  onFullscreen
}) => {
  const timeframes = [
    { value: 'week', label: '1W' },
    { value: 'month', label: '1M' },
    { value: 'quarter', label: '3M' },
    { value: 'year', label: '1Y' },
    { value: 'all', label: 'All' }
  ];

  const metrics = [
    { value: 'value', label: 'Portfolio Value' },
    { value: 'returns', label: 'Returns' },
    { value: 'allocation', label: 'Allocation' }
  ];

  return (
    <div className="flex items-center justify-between mb-4 p-4 bg-gray-50 rounded-lg">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <Calendar className="w-4 h-4 text-gray-600" />
          <span className="text-sm font-medium text-gray-700">Timeframe:</span>
          <div className="flex bg-white rounded-lg border border-gray-300">
            {timeframes.map((tf) => (
              <button
                key={tf.value}
                onClick={() => onTimeframeChange(tf.value)}
                className={`px-3 py-1 text-sm font-medium transition-colors first:rounded-l-lg last:rounded-r-lg ${
                  timeframe === tf.value
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                {tf.label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-gray-600" />
          <span className="text-sm font-medium text-gray-700">Metric:</span>
          <select
            value={metric}
            onChange={(e) => onMetricChange(e.target.value)}
            className="px-3 py-1 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {metrics.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="flex items-center gap-2">
        {onExport && (
          <button
            onClick={onExport}
            className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-200 rounded-lg transition-colors"
            title="Export Chart"
          >
            <Download className="w-4 h-4" />
          </button>
        )}
        {onFullscreen && (
          <button
            onClick={onFullscreen}
            className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-200 rounded-lg transition-colors"
            title="Fullscreen"
          >
            <Maximize2 className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
};

