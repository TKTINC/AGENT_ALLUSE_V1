import React, { useState, useEffect } from 'react';
import { TrendingUp, DollarSign, BarChart3, Eye, EyeOff, Calendar, Target, AlertTriangle, CheckCircle, ArrowUp, ArrowDown, Activity } from 'lucide-react';

interface Account {
  id: string;
  name: string;
  type: 'generation' | 'revenue' | 'compounding';
  balance: number;
  weeklyReturn: number;
  monthlyReturn: number;
  yearlyReturn: number;
  riskLevel: 'high' | 'medium' | 'low';
  strategy: string;
  nextAction: string;
  lastTradeDate: string;
  targetDelta: number;
  currentDelta: number;
  protocolCompliance: number;
}

interface AccountVisualizationProps {
  className?: string;
  accounts?: Account[];
  onAccountSelect?: (account: Account) => void;
}

export const AccountVisualization: React.FC<AccountVisualizationProps> = ({
  className = '',
  accounts: propAccounts,
  onAccountSelect
}) => {
  const [viewMode, setViewMode] = useState<'overview' | 'detailed' | 'performance'>('overview');
  const [timeframe, setTimeframe] = useState<'week' | 'month' | 'year'>('month');
  const [showBalances, setShowBalances] = useState(true);
  const [selectedAccount, setSelectedAccount] = useState<Account | null>(null);

  // Default account data if none provided
  const defaultAccounts: Account[] = [
    {
      id: 'gen-001',
      name: 'Generation Account',
      type: 'generation',
      balance: 75000,
      weeklyReturn: 2.3,
      monthlyReturn: 8.7,
      yearlyReturn: 24.5,
      riskLevel: 'high',
      strategy: 'Premium harvesting with aggressive delta targeting for maximum income generation',
      nextAction: 'Monitor for Green Week confirmation, prepare for premium collection',
      lastTradeDate: '2025-06-15',
      targetDelta: 0.30,
      currentDelta: 0.28,
      protocolCompliance: 94.2
    },
    {
      id: 'rev-001',
      name: 'Revenue Account',
      type: 'revenue',
      balance: 60000,
      weeklyReturn: 1.8,
      monthlyReturn: 6.2,
      yearlyReturn: 18.9,
      riskLevel: 'medium',
      strategy: 'Balanced income generation with moderate risk exposure and consistent returns',
      nextAction: 'Execute covered call strategy on existing positions',
      lastTradeDate: '2025-06-14',
      targetDelta: 0.20,
      currentDelta: 0.22,
      protocolCompliance: 97.8
    },
    {
      id: 'comp-001',
      name: 'Compounding Account',
      type: 'compounding',
      balance: 90000,
      weeklyReturn: 1.2,
      monthlyReturn: 4.8,
      yearlyReturn: 15.2,
      riskLevel: 'low',
      strategy: 'Conservative growth with geometric compounding and capital preservation focus',
      nextAction: 'Reinvest dividends and maintain long-term positions',
      lastTradeDate: '2025-06-12',
      targetDelta: 0.15,
      currentDelta: 0.16,
      protocolCompliance: 99.1
    }
  ];

  const accounts = propAccounts || defaultAccounts;
  const totalPortfolioValue = accounts.reduce((sum, account) => sum + account.balance, 0);

  const formatCurrency = (amount: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const formatPercentage = (value: number): string => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`;
  };

  const getReturnValue = (account: Account): number => {
    switch (timeframe) {
      case 'week': return account.weeklyReturn;
      case 'month': return account.monthlyReturn;
      case 'year': return account.yearlyReturn;
      default: return account.monthlyReturn;
    }
  };

  const getReturnColor = (value: number): string => {
    if (value > 0) return 'text-green-600';
    if (value < 0) return 'text-red-600';
    return 'text-gray-600';
  };

  const getRiskColor = (level: string): string => {
    switch (level) {
      case 'high': return 'bg-red-100 text-red-800 border-red-200';
      case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'low': return 'bg-green-100 text-green-800 border-green-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getAccountTypeColor = (type: string): string => {
    switch (type) {
      case 'generation': return 'bg-blue-600';
      case 'revenue': return 'bg-green-600';
      case 'compounding': return 'bg-purple-600';
      default: return 'bg-gray-600';
    }
  };

  const getComplianceColor = (compliance: number): string => {
    if (compliance >= 95) return 'text-green-600';
    if (compliance >= 85) return 'text-yellow-600';
    return 'text-red-600';
  };

  const handleAccountClick = (account: Account) => {
    setSelectedAccount(selectedAccount?.id === account.id ? null : account);
    onAccountSelect?.(account);
  };

  const renderOverviewMode = () => (
    <div className="space-y-6">
      {/* Portfolio Summary */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-700 text-white p-6 rounded-lg">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold mb-2">Total Portfolio Value</h3>
            <div className="text-3xl font-bold">
              {showBalances ? formatCurrency(totalPortfolioValue) : '••••••'}
            </div>
            <div className="text-blue-100 mt-1">
              Across {accounts.length} active accounts
            </div>
          </div>
          <div className="text-right">
            <div className="text-blue-100 text-sm">This {timeframe}</div>
            <div className="text-xl font-semibold">
              {formatPercentage(
                accounts.reduce((sum, acc) => sum + getReturnValue(acc) * (acc.balance / totalPortfolioValue), 0)
              )}
            </div>
            <button
              onClick={() => setShowBalances(!showBalances)}
              className="mt-2 p-1 hover:bg-blue-500 rounded"
            >
              {showBalances ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </div>

      {/* Account Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {accounts.map((account) => (
          <div
            key={account.id}
            className={`bg-white border-2 rounded-lg p-6 cursor-pointer transition-all duration-200 hover:shadow-lg ${
              selectedAccount?.id === account.id ? 'border-blue-500 shadow-lg' : 'border-gray-200'
            }`}
            onClick={() => handleAccountClick(account)}
          >
            <div className="flex items-center justify-between mb-4">
              <div className={`w-3 h-3 rounded-full ${getAccountTypeColor(account.type)}`}></div>
              <span className={`px-2 py-1 text-xs font-medium rounded-full border ${getRiskColor(account.riskLevel)}`}>
                {account.riskLevel.toUpperCase()} RISK
              </span>
            </div>

            <h4 className="text-lg font-semibold text-gray-900 mb-2">{account.name}</h4>
            
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Balance</span>
                <span className="font-semibold">
                  {showBalances ? formatCurrency(account.balance) : '••••••'}
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-gray-600">Return ({timeframe})</span>
                <span className={`font-semibold ${getReturnColor(getReturnValue(account))}`}>
                  {formatPercentage(getReturnValue(account))}
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-gray-600">Protocol Compliance</span>
                <span className={`font-semibold ${getComplianceColor(account.protocolCompliance)}`}>
                  {account.protocolCompliance.toFixed(1)}%
                </span>
              </div>

              <div className="pt-2 border-t border-gray-100">
                <div className="text-sm text-gray-600">
                  Portfolio Weight: {((account.balance / totalPortfolioValue) * 100).toFixed(0)}%
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderDetailedMode = () => (
    <div className="space-y-6">
      {accounts.map((account) => (
        <div key={account.id} className="bg-white border border-gray-200 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className={`w-4 h-4 rounded-full ${getAccountTypeColor(account.type)}`}></div>
              <h4 className="text-xl font-semibold text-gray-900">{account.name}</h4>
              <span className={`px-2 py-1 text-xs font-medium rounded-full border ${getRiskColor(account.riskLevel)}`}>
                {account.riskLevel.toUpperCase()} RISK
              </span>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold">
                {showBalances ? formatCurrency(account.balance) : '••••••'}
              </div>
              <div className="text-sm text-gray-600">
                {((account.balance / totalPortfolioValue) * 100).toFixed(1)}% of portfolio
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Weekly Return</div>
              <div className={`text-lg font-semibold ${getReturnColor(account.weeklyReturn)}`}>
                {formatPercentage(account.weeklyReturn)}
              </div>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Monthly Return</div>
              <div className={`text-lg font-semibold ${getReturnColor(account.monthlyReturn)}`}>
                {formatPercentage(account.monthlyReturn)}
              </div>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Yearly Return</div>
              <div className={`text-lg font-semibold ${getReturnColor(account.yearlyReturn)}`}>
                {formatPercentage(account.yearlyReturn)}
              </div>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Protocol Compliance</div>
              <div className={`text-lg font-semibold ${getComplianceColor(account.protocolCompliance)}`}>
                {account.protocolCompliance.toFixed(1)}%
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div>
              <h5 className="font-semibold text-gray-900 mb-2">Strategy</h5>
              <p className="text-gray-700">{account.strategy}</p>
            </div>

            <div>
              <h5 className="font-semibold text-gray-900 mb-2">Next Action</h5>
              <div className="flex items-center gap-2">
                <Target className="w-4 h-4 text-blue-600" />
                <span className="text-gray-700">{account.nextAction}</span>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <h6 className="font-medium text-gray-900 mb-1">Last Trade</h6>
                <div className="flex items-center gap-2">
                  <Calendar className="w-4 h-4 text-gray-500" />
                  <span className="text-gray-700">{account.lastTradeDate}</span>
                </div>
              </div>
              <div>
                <h6 className="font-medium text-gray-900 mb-1">Target Delta</h6>
                <div className="flex items-center gap-2">
                  <Target className="w-4 h-4 text-gray-500" />
                  <span className="text-gray-700">{account.targetDelta.toFixed(2)}</span>
                </div>
              </div>
              <div>
                <h6 className="font-medium text-gray-900 mb-1">Current Delta</h6>
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-gray-500" />
                  <span className={`font-medium ${
                    Math.abs(account.currentDelta - account.targetDelta) <= 0.05 ? 'text-green-600' : 'text-yellow-600'
                  }`}>
                    {account.currentDelta.toFixed(2)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );

  const renderPerformanceMode = () => (
    <div className="space-y-6">
      {/* Performance Summary */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <h4 className="text-lg font-semibold text-gray-900 mb-4">Performance Overview</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {['week', 'month', 'year'].map((period) => (
            <div key={period} className="text-center">
              <div className="text-sm text-gray-600 mb-1">{period.charAt(0).toUpperCase() + period.slice(1)}ly</div>
              <div className={`text-2xl font-bold ${getReturnColor(
                accounts.reduce((sum, acc) => sum + acc[`${period}lyReturn` as keyof Account] as number * (acc.balance / totalPortfolioValue), 0)
              )}`}>
                {formatPercentage(
                  accounts.reduce((sum, acc) => sum + (acc[`${period}lyReturn` as keyof Account] as number) * (acc.balance / totalPortfolioValue), 0)
                )}
              </div>
              <div className="text-sm text-gray-600">
                {showBalances ? formatCurrency(
                  accounts.reduce((sum, acc) => sum + acc.balance * ((acc[`${period}lyReturn` as keyof Account] as number) / 100), 0)
                ) : '••••••'}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Account Performance Comparison */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <h4 className="text-lg font-semibold text-gray-900 mb-4">Account Performance Comparison</h4>
        <div className="space-y-4">
          {accounts.map((account) => (
            <div key={account.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center gap-3">
                <div className={`w-3 h-3 rounded-full ${getAccountTypeColor(account.type)}`}></div>
                <span className="font-medium">{account.name}</span>
                <span className={`px-2 py-1 text-xs rounded-full ${getRiskColor(account.riskLevel)}`}>
                  {account.riskLevel.toUpperCase()}
                </span>
              </div>
              <div className="flex items-center gap-6">
                <div className="text-right">
                  <div className="text-sm text-gray-600">Balance</div>
                  <div className="font-semibold">
                    {showBalances ? formatCurrency(account.balance) : '••••••'}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm text-gray-600">{timeframe.charAt(0).toUpperCase() + timeframe.slice(1)}ly Return</div>
                  <div className={`font-semibold ${getReturnColor(getReturnValue(account))}`}>
                    {formatPercentage(getReturnValue(account))}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm text-gray-600">Compliance</div>
                  <div className={`font-semibold ${getComplianceColor(account.protocolCompliance)}`}>
                    {account.protocolCompliance.toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Risk Analysis */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <h4 className="text-lg font-semibold text-gray-900 mb-4">Risk Analysis</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {['high', 'medium', 'low'].map((riskLevel) => {
            const riskAccounts = accounts.filter(acc => acc.riskLevel === riskLevel);
            const riskValue = riskAccounts.reduce((sum, acc) => sum + acc.balance, 0);
            const riskPercentage = (riskValue / totalPortfolioValue) * 100;
            
            return (
              <div key={riskLevel} className={`p-4 rounded-lg border ${getRiskColor(riskLevel)}`}>
                <div className="font-semibold mb-2">{riskLevel.charAt(0).toUpperCase() + riskLevel.slice(1)} Risk</div>
                <div className="text-2xl font-bold">
                  {showBalances ? formatCurrency(riskValue) : '••••••'}
                </div>
                <div className="text-sm opacity-75">
                  {riskPercentage.toFixed(1)}% of portfolio
                </div>
                <div className="text-sm mt-1">
                  {riskAccounts.length} account{riskAccounts.length !== 1 ? 's' : ''}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );

  return (
    <div className={`bg-gray-50 ${className}`}>
      {/* Header Controls */}
      <div className="bg-white border-b border-gray-200 p-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <h3 className="text-xl font-semibold text-gray-900">Account Portfolio</h3>
          
          <div className="flex flex-col sm:flex-row gap-3">
            {/* View Mode Selector */}
            <div className="flex bg-gray-100 rounded-lg p-1">
              {[
                { key: 'overview', label: 'Overview', icon: BarChart3 },
                { key: 'detailed', label: 'Detailed', icon: TrendingUp },
                { key: 'performance', label: 'Performance', icon: DollarSign }
              ].map(({ key, label, icon: Icon }) => (
                <button
                  key={key}
                  onClick={() => setViewMode(key as any)}
                  className={`flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    viewMode === key
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
                { key: 'year', label: 'Year' }
              ].map(({ key, label }) => (
                <button
                  key={key}
                  onClick={() => setTimeframe(key as any)}
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

      {/* Content */}
      <div className="p-6">
        {viewMode === 'overview' && renderOverviewMode()}
        {viewMode === 'detailed' && renderDetailedMode()}
        {viewMode === 'performance' && renderPerformanceMode()}
      </div>
    </div>
  );
};

