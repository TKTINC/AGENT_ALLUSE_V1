import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { DragDropContext, Droppable, Draggable, DropResult } from 'react-beautiful-dnd';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Percent, 
  Target, 
  AlertTriangle, 
  CheckCircle, 
  Settings, 
  RefreshCw,
  PieChart,
  BarChart3,
  Calculator,
  Zap,
  Shield,
  Activity
} from 'lucide-react';
import { Button, Card, Modal, ProgressBar } from '../advanced/UIComponents';
import { formatCurrency, formatPercentage, calculateReturn } from '../../utils/chartUtils';

// Enhanced Account Management Interface
interface Account {
  id: string;
  name: string;
  type: 'generation' | 'revenue' | 'compounding';
  balance: number;
  targetAllocation: number;
  currentAllocation: number;
  performance: {
    dailyReturn: number;
    weeklyReturn: number;
    monthlyReturn: number;
    yearlyReturn: number;
  };
  positions: Position[];
  riskLevel: 'Low' | 'Medium' | 'High';
  strategy: string;
  lastRebalanced: string;
}

interface Position {
  id: string;
  symbol: string;
  quantity: number;
  averagePrice: number;
  currentPrice: number;
  marketValue: number;
  unrealizedPnL: number;
  unrealizedPnLPercent: number;
  positionSize: number;
  entryDate: string;
  strategy: string;
  expirationDate?: string;
  strikePrice?: number;
  optionType?: 'call' | 'put';
}

interface PortfolioRebalancingProps {
  accounts: Account[];
  onRebalance: (newAllocations: Record<string, number>) => void;
  totalPortfolioValue: number;
}

export const PortfolioRebalancing: React.FC<PortfolioRebalancingProps> = ({
  accounts,
  onRebalance,
  totalPortfolioValue
}) => {
  const [allocations, setAllocations] = useState<Record<string, number>>({});
  const [isRebalancing, setIsRebalancing] = useState(false);
  const [showConfirmation, setShowConfirmation] = useState(false);

  useEffect(() => {
    const initialAllocations = accounts.reduce((acc, account) => {
      acc[account.id] = account.currentAllocation;
      return acc;
    }, {} as Record<string, number>);
    setAllocations(initialAllocations);
  }, [accounts]);

  const totalAllocation = Object.values(allocations).reduce((sum, allocation) => sum + allocation, 0);
  const isValidAllocation = Math.abs(totalAllocation - 100) < 0.01;

  const handleAllocationChange = (accountId: string, newAllocation: number) => {
    setAllocations(prev => ({
      ...prev,
      [accountId]: Math.max(0, Math.min(100, newAllocation))
    }));
  };

  const handleAutoRebalance = () => {
    const targetAllocations = accounts.reduce((acc, account) => {
      acc[account.id] = account.targetAllocation;
      return acc;
    }, {} as Record<string, number>);
    setAllocations(targetAllocations);
  };

  const handleRebalanceConfirm = async () => {
    setIsRebalancing(true);
    try {
      await onRebalance(allocations);
      setShowConfirmation(false);
    } catch (error) {
      console.error('Rebalancing failed:', error);
    } finally {
      setIsRebalancing(false);
    }
  };

  const getRebalanceActions = () => {
    return accounts.map(account => {
      const currentValue = (account.currentAllocation / 100) * totalPortfolioValue;
      const targetValue = (allocations[account.id] / 100) * totalPortfolioValue;
      const difference = targetValue - currentValue;
      const action = difference > 0 ? 'Buy' : 'Sell';
      const amount = Math.abs(difference);

      return {
        accountId: account.id,
        accountName: account.name,
        action,
        amount,
        percentage: Math.abs(difference / currentValue) * 100
      };
    }).filter(action => action.amount > 100); // Only show significant changes
  };

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-gray-900">Portfolio Rebalancing</h3>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleAutoRebalance}
            icon={<Target className="w-4 h-4" />}
          >
            Auto Rebalance
          </Button>
          <Button
            variant="primary"
            size="sm"
            onClick={() => setShowConfirmation(true)}
            disabled={!isValidAllocation}
            icon={<RefreshCw className="w-4 h-4" />}
          >
            Rebalance Portfolio
          </Button>
        </div>
      </div>

      {/* Allocation Controls */}
      <div className="space-y-4 mb-6">
        {accounts.map(account => (
          <div key={account.id} className="bg-gray-50 p-4 rounded-lg">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-3">
                <div className={`w-4 h-4 rounded-full ${
                  account.type === 'generation' ? 'bg-red-500' :
                  account.type === 'revenue' ? 'bg-green-500' : 'bg-yellow-500'
                }`} />
                <span className="font-medium text-gray-900">{account.name}</span>
                <span className="text-sm text-gray-500">
                  {formatCurrency(account.balance)}
                </span>
              </div>
              <div className="text-right">
                <div className="text-sm text-gray-600">
                  Target: {formatPercentage(account.targetAllocation / 100)}
                </div>
                <div className="text-sm text-gray-600">
                  Current: {formatPercentage(account.currentAllocation / 100)}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className="flex-1">
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="0.1"
                  value={allocations[account.id] || 0}
                  onChange={(e) => handleAllocationChange(account.id, parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>
              <div className="w-20">
                <input
                  type="number"
                  min="0"
                  max="100"
                  step="0.1"
                  value={allocations[account.id]?.toFixed(1) || '0.0'}
                  onChange={(e) => handleAllocationChange(account.id, parseFloat(e.target.value))}
                  className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <span className="text-sm text-gray-600 w-8">%</span>
            </div>

            <ProgressBar
              value={allocations[account.id] || 0}
              max={100}
              size="sm"
              color={
                account.type === 'generation' ? 'red' :
                account.type === 'revenue' ? 'green' : 'yellow'
              }
              className="mt-2"
            />
          </div>
        ))}
      </div>

      {/* Allocation Summary */}
      <div className="bg-blue-50 p-4 rounded-lg">
        <div className="flex items-center justify-between">
          <span className="font-medium text-blue-900">Total Allocation:</span>
          <span className={`font-bold ${
            isValidAllocation ? 'text-green-600' : 'text-red-600'
          }`}>
            {totalAllocation.toFixed(1)}%
          </span>
        </div>
        {!isValidAllocation && (
          <div className="flex items-center gap-2 mt-2 text-sm text-red-600">
            <AlertTriangle className="w-4 h-4" />
            <span>Total allocation must equal 100%</span>
          </div>
        )}
      </div>

      {/* Rebalancing Confirmation Modal */}
      <Modal
        isOpen={showConfirmation}
        onClose={() => setShowConfirmation(false)}
        title="Confirm Portfolio Rebalancing"
        size="lg"
      >
        <div className="space-y-4">
          <p className="text-gray-600">
            The following actions will be executed to rebalance your portfolio:
          </p>

          <div className="space-y-2">
            {getRebalanceActions().map(action => (
              <div key={action.accountId} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center gap-3">
                  <span className={`px-2 py-1 text-xs font-medium rounded ${
                    action.action === 'Buy' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                  }`}>
                    {action.action}
                  </span>
                  <span className="font-medium">{action.accountName}</span>
                </div>
                <div className="text-right">
                  <div className="font-medium">{formatCurrency(action.amount)}</div>
                  <div className="text-sm text-gray-500">
                    {formatPercentage(action.percentage / 100)} change
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="flex gap-3 pt-4">
            <Button
              variant="outline"
              fullWidth
              onClick={() => setShowConfirmation(false)}
            >
              Cancel
            </Button>
            <Button
              variant="primary"
              fullWidth
              loading={isRebalancing}
              onClick={handleRebalanceConfirm}
            >
              Confirm Rebalancing
            </Button>
          </div>
        </div>
      </Modal>
    </Card>
  );
};

// Position Management Dashboard
interface PositionManagementProps {
  positions: Position[];
  onClosePosition: (positionId: string) => void;
  onAdjustPosition: (positionId: string, adjustment: any) => void;
}

export const PositionManagement: React.FC<PositionManagementProps> = ({
  positions,
  onClosePosition,
  onAdjustPosition
}) => {
  const [selectedPosition, setSelectedPosition] = useState<Position | null>(null);
  const [sortBy, setSortBy] = useState<'symbol' | 'pnl' | 'size' | 'expiration'>('pnl');
  const [filterBy, setFilterBy] = useState<'all' | 'profitable' | 'losing' | 'expiring'>('all');

  const filteredAndSortedPositions = useMemo(() => {
    let filtered = positions;

    // Apply filters
    switch (filterBy) {
      case 'profitable':
        filtered = positions.filter(p => p.unrealizedPnL > 0);
        break;
      case 'losing':
        filtered = positions.filter(p => p.unrealizedPnL < 0);
        break;
      case 'expiring':
        filtered = positions.filter(p => {
          if (!p.expirationDate) return false;
          const daysToExpiry = Math.ceil(
            (new Date(p.expirationDate).getTime() - new Date().getTime()) / (1000 * 60 * 60 * 24)
          );
          return daysToExpiry <= 7;
        });
        break;
    }

    // Apply sorting
    return filtered.sort((a, b) => {
      switch (sortBy) {
        case 'symbol':
          return a.symbol.localeCompare(b.symbol);
        case 'pnl':
          return b.unrealizedPnL - a.unrealizedPnL;
        case 'size':
          return b.positionSize - a.positionSize;
        case 'expiration':
          if (!a.expirationDate && !b.expirationDate) return 0;
          if (!a.expirationDate) return 1;
          if (!b.expirationDate) return -1;
          return new Date(a.expirationDate).getTime() - new Date(b.expirationDate).getTime();
        default:
          return 0;
      }
    });
  }, [positions, sortBy, filterBy]);

  const totalUnrealizedPnL = positions.reduce((sum, position) => sum + position.unrealizedPnL, 0);
  const totalMarketValue = positions.reduce((sum, position) => sum + position.marketValue, 0);

  const getDaysToExpiration = (expirationDate?: string) => {
    if (!expirationDate) return null;
    const days = Math.ceil(
      (new Date(expirationDate).getTime() - new Date().getTime()) / (1000 * 60 * 60 * 24)
    );
    return days;
  };

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-gray-900">Position Management</h3>
        <div className="flex items-center gap-4">
          <div className="text-right">
            <div className="text-sm text-gray-600">Total P&L</div>
            <div className={`text-lg font-bold ${
              totalUnrealizedPnL >= 0 ? 'text-green-600' : 'text-red-600'
            }`}>
              {formatCurrency(totalUnrealizedPnL)}
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-600">Market Value</div>
            <div className="text-lg font-bold text-gray-900">
              {formatCurrency(totalMarketValue)}
            </div>
          </div>
        </div>
      </div>

      {/* Filters and Sorting */}
      <div className="flex items-center gap-4 mb-6">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-700">Filter:</span>
          <select
            value={filterBy}
            onChange={(e) => setFilterBy(e.target.value as any)}
            className="px-3 py-1 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Positions</option>
            <option value="profitable">Profitable</option>
            <option value="losing">Losing</option>
            <option value="expiring">Expiring Soon</option>
          </select>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-700">Sort by:</span>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="px-3 py-1 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="pnl">P&L</option>
            <option value="symbol">Symbol</option>
            <option value="size">Position Size</option>
            <option value="expiration">Expiration</option>
          </select>
        </div>
      </div>

      {/* Positions Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-200">
              <th className="text-left py-3 px-4 font-medium text-gray-700">Symbol</th>
              <th className="text-left py-3 px-4 font-medium text-gray-700">Quantity</th>
              <th className="text-left py-3 px-4 font-medium text-gray-700">Avg Price</th>
              <th className="text-left py-3 px-4 font-medium text-gray-700">Current Price</th>
              <th className="text-left py-3 px-4 font-medium text-gray-700">Market Value</th>
              <th className="text-left py-3 px-4 font-medium text-gray-700">P&L</th>
              <th className="text-left py-3 px-4 font-medium text-gray-700">Expiration</th>
              <th className="text-left py-3 px-4 font-medium text-gray-700">Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredAndSortedPositions.map(position => {
              const daysToExpiry = getDaysToExpiration(position.expirationDate);
              const isExpiringSoon = daysToExpiry !== null && daysToExpiry <= 7;

              return (
                <tr key={position.id} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{position.symbol}</span>
                      {position.optionType && (
                        <span className={`px-2 py-1 text-xs rounded ${
                          position.optionType === 'call' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                        }`}>
                          {position.optionType.toUpperCase()}
                        </span>
                      )}
                      {isExpiringSoon && (
                        <AlertTriangle className="w-4 h-4 text-orange-500" />
                      )}
                    </div>
                    {position.strikePrice && (
                      <div className="text-sm text-gray-500">
                        Strike: {formatCurrency(position.strikePrice)}
                      </div>
                    )}
                  </td>
                  <td className="py-3 px-4">{position.quantity}</td>
                  <td className="py-3 px-4">{formatCurrency(position.averagePrice)}</td>
                  <td className="py-3 px-4">{formatCurrency(position.currentPrice)}</td>
                  <td className="py-3 px-4">{formatCurrency(position.marketValue)}</td>
                  <td className="py-3 px-4">
                    <div className={`font-medium ${
                      position.unrealizedPnL >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {formatCurrency(position.unrealizedPnL)}
                    </div>
                    <div className={`text-sm ${
                      position.unrealizedPnLPercent >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {formatPercentage(position.unrealizedPnLPercent / 100)}
                    </div>
                  </td>
                  <td className="py-3 px-4">
                    {position.expirationDate ? (
                      <div>
                        <div className="text-sm">
                          {new Date(position.expirationDate).toLocaleDateString()}
                        </div>
                        {daysToExpiry !== null && (
                          <div className={`text-xs ${
                            isExpiringSoon ? 'text-orange-600' : 'text-gray-500'
                          }`}>
                            {daysToExpiry} days
                          </div>
                        )}
                      </div>
                    ) : (
                      <span className="text-gray-400">-</span>
                    )}
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setSelectedPosition(position)}
                      >
                        Details
                      </Button>
                      <Button
                        variant="destructive"
                        size="sm"
                        onClick={() => onClosePosition(position.id)}
                      >
                        Close
                      </Button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {filteredAndSortedPositions.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          No positions found matching the current filter.
        </div>
      )}

      {/* Position Details Modal */}
      {selectedPosition && (
        <Modal
          isOpen={!!selectedPosition}
          onClose={() => setSelectedPosition(null)}
          title={`Position Details - ${selectedPosition.symbol}`}
          size="lg"
        >
          <div className="space-y-6">
            {/* Position Overview */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-3">
                <div>
                  <span className="text-sm text-gray-600">Quantity:</span>
                  <div className="font-medium">{selectedPosition.quantity}</div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Average Price:</span>
                  <div className="font-medium">{formatCurrency(selectedPosition.averagePrice)}</div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Current Price:</span>
                  <div className="font-medium">{formatCurrency(selectedPosition.currentPrice)}</div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Market Value:</span>
                  <div className="font-medium">{formatCurrency(selectedPosition.marketValue)}</div>
                </div>
              </div>
              <div className="space-y-3">
                <div>
                  <span className="text-sm text-gray-600">Unrealized P&L:</span>
                  <div className={`font-medium ${
                    selectedPosition.unrealizedPnL >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {formatCurrency(selectedPosition.unrealizedPnL)}
                  </div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">P&L Percentage:</span>
                  <div className={`font-medium ${
                    selectedPosition.unrealizedPnLPercent >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {formatPercentage(selectedPosition.unrealizedPnLPercent / 100)}
                  </div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Entry Date:</span>
                  <div className="font-medium">
                    {new Date(selectedPosition.entryDate).toLocaleDateString()}
                  </div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Strategy:</span>
                  <div className="font-medium">{selectedPosition.strategy}</div>
                </div>
              </div>
            </div>

            {/* Option Details */}
            {selectedPosition.optionType && (
              <div className="border-t pt-4">
                <h4 className="font-medium mb-3">Option Details</h4>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <span className="text-sm text-gray-600">Type:</span>
                    <div className="font-medium capitalize">{selectedPosition.optionType}</div>
                  </div>
                  <div>
                    <span className="text-sm text-gray-600">Strike Price:</span>
                    <div className="font-medium">{formatCurrency(selectedPosition.strikePrice || 0)}</div>
                  </div>
                  <div>
                    <span className="text-sm text-gray-600">Expiration:</span>
                    <div className="font-medium">
                      {selectedPosition.expirationDate ? 
                        new Date(selectedPosition.expirationDate).toLocaleDateString() : 
                        'N/A'
                      }
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-3 pt-4">
              <Button
                variant="outline"
                fullWidth
                onClick={() => setSelectedPosition(null)}
              >
                Close
              </Button>
              <Button
                variant="primary"
                fullWidth
                onClick={() => {
                  onAdjustPosition(selectedPosition.id, { action: 'adjust' });
                  setSelectedPosition(null);
                }}
              >
                Adjust Position
              </Button>
              <Button
                variant="destructive"
                fullWidth
                onClick={() => {
                  onClosePosition(selectedPosition.id);
                  setSelectedPosition(null);
                }}
              >
                Close Position
              </Button>
            </div>
          </div>
        </Modal>
      )}
    </Card>
  );
};

