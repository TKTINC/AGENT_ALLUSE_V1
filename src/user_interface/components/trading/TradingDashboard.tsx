import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { 
  Play, 
  Pause, 
  Square, 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Activity, 
  Clock, 
  CheckCircle, 
  XCircle, 
  AlertCircle,
  Edit3,
  Trash2,
  Plus,
  Filter,
  Search,
  BarChart3,
  PieChart,
  Target,
  Shield,
  Zap,
  Eye,
  Settings,
  RefreshCw,
  ArrowUp,
  ArrowDown,
  Minus
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  WS4MarketIntegration, 
  Order, 
  Position, 
  MarketData, 
  Account,
  Trade,
  OrderBook,
  createWS4Config,
  formatCurrency,
  formatPercent
} from '../../integrations/ws4/MarketIntegrationClient';

// Trading Dashboard Component
export const TradingDashboard: React.FC<{
  account?: string;
  className?: string;
}> = ({ account, className = '' }) => {
  const [ws4Client, setWs4Client] = useState<WS4MarketIntegration | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [activeTab, setActiveTab] = useState<'orders' | 'positions' | 'trades' | 'market'>('orders');
  const [orders, setOrders] = useState<Order[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [selectedAccount, setSelectedAccount] = useState<string>(account || '');

  // Initialize WS4 client
  useEffect(() => {
    const config = createWS4Config({
      defaultAccount: account || '',
      enableRealTimeData: true,
      enableOrderBook: true,
      enableTrades: true
    });

    const client = new WS4MarketIntegration(config);
    
    client.on('connected', () => setIsConnected(true));
    client.on('disconnected', () => setIsConnected(false));
    
    client.on('order_update', (order: Order) => {
      setOrders(prev => {
        const index = prev.findIndex(o => o.id === order.id);
        if (index >= 0) {
          const newOrders = [...prev];
          newOrders[index] = order;
          return newOrders;
        }
        return [order, ...prev];
      });
    });

    client.on('position_update', (position: Position) => {
      setPositions(prev => {
        const index = prev.findIndex(p => p.symbol === position.symbol);
        if (index >= 0) {
          const newPositions = [...prev];
          newPositions[index] = position;
          return newPositions;
        }
        return [position, ...prev];
      });
    });

    client.on('trade', (trade: Trade) => {
      setTrades(prev => [trade, ...prev.slice(0, 99)]);
    });

    client.on('market_data', (data: MarketData) => {
      setMarketData(prev => {
        const index = prev.findIndex(m => m.symbol === data.symbol);
        if (index >= 0) {
          const newData = [...prev];
          newData[index] = data;
          return newData;
        }
        return [data, ...prev];
      });
    });

    client.on('account_update', (account: Account) => {
      setAccounts(prev => {
        const index = prev.findIndex(a => a.id === account.id);
        if (index >= 0) {
          const newAccounts = [...prev];
          newAccounts[index] = account;
          return newAccounts;
        }
        return [account, ...prev];
      });
    });

    setWs4Client(client);

    return () => {
      client.disconnect();
    };
  }, [account]);

  const connectionStatus = useMemo(() => {
    if (!isConnected) return { color: 'text-red-500', bg: 'bg-red-500', text: 'Disconnected' };
    return { color: 'text-green-500', bg: 'bg-green-500', text: 'Live Trading' };
  }, [isConnected]);

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-lg ${className}`}>
      {/* Header */}
      <div className="border-b border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Activity className="w-6 h-6 text-blue-500" />
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
              Trading Dashboard
            </h2>
            <div className={`flex items-center gap-1 ${connectionStatus.color}`}>
              <div className={`w-2 h-2 rounded-full ${connectionStatus.bg}`} />
              <span className="text-xs font-medium">{connectionStatus.text}</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {accounts.length > 0 && (
              <select
                value={selectedAccount}
                onChange={(e) => setSelectedAccount(e.target.value)}
                className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                {accounts.map(acc => (
                  <option key={acc.id} value={acc.id}>
                    {acc.name} ({formatCurrency(acc.totalValue)})
                  </option>
                ))}
              </select>
            )}
            
            <button className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200">
              <Settings className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex gap-1 mt-4">
          {(['orders', 'positions', 'trades', 'market'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                activeTab === tab
                  ? 'bg-blue-500 text-white'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="p-6">
        <AnimatePresence mode="wait">
          {activeTab === 'orders' && (
            <OrdersPanel 
              key="orders"
              orders={orders} 
              ws4Client={ws4Client}
              account={selectedAccount}
            />
          )}
          {activeTab === 'positions' && (
            <PositionsPanel 
              key="positions"
              positions={positions}
              marketData={marketData}
            />
          )}
          {activeTab === 'trades' && (
            <TradesPanel 
              key="trades"
              trades={trades}
            />
          )}
          {activeTab === 'market' && (
            <MarketDataPanel 
              key="market"
              marketData={marketData}
              ws4Client={ws4Client}
            />
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

// Orders Panel
const OrdersPanel: React.FC<{
  orders: Order[];
  ws4Client: WS4MarketIntegration | null;
  account: string;
}> = ({ orders, ws4Client, account }) => {
  const [showOrderForm, setShowOrderForm] = useState(false);
  const [filterStatus, setFilterStatus] = useState<string>('all');

  const filteredOrders = useMemo(() => {
    if (filterStatus === 'all') return orders;
    return orders.filter(order => order.status === filterStatus);
  }, [orders, filterStatus]);

  const handleCancelOrder = useCallback(async (orderId: string) => {
    if (!ws4Client) return;
    
    try {
      await ws4Client.cancelOrder(orderId);
    } catch (error) {
      console.error('Failed to cancel order:', error);
    }
  }, [ws4Client]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'filled': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'cancelled': return <XCircle className="w-4 h-4 text-red-500" />;
      case 'rejected': return <XCircle className="w-4 h-4 text-red-500" />;
      case 'open': return <Clock className="w-4 h-4 text-blue-500" />;
      case 'pending': return <AlertCircle className="w-4 h-4 text-yellow-500" />;
      default: return <Minus className="w-4 h-4 text-gray-500" />;
    }
  };

  const getSideColor = (side: string) => {
    return side === 'buy' ? 'text-green-600' : 'text-red-600';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
    >
      {/* Controls */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          >
            <option value="all">All Orders</option>
            <option value="open">Open</option>
            <option value="filled">Filled</option>
            <option value="cancelled">Cancelled</option>
            <option value="pending">Pending</option>
          </select>
          
          <span className="text-sm text-gray-500 dark:text-gray-400">
            {filteredOrders.length} orders
          </span>
        </div>

        <button
          onClick={() => setShowOrderForm(true)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
        >
          <Plus className="w-4 h-4" />
          New Order
        </button>
      </div>

      {/* Orders Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <th className="text-left py-2 text-gray-600 dark:text-gray-400">Status</th>
              <th className="text-left py-2 text-gray-600 dark:text-gray-400">Symbol</th>
              <th className="text-left py-2 text-gray-600 dark:text-gray-400">Side</th>
              <th className="text-left py-2 text-gray-600 dark:text-gray-400">Type</th>
              <th className="text-right py-2 text-gray-600 dark:text-gray-400">Quantity</th>
              <th className="text-right py-2 text-gray-600 dark:text-gray-400">Price</th>
              <th className="text-right py-2 text-gray-600 dark:text-gray-400">Filled</th>
              <th className="text-left py-2 text-gray-600 dark:text-gray-400">Time</th>
              <th className="text-center py-2 text-gray-600 dark:text-gray-400">Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredOrders.map((order) => (
              <motion.tr
                key={order.id}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="border-b border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                <td className="py-3">
                  <div className="flex items-center gap-2">
                    {getStatusIcon(order.status)}
                    <span className="capitalize">{order.status}</span>
                  </div>
                </td>
                <td className="py-3 font-medium">{order.symbol}</td>
                <td className={`py-3 font-medium ${getSideColor(order.side)}`}>
                  {order.side.toUpperCase()}
                </td>
                <td className="py-3 capitalize">{order.type}</td>
                <td className="py-3 text-right">{order.quantity.toLocaleString()}</td>
                <td className="py-3 text-right">
                  {order.price ? formatCurrency(order.price) : 'Market'}
                </td>
                <td className="py-3 text-right">
                  {order.filledQuantity}/{order.quantity}
                </td>
                <td className="py-3">
                  {new Date(order.timestamp).toLocaleTimeString()}
                </td>
                <td className="py-3">
                  <div className="flex items-center justify-center gap-1">
                    {(order.status === 'open' || order.status === 'pending') && (
                      <>
                        <button
                          onClick={() => handleCancelOrder(order.id)}
                          className="p-1 text-red-500 hover:text-red-700"
                          title="Cancel Order"
                        >
                          <XCircle className="w-4 h-4" />
                        </button>
                        <button
                          className="p-1 text-blue-500 hover:text-blue-700"
                          title="Modify Order"
                        >
                          <Edit3 className="w-4 h-4" />
                        </button>
                      </>
                    )}
                  </div>
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>

        {filteredOrders.length === 0 && (
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            No orders found
          </div>
        )}
      </div>

      {/* Order Form Modal */}
      <AnimatePresence>
        {showOrderForm && (
          <OrderFormModal
            ws4Client={ws4Client}
            account={account}
            onClose={() => setShowOrderForm(false)}
          />
        )}
      </AnimatePresence>
    </motion.div>
  );
};

// Positions Panel
const PositionsPanel: React.FC<{
  positions: Position[];
  marketData: MarketData[];
}> = ({ positions, marketData }) => {
  const getMarketPrice = useCallback((symbol: string) => {
    const data = marketData.find(m => m.symbol === symbol);
    return data?.price || 0;
  }, [marketData]);

  const getPnLColor = (pnl: number) => {
    if (pnl > 0) return 'text-green-600';
    if (pnl < 0) return 'text-red-600';
    return 'text-gray-600';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
    >
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <th className="text-left py-2 text-gray-600 dark:text-gray-400">Symbol</th>
              <th className="text-right py-2 text-gray-600 dark:text-gray-400">Quantity</th>
              <th className="text-right py-2 text-gray-600 dark:text-gray-400">Avg Price</th>
              <th className="text-right py-2 text-gray-600 dark:text-gray-400">Market Price</th>
              <th className="text-right py-2 text-gray-600 dark:text-gray-400">Market Value</th>
              <th className="text-right py-2 text-gray-600 dark:text-gray-400">Unrealized P&L</th>
              <th className="text-right py-2 text-gray-600 dark:text-gray-400">Day P&L</th>
              <th className="text-center py-2 text-gray-600 dark:text-gray-400">Actions</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((position) => (
              <motion.tr
                key={position.symbol}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="border-b border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                <td className="py-3 font-medium">{position.symbol}</td>
                <td className={`py-3 text-right font-medium ${
                  position.quantity > 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {position.quantity.toLocaleString()}
                </td>
                <td className="py-3 text-right">{formatCurrency(position.averagePrice)}</td>
                <td className="py-3 text-right">{formatCurrency(getMarketPrice(position.symbol))}</td>
                <td className="py-3 text-right">{formatCurrency(position.marketValue)}</td>
                <td className={`py-3 text-right font-medium ${getPnLColor(position.unrealizedPnL)}`}>
                  {formatCurrency(position.unrealizedPnL)}
                </td>
                <td className={`py-3 text-right font-medium ${getPnLColor(position.dayPnL)}`}>
                  {formatCurrency(position.dayPnL)}
                </td>
                <td className="py-3">
                  <div className="flex items-center justify-center gap-1">
                    <button
                      className="p-1 text-blue-500 hover:text-blue-700"
                      title="Close Position"
                    >
                      <Square className="w-4 h-4" />
                    </button>
                    <button
                      className="p-1 text-green-500 hover:text-green-700"
                      title="Add to Position"
                    >
                      <Plus className="w-4 h-4" />
                    </button>
                  </div>
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>

        {positions.length === 0 && (
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            No positions found
          </div>
        )}
      </div>
    </motion.div>
  );
};

// Trades Panel
const TradesPanel: React.FC<{
  trades: Trade[];
}> = ({ trades }) => {
  const getSideColor = (side: string) => {
    return side === 'buy' ? 'text-green-600' : 'text-red-600';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
    >
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <th className="text-left py-2 text-gray-600 dark:text-gray-400">Time</th>
              <th className="text-left py-2 text-gray-600 dark:text-gray-400">Symbol</th>
              <th className="text-left py-2 text-gray-600 dark:text-gray-400">Side</th>
              <th className="text-right py-2 text-gray-600 dark:text-gray-400">Quantity</th>
              <th className="text-right py-2 text-gray-600 dark:text-gray-400">Price</th>
              <th className="text-right py-2 text-gray-600 dark:text-gray-400">Value</th>
              <th className="text-right py-2 text-gray-600 dark:text-gray-400">Commission</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((trade) => (
              <motion.tr
                key={trade.id}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="border-b border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                <td className="py-3">
                  {new Date(trade.timestamp).toLocaleTimeString()}
                </td>
                <td className="py-3 font-medium">{trade.symbol}</td>
                <td className={`py-3 font-medium ${getSideColor(trade.side)}`}>
                  {trade.side.toUpperCase()}
                </td>
                <td className="py-3 text-right">{trade.quantity.toLocaleString()}</td>
                <td className="py-3 text-right">{formatCurrency(trade.price)}</td>
                <td className="py-3 text-right">
                  {formatCurrency(trade.quantity * trade.price)}
                </td>
                <td className="py-3 text-right">{formatCurrency(trade.commission)}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>

        {trades.length === 0 && (
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            No trades found
          </div>
        )}
      </div>
    </motion.div>
  );
};

// Market Data Panel
const MarketDataPanel: React.FC<{
  marketData: MarketData[];
  ws4Client: WS4MarketIntegration | null;
}> = ({ marketData, ws4Client }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [newSymbol, setNewSymbol] = useState('');

  const filteredData = useMemo(() => {
    if (!searchTerm) return marketData;
    return marketData.filter(data => 
      data.symbol.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [marketData, searchTerm]);

  const handleAddSymbol = useCallback(() => {
    if (newSymbol && ws4Client) {
      ws4Client.subscribeToSymbol(newSymbol.toUpperCase());
      setNewSymbol('');
    }
  }, [newSymbol, ws4Client]);

  const getChangeColor = (change: number) => {
    if (change > 0) return 'text-green-600';
    if (change < 0) return 'text-red-600';
    return 'text-gray-600';
  };

  const getChangeIcon = (change: number) => {
    if (change > 0) return <ArrowUp className="w-3 h-3" />;
    if (change < 0) return <ArrowDown className="w-3 h-3" />;
    return <Minus className="w-3 h-3" />;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
    >
      {/* Controls */}
      <div className="flex items-center gap-3 mb-4">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search symbols..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          />
        </div>

        <div className="flex items-center gap-2">
          <input
            type="text"
            placeholder="Add symbol"
            value={newSymbol}
            onChange={(e) => setNewSymbol(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleAddSymbol()}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          />
          <button
            onClick={handleAddSymbol}
            className="px-3 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
          >
            <Plus className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Market Data Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <th className="text-left py-2 text-gray-600 dark:text-gray-400">Symbol</th>
              <th className="text-right py-2 text-gray-600 dark:text-gray-400">Price</th>
              <th className="text-right py-2 text-gray-600 dark:text-gray-400">Change</th>
              <th className="text-right py-2 text-gray-600 dark:text-gray-400">Change %</th>
              <th className="text-right py-2 text-gray-600 dark:text-gray-400">Volume</th>
              <th className="text-right py-2 text-gray-600 dark:text-gray-400">Bid</th>
              <th className="text-right py-2 text-gray-600 dark:text-gray-400">Ask</th>
              <th className="text-right py-2 text-gray-600 dark:text-gray-400">High</th>
              <th className="text-right py-2 text-gray-600 dark:text-gray-400">Low</th>
            </tr>
          </thead>
          <tbody>
            {filteredData.map((data) => (
              <motion.tr
                key={data.symbol}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="border-b border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                <td className="py-3 font-medium">{data.symbol}</td>
                <td className="py-3 text-right font-medium">{formatCurrency(data.price)}</td>
                <td className={`py-3 text-right font-medium ${getChangeColor(data.change)}`}>
                  <div className="flex items-center justify-end gap-1">
                    {getChangeIcon(data.change)}
                    {formatCurrency(Math.abs(data.change))}
                  </div>
                </td>
                <td className={`py-3 text-right font-medium ${getChangeColor(data.change)}`}>
                  {formatPercent(data.changePercent)}
                </td>
                <td className="py-3 text-right">{data.volume.toLocaleString()}</td>
                <td className="py-3 text-right">{formatCurrency(data.bid)}</td>
                <td className="py-3 text-right">{formatCurrency(data.ask)}</td>
                <td className="py-3 text-right">{formatCurrency(data.dayHigh)}</td>
                <td className="py-3 text-right">{formatCurrency(data.dayLow)}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>

        {filteredData.length === 0 && (
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            No market data found
          </div>
        )}
      </div>
    </motion.div>
  );
};

// Order Form Modal
const OrderFormModal: React.FC<{
  ws4Client: WS4MarketIntegration | null;
  account: string;
  onClose: () => void;
}> = ({ ws4Client, account, onClose }) => {
  const [formData, setFormData] = useState({
    symbol: '',
    side: 'buy' as 'buy' | 'sell',
    type: 'market' as 'market' | 'limit' | 'stop' | 'stop_limit',
    quantity: '',
    price: '',
    stopPrice: '',
    timeInForce: 'day' as 'day' | 'gtc' | 'ioc' | 'fok'
  });
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    if (!ws4Client || isSubmitting) return;

    setIsSubmitting(true);
    try {
      await ws4Client.placeOrder({
        symbol: formData.symbol.toUpperCase(),
        side: formData.side,
        type: formData.type,
        quantity: parseInt(formData.quantity),
        price: formData.price ? parseFloat(formData.price) : undefined,
        stopPrice: formData.stopPrice ? parseFloat(formData.stopPrice) : undefined,
        timeInForce: formData.timeInForce,
        account
      });
      onClose();
    } catch (error) {
      console.error('Failed to place order:', error);
    } finally {
      setIsSubmitting(false);
    }
  }, [formData, ws4Client, account, onClose, isSubmitting]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-md mx-4"
        onClick={(e) => e.stopPropagation()}
      >
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Place Order
        </h3>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Symbol
            </label>
            <input
              type="text"
              value={formData.symbol}
              onChange={(e) => setFormData(prev => ({ ...prev, symbol: e.target.value }))}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              placeholder="e.g., AAPL"
              required
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Side
              </label>
              <select
                value={formData.side}
                onChange={(e) => setFormData(prev => ({ ...prev, side: e.target.value as any }))}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                <option value="buy">Buy</option>
                <option value="sell">Sell</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Type
              </label>
              <select
                value={formData.type}
                onChange={(e) => setFormData(prev => ({ ...prev, type: e.target.value as any }))}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                <option value="market">Market</option>
                <option value="limit">Limit</option>
                <option value="stop">Stop</option>
                <option value="stop_limit">Stop Limit</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Quantity
            </label>
            <input
              type="number"
              value={formData.quantity}
              onChange={(e) => setFormData(prev => ({ ...prev, quantity: e.target.value }))}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              placeholder="Number of shares"
              required
            />
          </div>

          {(formData.type === 'limit' || formData.type === 'stop_limit') && (
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Limit Price
              </label>
              <input
                type="number"
                step="0.01"
                value={formData.price}
                onChange={(e) => setFormData(prev => ({ ...prev, price: e.target.value }))}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                placeholder="Price per share"
                required
              />
            </div>
          )}

          {(formData.type === 'stop' || formData.type === 'stop_limit') && (
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Stop Price
              </label>
              <input
                type="number"
                step="0.01"
                value={formData.stopPrice}
                onChange={(e) => setFormData(prev => ({ ...prev, stopPrice: e.target.value }))}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                placeholder="Stop price"
                required
              />
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Time in Force
            </label>
            <select
              value={formData.timeInForce}
              onChange={(e) => setFormData(prev => ({ ...prev, timeInForce: e.target.value as any }))}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="day">Day</option>
              <option value="gtc">Good Till Cancelled</option>
              <option value="ioc">Immediate or Cancel</option>
              <option value="fok">Fill or Kill</option>
            </select>
          </div>

          <div className="flex gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              className="flex-1 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isSubmitting ? 'Placing...' : 'Place Order'}
            </button>
          </div>
        </form>
      </motion.div>
    </motion.div>
  );
};

// Export all components
export {
  OrdersPanel,
  PositionsPanel,
  TradesPanel,
  MarketDataPanel,
  OrderFormModal
};

