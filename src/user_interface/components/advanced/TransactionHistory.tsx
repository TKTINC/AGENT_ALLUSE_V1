import React, { useState, useEffect, useMemo } from 'react';
import { 
  Clock, 
  DollarSign, 
  TrendingUp, 
  TrendingDown, 
  ArrowUpRight, 
  ArrowDownRight,
  Filter,
  Search,
  Download,
  Calendar,
  Tag,
  MapPin,
  CreditCard,
  Repeat,
  AlertCircle,
  CheckCircle,
  XCircle
} from 'lucide-react';
import { Button, Card, Modal, Tabs, ProgressBar } from '../advanced/UIComponents';
import { formatCurrency, formatDate, formatTime } from '../../utils/chartUtils';

// Transaction History and Portfolio Analytics
interface Transaction {
  id: string;
  date: string;
  time: string;
  type: 'buy' | 'sell' | 'dividend' | 'fee' | 'transfer' | 'split';
  symbol: string;
  description: string;
  quantity: number;
  price: number;
  amount: number;
  fees: number;
  account: string;
  status: 'completed' | 'pending' | 'failed' | 'cancelled';
  category: string;
  tags: string[];
  notes?: string;
}

interface TransactionFilter {
  dateRange: { start: string; end: string };
  accounts: string[];
  types: string[];
  symbols: string[];
  minAmount: number;
  maxAmount: number;
  status: string[];
}

interface TransactionHistoryProps {
  transactions: Transaction[];
  accounts: string[];
  onExportTransactions: (transactions: Transaction[], format: 'csv' | 'pdf') => void;
}

export const TransactionHistory: React.FC<TransactionHistoryProps> = ({
  transactions,
  accounts,
  onExportTransactions
}) => {
  const [filteredTransactions, setFilteredTransactions] = useState<Transaction[]>(transactions);
  const [searchTerm, setSearchTerm] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [selectedTransaction, setSelectedTransaction] = useState<Transaction | null>(null);
  const [sortBy, setSortBy] = useState<'date' | 'amount' | 'symbol'>('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  
  const [filters, setFilters] = useState<TransactionFilter>({
    dateRange: { 
      start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
      end: new Date().toISOString().split('T')[0]
    },
    accounts: [],
    types: [],
    symbols: [],
    minAmount: 0,
    maxAmount: 1000000,
    status: []
  });

  // Apply filters and search
  useEffect(() => {
    let filtered = transactions.filter(transaction => {
      // Date range filter
      const transactionDate = new Date(transaction.date);
      const startDate = new Date(filters.dateRange.start);
      const endDate = new Date(filters.dateRange.end);
      
      if (transactionDate < startDate || transactionDate > endDate) return false;

      // Account filter
      if (filters.accounts.length > 0 && !filters.accounts.includes(transaction.account)) return false;

      // Type filter
      if (filters.types.length > 0 && !filters.types.includes(transaction.type)) return false;

      // Symbol filter
      if (filters.symbols.length > 0 && !filters.symbols.includes(transaction.symbol)) return false;

      // Amount filter
      if (Math.abs(transaction.amount) < filters.minAmount || Math.abs(transaction.amount) > filters.maxAmount) return false;

      // Status filter
      if (filters.status.length > 0 && !filters.status.includes(transaction.status)) return false;

      // Search term
      if (searchTerm) {
        const searchLower = searchTerm.toLowerCase();
        return (
          transaction.symbol.toLowerCase().includes(searchLower) ||
          transaction.description.toLowerCase().includes(searchLower) ||
          transaction.account.toLowerCase().includes(searchLower) ||
          transaction.category.toLowerCase().includes(searchLower) ||
          transaction.tags.some(tag => tag.toLowerCase().includes(searchLower))
        );
      }

      return true;
    });

    // Apply sorting
    filtered.sort((a, b) => {
      let comparison = 0;
      
      switch (sortBy) {
        case 'date':
          comparison = new Date(a.date + ' ' + a.time).getTime() - new Date(b.date + ' ' + b.time).getTime();
          break;
        case 'amount':
          comparison = Math.abs(a.amount) - Math.abs(b.amount);
          break;
        case 'symbol':
          comparison = a.symbol.localeCompare(b.symbol);
          break;
      }

      return sortOrder === 'asc' ? comparison : -comparison;
    });

    setFilteredTransactions(filtered);
  }, [transactions, filters, searchTerm, sortBy, sortOrder]);

  const transactionSummary = useMemo(() => {
    const summary = {
      totalTransactions: filteredTransactions.length,
      totalBuys: 0,
      totalSells: 0,
      totalDividends: 0,
      totalFees: 0,
      netCashFlow: 0,
      byAccount: {} as Record<string, { count: number; amount: number }>,
      byType: {} as Record<string, { count: number; amount: number }>
    };

    filteredTransactions.forEach(transaction => {
      // By type
      if (!summary.byType[transaction.type]) {
        summary.byType[transaction.type] = { count: 0, amount: 0 };
      }
      summary.byType[transaction.type].count++;
      summary.byType[transaction.type].amount += transaction.amount;

      // By account
      if (!summary.byAccount[transaction.account]) {
        summary.byAccount[transaction.account] = { count: 0, amount: 0 };
      }
      summary.byAccount[transaction.account].count++;
      summary.byAccount[transaction.account].amount += transaction.amount;

      // Totals
      switch (transaction.type) {
        case 'buy':
          summary.totalBuys += Math.abs(transaction.amount);
          summary.netCashFlow -= Math.abs(transaction.amount);
          break;
        case 'sell':
          summary.totalSells += Math.abs(transaction.amount);
          summary.netCashFlow += Math.abs(transaction.amount);
          break;
        case 'dividend':
          summary.totalDividends += transaction.amount;
          summary.netCashFlow += transaction.amount;
          break;
        case 'fee':
          summary.totalFees += Math.abs(transaction.amount);
          summary.netCashFlow -= Math.abs(transaction.amount);
          break;
      }
    });

    return summary;
  }, [filteredTransactions]);

  const getTransactionIcon = (type: string) => {
    switch (type) {
      case 'buy': return <ArrowDownRight className="w-4 h-4 text-red-600" />;
      case 'sell': return <ArrowUpRight className="w-4 h-4 text-green-600" />;
      case 'dividend': return <DollarSign className="w-4 h-4 text-blue-600" />;
      case 'fee': return <CreditCard className="w-4 h-4 text-gray-600" />;
      case 'transfer': return <Repeat className="w-4 h-4 text-purple-600" />;
      case 'split': return <TrendingUp className="w-4 h-4 text-orange-600" />;
      default: return <Clock className="w-4 h-4 text-gray-600" />;
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'pending': return <Clock className="w-4 h-4 text-yellow-600" />;
      case 'failed': return <XCircle className="w-4 h-4 text-red-600" />;
      case 'cancelled': return <XCircle className="w-4 h-4 text-gray-600" />;
      default: return <AlertCircle className="w-4 h-4 text-gray-600" />;
    }
  };

  const getTransactionColor = (type: string) => {
    switch (type) {
      case 'buy': return 'text-red-600';
      case 'sell': return 'text-green-600';
      case 'dividend': return 'text-blue-600';
      case 'fee': return 'text-gray-600';
      case 'transfer': return 'text-purple-600';
      case 'split': return 'text-orange-600';
      default: return 'text-gray-600';
    }
  };

  const uniqueSymbols = Array.from(new Set(transactions.map(t => t.symbol))).sort();
  const uniqueTypes = Array.from(new Set(transactions.map(t => t.type))).sort();
  const uniqueStatuses = Array.from(new Set(transactions.map(t => t.status))).sort();

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold text-gray-900">Transaction History</h3>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            icon={<Filter className="w-4 h-4" />}
            onClick={() => setShowFilters(!showFilters)}
          >
            Filters
          </Button>
          <Button
            variant="outline"
            size="sm"
            icon={<Download className="w-4 h-4" />}
            onClick={() => onExportTransactions(filteredTransactions, 'csv')}
          >
            Export
          </Button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Total Transactions</span>
            <Clock className="w-5 h-5 text-blue-600" />
          </div>
          <div className="text-2xl font-bold text-gray-900">
            {transactionSummary.totalTransactions}
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Net Cash Flow</span>
            {transactionSummary.netCashFlow >= 0 ? 
              <TrendingUp className="w-5 h-5 text-green-600" /> :
              <TrendingDown className="w-5 h-5 text-red-600" />
            }
          </div>
          <div className={`text-2xl font-bold ${
            transactionSummary.netCashFlow >= 0 ? 'text-green-600' : 'text-red-600'
          }`}>
            {formatCurrency(transactionSummary.netCashFlow)}
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Total Dividends</span>
            <DollarSign className="w-5 h-5 text-blue-600" />
          </div>
          <div className="text-2xl font-bold text-blue-600">
            {formatCurrency(transactionSummary.totalDividends)}
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Total Fees</span>
            <CreditCard className="w-5 h-5 text-gray-600" />
          </div>
          <div className="text-2xl font-bold text-gray-600">
            {formatCurrency(transactionSummary.totalFees)}
          </div>
        </Card>
      </div>

      {/* Search and Filters */}
      <div className="space-y-4">
        <div className="flex items-center gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search transactions..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="date">Sort by Date</option>
            <option value="amount">Sort by Amount</option>
            <option value="symbol">Sort by Symbol</option>
          </select>

          <Button
            variant="outline"
            size="sm"
            onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
          >
            {sortOrder === 'asc' ? '↑' : '↓'}
          </Button>
        </div>

        {showFilters && (
          <Card className="p-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Date Range</label>
                <div className="flex gap-2">
                  <input
                    type="date"
                    value={filters.dateRange.start}
                    onChange={(e) => setFilters(prev => ({
                      ...prev,
                      dateRange: { ...prev.dateRange, start: e.target.value }
                    }))}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  <input
                    type="date"
                    value={filters.dateRange.end}
                    onChange={(e) => setFilters(prev => ({
                      ...prev,
                      dateRange: { ...prev.dateRange, end: e.target.value }
                    }))}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Transaction Type</label>
                <select
                  multiple
                  value={filters.types}
                  onChange={(e) => setFilters(prev => ({
                    ...prev,
                    types: Array.from(e.target.selectedOptions, option => option.value)
                  }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {uniqueTypes.map(type => (
                    <option key={type} value={type}>{type.charAt(0).toUpperCase() + type.slice(1)}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Account</label>
                <select
                  multiple
                  value={filters.accounts}
                  onChange={(e) => setFilters(prev => ({
                    ...prev,
                    accounts: Array.from(e.target.selectedOptions, option => option.value)
                  }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {accounts.map(account => (
                    <option key={account} value={account}>{account}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="flex gap-4 mt-4">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setFilters({
                  dateRange: { 
                    start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
                    end: new Date().toISOString().split('T')[0]
                  },
                  accounts: [],
                  types: [],
                  symbols: [],
                  minAmount: 0,
                  maxAmount: 1000000,
                  status: []
                })}
              >
                Clear Filters
              </Button>
            </div>
          </Card>
        )}
      </div>

      {/* Transactions Table */}
      <Card className="overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="text-left py-3 px-4 font-medium text-gray-700">Date/Time</th>
                <th className="text-left py-3 px-4 font-medium text-gray-700">Type</th>
                <th className="text-left py-3 px-4 font-medium text-gray-700">Symbol</th>
                <th className="text-left py-3 px-4 font-medium text-gray-700">Description</th>
                <th className="text-left py-3 px-4 font-medium text-gray-700">Quantity</th>
                <th className="text-left py-3 px-4 font-medium text-gray-700">Price</th>
                <th className="text-left py-3 px-4 font-medium text-gray-700">Amount</th>
                <th className="text-left py-3 px-4 font-medium text-gray-700">Account</th>
                <th className="text-left py-3 px-4 font-medium text-gray-700">Status</th>
                <th className="text-left py-3 px-4 font-medium text-gray-700">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredTransactions.map(transaction => (
                <tr key={transaction.id} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="py-3 px-4">
                    <div className="text-sm">
                      <div className="font-medium">{formatDate(transaction.date, 'short')}</div>
                      <div className="text-gray-500">{formatTime(transaction.date + ' ' + transaction.time)}</div>
                    </div>
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-2">
                      {getTransactionIcon(transaction.type)}
                      <span className={`text-sm font-medium capitalize ${getTransactionColor(transaction.type)}`}>
                        {transaction.type}
                      </span>
                    </div>
                  </td>
                  <td className="py-3 px-4">
                    <span className="font-medium">{transaction.symbol}</span>
                  </td>
                  <td className="py-3 px-4">
                    <div className="text-sm">
                      <div className="font-medium">{transaction.description}</div>
                      {transaction.tags.length > 0 && (
                        <div className="flex gap-1 mt-1">
                          {transaction.tags.slice(0, 2).map(tag => (
                            <span key={tag} className="px-2 py-1 text-xs bg-gray-100 text-gray-600 rounded">
                              {tag}
                            </span>
                          ))}
                          {transaction.tags.length > 2 && (
                            <span className="px-2 py-1 text-xs bg-gray-100 text-gray-600 rounded">
                              +{transaction.tags.length - 2}
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  </td>
                  <td className="py-3 px-4">
                    {transaction.quantity !== 0 ? transaction.quantity.toLocaleString() : '-'}
                  </td>
                  <td className="py-3 px-4">
                    {transaction.price !== 0 ? formatCurrency(transaction.price) : '-'}
                  </td>
                  <td className="py-3 px-4">
                    <div className={`font-medium ${getTransactionColor(transaction.type)}`}>
                      {formatCurrency(transaction.amount)}
                    </div>
                    {transaction.fees > 0 && (
                      <div className="text-xs text-gray-500">
                        Fee: {formatCurrency(transaction.fees)}
                      </div>
                    )}
                  </td>
                  <td className="py-3 px-4">
                    <span className="text-sm">{transaction.account}</span>
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(transaction.status)}
                      <span className={`text-sm capitalize ${
                        transaction.status === 'completed' ? 'text-green-600' :
                        transaction.status === 'pending' ? 'text-yellow-600' :
                        transaction.status === 'failed' ? 'text-red-600' :
                        'text-gray-600'
                      }`}>
                        {transaction.status}
                      </span>
                    </div>
                  </td>
                  <td className="py-3 px-4">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setSelectedTransaction(transaction)}
                    >
                      Details
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {filteredTransactions.length === 0 && (
          <div className="text-center py-8 text-gray-500">
            No transactions found matching the current filters.
          </div>
        )}
      </Card>

      {/* Transaction Details Modal */}
      {selectedTransaction && (
        <Modal
          isOpen={!!selectedTransaction}
          onClose={() => setSelectedTransaction(null)}
          title="Transaction Details"
          size="lg"
        >
          <div className="space-y-6">
            {/* Transaction Overview */}
            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-3">
                <div>
                  <span className="text-sm text-gray-600">Transaction ID:</span>
                  <div className="font-mono text-sm">{selectedTransaction.id}</div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Date & Time:</span>
                  <div className="font-medium">
                    {formatDate(selectedTransaction.date, 'long')} at {formatTime(selectedTransaction.date + ' ' + selectedTransaction.time)}
                  </div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Type:</span>
                  <div className="flex items-center gap-2">
                    {getTransactionIcon(selectedTransaction.type)}
                    <span className={`font-medium capitalize ${getTransactionColor(selectedTransaction.type)}`}>
                      {selectedTransaction.type}
                    </span>
                  </div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Status:</span>
                  <div className="flex items-center gap-2">
                    {getStatusIcon(selectedTransaction.status)}
                    <span className={`font-medium capitalize ${
                      selectedTransaction.status === 'completed' ? 'text-green-600' :
                      selectedTransaction.status === 'pending' ? 'text-yellow-600' :
                      selectedTransaction.status === 'failed' ? 'text-red-600' :
                      'text-gray-600'
                    }`}>
                      {selectedTransaction.status}
                    </span>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <div>
                  <span className="text-sm text-gray-600">Symbol:</span>
                  <div className="font-medium text-lg">{selectedTransaction.symbol}</div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Account:</span>
                  <div className="font-medium">{selectedTransaction.account}</div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Category:</span>
                  <div className="font-medium">{selectedTransaction.category}</div>
                </div>
                {selectedTransaction.quantity !== 0 && (
                  <div>
                    <span className="text-sm text-gray-600">Quantity:</span>
                    <div className="font-medium">{selectedTransaction.quantity.toLocaleString()}</div>
                  </div>
                )}
              </div>
            </div>

            {/* Financial Details */}
            <div className="border-t pt-4">
              <h4 className="font-medium mb-3">Financial Details</h4>
              <div className="grid grid-cols-2 gap-4">
                {selectedTransaction.price !== 0 && (
                  <div>
                    <span className="text-sm text-gray-600">Price per Share:</span>
                    <div className="font-medium">{formatCurrency(selectedTransaction.price)}</div>
                  </div>
                )}
                <div>
                  <span className="text-sm text-gray-600">Total Amount:</span>
                  <div className={`font-medium text-lg ${getTransactionColor(selectedTransaction.type)}`}>
                    {formatCurrency(selectedTransaction.amount)}
                  </div>
                </div>
                {selectedTransaction.fees > 0 && (
                  <div>
                    <span className="text-sm text-gray-600">Fees:</span>
                    <div className="font-medium text-red-600">{formatCurrency(selectedTransaction.fees)}</div>
                  </div>
                )}
                <div>
                  <span className="text-sm text-gray-600">Net Amount:</span>
                  <div className="font-medium">
                    {formatCurrency(selectedTransaction.amount - selectedTransaction.fees)}
                  </div>
                </div>
              </div>
            </div>

            {/* Description and Tags */}
            <div className="border-t pt-4">
              <h4 className="font-medium mb-3">Description</h4>
              <p className="text-gray-700">{selectedTransaction.description}</p>
              
              {selectedTransaction.tags.length > 0 && (
                <div className="mt-3">
                  <span className="text-sm text-gray-600 block mb-2">Tags:</span>
                  <div className="flex flex-wrap gap-2">
                    {selectedTransaction.tags.map(tag => (
                      <span key={tag} className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded-full">
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {selectedTransaction.notes && (
                <div className="mt-3">
                  <span className="text-sm text-gray-600 block mb-2">Notes:</span>
                  <p className="text-gray-700 text-sm bg-gray-50 p-3 rounded-lg">
                    {selectedTransaction.notes}
                  </p>
                </div>
              )}
            </div>

            <div className="flex gap-3 pt-4">
              <Button
                variant="outline"
                fullWidth
                onClick={() => setSelectedTransaction(null)}
              >
                Close
              </Button>
              <Button
                variant="outline"
                fullWidth
                icon={<Download className="w-4 h-4" />}
                onClick={() => onExportTransactions([selectedTransaction], 'pdf')}
              >
                Export Receipt
              </Button>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
};

