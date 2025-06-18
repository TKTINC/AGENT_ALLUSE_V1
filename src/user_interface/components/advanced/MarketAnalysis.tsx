import React, { useState, useEffect, useMemo } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Target, 
  AlertTriangle, 
  CheckCircle, 
  Clock,
  BarChart3,
  PieChart,
  Zap,
  Settings,
  RefreshCw,
  Play,
  Pause,
  Edit,
  Eye,
  DollarSign,
  Percent,
  Shield
} from 'lucide-react';
import { Button, Card, Modal, Tabs, ProgressBar } from '../advanced/UIComponents';
import { 
  useMarketData, 
  useTradingSignals, 
  useProtocolCompliance, 
  useAutomatedStrategies,
  useMarketAnalysis,
  TradingSignal,
  AutomatedStrategy,
  ProtocolCompliance,
  MarketAnalysis
} from '../../lib/ws2-integration';
import { formatCurrency, formatPercentage, formatDate } from '../../utils/chartUtils';

// Market Analysis Dashboard
interface MarketAnalysisDashboardProps {
  symbols: string[];
}

export const MarketAnalysisDashboard: React.FC<MarketAnalysisDashboardProps> = ({ symbols }) => {
  const { data: marketData, loading: marketLoading } = useMarketData(symbols);
  const { analysis, loading: analysisLoading, refreshAnalysis } = useMarketAnalysis();
  const [activeTab, setActiveTab] = useState('overview');

  const tabs = [
    { id: 'overview', label: 'Market Overview', icon: <BarChart3 className="w-4 h-4" /> },
    { id: 'sentiment', label: 'Sentiment Analysis', icon: <Activity className="w-4 h-4" /> },
    { id: 'technical', label: 'Technical Levels', icon: <TrendingUp className="w-4 h-4" /> },
    { id: 'economic', label: 'Economic Data', icon: <PieChart className="w-4 h-4" /> }
  ];

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'Bullish': return 'text-green-600 bg-green-100';
      case 'Bearish': return 'text-red-600 bg-red-100';
      case 'Neutral': return 'text-gray-600 bg-gray-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'Up': return 'text-green-600';
      case 'Down': return 'text-red-600';
      case 'Sideways': return 'text-gray-600';
      default: return 'text-gray-600';
    }
  };

  const renderOverview = () => (
    <div className="space-y-6">
      {/* Market Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Market Sentiment</span>
            <Activity className="w-5 h-5 text-blue-600" />
          </div>
          <div className={`text-lg font-bold px-3 py-1 rounded-full text-center ${
            analysis ? getSentimentColor(analysis.marketSentiment) : 'text-gray-600 bg-gray-100'
          }`}>
            {analysis?.marketSentiment || 'Loading...'}
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Volatility Index</span>
            <TrendingUp className="w-5 h-5 text-orange-600" />
          </div>
          <div className="text-2xl font-bold text-gray-900">
            {analysis?.volatilityIndex.toFixed(1) || '--'}
          </div>
          <div className="text-sm text-gray-600">VIX Level</div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Trend Direction</span>
            {analysis?.trendDirection === 'Up' ? <TrendingUp className="w-5 h-5 text-green-600" /> :
             analysis?.trendDirection === 'Down' ? <TrendingDown className="w-5 h-5 text-red-600" /> :
             <Activity className="w-5 h-5 text-gray-600" />}
          </div>
          <div className={`text-lg font-bold ${
            analysis ? getTrendColor(analysis.trendDirection) : 'text-gray-600'
          }`}>
            {analysis?.trendDirection || 'Loading...'}
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Active Symbols</span>
            <Eye className="w-5 h-5 text-blue-600" />
          </div>
          <div className="text-2xl font-bold text-gray-900">
            {marketData.length}
          </div>
          <div className="text-sm text-gray-600">Monitored</div>
        </Card>
      </div>

      {/* Key Market Events */}
      {analysis?.keyEvents && (
        <Card className="p-6">
          <h4 className="text-lg font-semibold mb-4">Upcoming Market Events</h4>
          <div className="space-y-3">
            {analysis.keyEvents.map((event, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center gap-3">
                  <div className={`w-3 h-3 rounded-full ${
                    event.impact === 'High' ? 'bg-red-500' :
                    event.impact === 'Medium' ? 'bg-yellow-500' : 'bg-green-500'
                  }`} />
                  <div>
                    <div className="font-medium">{event.event}</div>
                    <div className="text-sm text-gray-600">{formatDate(event.date, 'medium')}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`text-sm font-medium ${
                    event.sentiment === 'Positive' ? 'text-green-600' :
                    event.sentiment === 'Negative' ? 'text-red-600' : 'text-gray-600'
                  }`}>
                    {event.sentiment}
                  </div>
                  <div className="text-xs text-gray-500">{event.impact} Impact</div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Sector Rotation */}
      {analysis?.sectorRotation && (
        <Card className="p-6">
          <h4 className="text-lg font-semibold mb-4">Sector Performance</h4>
          <div className="space-y-3">
            {Object.entries(analysis.sectorRotation)
              .sort(([,a], [,b]) => b - a)
              .map(([sector, performance]) => (
                <div key={sector} className="flex items-center justify-between">
                  <span className="font-medium">{sector}</span>
                  <div className="flex items-center gap-3">
                    <div className={`font-medium ${
                      performance > 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {formatPercentage(performance)}
                    </div>
                    <ProgressBar
                      value={Math.abs(performance * 100)}
                      max={20}
                      color={performance > 0 ? 'green' : 'red'}
                      size="sm"
                      className="w-20"
                    />
                  </div>
                </div>
              ))}
          </div>
        </Card>
      )}
    </div>
  );

  const renderTechnicalLevels = () => (
    <div className="space-y-6">
      <Card className="p-6">
        <h4 className="text-lg font-semibold mb-4">Support & Resistance Levels</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h5 className="font-medium mb-3 text-green-600">Support Levels</h5>
            <div className="space-y-2">
              {analysis?.supportLevels.map((level, index) => (
                <div key={index} className="flex items-center justify-between p-2 bg-green-50 rounded">
                  <span className="text-sm text-gray-600">S{index + 1}</span>
                  <span className="font-medium">{level.toLocaleString()}</span>
                </div>
              ))}
            </div>
          </div>
          <div>
            <h5 className="font-medium mb-3 text-red-600">Resistance Levels</h5>
            <div className="space-y-2">
              {analysis?.resistanceLevels.map((level, index) => (
                <div key={index} className="flex items-center justify-between p-2 bg-red-50 rounded">
                  <span className="text-sm text-gray-600">R{index + 1}</span>
                  <span className="font-medium">{level.toLocaleString()}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Card>
    </div>
  );

  const renderEconomicData = () => (
    <div className="space-y-6">
      <Card className="p-6">
        <h4 className="text-lg font-semibold mb-4">Economic Indicators</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {analysis?.economicIndicators.gdpGrowth.toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">GDP Growth</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">
              {analysis?.economicIndicators.inflation.toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">Inflation</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {analysis?.economicIndicators.unemployment.toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">Unemployment</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {analysis?.economicIndicators.interestRates.toFixed(2)}%
            </div>
            <div className="text-sm text-gray-600">Interest Rates</div>
          </div>
        </div>
      </Card>
    </div>
  );

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-gray-900">Market Analysis</h3>
        <Button
          variant="outline"
          size="sm"
          icon={<RefreshCw className="w-4 h-4" />}
          onClick={refreshAnalysis}
          loading={analysisLoading}
        >
          Refresh
        </Button>
      </div>

      <Tabs
        tabs={tabs}
        activeTab={activeTab}
        onChange={setActiveTab}
        variant="underline"
      />

      <div className="mt-6">
        {activeTab === 'overview' && renderOverview()}
        {activeTab === 'sentiment' && renderOverview()}
        {activeTab === 'technical' && renderTechnicalLevels()}
        {activeTab === 'economic' && renderEconomicData()}
      </div>
    </Card>
  );
};

// Trading Signals Dashboard
interface TradingSignalsDashboardProps {
  symbols: string[];
  onExecuteSignal: (signal: TradingSignal) => void;
}

export const TradingSignalsDashboard: React.FC<TradingSignalsDashboardProps> = ({
  symbols,
  onExecuteSignal
}) => {
  const { signals, loading } = useTradingSignals(symbols);
  const [selectedSignal, setSelectedSignal] = useState<TradingSignal | null>(null);
  const [filterBy, setFilterBy] = useState<'all' | 'buy' | 'sell' | 'high-confidence'>('all');

  const filteredSignals = useMemo(() => {
    let filtered = signals;

    switch (filterBy) {
      case 'buy':
        filtered = signals.filter(s => s.type === 'buy');
        break;
      case 'sell':
        filtered = signals.filter(s => s.type === 'sell');
        break;
      case 'high-confidence':
        filtered = signals.filter(s => s.confidence > 0.8);
        break;
    }

    return filtered.sort((a, b) => b.confidence - a.confidence);
  }, [signals, filterBy]);

  const getSignalColor = (type: string) => {
    switch (type) {
      case 'buy': return 'text-green-600 bg-green-100';
      case 'sell': return 'text-red-600 bg-red-100';
      case 'hold': return 'text-gray-600 bg-gray-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'Low': return 'text-green-600 bg-green-100';
      case 'Medium': return 'text-yellow-600 bg-yellow-100';
      case 'High': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-gray-900">Trading Signals</h3>
        <div className="flex items-center gap-4">
          <select
            value={filterBy}
            onChange={(e) => setFilterBy(e.target.value as any)}
            className="px-3 py-1 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Signals</option>
            <option value="buy">Buy Signals</option>
            <option value="sell">Sell Signals</option>
            <option value="high-confidence">High Confidence</option>
          </select>
          <div className="text-sm text-gray-600">
            {filteredSignals.length} signals
          </div>
        </div>
      </div>

      {loading ? (
        <div className="text-center py-8 text-gray-500">Loading trading signals...</div>
      ) : (
        <div className="space-y-4">
          {filteredSignals.map(signal => (
            <div key={signal.id} className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <span className="text-lg font-bold">{signal.symbol}</span>
                  <span className={`px-3 py-1 text-sm font-medium rounded-full ${getSignalColor(signal.type)}`}>
                    {signal.type.toUpperCase()}
                  </span>
                  <span className={`px-2 py-1 text-xs rounded ${getRiskColor(signal.riskLevel)}`}>
                    {signal.riskLevel} Risk
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="text-right">
                    <div className="text-sm text-gray-600">Confidence</div>
                    <div className="font-bold">{(signal.confidence * 100).toFixed(0)}%</div>
                  </div>
                  <ProgressBar
                    value={signal.confidence * 100}
                    max={100}
                    color={signal.confidence > 0.8 ? 'green' : signal.confidence > 0.6 ? 'yellow' : 'red'}
                    size="sm"
                    className="w-16"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
                <div>
                  <span className="text-sm text-gray-600">Strategy:</span>
                  <div className="font-medium">{signal.strategy}</div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Target Price:</span>
                  <div className="font-medium">{formatCurrency(signal.targetPrice)}</div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Expected Return:</span>
                  <div className={`font-medium ${
                    signal.expectedReturn > 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {formatPercentage(signal.expectedReturn)}
                  </div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Timeframe:</span>
                  <div className="font-medium">{signal.timeframe}</div>
                </div>
              </div>

              <div className="text-sm text-gray-700 mb-3">
                {signal.reasoning}
              </div>

              <div className="flex items-center justify-between">
                <div className="text-xs text-gray-500">
                  Generated {new Date(signal.timestamp).toLocaleTimeString()}
                </div>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setSelectedSignal(signal)}
                  >
                    Details
                  </Button>
                  <Button
                    variant="primary"
                    size="sm"
                    onClick={() => onExecuteSignal(signal)}
                  >
                    Execute
                  </Button>
                </div>
              </div>
            </div>
          ))}

          {filteredSignals.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              No trading signals found matching the current filter.
            </div>
          )}
        </div>
      )}

      {/* Signal Details Modal */}
      {selectedSignal && (
        <Modal
          isOpen={!!selectedSignal}
          onClose={() => setSelectedSignal(null)}
          title={`Trading Signal: ${selectedSignal.symbol}`}
          size="lg"
        >
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-6">
              <div className="space-y-3">
                <div>
                  <span className="text-sm text-gray-600">Signal Type:</span>
                  <div className={`inline-block px-3 py-1 text-sm font-medium rounded-full ml-2 ${
                    getSignalColor(selectedSignal.type)
                  }`}>
                    {selectedSignal.type.toUpperCase()}
                  </div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Strategy:</span>
                  <div className="font-medium">{selectedSignal.strategy}</div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Confidence Level:</span>
                  <div className="font-medium">{(selectedSignal.confidence * 100).toFixed(1)}%</div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Risk Level:</span>
                  <div className={`inline-block px-2 py-1 text-xs rounded ml-2 ${
                    getRiskColor(selectedSignal.riskLevel)
                  }`}>
                    {selectedSignal.riskLevel}
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <div>
                  <span className="text-sm text-gray-600">Target Price:</span>
                  <div className="font-medium text-lg">{formatCurrency(selectedSignal.targetPrice)}</div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Stop Loss:</span>
                  <div className="font-medium">{formatCurrency(selectedSignal.stopLoss)}</div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Expected Return:</span>
                  <div className={`font-medium ${
                    selectedSignal.expectedReturn > 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {formatPercentage(selectedSignal.expectedReturn)}
                  </div>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Probability:</span>
                  <div className="font-medium">{(selectedSignal.probability * 100).toFixed(1)}%</div>
                </div>
              </div>
            </div>

            <div className="border-t pt-4">
              <h4 className="font-medium mb-2">Analysis</h4>
              <p className="text-gray-700">{selectedSignal.reasoning}</p>
            </div>

            <div className="flex gap-3 pt-4">
              <Button
                variant="outline"
                fullWidth
                onClick={() => setSelectedSignal(null)}
              >
                Close
              </Button>
              <Button
                variant="primary"
                fullWidth
                onClick={() => {
                  onExecuteSignal(selectedSignal);
                  setSelectedSignal(null);
                }}
              >
                Execute Signal
              </Button>
            </div>
          </div>
        </Modal>
      )}
    </Card>
  );
};

