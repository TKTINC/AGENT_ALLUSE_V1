import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Brain, 
  AlertTriangle, 
  Target, 
  BarChart3, 
  PieChart, 
  LineChart,
  Zap,
  Eye,
  Shield,
  Clock,
  Globe,
  Users,
  MessageSquare,
  Newspaper,
  DollarSign,
  Percent,
  ArrowUp,
  ArrowDown,
  Minus
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  WS3MarketIntelligence, 
  MarketIntelligenceData, 
  MarketSentiment, 
  TechnicalIndicators,
  MarketPredictions,
  createWS3Config 
} from '../../integrations/ws3/MarketIntelligenceClient';

// Intelligence Dashboard Component
export const IntelligenceDashboard: React.FC<{
  symbols?: string[];
  className?: string;
}> = ({ symbols = ['SPY', 'QQQ', 'IWM'], className = '' }) => {
  const [intelligence, setIntelligence] = useState<MarketIntelligenceData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState(symbols[0]);
  const [timeframe, setTimeframe] = useState<'1h' | '1d' | '1w' | '1m'>('1d');
  const [ws3Client, setWs3Client] = useState<WS3MarketIntelligence | null>(null);

  // Initialize WS3 client
  useEffect(() => {
    const config = createWS3Config({
      symbols,
      updateInterval: 3000 // 3 seconds for real-time updates
    });

    const client = new WS3MarketIntelligence(config);
    
    client.on('connected', () => setIsConnected(true));
    client.on('disconnected', () => setIsConnected(false));
    client.on('intelligence_update', (data: MarketIntelligenceData) => {
      setIntelligence(data);
    });

    setWs3Client(client);

    return () => {
      client.disconnect();
    };
  }, [symbols]);

  const sentimentColor = useMemo(() => {
    if (!intelligence?.marketSentiment) return 'text-gray-500';
    
    switch (intelligence.marketSentiment.overall) {
      case 'bullish': return 'text-green-500';
      case 'bearish': return 'text-red-500';
      default: return 'text-yellow-500';
    }
  }, [intelligence?.marketSentiment]);

  const confidenceLevel = useMemo(() => {
    if (!intelligence?.marketSentiment) return 'Low';
    
    const confidence = intelligence.marketSentiment.confidence;
    if (confidence >= 0.8) return 'Very High';
    if (confidence >= 0.6) return 'High';
    if (confidence >= 0.4) return 'Medium';
    return 'Low';
  }, [intelligence?.marketSentiment]);

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Brain className="w-6 h-6 text-blue-500" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            Market Intelligence
          </h2>
          <div className={`flex items-center gap-1 ${isConnected ? 'text-green-500' : 'text-red-500'}`}>
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="text-xs font-medium">
              {isConnected ? 'Live' : 'Disconnected'}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          >
            {symbols.map(symbol => (
              <option key={symbol} value={symbol}>{symbol}</option>
            ))}
          </select>
          
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value as any)}
            className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          >
            <option value="1h">1H</option>
            <option value="1d">1D</option>
            <option value="1w">1W</option>
            <option value="1m">1M</option>
          </select>
        </div>
      </div>

      {intelligence ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          {/* Market Sentiment */}
          <MarketSentimentCard sentiment={intelligence.marketSentiment} />
          
          {/* Technical Analysis */}
          <TechnicalAnalysisCard indicators={intelligence.technicalIndicators} />
          
          {/* Predictions */}
          <PredictionsCard predictions={intelligence.predictions} />
          
          {/* Risk Metrics */}
          <RiskMetricsCard metrics={intelligence.riskMetrics} />
          
          {/* News Analysis */}
          <NewsAnalysisCard news={intelligence.newsAnalysis} />
          
          {/* Social Sentiment */}
          <SocialSentimentCard social={intelligence.socialSentiment} />
        </div>
      ) : (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <Activity className="w-12 h-12 text-gray-400 mx-auto mb-4 animate-pulse" />
            <p className="text-gray-500 dark:text-gray-400">
              Loading market intelligence...
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

// Market Sentiment Card
const MarketSentimentCard: React.FC<{ sentiment: MarketSentiment }> = ({ sentiment }) => {
  const getSentimentIcon = () => {
    switch (sentiment.overall) {
      case 'bullish': return <TrendingUp className="w-5 h-5 text-green-500" />;
      case 'bearish': return <TrendingDown className="w-5 h-5 text-red-500" />;
      default: return <Minus className="w-5 h-5 text-yellow-500" />;
    }
  };

  const getSentimentColor = () => {
    switch (sentiment.overall) {
      case 'bullish': return 'text-green-500 bg-green-50 dark:bg-green-900/20';
      case 'bearish': return 'text-red-500 bg-red-50 dark:bg-red-900/20';
      default: return 'text-yellow-500 bg-yellow-50 dark:bg-yellow-900/20';
    }
  };

  return (
    <motion.div
      className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Market Sentiment
        </h3>
        {getSentimentIcon()}
      </div>

      <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getSentimentColor()}`}>
        {sentiment.overall.toUpperCase()}
      </div>

      <div className="mt-3 space-y-2">
        <div className="flex justify-between text-xs">
          <span className="text-gray-500 dark:text-gray-400">Confidence</span>
          <span className="font-medium">{(sentiment.confidence * 100).toFixed(1)}%</span>
        </div>
        
        <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
          <div
            className="bg-blue-500 h-2 rounded-full transition-all duration-300"
            style={{ width: `${sentiment.confidence * 100}%` }}
          />
        </div>

        <div className="grid grid-cols-2 gap-2 mt-3 text-xs">
          <div>
            <span className="text-gray-500 dark:text-gray-400">Fear & Greed</span>
            <div className="font-medium">{sentiment.fearGreedIndex}</div>
          </div>
          <div>
            <span className="text-gray-500 dark:text-gray-400">Volatility</span>
            <div className="font-medium">{sentiment.volatilityIndex}</div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

// Technical Analysis Card
const TechnicalAnalysisCard: React.FC<{ indicators: TechnicalIndicators }> = ({ indicators }) => {
  const getTrendIcon = () => {
    switch (indicators.trend.direction) {
      case 'up': return <ArrowUp className="w-4 h-4 text-green-500" />;
      case 'down': return <ArrowDown className="w-4 h-4 text-red-500" />;
      default: return <Minus className="w-4 h-4 text-yellow-500" />;
    }
  };

  const getRSIColor = () => {
    if (indicators.rsi >= 70) return 'text-red-500';
    if (indicators.rsi <= 30) return 'text-green-500';
    return 'text-gray-600 dark:text-gray-400';
  };

  return (
    <motion.div
      className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.1 }}
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Technical Analysis
        </h3>
        <BarChart3 className="w-5 h-5 text-blue-500" />
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500 dark:text-gray-400">Trend</span>
          <div className="flex items-center gap-1">
            {getTrendIcon()}
            <span className="text-xs font-medium capitalize">
              {indicators.trend.direction}
            </span>
          </div>
        </div>

        <div className="flex justify-between text-xs">
          <span className="text-gray-500 dark:text-gray-400">Strength</span>
          <span className="font-medium">{(indicators.trend.strength * 100).toFixed(1)}%</span>
        </div>

        <div className="flex justify-between text-xs">
          <span className="text-gray-500 dark:text-gray-400">RSI</span>
          <span className={`font-medium ${getRSIColor()}`}>
            {indicators.rsi.toFixed(1)}
          </span>
        </div>

        <div className="flex justify-between text-xs">
          <span className="text-gray-500 dark:text-gray-400">MACD Signal</span>
          <span className={`font-medium ${
            indicators.macd.crossover === 'bullish' ? 'text-green-500' :
            indicators.macd.crossover === 'bearish' ? 'text-red-500' : 'text-gray-600'
          }`}>
            {indicators.macd.crossover}
          </span>
        </div>

        <div className="flex justify-between text-xs">
          <span className="text-gray-500 dark:text-gray-400">Volume Trend</span>
          <span className="font-medium capitalize">{indicators.volume.trend}</span>
        </div>
      </div>
    </motion.div>
  );
};

// Predictions Card
const PredictionsCard: React.FC<{ predictions: MarketPredictions }> = ({ predictions }) => {
  const [selectedTimeframe, setSelectedTimeframe] = useState<'shortTerm' | 'mediumTerm' | 'longTerm'>('shortTerm');
  
  const currentPrediction = predictions[selectedTimeframe];

  const getDirectionIcon = (direction: string) => {
    switch (direction) {
      case 'up': return <ArrowUp className="w-4 h-4 text-green-500" />;
      case 'down': return <ArrowDown className="w-4 h-4 text-red-500" />;
      default: return <Minus className="w-4 h-4 text-yellow-500" />;
    }
  };

  return (
    <motion.div
      className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.2 }}
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
          AI Predictions
        </h3>
        <Target className="w-5 h-5 text-purple-500" />
      </div>

      <div className="flex gap-1 mb-3">
        {(['shortTerm', 'mediumTerm', 'longTerm'] as const).map((timeframe) => (
          <button
            key={timeframe}
            onClick={() => setSelectedTimeframe(timeframe)}
            className={`px-2 py-1 text-xs rounded transition-colors ${
              selectedTimeframe === timeframe
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 dark:bg-gray-600 text-gray-600 dark:text-gray-400 hover:bg-gray-300 dark:hover:bg-gray-500'
            }`}
          >
            {timeframe === 'shortTerm' ? '1-7d' : timeframe === 'mediumTerm' ? '1-4w' : '1-6m'}
          </button>
        ))}
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500 dark:text-gray-400">Direction</span>
          <div className="flex items-center gap-1">
            {getDirectionIcon(currentPrediction.direction)}
            <span className="text-xs font-medium capitalize">
              {currentPrediction.direction}
            </span>
          </div>
        </div>

        <div className="flex justify-between text-xs">
          <span className="text-gray-500 dark:text-gray-400">Confidence</span>
          <span className="font-medium">{(currentPrediction.confidence * 100).toFixed(1)}%</span>
        </div>

        <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
          <div
            className="bg-purple-500 h-2 rounded-full transition-all duration-300"
            style={{ width: `${currentPrediction.confidence * 100}%` }}
          />
        </div>

        <div className="flex justify-between text-xs">
          <span className="text-gray-500 dark:text-gray-400">Target Price</span>
          <span className="font-medium">${currentPrediction.targetPrice.toFixed(2)}</span>
        </div>

        <div className="flex justify-between text-xs">
          <span className="text-gray-500 dark:text-gray-400">Probability</span>
          <span className="font-medium">{(currentPrediction.probability * 100).toFixed(1)}%</span>
        </div>
      </div>
    </motion.div>
  );
};

// Risk Metrics Card
const RiskMetricsCard: React.FC<{ metrics: any }> = ({ metrics }) => {
  const getVaRColor = (var_value: number) => {
    if (var_value > 0.05) return 'text-red-500';
    if (var_value > 0.03) return 'text-yellow-500';
    return 'text-green-500';
  };

  return (
    <motion.div
      className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.3 }}
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Risk Metrics
        </h3>
        <Shield className="w-5 h-5 text-orange-500" />
      </div>

      <div className="space-y-3">
        <div className="flex justify-between text-xs">
          <span className="text-gray-500 dark:text-gray-400">Daily VaR</span>
          <span className={`font-medium ${getVaRColor(metrics.var.daily)}`}>
            {(metrics.var.daily * 100).toFixed(2)}%
          </span>
        </div>

        <div className="flex justify-between text-xs">
          <span className="text-gray-500 dark:text-gray-400">Beta</span>
          <span className="font-medium">{metrics.beta.toFixed(2)}</span>
        </div>

        <div className="flex justify-between text-xs">
          <span className="text-gray-500 dark:text-gray-400">Sharpe Ratio</span>
          <span className="font-medium">{metrics.sharpe.toFixed(2)}</span>
        </div>

        <div className="flex justify-between text-xs">
          <span className="text-gray-500 dark:text-gray-400">Max Drawdown</span>
          <span className="font-medium text-red-500">
            {(metrics.maxDrawdown * 100).toFixed(1)}%
          </span>
        </div>

        <div className="flex justify-between text-xs">
          <span className="text-gray-500 dark:text-gray-400">Volatility</span>
          <span className="font-medium">
            {(metrics.volatility * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    </motion.div>
  );
};

// News Analysis Card
const NewsAnalysisCard: React.FC<{ news: any }> = ({ news }) => {
  const getSentimentColor = () => {
    switch (news.sentiment) {
      case 'positive': return 'text-green-500 bg-green-50 dark:bg-green-900/20';
      case 'negative': return 'text-red-500 bg-red-50 dark:bg-red-900/20';
      default: return 'text-yellow-500 bg-yellow-50 dark:bg-yellow-900/20';
    }
  };

  const getImpactColor = () => {
    switch (news.impact) {
      case 'high': return 'text-red-500';
      case 'medium': return 'text-yellow-500';
      default: return 'text-green-500';
    }
  };

  return (
    <motion.div
      className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.4 }}
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
          News Analysis
        </h3>
        <Newspaper className="w-5 h-5 text-blue-500" />
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500 dark:text-gray-400">Sentiment</span>
          <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getSentimentColor()}`}>
            {news.sentiment}
          </div>
        </div>

        <div className="flex justify-between text-xs">
          <span className="text-gray-500 dark:text-gray-400">Relevance</span>
          <span className="font-medium">{(news.relevance * 100).toFixed(1)}%</span>
        </div>

        <div className="flex justify-between text-xs">
          <span className="text-gray-500 dark:text-gray-400">Impact</span>
          <span className={`font-medium ${getImpactColor()}`}>
            {news.impact}
          </span>
        </div>

        <div className="text-xs text-gray-600 dark:text-gray-400 line-clamp-3">
          {news.summary || 'No recent news analysis available.'}
        </div>

        {news.keyEvents && news.keyEvents.length > 0 && (
          <div className="text-xs">
            <span className="text-gray-500 dark:text-gray-400">Key Events: </span>
            <span className="font-medium">{news.keyEvents.length}</span>
          </div>
        )}
      </div>
    </motion.div>
  );
};

// Social Sentiment Card
const SocialSentimentCard: React.FC<{ social: any }> = ({ social }) => {
  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'bullish': return 'text-green-500';
      case 'bearish': return 'text-red-500';
      default: return 'text-yellow-500';
    }
  };

  const platforms = [
    { name: 'Twitter', key: 'twitter', icon: MessageSquare },
    { name: 'Reddit', key: 'reddit', icon: Users },
    { name: 'StockTwits', key: 'stocktwits', icon: TrendingUp },
    { name: 'Discord', key: 'discord', icon: MessageSquare }
  ];

  return (
    <motion.div
      className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.5 }}
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Social Sentiment
        </h3>
        <Users className="w-5 h-5 text-green-500" />
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500 dark:text-gray-400">Overall</span>
          <span className={`text-xs font-medium ${getSentimentColor(social.overall)}`}>
            {social.overall}
          </span>
        </div>

        <div className="flex justify-between text-xs">
          <span className="text-gray-500 dark:text-gray-400">Volume</span>
          <span className="font-medium">{social.volume.toLocaleString()}</span>
        </div>

        <div className="space-y-2">
          {platforms.map(({ name, key, icon: Icon }) => {
            const platformData = social.platforms[key];
            return (
              <div key={key} className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-1">
                  <Icon className="w-3 h-3 text-gray-400" />
                  <span className="text-gray-500 dark:text-gray-400">{name}</span>
                </div>
                <span className={`font-medium ${getSentimentColor(platformData.sentiment)}`}>
                  {platformData.sentiment}
                </span>
              </div>
            );
          })}
        </div>

        {social.trending && social.trending.length > 0 && (
          <div className="text-xs">
            <span className="text-gray-500 dark:text-gray-400">Trending: </span>
            <span className="font-medium">{social.trending.slice(0, 2).join(', ')}</span>
          </div>
        )}
      </div>
    </motion.div>
  );
};

// Export all components
export {
  MarketSentimentCard,
  TechnicalAnalysisCard,
  PredictionsCard,
  RiskMetricsCard,
  NewsAnalysisCard,
  SocialSentimentCard
};

