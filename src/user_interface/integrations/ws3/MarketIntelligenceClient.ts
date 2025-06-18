// WS3 Market Intelligence Integration Library for WS6-P3
// Deep integration with market intelligence systems, predictive analytics, and decision support

import { EventEmitter } from 'events';

// Market Intelligence Types and Interfaces
export interface MarketIntelligenceData {
  timestamp: number;
  marketSentiment: MarketSentiment;
  technicalIndicators: TechnicalIndicators;
  fundamentalAnalysis: FundamentalAnalysis;
  newsAnalysis: NewsAnalysis;
  socialSentiment: SocialSentiment;
  economicIndicators: EconomicIndicators;
  riskMetrics: RiskMetrics;
  predictions: MarketPredictions;
}

export interface MarketSentiment {
  overall: 'bullish' | 'bearish' | 'neutral';
  confidence: number; // 0-1
  volatilityIndex: number;
  fearGreedIndex: number;
  momentum: 'accelerating' | 'decelerating' | 'stable';
  sectors: SectorSentiment[];
}

export interface SectorSentiment {
  sector: string;
  sentiment: 'bullish' | 'bearish' | 'neutral';
  confidence: number;
  performance: number;
  outlook: string;
}

export interface TechnicalIndicators {
  trend: {
    direction: 'up' | 'down' | 'sideways';
    strength: number; // 0-1
    duration: number; // days
  };
  support: number[];
  resistance: number[];
  rsi: number;
  macd: {
    signal: number;
    histogram: number;
    crossover: 'bullish' | 'bearish' | 'none';
  };
  bollinger: {
    upper: number;
    middle: number;
    lower: number;
    squeeze: boolean;
  };
  volume: {
    average: number;
    current: number;
    trend: 'increasing' | 'decreasing' | 'stable';
  };
}

export interface FundamentalAnalysis {
  valuation: {
    pe: number;
    pb: number;
    ps: number;
    ev_ebitda: number;
    peg: number;
  };
  growth: {
    revenue: number;
    earnings: number;
    bookValue: number;
  };
  profitability: {
    roe: number;
    roa: number;
    grossMargin: number;
    operatingMargin: number;
    netMargin: number;
  };
  financial_health: {
    debtToEquity: number;
    currentRatio: number;
    quickRatio: number;
    interestCoverage: number;
  };
}

export interface NewsAnalysis {
  sentiment: 'positive' | 'negative' | 'neutral';
  relevance: number; // 0-1
  impact: 'high' | 'medium' | 'low';
  topics: string[];
  sources: NewsSource[];
  summary: string;
  keyEvents: NewsEvent[];
}

export interface NewsSource {
  name: string;
  credibility: number; // 0-1
  bias: 'left' | 'right' | 'center';
  sentiment: 'positive' | 'negative' | 'neutral';
}

export interface NewsEvent {
  title: string;
  timestamp: number;
  impact: 'high' | 'medium' | 'low';
  sentiment: 'positive' | 'negative' | 'neutral';
  relevance: number;
}

export interface SocialSentiment {
  platforms: {
    twitter: PlatformSentiment;
    reddit: PlatformSentiment;
    stocktwits: PlatformSentiment;
    discord: PlatformSentiment;
  };
  overall: 'bullish' | 'bearish' | 'neutral';
  volume: number;
  trending: string[];
}

export interface PlatformSentiment {
  sentiment: 'bullish' | 'bearish' | 'neutral';
  volume: number;
  engagement: number;
  influencerSentiment: 'bullish' | 'bearish' | 'neutral';
}

export interface EconomicIndicators {
  gdp: {
    current: number;
    forecast: number;
    trend: 'growing' | 'contracting' | 'stable';
  };
  inflation: {
    current: number;
    forecast: number;
    trend: 'rising' | 'falling' | 'stable';
  };
  unemployment: {
    current: number;
    forecast: number;
    trend: 'rising' | 'falling' | 'stable';
  };
  interestRates: {
    current: number;
    forecast: number;
    nextMeeting: number;
    probability: number;
  };
  currency: {
    strength: number;
    volatility: number;
    trend: 'strengthening' | 'weakening' | 'stable';
  };
}

export interface RiskMetrics {
  var: {
    daily: number;
    weekly: number;
    monthly: number;
  };
  beta: number;
  sharpe: number;
  sortino: number;
  maxDrawdown: number;
  volatility: number;
  correlation: { [symbol: string]: number };
}

export interface MarketPredictions {
  shortTerm: Prediction; // 1-7 days
  mediumTerm: Prediction; // 1-4 weeks
  longTerm: Prediction; // 1-6 months
  scenarios: Scenario[];
}

export interface Prediction {
  direction: 'up' | 'down' | 'sideways';
  confidence: number; // 0-1
  targetPrice: number;
  probability: number; // 0-1
  timeframe: number; // days
  factors: string[];
}

export interface Scenario {
  name: string;
  probability: number; // 0-1
  impact: 'high' | 'medium' | 'low';
  description: string;
  priceTarget: number;
  timeframe: number;
}

// Market Intelligence Configuration
export interface WS3Config {
  apiEndpoint: string;
  websocketEndpoint: string;
  apiKey: string;
  updateInterval: number; // milliseconds
  symbols: string[];
  enableRealTime: boolean;
  enablePredictions: boolean;
  enableSentiment: boolean;
  enableNews: boolean;
  enableSocial: boolean;
  riskThresholds: {
    var: number;
    volatility: number;
    drawdown: number;
  };
}

// WS3 Market Intelligence Client
export class WS3MarketIntelligence extends EventEmitter {
  private config: WS3Config;
  private websocket: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private updateTimer: NodeJS.Timeout | null = null;
  private isConnected = false;
  private cache = new Map<string, MarketIntelligenceData>();

  constructor(config: WS3Config) {
    super();
    this.config = config;
    this.initialize();
  }

  private async initialize(): Promise<void> {
    try {
      await this.connect();
      this.startPeriodicUpdates();
      this.emit('initialized');
    } catch (error) {
      this.emit('error', error);
    }
  }

  private async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.websocket = new WebSocket(this.config.websocketEndpoint);

        this.websocket.onopen = () => {
          this.isConnected = true;
          this.reconnectAttempts = 0;
          this.emit('connected');
          
          // Subscribe to market intelligence feeds
          this.subscribe();
          resolve();
        };

        this.websocket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.handleIntelligenceUpdate(data);
          } catch (error) {
            this.emit('error', new Error('Failed to parse intelligence data'));
          }
        };

        this.websocket.onclose = () => {
          this.isConnected = false;
          this.emit('disconnected');
          this.handleReconnect();
        };

        this.websocket.onerror = (error) => {
          this.emit('error', error);
          reject(error);
        };

      } catch (error) {
        reject(error);
      }
    });
  }

  private subscribe(): void {
    if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) return;

    const subscriptionMessage = {
      type: 'subscribe',
      channels: [
        'market_sentiment',
        'technical_indicators',
        'fundamental_analysis',
        'news_analysis',
        'social_sentiment',
        'economic_indicators',
        'risk_metrics',
        'predictions'
      ],
      symbols: this.config.symbols,
      apiKey: this.config.apiKey
    };

    this.websocket.send(JSON.stringify(subscriptionMessage));
  }

  private handleIntelligenceUpdate(data: any): void {
    try {
      const intelligence: MarketIntelligenceData = this.processIntelligenceData(data);
      
      // Cache the data
      this.cache.set(data.symbol || 'market', intelligence);
      
      // Emit specific events based on data type
      this.emit('intelligence_update', intelligence);
      
      if (data.type === 'market_sentiment') {
        this.emit('sentiment_update', intelligence.marketSentiment);
      }
      
      if (data.type === 'predictions') {
        this.emit('prediction_update', intelligence.predictions);
      }
      
      if (data.type === 'risk_alert') {
        this.emit('risk_alert', intelligence.riskMetrics);
      }

      // Check for significant changes
      this.analyzeSignificantChanges(intelligence);
      
    } catch (error) {
      this.emit('error', new Error('Failed to process intelligence update'));
    }
  }

  private processIntelligenceData(rawData: any): MarketIntelligenceData {
    // Process and normalize the raw data from WS3
    return {
      timestamp: Date.now(),
      marketSentiment: this.processMarketSentiment(rawData.sentiment),
      technicalIndicators: this.processTechnicalIndicators(rawData.technical),
      fundamentalAnalysis: this.processFundamentalAnalysis(rawData.fundamental),
      newsAnalysis: this.processNewsAnalysis(rawData.news),
      socialSentiment: this.processSocialSentiment(rawData.social),
      economicIndicators: this.processEconomicIndicators(rawData.economic),
      riskMetrics: this.processRiskMetrics(rawData.risk),
      predictions: this.processPredictions(rawData.predictions)
    };
  }

  private processMarketSentiment(data: any): MarketSentiment {
    return {
      overall: data?.overall || 'neutral',
      confidence: data?.confidence || 0.5,
      volatilityIndex: data?.volatilityIndex || 50,
      fearGreedIndex: data?.fearGreedIndex || 50,
      momentum: data?.momentum || 'stable',
      sectors: data?.sectors || []
    };
  }

  private processTechnicalIndicators(data: any): TechnicalIndicators {
    return {
      trend: {
        direction: data?.trend?.direction || 'sideways',
        strength: data?.trend?.strength || 0.5,
        duration: data?.trend?.duration || 0
      },
      support: data?.support || [],
      resistance: data?.resistance || [],
      rsi: data?.rsi || 50,
      macd: {
        signal: data?.macd?.signal || 0,
        histogram: data?.macd?.histogram || 0,
        crossover: data?.macd?.crossover || 'none'
      },
      bollinger: {
        upper: data?.bollinger?.upper || 0,
        middle: data?.bollinger?.middle || 0,
        lower: data?.bollinger?.lower || 0,
        squeeze: data?.bollinger?.squeeze || false
      },
      volume: {
        average: data?.volume?.average || 0,
        current: data?.volume?.current || 0,
        trend: data?.volume?.trend || 'stable'
      }
    };
  }

  private processFundamentalAnalysis(data: any): FundamentalAnalysis {
    return {
      valuation: {
        pe: data?.valuation?.pe || 0,
        pb: data?.valuation?.pb || 0,
        ps: data?.valuation?.ps || 0,
        ev_ebitda: data?.valuation?.ev_ebitda || 0,
        peg: data?.valuation?.peg || 0
      },
      growth: {
        revenue: data?.growth?.revenue || 0,
        earnings: data?.growth?.earnings || 0,
        bookValue: data?.growth?.bookValue || 0
      },
      profitability: {
        roe: data?.profitability?.roe || 0,
        roa: data?.profitability?.roa || 0,
        grossMargin: data?.profitability?.grossMargin || 0,
        operatingMargin: data?.profitability?.operatingMargin || 0,
        netMargin: data?.profitability?.netMargin || 0
      },
      financial_health: {
        debtToEquity: data?.financial_health?.debtToEquity || 0,
        currentRatio: data?.financial_health?.currentRatio || 0,
        quickRatio: data?.financial_health?.quickRatio || 0,
        interestCoverage: data?.financial_health?.interestCoverage || 0
      }
    };
  }

  private processNewsAnalysis(data: any): NewsAnalysis {
    return {
      sentiment: data?.sentiment || 'neutral',
      relevance: data?.relevance || 0.5,
      impact: data?.impact || 'medium',
      topics: data?.topics || [],
      sources: data?.sources || [],
      summary: data?.summary || '',
      keyEvents: data?.keyEvents || []
    };
  }

  private processSocialSentiment(data: any): SocialSentiment {
    return {
      platforms: {
        twitter: data?.platforms?.twitter || { sentiment: 'neutral', volume: 0, engagement: 0, influencerSentiment: 'neutral' },
        reddit: data?.platforms?.reddit || { sentiment: 'neutral', volume: 0, engagement: 0, influencerSentiment: 'neutral' },
        stocktwits: data?.platforms?.stocktwits || { sentiment: 'neutral', volume: 0, engagement: 0, influencerSentiment: 'neutral' },
        discord: data?.platforms?.discord || { sentiment: 'neutral', volume: 0, engagement: 0, influencerSentiment: 'neutral' }
      },
      overall: data?.overall || 'neutral',
      volume: data?.volume || 0,
      trending: data?.trending || []
    };
  }

  private processEconomicIndicators(data: any): EconomicIndicators {
    return {
      gdp: {
        current: data?.gdp?.current || 0,
        forecast: data?.gdp?.forecast || 0,
        trend: data?.gdp?.trend || 'stable'
      },
      inflation: {
        current: data?.inflation?.current || 0,
        forecast: data?.inflation?.forecast || 0,
        trend: data?.inflation?.trend || 'stable'
      },
      unemployment: {
        current: data?.unemployment?.current || 0,
        forecast: data?.unemployment?.forecast || 0,
        trend: data?.unemployment?.trend || 'stable'
      },
      interestRates: {
        current: data?.interestRates?.current || 0,
        forecast: data?.interestRates?.forecast || 0,
        nextMeeting: data?.interestRates?.nextMeeting || 0,
        probability: data?.interestRates?.probability || 0
      },
      currency: {
        strength: data?.currency?.strength || 0,
        volatility: data?.currency?.volatility || 0,
        trend: data?.currency?.trend || 'stable'
      }
    };
  }

  private processRiskMetrics(data: any): RiskMetrics {
    return {
      var: {
        daily: data?.var?.daily || 0,
        weekly: data?.var?.weekly || 0,
        monthly: data?.var?.monthly || 0
      },
      beta: data?.beta || 1,
      sharpe: data?.sharpe || 0,
      sortino: data?.sortino || 0,
      maxDrawdown: data?.maxDrawdown || 0,
      volatility: data?.volatility || 0,
      correlation: data?.correlation || {}
    };
  }

  private processPredictions(data: any): MarketPredictions {
    return {
      shortTerm: data?.shortTerm || {
        direction: 'sideways',
        confidence: 0.5,
        targetPrice: 0,
        probability: 0.5,
        timeframe: 7,
        factors: []
      },
      mediumTerm: data?.mediumTerm || {
        direction: 'sideways',
        confidence: 0.5,
        targetPrice: 0,
        probability: 0.5,
        timeframe: 30,
        factors: []
      },
      longTerm: data?.longTerm || {
        direction: 'sideways',
        confidence: 0.5,
        targetPrice: 0,
        probability: 0.5,
        timeframe: 180,
        factors: []
      },
      scenarios: data?.scenarios || []
    };
  }

  private analyzeSignificantChanges(intelligence: MarketIntelligenceData): void {
    // Analyze for significant changes and emit alerts
    const { marketSentiment, riskMetrics, predictions } = intelligence;

    // Sentiment change alerts
    if (marketSentiment.confidence > 0.8) {
      this.emit('high_confidence_signal', {
        type: 'sentiment',
        sentiment: marketSentiment.overall,
        confidence: marketSentiment.confidence
      });
    }

    // Risk alerts
    if (riskMetrics.var.daily > this.config.riskThresholds.var) {
      this.emit('risk_alert', {
        type: 'var_exceeded',
        value: riskMetrics.var.daily,
        threshold: this.config.riskThresholds.var
      });
    }

    // Prediction alerts
    if (predictions.shortTerm.confidence > 0.8) {
      this.emit('high_confidence_prediction', {
        prediction: predictions.shortTerm,
        timeframe: 'short'
      });
    }
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
      
      setTimeout(() => {
        this.connect().catch(() => {
          // Reconnection failed, will try again
        });
      }, delay);
    } else {
      this.emit('max_reconnect_attempts_reached');
    }
  }

  private startPeriodicUpdates(): void {
    if (this.updateTimer) {
      clearInterval(this.updateTimer);
    }

    this.updateTimer = setInterval(() => {
      this.requestIntelligenceUpdate();
    }, this.config.updateInterval);
  }

  private async requestIntelligenceUpdate(): Promise<void> {
    try {
      const response = await fetch(`${this.config.apiEndpoint}/intelligence`, {
        headers: {
          'Authorization': `Bearer ${this.config.apiKey}`,
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        const data = await response.json();
        this.handleIntelligenceUpdate(data);
      }
    } catch (error) {
      this.emit('error', error);
    }
  }

  // Public API Methods
  public getLatestIntelligence(symbol?: string): MarketIntelligenceData | null {
    return this.cache.get(symbol || 'market') || null;
  }

  public async getHistoricalIntelligence(
    symbol: string, 
    timeframe: string, 
    limit: number = 100
  ): Promise<MarketIntelligenceData[]> {
    try {
      const response = await fetch(
        `${this.config.apiEndpoint}/intelligence/historical?symbol=${symbol}&timeframe=${timeframe}&limit=${limit}`,
        {
          headers: {
            'Authorization': `Bearer ${this.config.apiKey}`,
            'Content-Type': 'application/json'
          }
        }
      );

      if (response.ok) {
        const data = await response.json();
        return data.map((item: any) => this.processIntelligenceData(item));
      }
      
      return [];
    } catch (error) {
      this.emit('error', error);
      return [];
    }
  }

  public async getPredictionAccuracy(timeframe: string = '30d'): Promise<any> {
    try {
      const response = await fetch(
        `${this.config.apiEndpoint}/intelligence/accuracy?timeframe=${timeframe}`,
        {
          headers: {
            'Authorization': `Bearer ${this.config.apiKey}`,
            'Content-Type': 'application/json'
          }
        }
      );

      if (response.ok) {
        return await response.json();
      }
      
      return null;
    } catch (error) {
      this.emit('error', error);
      return null;
    }
  }

  public updateConfig(newConfig: Partial<WS3Config>): void {
    this.config = { ...this.config, ...newConfig };
    
    if (newConfig.symbols) {
      this.subscribe();
    }
    
    if (newConfig.updateInterval) {
      this.startPeriodicUpdates();
    }
  }

  public disconnect(): void {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
    
    if (this.updateTimer) {
      clearInterval(this.updateTimer);
      this.updateTimer = null;
    }
    
    this.isConnected = false;
    this.emit('disconnected');
  }

  public isConnectedToWS3(): boolean {
    return this.isConnected;
  }
}

// Utility Functions
export const createWS3Config = (overrides: Partial<WS3Config> = {}): WS3Config => {
  return {
    apiEndpoint: process.env.WS3_API_ENDPOINT || 'https://api.ws3.alluse.com',
    websocketEndpoint: process.env.WS3_WS_ENDPOINT || 'wss://ws.ws3.alluse.com',
    apiKey: process.env.WS3_API_KEY || '',
    updateInterval: 5000, // 5 seconds
    symbols: ['SPY', 'QQQ', 'IWM', 'VTI'],
    enableRealTime: true,
    enablePredictions: true,
    enableSentiment: true,
    enableNews: true,
    enableSocial: true,
    riskThresholds: {
      var: 0.05, // 5%
      volatility: 0.3, // 30%
      drawdown: 0.2 // 20%
    },
    ...overrides
  };
};

export const formatIntelligenceData = (data: MarketIntelligenceData): string => {
  return JSON.stringify(data, null, 2);
};

export const calculateIntelligenceScore = (data: MarketIntelligenceData): number => {
  const sentimentScore = data.marketSentiment.confidence;
  const technicalScore = data.technicalIndicators.trend.strength;
  const predictionScore = data.predictions.shortTerm.confidence;
  
  return (sentimentScore + technicalScore + predictionScore) / 3;
};

// Export all types and classes
export default WS3MarketIntelligence;

