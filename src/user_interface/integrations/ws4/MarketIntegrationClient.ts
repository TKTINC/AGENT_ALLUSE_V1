// WS4 Market Integration Client for WS6-P3
// Real-time trading, order management, execution, and market data integration

import { EventEmitter } from 'events';

// Trading Types and Interfaces
export interface Order {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit' | 'stop' | 'stop_limit';
  quantity: number;
  price?: number;
  stopPrice?: number;
  timeInForce: 'day' | 'gtc' | 'ioc' | 'fok';
  status: 'pending' | 'open' | 'filled' | 'cancelled' | 'rejected' | 'expired';
  filledQuantity: number;
  averagePrice: number;
  timestamp: number;
  lastUpdate: number;
  account: string;
  strategy?: string;
  tags?: string[];
}

export interface Position {
  symbol: string;
  quantity: number;
  averagePrice: number;
  marketValue: number;
  unrealizedPnL: number;
  realizedPnL: number;
  dayPnL: number;
  side: 'long' | 'short';
  account: string;
  lastUpdate: number;
}

export interface MarketData {
  symbol: string;
  price: number;
  bid: number;
  ask: number;
  bidSize: number;
  askSize: number;
  volume: number;
  dayHigh: number;
  dayLow: number;
  dayOpen: number;
  previousClose: number;
  change: number;
  changePercent: number;
  timestamp: number;
}

export interface Trade {
  id: string;
  orderId: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  timestamp: number;
  commission: number;
  account: string;
}

export interface Account {
  id: string;
  name: string;
  type: 'cash' | 'margin' | 'ira';
  buyingPower: number;
  totalValue: number;
  cashBalance: number;
  marginUsed: number;
  dayTradingBuyingPower: number;
  maintenanceMargin: number;
  equity: number;
  lastUpdate: number;
}

export interface OrderBook {
  symbol: string;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  timestamp: number;
}

export interface OrderBookLevel {
  price: number;
  size: number;
  orders: number;
}

export interface TradingSession {
  market: string;
  status: 'pre_market' | 'open' | 'closed' | 'after_hours';
  nextOpen?: number;
  nextClose?: number;
}

export interface RiskLimits {
  maxOrderSize: number;
  maxDayLoss: number;
  maxPositionSize: number;
  allowedSymbols: string[];
  blockedSymbols: string[];
  maxLeverage: number;
}

// WS4 Configuration
export interface WS4Config {
  apiEndpoint: string;
  websocketEndpoint: string;
  apiKey: string;
  secretKey: string;
  environment: 'sandbox' | 'live';
  defaultAccount: string;
  enableRealTimeData: boolean;
  enableOrderBook: boolean;
  enableTrades: boolean;
  riskLimits: RiskLimits;
  reconnectAttempts: number;
  heartbeatInterval: number;
}

// WS4 Market Integration Client
export class WS4MarketIntegration extends EventEmitter {
  private config: WS4Config;
  private websocket: WebSocket | null = null;
  private reconnectAttempts = 0;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private isConnected = false;
  private isAuthenticated = false;
  
  // Data caches
  private orders = new Map<string, Order>();
  private positions = new Map<string, Position>();
  private marketData = new Map<string, MarketData>();
  private accounts = new Map<string, Account>();
  private orderBooks = new Map<string, OrderBook>();
  private trades: Trade[] = [];

  constructor(config: WS4Config) {
    super();
    this.config = config;
    this.initialize();
  }

  private async initialize(): Promise<void> {
    try {
      await this.connect();
      await this.authenticate();
      this.startHeartbeat();
      this.subscribeToDataFeeds();
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
          resolve();
        };

        this.websocket.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            this.emit('error', new Error('Failed to parse WebSocket message'));
          }
        };

        this.websocket.onclose = () => {
          this.isConnected = false;
          this.isAuthenticated = false;
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

  private async authenticate(): Promise<void> {
    if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected');
    }

    const timestamp = Date.now();
    const signature = this.generateSignature(timestamp);

    const authMessage = {
      type: 'auth',
      apiKey: this.config.apiKey,
      timestamp,
      signature
    };

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Authentication timeout'));
      }, 10000);

      const handleAuth = (message: any) => {
        if (message.type === 'auth_response') {
          clearTimeout(timeout);
          if (message.success) {
            this.isAuthenticated = true;
            this.emit('authenticated');
            resolve();
          } else {
            reject(new Error(message.error || 'Authentication failed'));
          }
        }
      };

      this.once('message', handleAuth);
      this.websocket!.send(JSON.stringify(authMessage));
    });
  }

  private generateSignature(timestamp: number): string {
    // In a real implementation, this would use HMAC-SHA256 or similar
    // For demo purposes, we'll use a simple hash
    const message = `${timestamp}${this.config.apiKey}`;
    return btoa(message); // Base64 encode for demo
  }

  private handleMessage(message: any): void {
    this.emit('message', message);

    switch (message.type) {
      case 'heartbeat':
        this.handleHeartbeat(message);
        break;
      case 'order_update':
        this.handleOrderUpdate(message);
        break;
      case 'position_update':
        this.handlePositionUpdate(message);
        break;
      case 'market_data':
        this.handleMarketData(message);
        break;
      case 'trade':
        this.handleTrade(message);
        break;
      case 'account_update':
        this.handleAccountUpdate(message);
        break;
      case 'order_book':
        this.handleOrderBook(message);
        break;
      case 'error':
        this.emit('error', new Error(message.error));
        break;
    }
  }

  private handleHeartbeat(message: any): void {
    // Respond to heartbeat to maintain connection
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify({ type: 'heartbeat_response' }));
    }
  }

  private handleOrderUpdate(message: any): void {
    const order: Order = {
      id: message.id,
      symbol: message.symbol,
      side: message.side,
      type: message.type,
      quantity: message.quantity,
      price: message.price,
      stopPrice: message.stopPrice,
      timeInForce: message.timeInForce,
      status: message.status,
      filledQuantity: message.filledQuantity || 0,
      averagePrice: message.averagePrice || 0,
      timestamp: message.timestamp,
      lastUpdate: Date.now(),
      account: message.account,
      strategy: message.strategy,
      tags: message.tags
    };

    this.orders.set(order.id, order);
    this.emit('order_update', order);

    // Emit specific status events
    if (message.status === 'filled') {
      this.emit('order_filled', order);
    } else if (message.status === 'cancelled') {
      this.emit('order_cancelled', order);
    } else if (message.status === 'rejected') {
      this.emit('order_rejected', order);
    }
  }

  private handlePositionUpdate(message: any): void {
    const position: Position = {
      symbol: message.symbol,
      quantity: message.quantity,
      averagePrice: message.averagePrice,
      marketValue: message.marketValue,
      unrealizedPnL: message.unrealizedPnL,
      realizedPnL: message.realizedPnL,
      dayPnL: message.dayPnL,
      side: message.quantity > 0 ? 'long' : 'short',
      account: message.account,
      lastUpdate: Date.now()
    };

    this.positions.set(position.symbol, position);
    this.emit('position_update', position);
  }

  private handleMarketData(message: any): void {
    const data: MarketData = {
      symbol: message.symbol,
      price: message.price,
      bid: message.bid,
      ask: message.ask,
      bidSize: message.bidSize,
      askSize: message.askSize,
      volume: message.volume,
      dayHigh: message.dayHigh,
      dayLow: message.dayLow,
      dayOpen: message.dayOpen,
      previousClose: message.previousClose,
      change: message.change,
      changePercent: message.changePercent,
      timestamp: message.timestamp
    };

    this.marketData.set(data.symbol, data);
    this.emit('market_data', data);
  }

  private handleTrade(message: any): void {
    const trade: Trade = {
      id: message.id,
      orderId: message.orderId,
      symbol: message.symbol,
      side: message.side,
      quantity: message.quantity,
      price: message.price,
      timestamp: message.timestamp,
      commission: message.commission,
      account: message.account
    };

    this.trades.unshift(trade);
    // Keep only last 1000 trades
    if (this.trades.length > 1000) {
      this.trades = this.trades.slice(0, 1000);
    }

    this.emit('trade', trade);
  }

  private handleAccountUpdate(message: any): void {
    const account: Account = {
      id: message.id,
      name: message.name,
      type: message.type,
      buyingPower: message.buyingPower,
      totalValue: message.totalValue,
      cashBalance: message.cashBalance,
      marginUsed: message.marginUsed,
      dayTradingBuyingPower: message.dayTradingBuyingPower,
      maintenanceMargin: message.maintenanceMargin,
      equity: message.equity,
      lastUpdate: Date.now()
    };

    this.accounts.set(account.id, account);
    this.emit('account_update', account);
  }

  private handleOrderBook(message: any): void {
    const orderBook: OrderBook = {
      symbol: message.symbol,
      bids: message.bids || [],
      asks: message.asks || [],
      timestamp: message.timestamp
    };

    this.orderBooks.set(orderBook.symbol, orderBook);
    this.emit('order_book', orderBook);
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts < this.config.reconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts - 1), 30000);
      
      setTimeout(() => {
        this.initialize().catch(() => {
          // Reconnection failed, will try again
        });
      }, delay);
    } else {
      this.emit('max_reconnect_attempts_reached');
    }
  }

  private startHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
    }

    this.heartbeatTimer = setInterval(() => {
      if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
        this.websocket.send(JSON.stringify({ type: 'heartbeat' }));
      }
    }, this.config.heartbeatInterval);
  }

  private subscribeToDataFeeds(): void {
    if (!this.websocket || !this.isAuthenticated) return;

    const subscriptions = [];

    if (this.config.enableRealTimeData) {
      subscriptions.push('market_data');
    }

    if (this.config.enableOrderBook) {
      subscriptions.push('order_book');
    }

    if (this.config.enableTrades) {
      subscriptions.push('trades');
    }

    // Always subscribe to account and order updates
    subscriptions.push('orders', 'positions', 'accounts');

    const subscribeMessage = {
      type: 'subscribe',
      channels: subscriptions,
      account: this.config.defaultAccount
    };

    this.websocket.send(JSON.stringify(subscribeMessage));
  }

  // Public Trading API
  public async placeOrder(orderRequest: {
    symbol: string;
    side: 'buy' | 'sell';
    type: 'market' | 'limit' | 'stop' | 'stop_limit';
    quantity: number;
    price?: number;
    stopPrice?: number;
    timeInForce?: 'day' | 'gtc' | 'ioc' | 'fok';
    account?: string;
    strategy?: string;
    tags?: string[];
  }): Promise<Order> {
    // Validate order against risk limits
    this.validateOrder(orderRequest);

    const order = {
      type: 'place_order',
      ...orderRequest,
      account: orderRequest.account || this.config.defaultAccount,
      timeInForce: orderRequest.timeInForce || 'day',
      timestamp: Date.now()
    };

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Order placement timeout'));
      }, 30000);

      const handleOrderResponse = (message: any) => {
        if (message.type === 'order_response') {
          clearTimeout(timeout);
          if (message.success) {
            resolve(message.order);
          } else {
            reject(new Error(message.error || 'Order placement failed'));
          }
        }
      };

      this.once('message', handleOrderResponse);
      
      if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
        this.websocket.send(JSON.stringify(order));
      } else {
        clearTimeout(timeout);
        reject(new Error('WebSocket not connected'));
      }
    });
  }

  public async cancelOrder(orderId: string): Promise<boolean> {
    const cancelRequest = {
      type: 'cancel_order',
      orderId,
      timestamp: Date.now()
    };

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Order cancellation timeout'));
      }, 10000);

      const handleCancelResponse = (message: any) => {
        if (message.type === 'cancel_response' && message.orderId === orderId) {
          clearTimeout(timeout);
          resolve(message.success);
        }
      };

      this.once('message', handleCancelResponse);
      
      if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
        this.websocket.send(JSON.stringify(cancelRequest));
      } else {
        clearTimeout(timeout);
        reject(new Error('WebSocket not connected'));
      }
    });
  }

  public async modifyOrder(orderId: string, modifications: {
    quantity?: number;
    price?: number;
    stopPrice?: number;
  }): Promise<Order> {
    const modifyRequest = {
      type: 'modify_order',
      orderId,
      ...modifications,
      timestamp: Date.now()
    };

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Order modification timeout'));
      }, 10000);

      const handleModifyResponse = (message: any) => {
        if (message.type === 'modify_response' && message.orderId === orderId) {
          clearTimeout(timeout);
          if (message.success) {
            resolve(message.order);
          } else {
            reject(new Error(message.error || 'Order modification failed'));
          }
        }
      };

      this.once('message', handleModifyResponse);
      
      if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
        this.websocket.send(JSON.stringify(modifyRequest));
      } else {
        clearTimeout(timeout);
        reject(new Error('WebSocket not connected'));
      }
    });
  }

  private validateOrder(orderRequest: any): void {
    const { riskLimits } = this.config;

    // Check order size
    if (orderRequest.quantity > riskLimits.maxOrderSize) {
      throw new Error(`Order size exceeds maximum allowed: ${riskLimits.maxOrderSize}`);
    }

    // Check allowed symbols
    if (riskLimits.allowedSymbols.length > 0 && !riskLimits.allowedSymbols.includes(orderRequest.symbol)) {
      throw new Error(`Symbol ${orderRequest.symbol} is not in allowed list`);
    }

    // Check blocked symbols
    if (riskLimits.blockedSymbols.includes(orderRequest.symbol)) {
      throw new Error(`Symbol ${orderRequest.symbol} is blocked`);
    }

    // Additional risk checks can be added here
  }

  // Data Access Methods
  public getOrders(status?: string): Order[] {
    const orders = Array.from(this.orders.values());
    return status ? orders.filter(order => order.status === status) : orders;
  }

  public getOrder(orderId: string): Order | null {
    return this.orders.get(orderId) || null;
  }

  public getPositions(account?: string): Position[] {
    const positions = Array.from(this.positions.values());
    return account ? positions.filter(pos => pos.account === account) : positions;
  }

  public getPosition(symbol: string): Position | null {
    return this.positions.get(symbol) || null;
  }

  public getMarketData(symbol: string): MarketData | null {
    return this.marketData.get(symbol) || null;
  }

  public getAllMarketData(): MarketData[] {
    return Array.from(this.marketData.values());
  }

  public getAccounts(): Account[] {
    return Array.from(this.accounts.values());
  }

  public getAccount(accountId: string): Account | null {
    return this.accounts.get(accountId) || null;
  }

  public getTrades(limit: number = 100): Trade[] {
    return this.trades.slice(0, limit);
  }

  public getOrderBook(symbol: string): OrderBook | null {
    return this.orderBooks.get(symbol) || null;
  }

  public subscribeToSymbol(symbol: string): void {
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      const subscribeMessage = {
        type: 'subscribe_symbol',
        symbol,
        timestamp: Date.now()
      };
      this.websocket.send(JSON.stringify(subscribeMessage));
    }
  }

  public unsubscribeFromSymbol(symbol: string): void {
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      const unsubscribeMessage = {
        type: 'unsubscribe_symbol',
        symbol,
        timestamp: Date.now()
      };
      this.websocket.send(JSON.stringify(unsubscribeMessage));
    }
  }

  public isConnectedToWS4(): boolean {
    return this.isConnected && this.isAuthenticated;
  }

  public disconnect(): void {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
    
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
    
    this.isConnected = false;
    this.isAuthenticated = false;
    this.emit('disconnected');
  }
}

// Utility Functions
export const createWS4Config = (overrides: Partial<WS4Config> = {}): WS4Config => {
  return {
    apiEndpoint: process.env.WS4_API_ENDPOINT || 'https://api.ws4.alluse.com',
    websocketEndpoint: process.env.WS4_WS_ENDPOINT || 'wss://ws.ws4.alluse.com',
    apiKey: process.env.WS4_API_KEY || '',
    secretKey: process.env.WS4_SECRET_KEY || '',
    environment: (process.env.WS4_ENVIRONMENT as 'sandbox' | 'live') || 'sandbox',
    defaultAccount: process.env.WS4_DEFAULT_ACCOUNT || '',
    enableRealTimeData: true,
    enableOrderBook: true,
    enableTrades: true,
    riskLimits: {
      maxOrderSize: 10000,
      maxDayLoss: 50000,
      maxPositionSize: 100000,
      allowedSymbols: [],
      blockedSymbols: [],
      maxLeverage: 4
    },
    reconnectAttempts: 5,
    heartbeatInterval: 30000,
    ...overrides
  };
};

export const formatCurrency = (amount: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD'
  }).format(amount);
};

export const formatPercent = (value: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(value / 100);
};

// Export all types and classes
export default WS4MarketIntegration;

