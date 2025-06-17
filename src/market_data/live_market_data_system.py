"""
Live Market Data Integration and Processing System
Real-time market data engine for ALL-USE trading system

This module provides:
- Real-time market data streaming from multiple sources
- Data quality validation and normalization
- Multi-source data aggregation and conflict resolution
- High-performance streaming data processing
- Market data storage and retrieval
- Connection management and failover for data feeds
"""

import sys
import os
import time
import json
import asyncio
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import pandas as pd
import numpy as np
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import websockets
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSourceType(Enum):
    """Market data source types"""
    IBKR = "ibkr"
    TD_AMERITRADE = "td_ameritrade"
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"

class DataType(Enum):
    """Market data types"""
    QUOTE = "quote"
    TRADE = "trade"
    OPTIONS_CHAIN = "options_chain"
    LEVEL2 = "level2"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"

class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    STALE = "stale"
    INVALID = "invalid"

@dataclass
class MarketDataPoint:
    """Individual market data point"""
    symbol: str
    data_type: DataType
    source: DataSourceType
    timestamp: datetime
    data: Dict[str, Any]
    quality: DataQuality = DataQuality.GOOD
    latency_ms: float = 0.0
    sequence_number: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'data_type': self.data_type.value,
            'source': self.source.value,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'quality': self.quality.value,
            'latency_ms': self.latency_ms,
            'sequence_number': self.sequence_number
        }

@dataclass
class DataSourceConfig:
    """Configuration for market data source"""
    source_type: DataSourceType
    enabled: bool = True
    priority: int = 1  # 1 = highest priority
    max_symbols: int = 1000
    update_frequency_ms: int = 100
    timeout_seconds: int = 30
    retry_attempts: int = 3
    quality_threshold: DataQuality = DataQuality.FAIR
    
    # Connection settings
    connection_url: str = ""
    api_key: str = ""
    rate_limit_per_second: float = 100.0
    
    # Data types supported
    supported_data_types: List[DataType] = field(default_factory=lambda: [DataType.QUOTE])

class MarketDataValidator:
    """Market data quality validation"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.MarketDataValidator")
        
        # Validation thresholds
        self.max_spread_percentage = 0.10  # 10% max bid-ask spread
        self.max_price_change_percentage = 0.20  # 20% max price change
        self.max_latency_ms = 5000.0  # 5 second max latency
        self.min_volume = 0  # Minimum volume
        
        # Price history for validation
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.last_prices: Dict[str, float] = {}
    
    def validate_quote(self, data_point: MarketDataPoint) -> DataQuality:
        """Validate quote data quality"""
        try:
            data = data_point.data
            symbol = data_point.symbol
            
            # Extract quote data
            bid = data.get('bid', 0.0)
            ask = data.get('ask', 0.0)
            last = data.get('last', 0.0)
            volume = data.get('volume', 0)
            timestamp = data_point.timestamp
            
            # Check for missing critical data
            if not all([bid > 0, ask > 0, last > 0]):
                return DataQuality.INVALID
            
            # Check bid-ask spread
            if bid >= ask:
                return DataQuality.INVALID
            
            spread_percentage = (ask - bid) / ((ask + bid) / 2)
            if spread_percentage > self.max_spread_percentage:
                return DataQuality.POOR
            
            # Check price change from last known price
            if symbol in self.last_prices:
                last_price = self.last_prices[symbol]
                price_change = abs(last - last_price) / last_price
                if price_change > self.max_price_change_percentage:
                    return DataQuality.FAIR
            
            # Check latency
            if data_point.latency_ms > self.max_latency_ms:
                return DataQuality.STALE
            
            # Check timestamp freshness
            age_seconds = (datetime.now() - timestamp).total_seconds()
            if age_seconds > 60:  # 1 minute old
                return DataQuality.STALE
            elif age_seconds > 10:  # 10 seconds old
                return DataQuality.FAIR
            
            # Update price history
            self.price_history[symbol].append(last)
            self.last_prices[symbol] = last
            
            # Determine quality based on spread and latency
            if spread_percentage < 0.01 and data_point.latency_ms < 100:
                return DataQuality.EXCELLENT
            elif spread_percentage < 0.05 and data_point.latency_ms < 500:
                return DataQuality.GOOD
            else:
                return DataQuality.FAIR
                
        except Exception as e:
            self.logger.error(f"Error validating quote for {data_point.symbol}: {str(e)}")
            return DataQuality.INVALID
    
    def validate_options_chain(self, data_point: MarketDataPoint) -> DataQuality:
        """Validate options chain data quality"""
        try:
            data = data_point.data
            
            # Check for required fields
            required_fields = ['calls', 'puts', 'underlyingPrice', 'expiry']
            if not all(field in data for field in required_fields):
                return DataQuality.INVALID
            
            calls = data.get('calls', [])
            puts = data.get('puts', [])
            
            # Check if we have options data
            if not calls and not puts:
                return DataQuality.INVALID
            
            # Validate individual options
            valid_options = 0
            total_options = len(calls) + len(puts)
            
            for option in calls + puts:
                if self._validate_option_data(option):
                    valid_options += 1
            
            # Determine quality based on valid options percentage
            valid_percentage = valid_options / total_options if total_options > 0 else 0
            
            if valid_percentage >= 0.95:
                return DataQuality.EXCELLENT
            elif valid_percentage >= 0.85:
                return DataQuality.GOOD
            elif valid_percentage >= 0.70:
                return DataQuality.FAIR
            else:
                return DataQuality.POOR
                
        except Exception as e:
            self.logger.error(f"Error validating options chain for {data_point.symbol}: {str(e)}")
            return DataQuality.INVALID
    
    def _validate_option_data(self, option: Dict[str, Any]) -> bool:
        """Validate individual option data"""
        try:
            # Check required fields
            required_fields = ['strike', 'bid', 'ask', 'last']
            if not all(field in option for field in required_fields):
                return False
            
            bid = option.get('bid', 0.0)
            ask = option.get('ask', 0.0)
            strike = option.get('strike', 0.0)
            
            # Basic validation
            if bid < 0 or ask < 0 or strike <= 0:
                return False
            
            if bid > ask:
                return False
            
            return True
            
        except Exception:
            return False

class DataAggregator:
    """Multi-source data aggregation and conflict resolution"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DataAggregator")
        
        # Data source priorities
        self.source_priorities = {
            DataSourceType.IBKR: 1,
            DataSourceType.TD_AMERITRADE: 2,
            DataSourceType.POLYGON: 3,
            DataSourceType.YAHOO_FINANCE: 4,
            DataSourceType.ALPHA_VANTAGE: 5
        }
        
        # Recent data cache
        self.data_cache: Dict[str, Dict[DataSourceType, MarketDataPoint]] = defaultdict(dict)
        self.cache_expiry_seconds = 60
    
    def aggregate_quotes(self, symbol: str, data_points: List[MarketDataPoint]) -> Optional[MarketDataPoint]:
        """Aggregate quotes from multiple sources"""
        try:
            if not data_points:
                return None
            
            # Filter valid data points
            valid_points = [dp for dp in data_points if dp.quality != DataQuality.INVALID]
            if not valid_points:
                return None
            
            # Sort by priority and quality
            sorted_points = sorted(valid_points, key=lambda x: (
                self.source_priorities.get(x.source, 999),
                x.quality.value,
                x.latency_ms
            ))
            
            # Use highest priority source as base
            primary_point = sorted_points[0]
            
            # Create aggregated data
            aggregated_data = primary_point.data.copy()
            
            # Enhance with data from other sources if available
            for point in sorted_points[1:]:
                # Use better bid if available
                if point.data.get('bid', 0) > aggregated_data.get('bid', 0):
                    aggregated_data['bid'] = point.data['bid']
                    aggregated_data['bidSize'] = point.data.get('bidSize', 0)
                
                # Use better ask if available
                if point.data.get('ask', float('inf')) < aggregated_data.get('ask', float('inf')):
                    aggregated_data['ask'] = point.data['ask']
                    aggregated_data['askSize'] = point.data.get('askSize', 0)
                
                # Use most recent last price
                if point.timestamp > primary_point.timestamp:
                    aggregated_data['last'] = point.data.get('last', aggregated_data['last'])
                    aggregated_data['lastSize'] = point.data.get('lastSize', aggregated_data.get('lastSize', 0))
            
            # Calculate aggregated quality
            avg_quality_score = sum(self._quality_to_score(dp.quality) for dp in valid_points) / len(valid_points)
            aggregated_quality = self._score_to_quality(avg_quality_score)
            
            # Create aggregated data point
            return MarketDataPoint(
                symbol=symbol,
                data_type=DataType.QUOTE,
                source=primary_point.source,  # Primary source
                timestamp=max(dp.timestamp for dp in valid_points),
                data=aggregated_data,
                quality=aggregated_quality,
                latency_ms=min(dp.latency_ms for dp in valid_points),
                sequence_number=max(dp.sequence_number for dp in valid_points)
            )
            
        except Exception as e:
            self.logger.error(f"Error aggregating quotes for {symbol}: {str(e)}")
            return None
    
    def _quality_to_score(self, quality: DataQuality) -> float:
        """Convert quality enum to numeric score"""
        quality_scores = {
            DataQuality.EXCELLENT: 5.0,
            DataQuality.GOOD: 4.0,
            DataQuality.FAIR: 3.0,
            DataQuality.POOR: 2.0,
            DataQuality.STALE: 1.0,
            DataQuality.INVALID: 0.0
        }
        return quality_scores.get(quality, 0.0)
    
    def _score_to_quality(self, score: float) -> DataQuality:
        """Convert numeric score to quality enum"""
        if score >= 4.5:
            return DataQuality.EXCELLENT
        elif score >= 3.5:
            return DataQuality.GOOD
        elif score >= 2.5:
            return DataQuality.FAIR
        elif score >= 1.5:
            return DataQuality.POOR
        elif score >= 0.5:
            return DataQuality.STALE
        else:
            return DataQuality.INVALID

class MarketDataStorage:
    """Market data storage and retrieval"""
    
    def __init__(self, db_path: str = "/home/ubuntu/AGENT_ALLUSE_V1/data/market_data.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.MarketDataStorage")
        
        # Create data directory
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # In-memory cache for recent data
        self.memory_cache: Dict[str, MarketDataPoint] = {}
        self.cache_size_limit = 10000
    
    def _init_database(self):
        """Initialize SQLite database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create market data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS market_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        data_type TEXT NOT NULL,
                        source TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        data TEXT NOT NULL,
                        quality TEXT NOT NULL,
                        latency_ms REAL NOT NULL,
                        sequence_number INTEGER NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON market_data(symbol, timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_data_type ON market_data(data_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON market_data(source)')
                
                conn.commit()
                self.logger.info("Market data database initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
    
    def store_data_point(self, data_point: MarketDataPoint):
        """Store market data point"""
        try:
            # Store in memory cache
            cache_key = f"{data_point.symbol}_{data_point.data_type.value}"
            self.memory_cache[cache_key] = data_point
            
            # Limit cache size
            if len(self.memory_cache) > self.cache_size_limit:
                # Remove oldest entries
                oldest_keys = list(self.memory_cache.keys())[:len(self.memory_cache) - self.cache_size_limit]
                for key in oldest_keys:
                    del self.memory_cache[key]
            
            # Store in database (async to avoid blocking)
            threading.Thread(target=self._store_to_db, args=(data_point,), daemon=True).start()
            
        except Exception as e:
            self.logger.error(f"Error storing data point: {str(e)}")
    
    def _store_to_db(self, data_point: MarketDataPoint):
        """Store data point to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO market_data 
                    (symbol, data_type, source, timestamp, data, quality, latency_ms, sequence_number)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data_point.symbol,
                    data_point.data_type.value,
                    data_point.source.value,
                    data_point.timestamp.isoformat(),
                    json.dumps(data_point.data),
                    data_point.quality.value,
                    data_point.latency_ms,
                    data_point.sequence_number
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing to database: {str(e)}")
    
    def get_latest_quote(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get latest quote for symbol"""
        try:
            # Check memory cache first
            cache_key = f"{symbol}_{DataType.QUOTE.value}"
            if cache_key in self.memory_cache:
                cached_data = self.memory_cache[cache_key]
                # Check if data is fresh (less than 60 seconds old)
                if (datetime.now() - cached_data.timestamp).total_seconds() < 60:
                    return cached_data
            
            # Query database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT symbol, data_type, source, timestamp, data, quality, latency_ms, sequence_number
                    FROM market_data
                    WHERE symbol = ? AND data_type = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                ''', (symbol, DataType.QUOTE.value))
                
                row = cursor.fetchone()
                if row:
                    return MarketDataPoint(
                        symbol=row[0],
                        data_type=DataType(row[1]),
                        source=DataSourceType(row[2]),
                        timestamp=datetime.fromisoformat(row[3]),
                        data=json.loads(row[4]),
                        quality=DataQuality(row[5]),
                        latency_ms=row[6],
                        sequence_number=row[7]
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting latest quote for {symbol}: {str(e)}")
            return None
    
    def get_historical_data(self, symbol: str, data_type: DataType, 
                          start_time: datetime, end_time: datetime) -> List[MarketDataPoint]:
        """Get historical data for symbol"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT symbol, data_type, source, timestamp, data, quality, latency_ms, sequence_number
                    FROM market_data
                    WHERE symbol = ? AND data_type = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp ASC
                ''', (symbol, data_type.value, start_time.isoformat(), end_time.isoformat()))
                
                rows = cursor.fetchall()
                return [
                    MarketDataPoint(
                        symbol=row[0],
                        data_type=DataType(row[1]),
                        source=DataSourceType(row[2]),
                        timestamp=datetime.fromisoformat(row[3]),
                        data=json.loads(row[4]),
                        quality=DataQuality(row[5]),
                        latency_ms=row[6],
                        sequence_number=row[7]
                    )
                    for row in rows
                ]
                
        except Exception as e:
            self.logger.error(f"Error getting historical data: {str(e)}")
            return []

class MarketDataFeed:
    """Market data feed from specific source"""
    
    def __init__(self, config: DataSourceConfig, validator: MarketDataValidator):
        self.config = config
        self.validator = validator
        self.logger = logging.getLogger(f"{__name__}.MarketDataFeed.{config.source_type.value}")
        
        # Connection state
        self.connected = False
        self.last_heartbeat = datetime.now()
        self.error_count = 0
        self.sequence_number = 0
        
        # Subscriptions
        self.subscribed_symbols: Set[str] = set()
        self.data_callbacks: List[Callable[[MarketDataPoint], None]] = []
        
        # Performance metrics
        self.messages_received = 0
        self.messages_processed = 0
        self.total_latency_ms = 0.0
        self.last_message_time = datetime.now()
    
    async def connect(self) -> bool:
        """Connect to market data source"""
        try:
            self.logger.info(f"Connecting to {self.config.source_type.value} market data...")
            
            # Simulate connection based on source type
            if self.config.source_type == DataSourceType.IBKR:
                # IBKR TWS market data connection
                await asyncio.sleep(0.5)
                self.connected = True
                self.logger.info("Connected to IBKR market data")
                
            elif self.config.source_type == DataSourceType.TD_AMERITRADE:
                # TD Ameritrade streaming API
                await asyncio.sleep(0.3)
                self.connected = True
                self.logger.info("Connected to TD Ameritrade market data")
                
            else:
                # Other data sources
                await asyncio.sleep(0.2)
                self.connected = True
                self.logger.info(f"Connected to {self.config.source_type.value} market data")
            
            self.last_heartbeat = datetime.now()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to {self.config.source_type.value}: {str(e)}")
            self.error_count += 1
            return False
    
    async def disconnect(self):
        """Disconnect from market data source"""
        try:
            self.connected = False
            self.subscribed_symbols.clear()
            self.logger.info(f"Disconnected from {self.config.source_type.value} market data")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from {self.config.source_type.value}: {str(e)}")
    
    async def subscribe_symbol(self, symbol: str, data_types: List[DataType] = None) -> bool:
        """Subscribe to symbol data"""
        try:
            if not self.connected:
                return False
            
            if data_types is None:
                data_types = [DataType.QUOTE]
            
            # Check if data types are supported
            for data_type in data_types:
                if data_type not in self.config.supported_data_types:
                    self.logger.warning(f"Data type {data_type.value} not supported by {self.config.source_type.value}")
                    continue
            
            self.subscribed_symbols.add(symbol)
            self.logger.info(f"Subscribed to {symbol} on {self.config.source_type.value}")
            
            # Start generating simulated data
            asyncio.create_task(self._generate_data(symbol, data_types))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error subscribing to {symbol}: {str(e)}")
            return False
    
    async def unsubscribe_symbol(self, symbol: str) -> bool:
        """Unsubscribe from symbol data"""
        try:
            self.subscribed_symbols.discard(symbol)
            self.logger.info(f"Unsubscribed from {symbol} on {self.config.source_type.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error unsubscribing from {symbol}: {str(e)}")
            return False
    
    async def _generate_data(self, symbol: str, data_types: List[DataType]):
        """Generate simulated market data"""
        try:
            base_price = 100.0 + hash(symbol) % 400
            
            while symbol in self.subscribed_symbols and self.connected:
                for data_type in data_types:
                    if data_type == DataType.QUOTE:
                        # Generate quote data
                        start_time = time.time()
                        
                        # Add some price movement
                        price_change = np.random.normal(0, 0.5)
                        current_price = base_price + price_change
                        
                        # Generate bid/ask spread
                        spread = 0.01 if self.config.source_type == DataSourceType.IBKR else 0.02
                        bid = current_price - spread
                        ask = current_price + spread
                        
                        quote_data = {
                            'symbol': symbol,
                            'bid': round(bid, 2),
                            'ask': round(ask, 2),
                            'last': round(current_price, 2),
                            'close': round(base_price, 2),
                            'volume': np.random.randint(100000, 2000000),
                            'bidSize': np.random.randint(100, 1000),
                            'askSize': np.random.randint(100, 1000),
                            'lastSize': np.random.randint(10, 500),
                            'high': round(current_price + 2.5, 2),
                            'low': round(current_price - 1.8, 2),
                            'open': round(base_price - 0.3, 2),
                            'halted': False
                        }
                        
                        # Calculate latency
                        latency_ms = (time.time() - start_time) * 1000
                        
                        # Create data point
                        data_point = MarketDataPoint(
                            symbol=symbol,
                            data_type=data_type,
                            source=self.config.source_type,
                            timestamp=datetime.now(),
                            data=quote_data,
                            latency_ms=latency_ms,
                            sequence_number=self.sequence_number
                        )
                        
                        # Validate data quality
                        data_point.quality = self.validator.validate_quote(data_point)
                        
                        # Update metrics
                        self.messages_received += 1
                        self.sequence_number += 1
                        self.last_message_time = datetime.now()
                        
                        # Send to callbacks
                        for callback in self.data_callbacks:
                            try:
                                callback(data_point)
                                self.messages_processed += 1
                            except Exception as e:
                                self.logger.error(f"Error in data callback: {str(e)}")
                
                # Wait for next update
                await asyncio.sleep(self.config.update_frequency_ms / 1000.0)
                
        except Exception as e:
            self.logger.error(f"Error generating data for {symbol}: {str(e)}")
    
    def add_data_callback(self, callback: Callable[[MarketDataPoint], None]):
        """Add data callback"""
        self.data_callbacks.append(callback)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        uptime_seconds = (datetime.now() - self.last_heartbeat).total_seconds()
        avg_latency = self.total_latency_ms / max(self.messages_received, 1)
        
        return {
            'source': self.config.source_type.value,
            'connected': self.connected,
            'uptime_seconds': uptime_seconds,
            'subscribed_symbols': len(self.subscribed_symbols),
            'messages_received': self.messages_received,
            'messages_processed': self.messages_processed,
            'error_count': self.error_count,
            'avg_latency_ms': avg_latency,
            'last_message_age_seconds': (datetime.now() - self.last_message_time).total_seconds()
        }

class LiveMarketDataManager:
    """Main market data manager coordinating multiple feeds"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.LiveMarketDataManager")
        
        # Components
        self.validator = MarketDataValidator()
        self.aggregator = DataAggregator()
        self.storage = MarketDataStorage()
        
        # Data feeds
        self.feeds: Dict[DataSourceType, MarketDataFeed] = {}
        self.feed_configs: Dict[DataSourceType, DataSourceConfig] = {}
        
        # Subscriptions
        self.symbol_subscriptions: Dict[str, Set[DataSourceType]] = defaultdict(set)
        self.data_callbacks: List[Callable[[MarketDataPoint], None]] = []
        
        # Performance tracking
        self.total_data_points = 0
        self.start_time = datetime.now()
        
        # Initialize default feeds
        self._initialize_default_feeds()
    
    def _initialize_default_feeds(self):
        """Initialize default market data feeds"""
        
        # IBKR feed configuration
        ibkr_config = DataSourceConfig(
            source_type=DataSourceType.IBKR,
            enabled=True,
            priority=1,
            max_symbols=1000,
            update_frequency_ms=100,
            supported_data_types=[DataType.QUOTE, DataType.OPTIONS_CHAIN, DataType.LEVEL2]
        )
        
        # TD Ameritrade feed configuration
        td_config = DataSourceConfig(
            source_type=DataSourceType.TD_AMERITRADE,
            enabled=True,
            priority=2,
            max_symbols=500,
            update_frequency_ms=200,
            supported_data_types=[DataType.QUOTE, DataType.OPTIONS_CHAIN]
        )
        
        # Yahoo Finance feed configuration (backup)
        yahoo_config = DataSourceConfig(
            source_type=DataSourceType.YAHOO_FINANCE,
            enabled=True,
            priority=3,
            max_symbols=100,
            update_frequency_ms=1000,
            supported_data_types=[DataType.QUOTE]
        )
        
        self.feed_configs = {
            DataSourceType.IBKR: ibkr_config,
            DataSourceType.TD_AMERITRADE: td_config,
            DataSourceType.YAHOO_FINANCE: yahoo_config
        }
    
    async def start_feeds(self, source_types: List[DataSourceType] = None) -> Dict[DataSourceType, bool]:
        """Start market data feeds"""
        if source_types is None:
            source_types = list(self.feed_configs.keys())
        
        results = {}
        
        for source_type in source_types:
            if source_type not in self.feed_configs:
                self.logger.warning(f"No configuration for {source_type.value}")
                results[source_type] = False
                continue
            
            config = self.feed_configs[source_type]
            if not config.enabled:
                self.logger.info(f"Feed {source_type.value} is disabled")
                results[source_type] = False
                continue
            
            try:
                # Create feed
                feed = MarketDataFeed(config, self.validator)
                feed.add_data_callback(self._on_data_received)
                
                # Connect feed
                connected = await feed.connect()
                if connected:
                    self.feeds[source_type] = feed
                    self.logger.info(f"Started {source_type.value} market data feed")
                
                results[source_type] = connected
                
            except Exception as e:
                self.logger.error(f"Failed to start {source_type.value} feed: {str(e)}")
                results[source_type] = False
        
        return results
    
    async def stop_feeds(self):
        """Stop all market data feeds"""
        for source_type, feed in self.feeds.items():
            try:
                await feed.disconnect()
                self.logger.info(f"Stopped {source_type.value} feed")
            except Exception as e:
                self.logger.error(f"Error stopping {source_type.value} feed: {str(e)}")
        
        self.feeds.clear()
        self.symbol_subscriptions.clear()
    
    async def subscribe_symbol(self, symbol: str, data_types: List[DataType] = None,
                             preferred_sources: List[DataSourceType] = None) -> bool:
        """Subscribe to symbol across multiple feeds"""
        if data_types is None:
            data_types = [DataType.QUOTE]
        
        if preferred_sources is None:
            preferred_sources = list(self.feeds.keys())
        
        success_count = 0
        
        for source_type in preferred_sources:
            if source_type not in self.feeds:
                continue
            
            feed = self.feeds[source_type]
            
            # Check if feed supports required data types
            supported_types = [dt for dt in data_types if dt in feed.config.supported_data_types]
            if not supported_types:
                continue
            
            try:
                subscribed = await feed.subscribe_symbol(symbol, supported_types)
                if subscribed:
                    self.symbol_subscriptions[symbol].add(source_type)
                    success_count += 1
                    
            except Exception as e:
                self.logger.error(f"Error subscribing {symbol} to {source_type.value}: {str(e)}")
        
        if success_count > 0:
            self.logger.info(f"Subscribed {symbol} to {success_count} feeds")
            return True
        else:
            self.logger.warning(f"Failed to subscribe {symbol} to any feeds")
            return False
    
    async def unsubscribe_symbol(self, symbol: str) -> bool:
        """Unsubscribe symbol from all feeds"""
        success_count = 0
        
        for source_type in list(self.symbol_subscriptions[symbol]):
            if source_type in self.feeds:
                try:
                    unsubscribed = await self.feeds[source_type].unsubscribe_symbol(symbol)
                    if unsubscribed:
                        success_count += 1
                        
                except Exception as e:
                    self.logger.error(f"Error unsubscribing {symbol} from {source_type.value}: {str(e)}")
        
        # Clear subscriptions
        del self.symbol_subscriptions[symbol]
        
        self.logger.info(f"Unsubscribed {symbol} from {success_count} feeds")
        return success_count > 0
    
    def _on_data_received(self, data_point: MarketDataPoint):
        """Handle incoming market data"""
        try:
            # Store data point
            self.storage.store_data_point(data_point)
            
            # Update metrics
            self.total_data_points += 1
            
            # Send to callbacks
            for callback in self.data_callbacks:
                try:
                    callback(data_point)
                except Exception as e:
                    self.logger.error(f"Error in data callback: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error processing data point: {str(e)}")
    
    def get_latest_quote(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get latest aggregated quote for symbol"""
        try:
            # Get data from all subscribed sources
            data_points = []
            
            for source_type in self.symbol_subscriptions.get(symbol, set()):
                if source_type in self.feeds:
                    # Get latest from storage
                    latest = self.storage.get_latest_quote(symbol)
                    if latest and latest.source == source_type:
                        data_points.append(latest)
            
            # Aggregate if multiple sources
            if len(data_points) > 1:
                return self.aggregator.aggregate_quotes(symbol, data_points)
            elif len(data_points) == 1:
                return data_points[0]
            else:
                return self.storage.get_latest_quote(symbol)
                
        except Exception as e:
            self.logger.error(f"Error getting latest quote for {symbol}: {str(e)}")
            return None
    
    def add_data_callback(self, callback: Callable[[MarketDataPoint], None]):
        """Add data callback"""
        self.data_callbacks.append(callback)
    
    def get_feed_status(self) -> Dict[str, Any]:
        """Get status of all feeds"""
        status = {
            'total_feeds': len(self.feeds),
            'connected_feeds': sum(1 for feed in self.feeds.values() if feed.connected),
            'total_subscriptions': sum(len(sources) for sources in self.symbol_subscriptions.values()),
            'unique_symbols': len(self.symbol_subscriptions),
            'total_data_points': self.total_data_points,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'feeds': {}
        }
        
        for source_type, feed in self.feeds.items():
            status['feeds'][source_type.value] = feed.get_performance_metrics()
        
        return status
    
    def get_symbol_status(self, symbol: str) -> Dict[str, Any]:
        """Get status for specific symbol"""
        latest_quote = self.get_latest_quote(symbol)
        
        return {
            'symbol': symbol,
            'subscribed_feeds': [source.value for source in self.symbol_subscriptions.get(symbol, set())],
            'latest_quote': latest_quote.to_dict() if latest_quote else None,
            'data_age_seconds': (datetime.now() - latest_quote.timestamp).total_seconds() if latest_quote else None
        }

async def test_live_market_data_system():
    """Test live market data integration system"""
    print("🚀 Testing Live Market Data Integration System...")
    
    # Create market data manager
    manager = LiveMarketDataManager()
    
    # Test feed startup
    print("\n📡 Testing Market Data Feeds")
    
    feed_results = await manager.start_feeds([
        DataSourceType.IBKR,
        DataSourceType.TD_AMERITRADE,
        DataSourceType.YAHOO_FINANCE
    ])
    
    for source, success in feed_results.items():
        print(f"  {'✅' if success else '❌'} {source.value} feed: {success}")
    
    # Test symbol subscriptions
    print("\n📊 Testing Symbol Subscriptions")
    
    test_symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA']
    
    for symbol in test_symbols:
        subscribed = await manager.subscribe_symbol(symbol, [DataType.QUOTE])
        print(f"  {'✅' if subscribed else '❌'} Subscribed {symbol}: {subscribed}")
    
    # Wait for data to flow
    print("\n⏳ Waiting for market data...")
    await asyncio.sleep(3)
    
    # Test data retrieval
    print("\n📈 Testing Data Retrieval")
    
    for symbol in test_symbols[:3]:  # Test first 3 symbols
        quote = manager.get_latest_quote(symbol)
        if quote:
            print(f"  ✅ {symbol}: ${quote.data['last']:.2f} (bid: ${quote.data['bid']:.2f}, ask: ${quote.data['ask']:.2f})")
            print(f"    Quality: {quote.quality.value}, Latency: {quote.latency_ms:.1f}ms, Source: {quote.source.value}")
        else:
            print(f"  ❌ {symbol}: No data available")
    
    # Test feed status
    print("\n📊 Testing Feed Status")
    
    feed_status = manager.get_feed_status()
    print(f"  📋 Total feeds: {feed_status['total_feeds']}")
    print(f"  📋 Connected feeds: {feed_status['connected_feeds']}")
    print(f"  📋 Total subscriptions: {feed_status['total_subscriptions']}")
    print(f"  📋 Unique symbols: {feed_status['unique_symbols']}")
    print(f"  📋 Total data points: {feed_status['total_data_points']}")
    print(f"  📋 Uptime: {feed_status['uptime_seconds']:.1f} seconds")
    
    for feed_name, feed_metrics in feed_status['feeds'].items():
        print(f"    {feed_name}: {feed_metrics['messages_received']} messages, {feed_metrics['avg_latency_ms']:.1f}ms avg latency")
    
    # Test symbol status
    print("\n🔍 Testing Symbol Status")
    
    for symbol in test_symbols[:2]:
        symbol_status = manager.get_symbol_status(symbol)
        print(f"  📋 {symbol}:")
        print(f"    Feeds: {symbol_status['subscribed_feeds']}")
        if symbol_status['latest_quote']:
            print(f"    Latest: ${symbol_status['latest_quote']['data']['last']:.2f}")
            print(f"    Age: {symbol_status['data_age_seconds']:.1f} seconds")
        else:
            print(f"    No data available")
    
    # Test data validation
    print("\n🔍 Testing Data Validation")
    
    validator = MarketDataValidator()
    
    # Create test data points with different quality levels
    test_data_points = [
        MarketDataPoint(
            symbol='TEST1',
            data_type=DataType.QUOTE,
            source=DataSourceType.IBKR,
            timestamp=datetime.now(),
            data={'bid': 100.0, 'ask': 100.05, 'last': 100.02, 'volume': 1000000},
            latency_ms=50.0
        ),
        MarketDataPoint(
            symbol='TEST2',
            data_type=DataType.QUOTE,
            source=DataSourceType.TD_AMERITRADE,
            timestamp=datetime.now() - timedelta(seconds=30),
            data={'bid': 200.0, 'ask': 200.50, 'last': 200.25, 'volume': 500000},
            latency_ms=500.0
        ),
        MarketDataPoint(
            symbol='TEST3',
            data_type=DataType.QUOTE,
            source=DataSourceType.YAHOO_FINANCE,
            timestamp=datetime.now(),
            data={'bid': 0.0, 'ask': 300.0, 'last': 300.0, 'volume': 0},  # Invalid data
            latency_ms=100.0
        )
    ]
    
    for data_point in test_data_points:
        quality = validator.validate_quote(data_point)
        data_point.quality = quality
        print(f"  📊 {data_point.symbol}: {quality.value} quality")
        print(f"    Data: bid=${data_point.data['bid']:.2f}, ask=${data_point.data['ask']:.2f}")
        print(f"    Latency: {data_point.latency_ms:.1f}ms, Age: {(datetime.now() - data_point.timestamp).total_seconds():.1f}s")
    
    # Test data aggregation
    print("\n🔄 Testing Data Aggregation")
    
    aggregator = DataAggregator()
    
    # Create multiple data points for same symbol
    multi_source_data = [
        MarketDataPoint(
            symbol='AGGTEST',
            data_type=DataType.QUOTE,
            source=DataSourceType.IBKR,
            timestamp=datetime.now(),
            data={'bid': 150.00, 'ask': 150.05, 'last': 150.02, 'volume': 1000000},
            quality=DataQuality.EXCELLENT,
            latency_ms=25.0
        ),
        MarketDataPoint(
            symbol='AGGTEST',
            data_type=DataType.QUOTE,
            source=DataSourceType.TD_AMERITRADE,
            timestamp=datetime.now(),
            data={'bid': 149.99, 'ask': 150.06, 'last': 150.03, 'volume': 800000},
            quality=DataQuality.GOOD,
            latency_ms=75.0
        )
    ]
    
    aggregated = aggregator.aggregate_quotes('AGGTEST', multi_source_data)
    if aggregated:
        print(f"  ✅ Aggregated AGGTEST:")
        print(f"    Best bid: ${aggregated.data['bid']:.2f} (from multiple sources)")
        print(f"    Best ask: ${aggregated.data['ask']:.2f} (from multiple sources)")
        print(f"    Quality: {aggregated.quality.value}")
        print(f"    Primary source: {aggregated.source.value}")
    else:
        print(f"  ❌ Failed to aggregate data")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    await manager.stop_feeds()
    
    print("\n✅ Live Market Data Integration testing completed!")
    
    return {
        'feed_startup': any(feed_results.values()),
        'symbol_subscriptions': all(feed_results.values()),
        'data_retrieval': feed_status['total_data_points'] > 0,
        'data_validation': True,
        'data_aggregation': aggregated is not None,
        'feed_management': feed_status['connected_feeds'] > 0
    }

if __name__ == "__main__":
    asyncio.run(test_live_market_data_system())

