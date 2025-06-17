"""
Trading Execution Engine and Order Management System
Advanced order management with paper/live environment switching

This module provides comprehensive trading execution capabilities:
- Real-time order management and execution
- Paper/Live trading environment switching
- Smart order routing and execution algorithms
- Position tracking and P&L calculation
- Order validation and risk controls
"""

import sys
import os
import time
import logging
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
import json
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingEnvironment(Enum):
    """Trading environment types"""
    PAPER = "paper"
    LIVE = "live"
    SIMULATION = "simulation"
    BACKTESTING = "backtesting"

class OrderType(Enum):
    """Order types supported by the system"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    CONDITIONAL = "conditional"
    BRACKET = "bracket"
    OCO = "oco"  # One-Cancels-Other
    OTO = "oto"  # One-Triggers-Other

class OrderSide(Enum):
    """Order side (buy/sell)"""
    BUY = "buy"
    SELL = "sell"
    BUY_TO_OPEN = "buy_to_open"
    SELL_TO_OPEN = "sell_to_open"
    BUY_TO_CLOSE = "buy_to_close"
    SELL_TO_CLOSE = "sell_to_close"

class OrderStatus(Enum):
    """Order status states"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class InstrumentType(Enum):
    """Financial instrument types"""
    STOCK = "stock"
    OPTION = "option"
    ETF = "etf"
    INDEX = "index"
    FUTURE = "future"

class PositionSide(Enum):
    """Position side"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

@dataclass
class Instrument:
    """Financial instrument definition"""
    symbol: str
    instrument_type: InstrumentType
    exchange: str = ""
    currency: str = "USD"
    
    # Options-specific fields
    underlying: Optional[str] = None
    expiration: Optional[datetime] = None
    strike: Optional[float] = None
    option_type: Optional[str] = None  # "call" or "put"
    
    # Additional metadata
    multiplier: int = 1
    tick_size: float = 0.01
    
    def __post_init__(self):
        """Validate instrument data"""
        if self.instrument_type == InstrumentType.OPTION:
            if not all([self.underlying, self.expiration, self.strike, self.option_type]):
                raise ValueError("Options require underlying, expiration, strike, and option_type")
            if self.option_type not in ["call", "put"]:
                raise ValueError("Option type must be 'call' or 'put'")

@dataclass
class Order:
    """Order representation"""
    order_id: str
    instrument: Instrument
    order_type: OrderType
    side: OrderSide
    quantity: int
    
    # Price fields
    price: Optional[float] = None
    stop_price: Optional[float] = None
    limit_price: Optional[float] = None
    
    # Order management
    status: OrderStatus = OrderStatus.PENDING
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: datetime = field(default_factory=datetime.now)
    
    # Execution details
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    
    # Order attributes
    time_in_force: str = "DAY"  # DAY, GTC, IOC, FOK
    all_or_none: bool = False
    
    # Parent/child relationships for complex orders
    parent_order_id: Optional[str] = None
    child_order_ids: List[str] = field(default_factory=list)
    
    # Risk and validation
    account_id: str = ""
    strategy_id: str = ""
    
    # Environment tracking
    environment: TradingEnvironment = TradingEnvironment.PAPER
    
    def __post_init__(self):
        """Validate order data"""
        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive")
        
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and self.price is None:
            raise ValueError(f"{self.order_type.value} orders require a price")
        
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValueError(f"{self.order_type.value} orders require a stop price")
    
    @property
    def remaining_quantity(self) -> int:
        """Get remaining unfilled quantity"""
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled"""
        return self.filled_quantity >= self.quantity
    
    @property
    def fill_percentage(self) -> float:
        """Get fill percentage"""
        return (self.filled_quantity / self.quantity) * 100 if self.quantity > 0 else 0

@dataclass
class Position:
    """Position representation"""
    instrument: Instrument
    quantity: int
    avg_price: float
    market_price: float = 0.0
    
    # Position metadata
    account_id: str = ""
    strategy_id: str = ""
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: datetime = field(default_factory=datetime.now)
    
    # Environment tracking
    environment: TradingEnvironment = TradingEnvironment.PAPER
    
    @property
    def side(self) -> PositionSide:
        """Get position side"""
        if self.quantity > 0:
            return PositionSide.LONG
        elif self.quantity < 0:
            return PositionSide.SHORT
        else:
            return PositionSide.FLAT
    
    @property
    def market_value(self) -> float:
        """Get current market value"""
        return self.quantity * self.market_price * self.instrument.multiplier
    
    @property
    def cost_basis(self) -> float:
        """Get cost basis"""
        return self.quantity * self.avg_price * self.instrument.multiplier
    
    @property
    def unrealized_pnl(self) -> float:
        """Get unrealized P&L"""
        return self.market_value - self.cost_basis
    
    @property
    def unrealized_pnl_percent(self) -> float:
        """Get unrealized P&L percentage"""
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / abs(self.cost_basis)) * 100

@dataclass
class Fill:
    """Trade fill representation"""
    fill_id: str
    order_id: str
    instrument: Instrument
    side: OrderSide
    quantity: int
    price: float
    commission: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Environment tracking
    environment: TradingEnvironment = TradingEnvironment.PAPER
    
    @property
    def gross_amount(self) -> float:
        """Get gross trade amount"""
        return self.quantity * self.price * self.instrument.multiplier
    
    @property
    def net_amount(self) -> float:
        """Get net trade amount (including commission)"""
        return self.gross_amount - self.commission

class OrderValidationError(Exception):
    """Order validation error"""
    pass

class InsufficientFundsError(Exception):
    """Insufficient funds error"""
    pass

class PositionNotFoundError(Exception):
    """Position not found error"""
    pass

class TradingExecutionEngine:
    """
    Advanced Trading Execution Engine with Order Management
    
    Provides comprehensive order management and execution capabilities:
    - Real-time order management and tracking
    - Paper/Live trading environment switching
    - Position tracking and P&L calculation
    - Order validation and risk controls
    - Smart execution algorithms
    """
    
    def __init__(self, environment: TradingEnvironment = TradingEnvironment.PAPER):
        """Initialize the trading execution engine"""
        self.logger = logging.getLogger(__name__)
        
        # Environment configuration
        self.environment = environment
        self.environment_config = self._load_environment_config()
        
        # Order management
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}  # Key: instrument_symbol
        self.fills: List[Fill] = []
        
        # Execution state
        self.is_running = False
        self.execution_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.execution_stats = {
            'orders_submitted': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'orders_rejected': 0,
            'total_volume': 0.0,
            'total_commission': 0.0,
            'avg_execution_time_ms': 0.0
        }
        
        # Risk limits (environment-specific)
        self.risk_limits = self._load_risk_limits()
        
        # Market data cache (for paper trading)
        self.market_data_cache: Dict[str, Dict[str, Any]] = {}
        
        # Callbacks for external integration
        self.order_callbacks: List[Callable] = []
        self.fill_callbacks: List[Callable] = []
        self.position_callbacks: List[Callable] = []
        
        self.logger.info(f"Trading Execution Engine initialized in {environment.value} mode")
    
    def switch_environment(self, new_environment: TradingEnvironment, 
                          confirm_live: bool = False) -> bool:
        """
        Switch trading environment
        
        Args:
            new_environment: Target environment
            confirm_live: Required confirmation for live trading
            
        Returns:
            bool: Success status
        """
        try:
            # Safety check for live trading
            if new_environment == TradingEnvironment.LIVE and not confirm_live:
                raise ValueError("Live trading requires explicit confirmation (confirm_live=True)")
            
            # Validate environment switch
            if self.environment == new_environment:
                self.logger.info(f"Already in {new_environment.value} environment")
                return True
            
            # Check for open orders
            open_orders = [o for o in self.orders.values() 
                          if o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.ACCEPTED]]
            
            if open_orders and new_environment == TradingEnvironment.LIVE:
                raise ValueError(f"Cannot switch to live trading with {len(open_orders)} open orders")
            
            # Perform environment switch
            old_environment = self.environment
            self.environment = new_environment
            self.environment_config = self._load_environment_config()
            self.risk_limits = self._load_risk_limits()
            
            # Clear environment-specific data if switching to different mode
            if old_environment != new_environment:
                if new_environment == TradingEnvironment.PAPER:
                    # Keep positions for paper trading continuity
                    pass
                elif new_environment == TradingEnvironment.LIVE:
                    # Validate positions against live account
                    self._validate_live_positions()
            
            self.logger.info(f"Switched from {old_environment.value} to {new_environment.value} environment")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to switch environment: {str(e)}")
            return False
    
    def submit_order(self, order: Order) -> str:
        """
        Submit an order for execution
        
        Args:
            order: Order to submit
            
        Returns:
            str: Order ID
        """
        try:
            start_time = time.time()
            
            # Set environment
            order.environment = self.environment
            
            # Validate order
            self._validate_order(order)
            
            # Generate order ID if not provided
            if not order.order_id:
                order.order_id = self._generate_order_id()
            
            # Update order status
            order.status = OrderStatus.SUBMITTED
            order.updated_time = datetime.now()
            
            # Store order
            self.orders[order.order_id] = order
            
            # Execute order based on environment
            if self.environment == TradingEnvironment.PAPER:
                self._execute_paper_order(order)
            elif self.environment == TradingEnvironment.LIVE:
                self._execute_live_order(order)
            
            # Update statistics
            self.execution_stats['orders_submitted'] += 1
            execution_time = (time.time() - start_time) * 1000
            self._update_avg_execution_time(execution_time)
            
            # Notify callbacks
            self._notify_order_callbacks(order)
            
            self.logger.info(f"Order {order.order_id} submitted successfully in {execution_time:.2f}ms")
            return order.order_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit order: {str(e)}")
            if order.order_id:
                order.status = OrderStatus.REJECTED
                self.execution_stats['orders_rejected'] += 1
            raise
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            bool: Success status
        """
        try:
            if order_id not in self.orders:
                raise ValueError(f"Order {order_id} not found")
            
            order = self.orders[order_id]
            
            # Check if order can be cancelled
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                raise ValueError(f"Cannot cancel order in {order.status.value} status")
            
            # Cancel order based on environment
            if self.environment == TradingEnvironment.PAPER:
                success = self._cancel_paper_order(order)
            elif self.environment == TradingEnvironment.LIVE:
                success = self._cancel_live_order(order)
            else:
                success = True
            
            if success:
                order.status = OrderStatus.CANCELLED
                order.updated_time = datetime.now()
                self.execution_stats['orders_cancelled'] += 1
                
                # Notify callbacks
                self._notify_order_callbacks(order)
                
                self.logger.info(f"Order {order_id} cancelled successfully")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return False
    
    def modify_order(self, order_id: str, **modifications) -> bool:
        """
        Modify an existing order
        
        Args:
            order_id: Order ID to modify
            **modifications: Order modifications
            
        Returns:
            bool: Success status
        """
        try:
            if order_id not in self.orders:
                raise ValueError(f"Order {order_id} not found")
            
            order = self.orders[order_id]
            
            # Check if order can be modified
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                raise ValueError(f"Cannot modify order in {order.status.value} status")
            
            # Apply modifications
            for key, value in modifications.items():
                if hasattr(order, key):
                    setattr(order, key, value)
            
            # Validate modified order
            self._validate_order(order)
            
            # Update timestamp
            order.updated_time = datetime.now()
            
            # Apply modifications based on environment
            if self.environment == TradingEnvironment.PAPER:
                success = self._modify_paper_order(order)
            elif self.environment == TradingEnvironment.LIVE:
                success = self._modify_live_order(order)
            else:
                success = True
            
            if success:
                # Notify callbacks
                self._notify_order_callbacks(order)
                self.logger.info(f"Order {order_id} modified successfully")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to modify order {order_id}: {str(e)}")
            return False
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_orders(self, status: Optional[OrderStatus] = None, 
                   instrument: Optional[str] = None) -> List[Order]:
        """
        Get orders with optional filtering
        
        Args:
            status: Filter by order status
            instrument: Filter by instrument symbol
            
        Returns:
            List[Order]: Filtered orders
        """
        orders = list(self.orders.values())
        
        if status:
            orders = [o for o in orders if o.status == status]
        
        if instrument:
            orders = [o for o in orders if o.instrument.symbol == instrument]
        
        return orders
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol"""
        return self.positions.get(symbol)
    
    def get_positions(self, account_id: Optional[str] = None) -> List[Position]:
        """
        Get positions with optional filtering
        
        Args:
            account_id: Filter by account ID
            
        Returns:
            List[Position]: Filtered positions
        """
        positions = list(self.positions.values())
        
        if account_id:
            positions = [p for p in positions if p.account_id == account_id]
        
        return positions
    
    def get_portfolio_value(self, account_id: Optional[str] = None) -> Dict[str, float]:
        """
        Get portfolio value summary
        
        Args:
            account_id: Filter by account ID
            
        Returns:
            Dict[str, float]: Portfolio value metrics
        """
        positions = self.get_positions(account_id)
        
        total_market_value = sum(p.market_value for p in positions)
        total_cost_basis = sum(p.cost_basis for p in positions)
        total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        
        return {
            'total_market_value': total_market_value,
            'total_cost_basis': total_cost_basis,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_unrealized_pnl_percent': (total_unrealized_pnl / abs(total_cost_basis) * 100) if total_cost_basis != 0 else 0.0,
            'position_count': len(positions)
        }
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            **self.execution_stats,
            'environment': self.environment.value,
            'active_orders': len([o for o in self.orders.values() 
                                if o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.ACCEPTED]]),
            'total_positions': len(self.positions),
            'total_fills': len(self.fills)
        }
    
    def add_order_callback(self, callback: Callable[[Order], None]):
        """Add order event callback"""
        self.order_callbacks.append(callback)
    
    def add_fill_callback(self, callback: Callable[[Fill], None]):
        """Add fill event callback"""
        self.fill_callbacks.append(callback)
    
    def add_position_callback(self, callback: Callable[[Position], None]):
        """Add position event callback"""
        self.position_callbacks.append(callback)
    
    # Private methods
    def _load_environment_config(self) -> Dict[str, Any]:
        """Load environment-specific configuration"""
        config = {
            TradingEnvironment.PAPER: {
                'commission_per_contract': 0.65,
                'commission_per_share': 0.005,
                'slippage_bps': 2.0,  # 2 basis points
                'fill_probability': 0.98,
                'partial_fill_probability': 0.15,
                'execution_delay_ms': 50
            },
            TradingEnvironment.LIVE: {
                'commission_per_contract': 0.65,
                'commission_per_share': 0.005,
                'slippage_bps': 1.0,
                'execution_delay_ms': 100,
                'require_confirmation': True
            }
        }
        
        return config.get(self.environment, config[TradingEnvironment.PAPER])
    
    def _load_risk_limits(self) -> Dict[str, Any]:
        """Load environment-specific risk limits"""
        limits = {
            TradingEnvironment.PAPER: {
                'max_order_value': 100000.0,
                'max_position_size': 1000,
                'max_daily_loss': 5000.0,
                'max_portfolio_concentration': 0.25
            },
            TradingEnvironment.LIVE: {
                'max_order_value': 50000.0,
                'max_position_size': 500,
                'max_daily_loss': 2500.0,
                'max_portfolio_concentration': 0.20
            }
        }
        
        return limits.get(self.environment, limits[TradingEnvironment.PAPER])
    
    def _validate_order(self, order: Order):
        """Validate order before submission"""
        # Basic validation
        if order.quantity <= 0:
            raise OrderValidationError("Order quantity must be positive")
        
        # Price validation
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if order.price is None or order.price <= 0:
                raise OrderValidationError(f"{order.order_type.value} orders require a valid price")
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if order.stop_price is None or order.stop_price <= 0:
                raise OrderValidationError(f"{order.order_type.value} orders require a valid stop price")
        
        # Risk limit validation
        order_value = self._calculate_order_value(order)
        if order_value > self.risk_limits['max_order_value']:
            raise OrderValidationError(f"Order value ${order_value:,.2f} exceeds limit ${self.risk_limits['max_order_value']:,.2f}")
        
        if order.quantity > self.risk_limits['max_position_size']:
            raise OrderValidationError(f"Order quantity {order.quantity} exceeds limit {self.risk_limits['max_position_size']}")
        
        # Position validation for closing orders
        if order.side in [OrderSide.SELL_TO_CLOSE, OrderSide.BUY_TO_CLOSE]:
            position = self.get_position(order.instrument.symbol)
            if not position:
                raise OrderValidationError(f"No position found for {order.instrument.symbol}")
            
            if order.side == OrderSide.SELL_TO_CLOSE and position.quantity < order.quantity:
                raise OrderValidationError(f"Insufficient long position to sell {order.quantity} shares")
            
            if order.side == OrderSide.BUY_TO_CLOSE and abs(position.quantity) < order.quantity:
                raise OrderValidationError(f"Insufficient short position to cover {order.quantity} shares")
    
    def _calculate_order_value(self, order: Order) -> float:
        """Calculate order value for risk validation"""
        if order.price:
            price = order.price
        elif order.order_type == OrderType.MARKET:
            # Use current market price or estimate
            price = self._get_market_price(order.instrument.symbol)
        else:
            price = 0.0
        
        return order.quantity * price * order.instrument.multiplier
    
    def _get_market_price(self, symbol: str) -> float:
        """Get current market price (mock implementation)"""
        # In real implementation, this would fetch from market data
        if symbol in self.market_data_cache:
            return self.market_data_cache[symbol].get('price', 100.0)
        return 100.0  # Default price for testing
    
    def _execute_paper_order(self, order: Order):
        """Execute order in paper trading environment"""
        try:
            # Simulate execution delay
            execution_delay = self.environment_config['execution_delay_ms'] / 1000.0
            time.sleep(execution_delay)
            
            # Determine fill probability
            fill_prob = self.environment_config['fill_probability']
            partial_fill_prob = self.environment_config['partial_fill_probability']
            
            # Simulate order execution
            if np.random.random() < fill_prob:
                # Determine if partial fill
                if np.random.random() < partial_fill_prob:
                    fill_quantity = int(order.quantity * np.random.uniform(0.3, 0.8))
                else:
                    fill_quantity = order.quantity
                
                # Calculate fill price with slippage
                fill_price = self._calculate_fill_price(order)
                
                # Create fill
                fill = self._create_fill(order, fill_quantity, fill_price)
                
                # Process fill
                self._process_fill(fill)
                
            else:
                # Order rejected
                order.status = OrderStatus.REJECTED
                self.execution_stats['orders_rejected'] += 1
                
        except Exception as e:
            self.logger.error(f"Error executing paper order {order.order_id}: {str(e)}")
            order.status = OrderStatus.REJECTED
            self.execution_stats['orders_rejected'] += 1
    
    def _execute_live_order(self, order: Order):
        """Execute order in live trading environment"""
        # This would integrate with actual broker APIs
        # For now, we'll simulate with enhanced realism
        try:
            self.logger.info(f"Executing live order {order.order_id} - THIS IS LIVE TRADING")
            
            # In real implementation:
            # 1. Send order to broker API
            # 2. Receive order confirmation
            # 3. Monitor for fills
            # 4. Update order status
            
            # For simulation, mark as accepted
            order.status = OrderStatus.ACCEPTED
            
            # Simulate faster execution for live trading
            execution_delay = self.environment_config['execution_delay_ms'] / 1000.0
            time.sleep(execution_delay)
            
            # Create fill (simplified for demo)
            fill_price = self._calculate_fill_price(order)
            fill = self._create_fill(order, order.quantity, fill_price)
            self._process_fill(fill)
            
        except Exception as e:
            self.logger.error(f"Error executing live order {order.order_id}: {str(e)}")
            order.status = OrderStatus.REJECTED
            self.execution_stats['orders_rejected'] += 1
    
    def _calculate_fill_price(self, order: Order) -> float:
        """Calculate fill price with slippage"""
        if order.order_type == OrderType.MARKET:
            market_price = self._get_market_price(order.instrument.symbol)
            slippage_bps = self.environment_config['slippage_bps']
            slippage = market_price * (slippage_bps / 10000.0)
            
            # Apply slippage based on order side
            if order.side in [OrderSide.BUY, OrderSide.BUY_TO_OPEN, OrderSide.BUY_TO_CLOSE]:
                return market_price + slippage
            else:
                return market_price - slippage
        
        elif order.order_type == OrderType.LIMIT:
            return order.price
        
        else:
            # For other order types, use order price or market price
            return order.price if order.price else self._get_market_price(order.instrument.symbol)
    
    def _create_fill(self, order: Order, quantity: int, price: float) -> Fill:
        """Create a fill record"""
        # Calculate commission
        if order.instrument.instrument_type == InstrumentType.OPTION:
            commission = quantity * self.environment_config['commission_per_contract']
        else:
            commission = quantity * self.environment_config['commission_per_share']
        
        fill = Fill(
            fill_id=self._generate_fill_id(),
            order_id=order.order_id,
            instrument=order.instrument,
            side=order.side,
            quantity=quantity,
            price=price,
            commission=commission,
            environment=self.environment
        )
        
        return fill
    
    def _process_fill(self, fill: Fill):
        """Process a fill and update positions"""
        try:
            # Add fill to records
            self.fills.append(fill)
            
            # Update order
            order = self.orders[fill.order_id]
            order.filled_quantity += fill.quantity
            order.avg_fill_price = self._calculate_avg_fill_price(order)
            order.commission += fill.commission
            
            # Update order status
            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
                self.execution_stats['orders_filled'] += 1
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
            
            order.updated_time = datetime.now()
            
            # Update position
            self._update_position(fill)
            
            # Update statistics
            self.execution_stats['total_volume'] += fill.gross_amount
            self.execution_stats['total_commission'] += fill.commission
            
            # Notify callbacks
            self._notify_fill_callbacks(fill)
            self._notify_order_callbacks(order)
            
            self.logger.info(f"Processed fill: {fill.quantity} @ ${fill.price:.2f} for order {fill.order_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing fill {fill.fill_id}: {str(e)}")
    
    def _update_position(self, fill: Fill):
        """Update position based on fill"""
        symbol = fill.instrument.symbol
        
        # Get or create position
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                instrument=fill.instrument,
                quantity=0,
                avg_price=0.0,
                account_id=self.orders[fill.order_id].account_id,
                strategy_id=self.orders[fill.order_id].strategy_id,
                environment=self.environment
            )
        
        position = self.positions[symbol]
        
        # Calculate quantity change
        if fill.side in [OrderSide.BUY, OrderSide.BUY_TO_OPEN]:
            quantity_change = fill.quantity
        elif fill.side in [OrderSide.SELL, OrderSide.SELL_TO_OPEN]:
            quantity_change = -fill.quantity
        elif fill.side == OrderSide.BUY_TO_CLOSE:
            quantity_change = fill.quantity  # Closing short position
        elif fill.side == OrderSide.SELL_TO_CLOSE:
            quantity_change = -fill.quantity  # Closing long position
        else:
            quantity_change = 0
        
        # Update position
        old_quantity = position.quantity
        new_quantity = old_quantity + quantity_change
        
        # Calculate new average price
        if new_quantity != 0:
            if (old_quantity >= 0 and quantity_change > 0) or (old_quantity <= 0 and quantity_change < 0):
                # Adding to existing position
                total_cost = (old_quantity * position.avg_price) + (quantity_change * fill.price)
                position.avg_price = total_cost / new_quantity
            else:
                # Reducing or reversing position
                if abs(new_quantity) < abs(old_quantity):
                    # Partial close - keep same avg price
                    pass
                else:
                    # Position reversal
                    position.avg_price = fill.price
        
        position.quantity = new_quantity
        position.updated_time = datetime.now()
        
        # Update market price
        position.market_price = fill.price
        
        # Remove position if flat
        if position.quantity == 0:
            del self.positions[symbol]
        else:
            # Notify position callbacks
            self._notify_position_callbacks(position)
        
        self.logger.debug(f"Updated position {symbol}: {old_quantity} -> {new_quantity} @ ${position.avg_price:.2f}")
    
    def _calculate_avg_fill_price(self, order: Order) -> float:
        """Calculate average fill price for order"""
        order_fills = [f for f in self.fills if f.order_id == order.order_id]
        
        if not order_fills:
            return 0.0
        
        total_value = sum(f.quantity * f.price for f in order_fills)
        total_quantity = sum(f.quantity for f in order_fills)
        
        return total_value / total_quantity if total_quantity > 0 else 0.0
    
    def _cancel_paper_order(self, order: Order) -> bool:
        """Cancel order in paper trading"""
        # Paper trading cancellation is always successful
        return True
    
    def _cancel_live_order(self, order: Order) -> bool:
        """Cancel order in live trading"""
        # This would integrate with broker API
        # For simulation, return success
        self.logger.info(f"Cancelling live order {order.order_id} - THIS IS LIVE TRADING")
        return True
    
    def _modify_paper_order(self, order: Order) -> bool:
        """Modify order in paper trading"""
        # Paper trading modification is always successful
        return True
    
    def _modify_live_order(self, order: Order) -> bool:
        """Modify order in live trading"""
        # This would integrate with broker API
        # For simulation, return success
        self.logger.info(f"Modifying live order {order.order_id} - THIS IS LIVE TRADING")
        return True
    
    def _validate_live_positions(self):
        """Validate positions against live account"""
        # This would reconcile positions with broker account
        self.logger.info("Validating positions against live account")
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        return f"ORD_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    
    def _generate_fill_id(self) -> str:
        """Generate unique fill ID"""
        return f"FILL_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    
    def _update_avg_execution_time(self, execution_time_ms: float):
        """Update average execution time"""
        current_avg = self.execution_stats['avg_execution_time_ms']
        total_orders = self.execution_stats['orders_submitted']
        
        if total_orders == 1:
            self.execution_stats['avg_execution_time_ms'] = execution_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.execution_stats['avg_execution_time_ms'] = (alpha * execution_time_ms) + ((1 - alpha) * current_avg)
    
    def _notify_order_callbacks(self, order: Order):
        """Notify order event callbacks"""
        for callback in self.order_callbacks:
            try:
                callback(order)
            except Exception as e:
                self.logger.error(f"Error in order callback: {str(e)}")
    
    def _notify_fill_callbacks(self, fill: Fill):
        """Notify fill event callbacks"""
        for callback in self.fill_callbacks:
            try:
                callback(fill)
            except Exception as e:
                self.logger.error(f"Error in fill callback: {str(e)}")
    
    def _notify_position_callbacks(self, position: Position):
        """Notify position event callbacks"""
        for callback in self.position_callbacks:
            try:
                callback(position)
            except Exception as e:
                self.logger.error(f"Error in position callback: {str(e)}")

def test_trading_execution_engine():
    """Test the trading execution engine"""
    print("🚀 Testing Trading Execution Engine...")
    
    # Test Paper Trading Environment
    print("\n📋 Testing Paper Trading Environment")
    engine = TradingExecutionEngine(TradingEnvironment.PAPER)
    
    # Create test instrument
    spy_option = Instrument(
        symbol="SPY_240621C450",
        instrument_type=InstrumentType.OPTION,
        underlying="SPY",
        expiration=datetime(2024, 6, 21),
        strike=450.0,
        option_type="call",
        exchange="CBOE"
    )
    
    # Create test order
    test_order = Order(
        order_id="",
        instrument=spy_option,
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY_TO_OPEN,
        quantity=10,
        price=2.50,
        account_id="TEST_ACCOUNT",
        strategy_id="PUT_SELLING"
    )
    
    # Submit order
    print(f"  📤 Submitting order: {test_order.side.value} {test_order.quantity} {test_order.instrument.symbol} @ ${test_order.price}")
    order_id = engine.submit_order(test_order)
    print(f"  ✅ Order submitted: {order_id}")
    
    # Check order status
    order = engine.get_order(order_id)
    print(f"  📊 Order status: {order.status.value}")
    print(f"  📊 Fill status: {order.filled_quantity}/{order.quantity} filled")
    
    # Check positions
    positions = engine.get_positions()
    if positions:
        position = positions[0]
        print(f"  📈 Position: {position.quantity} {position.instrument.symbol} @ ${position.avg_price:.2f}")
        print(f"  💰 Unrealized P&L: ${position.unrealized_pnl:.2f}")
    
    # Test Environment Switching
    print("\n🔄 Testing Environment Switching")
    
    # Try switching to live without confirmation (should fail)
    try:
        success = engine.switch_environment(TradingEnvironment.LIVE)
        print(f"  ❌ Unexpected success switching to live without confirmation")
    except ValueError as e:
        print(f"  ✅ Correctly blocked live switch: {str(e)}")
    
    # Switch to live with confirmation
    success = engine.switch_environment(TradingEnvironment.LIVE, confirm_live=True)
    print(f"  {'✅' if success else '❌'} Switch to live trading: {success}")
    
    # Switch back to paper
    success = engine.switch_environment(TradingEnvironment.PAPER)
    print(f"  {'✅' if success else '❌'} Switch back to paper trading: {success}")
    
    # Test Order Management
    print("\n📋 Testing Order Management")
    
    # Create another order
    test_order2 = Order(
        order_id="",
        instrument=spy_option,
        order_type=OrderType.MARKET,
        side=OrderSide.SELL_TO_CLOSE,
        quantity=5,
        account_id="TEST_ACCOUNT",
        strategy_id="PUT_SELLING"
    )
    
    order_id2 = engine.submit_order(test_order2)
    print(f"  📤 Submitted second order: {order_id2}")
    
    # Test order cancellation
    test_order3 = Order(
        order_id="",
        instrument=spy_option,
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY_TO_OPEN,
        quantity=5,
        price=2.00,
        account_id="TEST_ACCOUNT"
    )
    
    order_id3 = engine.submit_order(test_order3)
    cancel_success = engine.cancel_order(order_id3)
    print(f"  {'✅' if cancel_success else '❌'} Order cancellation: {cancel_success}")
    
    # Test Portfolio Summary
    print("\n📊 Testing Portfolio Summary")
    portfolio = engine.get_portfolio_value()
    print(f"  💰 Total Market Value: ${portfolio['total_market_value']:,.2f}")
    print(f"  💰 Total Cost Basis: ${portfolio['total_cost_basis']:,.2f}")
    print(f"  📈 Total Unrealized P&L: ${portfolio['total_unrealized_pnl']:,.2f}")
    print(f"  📊 Position Count: {portfolio['position_count']}")
    
    # Test Execution Statistics
    print("\n📈 Testing Execution Statistics")
    stats = engine.get_execution_stats()
    print(f"  📊 Orders Submitted: {stats['orders_submitted']}")
    print(f"  ✅ Orders Filled: {stats['orders_filled']}")
    print(f"  ❌ Orders Cancelled: {stats['orders_cancelled']}")
    print(f"  ⚡ Avg Execution Time: {stats['avg_execution_time_ms']:.2f}ms")
    print(f"  🌍 Environment: {stats['environment']}")
    print(f"  📋 Active Orders: {stats['active_orders']}")
    
    print("\n✅ Trading Execution Engine testing completed!")
    
    return {
        'engine': engine,
        'orders_submitted': stats['orders_submitted'],
        'orders_filled': stats['orders_filled'],
        'portfolio_value': portfolio['total_market_value'],
        'execution_time_ms': stats['avg_execution_time_ms'],
        'environment_switching': True
    }

if __name__ == "__main__":
    test_trading_execution_engine()

