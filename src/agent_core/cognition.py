"""
ALL-USE Agent: Cognition Module

This module handles the cognitive processing for the ALL-USE agent, implementing
decision-making, context management, and protocol application logic.

Key components:
- ContextManager: Maintains and updates the agent's understanding of the current context
- DecisionEngine: Makes decisions based on the current context and ALL-USE protocol
- ProtocolApplicator: Applies the ALL-USE protocol rules to specific situations
- CognitiveController: Coordinates the cognitive processing pipeline
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum

from .perception import EventType, Intent, Entity, PerceptionEvent
from .memory import MemoryType, ConversationEntry, ProtocolState, UserPreferences, MemoryManager

class DecisionType(Enum):
    """Types of decisions that the agent can make."""
    RESPONSE = "response"
    TRADE = "trade"
    ACCOUNT_ACTION = "account_action"
    NOTIFICATION = "notification"
    ANALYSIS = "analysis"

class Decision:
    """Represents a decision made by the agent."""
    def __init__(
        self,
        decision_type: DecisionType,
        content: Dict[str, Any],
        confidence: float = 1.0,
        reasoning: List[str] = None,
        timestamp: datetime = None
    ):
        self.decision_type = decision_type
        self.content = content
        self.confidence = confidence
        self.reasoning = reasoning or []
        self.timestamp = timestamp or datetime.now()
    
    def __repr__(self):
        return f"Decision({self.decision_type}, confidence={self.confidence})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the decision to a dictionary."""
        return {
            "decision_type": self.decision_type.value,
            "content": self.content,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat()
        }

class WeekClassification(Enum):
    """Classification of trading weeks in the ALL-USE protocol."""
    P_EW = "P-EW"    # Puts Expired Worthless
    P_AWL = "P-AWL"  # Puts Assigned Within Limit
    P_RO = "P-RO"    # Puts Roll Over
    P_AOL = "P-AOL"  # Puts Assigned Over Limit
    P_DD = "P-DD"    # Puts Deep Drawdown
    C_WAP = "C-WAP"  # Calls With Appreciation Profit
    C_WAP_PLUS = "C-WAP+"  # Calls With Strong Appreciation Profit
    C_PNO = "C-PNO"  # Calls Premium-Only
    C_RO = "C-RO"    # Calls Roll Over
    C_REC = "C-REC"  # Calls Recovery Mode
    W_IDL = "W-IDL"  # Week Idle

class ContextManager:
    """Maintains and updates the agent's understanding of the current context."""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.current_context = {
            "conversation": [],
            "protocol_state": {},
            "user_preferences": {},
            "current_focus": None,
            "last_updated": datetime.now().isoformat()
        }
    
    def update_context(self, event: PerceptionEvent, user_id: str) -> Dict[str, Any]:
        """Update the context based on a new perception event."""
        # Update conversation context
        self.current_context["conversation"] = self.memory_manager.get_conversation_context()
        
        # Update protocol state context
        protocol_context = self.memory_manager.get_protocol_context(user_id)
        if protocol_context:
            self.current_context["protocol_state"] = protocol_context
        
        # Update user preferences context
        user_context = self.memory_manager.get_user_context(user_id)
        if user_context:
            self.current_context["user_preferences"] = user_context
        
        # Update focus based on event type and intent
        if event.event_type == EventType.USER_MESSAGE and event.intent:
            self.current_context["current_focus"] = event.intent.value
        
        self.current_context["last_updated"] = datetime.now().isoformat()
        
        return self.current_context
    
    def get_context(self) -> Dict[str, Any]:
        """Get the current context."""
        return self.current_context

class ProtocolApplicator:
    """Applies the ALL-USE protocol rules to specific situations."""
    
    def __init__(self):
        # Weekly return targets for each account type
        self.weekly_return_targets = {
            "Gen-Acc": 0.015,  # 1.5%
            "Rev-Acc": 0.010,  # 1.0%
            "Com-Acc": 0.005   # 0.5%
        }
        
        # Delta ranges for each account type
        self.delta_ranges = {
            "Gen-Acc": (0.40, 0.50),  # 40-50 delta
            "Rev-Acc": (0.30, 0.40),  # 30-40 delta
            "Com-Acc": (0.20, 0.30)   # 20-30 delta
        }
        
        # Forking and merging thresholds
        self.fork_threshold = 100000.0  # $100K
        self.merge_threshold = 500000.0  # $500K
        
        # Reinvestment parameters
        self.reinvestment_frequency = "quarterly"
        self.reinvestment_contract_allocation = 0.75  # 75% to contracts
        self.reinvestment_leap_allocation = 0.25     # 25% to LEAPS
    
    def evaluate_fork_condition(self, account_state: Dict[str, Any]) -> bool:
        """Evaluate if an account meets the forking condition."""
        if account_state["account_type"] != "Gen-Acc":
            return False
        
        initial_balance = account_state["initial_balance"]
        current_balance = account_state["current_balance"]
        
        # Check if surplus exceeds fork threshold
        surplus = current_balance - initial_balance
        return surplus >= self.fork_threshold
    
    def evaluate_merge_condition(self, account_state: Dict[str, Any]) -> bool:
        """Evaluate if an account meets the merging condition."""
        if account_state.get("is_parent", True) or not account_state.get("parent_id"):
            return False
        
        current_balance = account_state["current_balance"]
        return current_balance >= self.merge_threshold
    
    def calculate_reinvestment_amounts(self, pending_amount: float) -> Dict[str, float]:
        """Calculate reinvestment amounts based on protocol rules."""
        contract_amount = pending_amount * self.reinvestment_contract_allocation
        leap_amount = pending_amount * self.reinvestment_leap_allocation
        
        return {
            "contract_amount": contract_amount,
            "leap_amount": leap_amount,
            "total_amount": pending_amount
        }
    
    def classify_week(self, week_data: Dict[str, Any]) -> WeekClassification:
        """Classify a trading week based on its outcome."""
        # Extract relevant data
        option_type = week_data.get("option_type", "put").lower()
        strike_price = week_data.get("strike_price", 0.0)
        entry_price = week_data.get("entry_price", 0.0)
        exit_price = week_data.get("exit_price", 0.0)
        underlying_price = week_data.get("underlying_price", 0.0)
        final_price = week_data.get("final_price", 0.0)
        premium = week_data.get("premium", 0.0)
        
        # Calculate key metrics
        price_change_pct = (final_price - underlying_price) / underlying_price if underlying_price else 0
        premium_pct = premium / strike_price if strike_price else 0
        
        # Classify based on option type and outcome
        if option_type == "put":
            if final_price >= strike_price:
                return WeekClassification.P_EW  # Puts Expired Worthless
            elif final_price >= strike_price * 0.98:
                return WeekClassification.P_AWL  # Puts Assigned Within Limit
            elif final_price >= strike_price * 0.95:
                return WeekClassification.P_RO  # Puts Roll Over
            elif final_price >= strike_price * 0.90:
                return WeekClassification.P_AOL  # Puts Assigned Over Limit
            else:
                return WeekClassification.P_DD  # Puts Deep Drawdown
        
        elif option_type == "call":
            if price_change_pct >= 0.05:
                return WeekClassification.C_WAP_PLUS  # Calls With Strong Appreciation Profit
            elif price_change_pct >= 0.02:
                return WeekClassification.C_WAP  # Calls With Appreciation Profit
            elif final_price <= strike_price:
                return WeekClassification.C_PNO  # Calls Premium-Only
            elif final_price <= strike_price * 1.02:
                return WeekClassification.C_RO  # Calls Roll Over
            else:
                return WeekClassification.C_REC  # Calls Recovery Mode
        
        else:
            return WeekClassification.W_IDL  # Week Idle
    
    def get_week_action(self, classification: WeekClassification) -> Dict[str, Any]:
        """Get the recommended action for a classified week."""
        actions = {
            WeekClassification.P_EW: {
                "action": "collect_premium",
                "next_trade": "new_put_position",
                "expected_return": (0.018, 0.022)  # 1.8-2.2%
            },
            WeekClassification.P_AWL: {
                "action": "accept_assignment",
                "next_trade": "sell_calls",
                "expected_return": (0.015, 0.018)  # 1.5-1.8%
            },
            WeekClassification.P_RO: {
                "action": "roll_position",
                "next_trade": "roll_put_down_and_out",
                "expected_return": (0.010, 0.015)  # 1.0-1.5%
            },
            WeekClassification.P_AOL: {
                "action": "accept_assignment",
                "next_trade": "sell_calls_reduced_delta",
                "expected_return": (0.008, 0.012)  # 0.8-1.2%
            },
            WeekClassification.P_DD: {
                "action": "manage_loss",
                "next_trade": "recovery_strategy",
                "expected_return": (-0.005, 0.000)  # -0.5-0.0%
            },
            WeekClassification.C_WAP: {
                "action": "collect_premium_and_appreciation",
                "next_trade": "new_put_position",
                "expected_return": (0.030, 0.040)  # 3.0-4.0%
            },
            WeekClassification.C_WAP_PLUS: {
                "action": "collect_premium_and_strong_appreciation",
                "next_trade": "new_put_position",
                "expected_return": (0.050, 0.060)  # 5.0-6.0%
            },
            WeekClassification.C_PNO: {
                "action": "collect_premium_only",
                "next_trade": "new_put_position",
                "expected_return": (0.018, 0.022)  # 1.8-2.2%
            },
            WeekClassification.C_RO: {
                "action": "roll_position",
                "next_trade": "roll_call_up_and_out",
                "expected_return": (0.008, 0.012)  # 0.8-1.2%
            },
            WeekClassification.C_REC: {
                "action": "recovery_mode",
                "next_trade": "balanced_delta_call",
                "expected_return": (0.005, 0.008)  # 0.5-0.8%
            },
            WeekClassification.W_IDL: {
                "action": "wait",
                "next_trade": "evaluate_market",
                "expected_return": (0.000, 0.000)  # 0.0%
            }
        }
        
        return actions.get(classification, actions[WeekClassification.W_IDL])

class DecisionEngine:
    """Makes decisions based on the current context and ALL-USE protocol."""
    
    def __init__(self, protocol_applicator: ProtocolApplicator):
        self.protocol_applicator = protocol_applicator
    
    def make_decision(self, context: Dict[str, Any], event: PerceptionEvent) -> Decision:
        """Make a decision based on the current context and event."""
        if event.event_type == EventType.USER_MESSAGE:
            return self._make_response_decision(context, event)
        elif event.event_type == EventType.MARKET_DATA:
            return self._make_trade_decision(context, event)
        elif event.event_type == EventType.ACCOUNT_UPDATE:
            return self._make_account_action_decision(context, event)
        elif event.event_type == EventType.SYSTEM_NOTIFICATION:
            return self._make_notification_decision(context, event)
        elif event.event_type == EventType.SCHEDULED_EVENT:
            return self._make_scheduled_event_decision(context, event)
        else:
            # Default to a response decision if event type is unknown
            return self._make_response_decision(context, event)
    
    def _make_response_decision(self, context: Dict[str, Any], event: PerceptionEvent) -> Decision:
        """Make a decision about how to respond to a user message."""
        intent = event.intent
        entities = event.entities
        
        # Default response content
        response_content = {
            "message_type": "text",
            "content": "I understand your message. How can I assist you with ALL-USE today?",
            "suggestions": []
        }
        
        reasoning = ["User sent a message", f"Detected intent: {intent.value if intent else 'Unknown'}"]
        
        # Customize response based on intent
        if intent == Intent.ACCOUNT_INQUIRY:
            account_entities = [e for e in entities if e.entity_type == "account"]
            
            if account_entities:
                # User is asking about specific accounts
                account_types = [e.value for e in account_entities]
                reasoning.append(f"User is inquiring about specific accounts: {', '.join(account_types)}")
                
                # Get account information from context
                accounts = context.get("protocol_state", {}).get("accounts", {})
                account_info = {}
                
                for account_type in account_types:
                    if account_type in accounts:
                        account_info[account_type] = accounts[account_type]
                
                if account_info:
                    response_content["message_type"] = "account_info"
                    response_content["content"] = f"Here's the information about your {', '.join(account_types)} accounts:"
                    response_content["account_info"] = account_info
                else:
                    response_content["content"] = f"I don't have information about the requested accounts. Would you like to set up these accounts?"
            else:
                # User is asking about accounts in general
                reasoning.append("User is inquiring about accounts in general")
                
                # Get general account information from context
                accounts = context.get("protocol_state", {}).get("accounts", {})
                total_value = context.get("protocol_state", {}).get("total_value", 0)
                
                if accounts:
                    response_content["message_type"] = "account_summary"
                    response_content["content"] = "Here's a summary of your ALL-USE accounts:"
                    response_content["account_summary"] = {
                        "accounts": accounts,
                        "total_value": total_value,
                        "forked_accounts": context.get("protocol_state", {}).get("forked_accounts", 0)
                    }
                else:
                    response_content["content"] = "You don't have any ALL-USE accounts set up yet. Would you like to create your initial accounts?"
        
        elif intent == Intent.PERFORMANCE_INQUIRY:
            reasoning.append("User is inquiring about performance")
            
            # Get performance information from context
            protocol_state = context.get("protocol_state", {})
            
            if protocol_state:
                response_content["message_type"] = "performance_summary"
                response_content["content"] = "Here's a summary of your ALL-USE performance:"
                
                # Calculate basic performance metrics
                initial_investment = protocol_state.get("initial_investment", 0)
                total_value = protocol_state.get("total_value", 0)
                
                if initial_investment > 0:
                    total_return = (total_value - initial_investment) / initial_investment
                    response_content["performance_summary"] = {
                        "initial_investment": initial_investment,
                        "current_value": total_value,
                        "total_return": total_return,
                        "total_return_pct": f"{total_return * 100:.2f}%"
                    }
                else:
                    response_content["content"] = "I don't have enough information to calculate your performance yet."
            else:
                response_content["content"] = "I don't have any performance data for your ALL-USE accounts yet."
        
        elif intent == Intent.STRATEGY_INQUIRY:
            reasoning.append("User is inquiring about strategy")
            
            response_content["message_type"] = "strategy_explanation"
            response_content["content"] = "The ALL-USE strategy is based on a three-tiered account structure:"
            response_content["strategy_details"] = {
                "account_structure": {
                    "Gen-Acc": "Generation Account: 40-50 delta puts on volatile stocks like TSLA, NVDA",
                    "Rev-Acc": "Revenue Account: 30-40 delta puts on stable market leaders like AAPL, AMZN, MSFT",
                    "Com-Acc": "Compounding Account: 20-30 delta puts/calls on stable market leaders"
                },
                "weekly_returns": {
                    "Gen-Acc": "Target 1.5% weekly return",
                    "Rev-Acc": "Target 1.0% weekly return",
                    "Com-Acc": "Target 0.5% weekly return"
                },
                "reinvestment": "Quarterly reinvestment with 75% to contracts and 25% to LEAPS",
                "account_management": f"Forking at ${self.protocol_applicator.fork_threshold/1000:.0f}K surplus, merging at ${self.protocol_applicator.merge_threshold/1000:.0f}K"
            }
        
        elif intent == Intent.PROJECTION_REQUEST:
            reasoning.append("User is requesting projections")
            
            # Extract any amount entities
            amount_entities = [e for e in entities if e.entity_type == "amount"]
            initial_amount = 300000.0  # Default amount
            
            if amount_entities:
                initial_amount = amount_entities[0].value
                reasoning.append(f"User specified initial amount: ${initial_amount:,.2f}")
            
            response_content["message_type"] = "projection_summary"
            response_content["content"] = f"Here's a 10-year projection for an initial investment of ${initial_amount:,.2f}:"
            response_content["projection_summary"] = {
                "initial_investment": initial_amount,
                "year_10_value": initial_amount * 7.27,  # Conservative estimate from our projections
                "cagr": "21.94%",
                "weekly_income_year_10": initial_amount * 0.0313 / 52,  # Based on our projections
                "note": "This is a conservative projection based on the ALL-USE protocol with realistic market conditions."
            }
        
        elif intent == Intent.GREETING:
            reasoning.append("User sent a greeting")
            
            response_content["content"] = "Hello! I'm your ALL-USE agent. I can help you manage your three-tiered account structure, track performance, and make protocol-based decisions. How can I assist you today?"
        
        elif intent == Intent.HELP_REQUEST:
            reasoning.append("User is requesting help")
            
            response_content["message_type"] = "help"
            response_content["content"] = "I'm your ALL-USE agent. Here's how I can help you:"
            response_content["help_topics"] = {
                "account_management": "Set up and manage your Gen-Acc, Rev-Acc, and Com-Acc accounts",
                "performance_tracking": "Track the performance of your accounts and overall portfolio",
                "trade_recommendations": "Get recommendations for trades based on the ALL-USE protocol",
                "projections": "See projections for your portfolio growth over time",
                "strategy_explanation": "Learn more about the ALL-USE strategy and protocol"
            }
        
        return Decision(
            decision_type=DecisionType.RESPONSE,
            content=response_content,
            confidence=0.9,
            reasoning=reasoning
        )
    
    def _make_trade_decision(self, context: Dict[str, Any], event: PerceptionEvent) -> Decision:
        """Make a decision about trades based on market data."""
        market_data = event.processed_content
        
        # Default trade decision content
        trade_content = {
            "action": "no_action",
            "reason": "Insufficient market data for trade decision"
        }
        
        reasoning = ["Received market data", "Analyzing for trade opportunities"]
        
        # Extract tickers from market data
        tickers = market_data.get("tickers", [])
        
        if tickers:
            reasoning.append(f"Market data includes tickers: {', '.join(tickers)}")
            
            # Simple example logic - in a real implementation, this would be much more sophisticated
            # and would use the protocol_applicator to apply ALL-USE rules
            
            # Get account information from context
            accounts = context.get("protocol_state", {}).get("accounts", {})
            
            if "Gen-Acc" in accounts:
                # For Gen-Acc, look for volatile stocks
                volatile_tickers = [t for t in tickers if t in ["TSLA", "NVDA"]]
                
                if volatile_tickers:
                    ticker = volatile_tickers[0]
                    delta_range = self.protocol_applicator.delta_ranges["Gen-Acc"]
                    
                    trade_content = {
                        "action": "enter_put_position",
                        "account": "Gen-Acc",
                        "ticker": ticker,
                        "delta_range": delta_range,
                        "reason": f"Opportunity identified for Gen-Acc with {ticker}"
                    }
                    
                    reasoning.append(f"Identified opportunity for Gen-Acc with {ticker}")
                    reasoning.append(f"Recommended delta range: {delta_range[0]}-{delta_range[1]}")
            
            if "Rev-Acc" in accounts and trade_content["action"] == "no_action":
                # For Rev-Acc, look for stable market leaders
                stable_tickers = [t for t in tickers if t in ["AAPL", "AMZN", "MSFT"]]
                
                if stable_tickers:
                    ticker = stable_tickers[0]
                    delta_range = self.protocol_applicator.delta_ranges["Rev-Acc"]
                    
                    trade_content = {
                        "action": "enter_put_position",
                        "account": "Rev-Acc",
                        "ticker": ticker,
                        "delta_range": delta_range,
                        "reason": f"Opportunity identified for Rev-Acc with {ticker}"
                    }
                    
                    reasoning.append(f"Identified opportunity for Rev-Acc with {ticker}")
                    reasoning.append(f"Recommended delta range: {delta_range[0]}-{delta_range[1]}")
        
        return Decision(
            decision_type=DecisionType.TRADE,
            content=trade_content,
            confidence=0.8,
            reasoning=reasoning
        )
    
    def _make_account_action_decision(self, context: Dict[str, Any], event: PerceptionEvent) -> Decision:
        """Make a decision about account actions based on account updates."""
        account_update = event.processed_content
        
        # Default account action decision content
        account_action_content = {
            "action": "no_action",
            "reason": "No account action required"
        }
        
        reasoning = ["Received account update", "Analyzing for account actions"]
        
        # Check for forking opportunities
        accounts = account_update.get("accounts", [])
        
        for account_id in accounts:
            account = accounts[account_id]
            
            # Check if this is a Gen-Acc that meets forking criteria
            if account.get("account_type") == "Gen-Acc":
                if self.protocol_applicator.evaluate_fork_condition(account):
                    account_action_content = {
                        "action": "fork_account",
                        "account_id": account_id,
                        "surplus": account["current_balance"] - account["initial_balance"],
                        "reason": f"Gen-Acc has surplus exceeding ${self.protocol_applicator.fork_threshold/1000:.0f}K threshold"
                    }
                    
                    reasoning.append(f"Gen-Acc {account_id} meets forking criteria")
                    reasoning.append(f"Surplus: ${account['current_balance'] - account['initial_balance']:,.2f}")
                    reasoning.append(f"Threshold: ${self.protocol_applicator.fork_threshold:,.2f}")
                    break
            
            # Check if this is a forked account that meets merging criteria
            elif not account.get("is_parent", True) and account.get("parent_id"):
                if self.protocol_applicator.evaluate_merge_condition(account):
                    account_action_content = {
                        "action": "merge_account",
                        "account_id": account_id,
                        "parent_id": account["parent_id"],
                        "reason": f"Forked account has reached ${self.protocol_applicator.merge_threshold/1000:.0f}K threshold"
                    }
                    
                    reasoning.append(f"Forked account {account_id} meets merging criteria")
                    reasoning.append(f"Current balance: ${account['current_balance']:,.2f}")
                    reasoning.append(f"Threshold: ${self.protocol_applicator.merge_threshold:,.2f}")
                    break
        
        # Check for reinvestment opportunities
        current_week = context.get("protocol_state", {}).get("current_week", 0)
        
        # Quarterly reinvestment (every 13 weeks)
        if current_week > 0 and current_week % 13 == 0:
            pending_reinvestment = 0
            
            for account_id in accounts:
                account = accounts[account_id]
                pending_reinvestment += account.get("pending_reinvestment", 0)
            
            if pending_reinvestment > 0:
                reinvestment_amounts = self.protocol_applicator.calculate_reinvestment_amounts(pending_reinvestment)
                
                account_action_content = {
                    "action": "reinvest",
                    "total_amount": pending_reinvestment,
                    "contract_amount": reinvestment_amounts["contract_amount"],
                    "leap_amount": reinvestment_amounts["leap_amount"],
                    "reason": "Quarterly reinvestment of accumulated premiums"
                }
                
                reasoning.append("Quarterly reinvestment period reached")
                reasoning.append(f"Total reinvestment amount: ${pending_reinvestment:,.2f}")
                reasoning.append(f"Contract allocation: ${reinvestment_amounts['contract_amount']:,.2f}")
                reasoning.append(f"LEAP allocation: ${reinvestment_amounts['leap_amount']:,.2f}")
        
        return Decision(
            decision_type=DecisionType.ACCOUNT_ACTION,
            content=account_action_content,
            confidence=0.9,
            reasoning=reasoning
        )
    
    def _make_notification_decision(self, context: Dict[str, Any], event: PerceptionEvent) -> Decision:
        """Make a decision about notifications based on system notifications."""
        notification = event.processed_content
        
        # Default notification decision content
        notification_content = {
            "send_notification": False,
            "notification_type": "info",
            "message": "",
            "reason": "Notification not required"
        }
        
        reasoning = ["Received system notification", f"Type: {notification.get('type', 'unknown')}"]
        
        # Check user preferences for notifications
        user_preferences = context.get("user_preferences", {})
        notification_preferences = user_preferences.get("notification_preferences", {})
        
        # Process based on notification type
        notification_type = notification.get("type", "unknown")
        
        if notification_type == "trade_execution" and notification_preferences.get("trade_execution", True):
            notification_content = {
                "send_notification": True,
                "notification_type": "trade",
                "message": "A trade has been executed in your ALL-USE account.",
                "details": notification.get("details", {}),
                "reason": "User has enabled trade execution notifications"
            }
            
            reasoning.append("User has enabled trade execution notifications")
            reasoning.append("Sending trade execution notification")
        
        elif notification_type == "account_update" and notification_preferences.get("account_updates", True):
            notification_content = {
                "send_notification": True,
                "notification_type": "account",
                "message": "Your ALL-USE account has been updated.",
                "details": notification.get("details", {}),
                "reason": "User has enabled account update notifications"
            }
            
            reasoning.append("User has enabled account update notifications")
            reasoning.append("Sending account update notification")
        
        elif notification_type == "weekly_summary" and notification_preferences.get("weekly_summary", True):
            notification_content = {
                "send_notification": True,
                "notification_type": "summary",
                "message": "Your weekly ALL-USE summary is ready.",
                "details": notification.get("details", {}),
                "reason": "User has enabled weekly summary notifications"
            }
            
            reasoning.append("User has enabled weekly summary notifications")
            reasoning.append("Sending weekly summary notification")
        
        return Decision(
            decision_type=DecisionType.NOTIFICATION,
            content=notification_content,
            confidence=0.9,
            reasoning=reasoning
        )
    
    def _make_scheduled_event_decision(self, context: Dict[str, Any], event: PerceptionEvent) -> Decision:
        """Make a decision about scheduled events."""
        scheduled_event = event.processed_content
        
        # Default scheduled event decision content
        scheduled_event_content = {
            "action": "no_action",
            "reason": "No action required for this scheduled event"
        }
        
        reasoning = ["Received scheduled event", f"Event name: {scheduled_event.get('event_name', 'unknown')}"]
        
        # Process based on event name
        event_name = scheduled_event.get("event_name", "unknown")
        
        if event_name == "weekly_analysis":
            scheduled_event_content = {
                "action": "perform_weekly_analysis",
                "reason": "Weekly analysis scheduled event triggered"
            }
            
            reasoning.append("Weekly analysis scheduled event triggered")
            reasoning.append("Performing weekly analysis")
        
        elif event_name == "quarterly_reinvestment":
            scheduled_event_content = {
                "action": "check_reinvestment_opportunities",
                "reason": "Quarterly reinvestment scheduled event triggered"
            }
            
            reasoning.append("Quarterly reinvestment scheduled event triggered")
            reasoning.append("Checking for reinvestment opportunities")
        
        return Decision(
            decision_type=DecisionType.ANALYSIS,
            content=scheduled_event_content,
            confidence=0.9,
            reasoning=reasoning
        )

class CognitiveController:
    """Coordinates the cognitive processing pipeline."""
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        context_manager: Optional[ContextManager] = None,
        protocol_applicator: Optional[ProtocolApplicator] = None,
        decision_engine: Optional[DecisionEngine] = None
    ):
        self.memory_manager = memory_manager
        self.context_manager = context_manager or ContextManager(memory_manager)
        self.protocol_applicator = protocol_applicator or ProtocolApplicator()
        self.decision_engine = decision_engine or DecisionEngine(self.protocol_applicator)
    
    def process_event(self, event: PerceptionEvent, user_id: str) -> Decision:
        """Process a perception event through the cognitive pipeline."""
        # Update memory with the event
        self.memory_manager.process_perception_event(event, user_id)
        
        # Update context based on the event and memory
        context = self.context_manager.update_context(event, user_id)
        
        # Make a decision based on the context and event
        decision = self.decision_engine.make_decision(context, event)
        
        return decision
