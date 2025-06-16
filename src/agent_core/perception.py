"""
ALL-USE Agent: Perception Module

This module handles the perception component of the agent's perception-cognition-action loop.
It processes incoming events, extracts relevant information, and prepares it for cognitive processing.

Key components:
- EventProcessor: Processes different types of events (messages, market data, etc.)
- EntityExtractor: Extracts entities from text (account names, amounts, dates, etc.)
- IntentDetector: Detects user intents from messages
"""

import re
import json
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union

class EventType(Enum):
    """Types of events that the agent can perceive."""
    USER_MESSAGE = "user_message"
    MARKET_DATA = "market_data"
    ACCOUNT_UPDATE = "account_update"
    SYSTEM_NOTIFICATION = "system_notification"
    SCHEDULED_EVENT = "scheduled_event"

class Intent(Enum):
    """Possible user intents that can be detected from messages."""
    ACCOUNT_INQUIRY = "account_inquiry"
    PERFORMANCE_INQUIRY = "performance_inquiry"
    TRADE_REQUEST = "trade_request"
    STRATEGY_INQUIRY = "strategy_inquiry"
    PROJECTION_REQUEST = "projection_request"
    GENERAL_QUESTION = "general_question"
    SETTINGS_UPDATE = "settings_update"
    HELP_REQUEST = "help_request"
    GREETING = "greeting"
    UNKNOWN = "unknown"

class Entity:
    """Represents an entity extracted from text."""
    def __init__(self, entity_type: str, value: Any, confidence: float = 1.0):
        self.entity_type = entity_type
        self.value = value
        self.confidence = confidence
    
    def __repr__(self):
        return f"Entity({self.entity_type}, {self.value}, {self.confidence})"

class PerceptionEvent:
    """Represents a processed event ready for cognitive processing."""
    def __init__(
        self, 
        event_type: EventType, 
        raw_content: Any,
        processed_content: Dict[str, Any],
        entities: List[Entity] = None,
        intent: Intent = None,
        timestamp: datetime = None,
        confidence: float = 1.0
    ):
        self.event_type = event_type
        self.raw_content = raw_content
        self.processed_content = processed_content
        self.entities = entities or []
        self.intent = intent
        self.timestamp = timestamp or datetime.now()
        self.confidence = confidence
    
    def __repr__(self):
        return f"PerceptionEvent({self.event_type}, intent={self.intent}, entities={len(self.entities)})"

class EntityExtractor:
    """Extracts entities from text."""
    
    def __init__(self):
        # Regular expressions for entity extraction
        self.patterns = {
            "account": r"(gen[-\s]?acc|rev[-\s]?acc|com[-\s]?acc)",
            "amount": r"\$?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)[k|K|m|M]?",
            "percentage": r"(\d+(?:\.\d{1,2})?\s*%)",
            "date": r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            "ticker": r"\b([A-Z]{1,5})\b"
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {
            entity_type: re.compile(pattern, re.IGNORECASE) 
            for entity_type, pattern in self.patterns.items()
        }
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text using regex patterns."""
        entities = []
        
        for entity_type, pattern in self.compiled_patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]  # Take the first group if multiple groups
                
                # Process specific entity types
                if entity_type == "amount":
                    # Convert k/m suffixes to actual numbers
                    clean_match = match.replace(",", "").replace("$", "")
                    if clean_match.lower().endswith('k'):
                        value = float(clean_match[:-1]) * 1000
                    elif clean_match.lower().endswith('m'):
                        value = float(clean_match[:-1]) * 1000000
                    else:
                        value = float(clean_match)
                    entities.append(Entity(entity_type, value, 0.9))
                
                elif entity_type == "percentage":
                    # Extract numeric value from percentage
                    value = float(match.replace("%", "").strip()) / 100
                    entities.append(Entity(entity_type, value, 0.9))
                
                elif entity_type == "account":
                    # Normalize account names
                    if "gen" in match.lower():
                        entities.append(Entity(entity_type, "Gen-Acc", 0.95))
                    elif "rev" in match.lower():
                        entities.append(Entity(entity_type, "Rev-Acc", 0.95))
                    elif "com" in match.lower():
                        entities.append(Entity(entity_type, "Com-Acc", 0.95))
                
                else:
                    entities.append(Entity(entity_type, match, 0.8))
        
        return entities

class IntentDetector:
    """Detects user intents from messages."""
    
    def __init__(self):
        # Keywords associated with different intents
        self.intent_keywords = {
            Intent.ACCOUNT_INQUIRY: [
                "balance", "account", "how much", "gen-acc", "rev-acc", "com-acc", 
                "how are my accounts", "account status"
            ],
            Intent.PERFORMANCE_INQUIRY: [
                "performance", "return", "how did", "profit", "loss", "gain",
                "how am i doing", "results", "weekly results"
            ],
            Intent.TRADE_REQUEST: [
                "trade", "buy", "sell", "enter", "exit", "position", "option",
                "put", "call", "roll", "adjust"
            ],
            Intent.STRATEGY_INQUIRY: [
                "strategy", "approach", "protocol", "why did you", "explain",
                "how does", "methodology", "delta", "weekly scenario"
            ],
            Intent.PROJECTION_REQUEST: [
                "project", "forecast", "predict", "future", "estimate",
                "what if", "scenario", "10-year", "growth"
            ],
            Intent.SETTINGS_UPDATE: [
                "change", "update", "modify", "setting", "preference",
                "risk", "allocation", "threshold", "fork", "merge"
            ],
            Intent.HELP_REQUEST: [
                "help", "guide", "assist", "support", "how to", "explain",
                "what can you do", "capabilities"
            ],
            Intent.GREETING: [
                "hello", "hi", "hey", "good morning", "good afternoon",
                "good evening", "greetings", "howdy"
            ]
        }
    
    def detect_intent(self, text: str) -> Tuple[Intent, float]:
        """Detect the most likely intent from text."""
        text = text.lower()
        
        # Count keyword matches for each intent
        intent_scores = {intent: 0 for intent in Intent}
        
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    intent_scores[intent] += 1
        
        # Find the intent with the highest score
        max_score = max(intent_scores.values())
        if max_score == 0:
            return Intent.UNKNOWN, 0.5
        
        # If multiple intents have the same score, choose the first one
        for intent, score in intent_scores.items():
            if score == max_score:
                # Calculate confidence based on score and text length
                confidence = min(0.5 + (score / 5), 0.95)
                return intent, confidence
        
        return Intent.UNKNOWN, 0.5

class EventProcessor:
    """Processes different types of events."""
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.intent_detector = IntentDetector()
    
    def process_event(self, event_type: EventType, content: Any) -> PerceptionEvent:
        """Process an event based on its type."""
        if event_type == EventType.USER_MESSAGE:
            return self._process_user_message(content)
        elif event_type == EventType.MARKET_DATA:
            return self._process_market_data(content)
        elif event_type == EventType.ACCOUNT_UPDATE:
            return self._process_account_update(content)
        elif event_type == EventType.SYSTEM_NOTIFICATION:
            return self._process_system_notification(content)
        elif event_type == EventType.SCHEDULED_EVENT:
            return self._process_scheduled_event(content)
        else:
            raise ValueError(f"Unknown event type: {event_type}")
    
    def _process_user_message(self, message: str) -> PerceptionEvent:
        """Process a user message event."""
        entities = self.entity_extractor.extract_entities(message)
        intent, confidence = self.intent_detector.detect_intent(message)
        
        processed_content = {
            "text": message,
            "word_count": len(message.split()),
            "processed_at": datetime.now().isoformat()
        }
        
        return PerceptionEvent(
            event_type=EventType.USER_MESSAGE,
            raw_content=message,
            processed_content=processed_content,
            entities=entities,
            intent=intent,
            confidence=confidence
        )
    
    def _process_market_data(self, data: Dict[str, Any]) -> PerceptionEvent:
        """Process market data event."""
        processed_content = {
            "tickers": list(data.keys()),
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "processed_at": datetime.now().isoformat()
        }
        
        return PerceptionEvent(
            event_type=EventType.MARKET_DATA,
            raw_content=data,
            processed_content=processed_content
        )
    
    def _process_account_update(self, update: Dict[str, Any]) -> PerceptionEvent:
        """Process account update event."""
        processed_content = {
            "accounts": update.get("accounts", []),
            "total_value": update.get("total_value", 0),
            "timestamp": update.get("timestamp", datetime.now().isoformat()),
            "processed_at": datetime.now().isoformat()
        }
        
        return PerceptionEvent(
            event_type=EventType.ACCOUNT_UPDATE,
            raw_content=update,
            processed_content=processed_content
        )
    
    def _process_system_notification(self, notification: Dict[str, Any]) -> PerceptionEvent:
        """Process system notification event."""
        processed_content = {
            "type": notification.get("type", "unknown"),
            "priority": notification.get("priority", "normal"),
            "timestamp": notification.get("timestamp", datetime.now().isoformat()),
            "processed_at": datetime.now().isoformat()
        }
        
        return PerceptionEvent(
            event_type=EventType.SYSTEM_NOTIFICATION,
            raw_content=notification,
            processed_content=processed_content
        )
    
    def _process_scheduled_event(self, event: Dict[str, Any]) -> PerceptionEvent:
        """Process scheduled event."""
        processed_content = {
            "event_name": event.get("name", "unknown"),
            "scheduled_time": event.get("scheduled_time", datetime.now().isoformat()),
            "processed_at": datetime.now().isoformat()
        }
        
        return PerceptionEvent(
            event_type=EventType.SCHEDULED_EVENT,
            raw_content=event,
            processed_content=processed_content
        )
