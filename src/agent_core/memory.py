"""
ALL-USE Agent: Memory Module

This module handles the memory systems for the ALL-USE agent, providing storage and retrieval
capabilities for conversation history, protocol state, and user preferences.

Key components:
- ConversationMemory: Stores and retrieves conversation history
- ProtocolStateMemory: Maintains the state of ALL-USE protocol execution
- UserPreferencesMemory: Stores user preferences and settings
- MemoryManager: Coordinates between different memory systems
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import os
import uuid

from .perception import EventType, Intent, Entity, PerceptionEvent

class MemoryType(Enum):
    """Types of memory in the ALL-USE agent."""
    CONVERSATION = "conversation"
    PROTOCOL_STATE = "protocol_state"
    USER_PREFERENCES = "user_preferences"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"

class ConversationEntry:
    """Represents a single entry in the conversation memory."""
    def __init__(
        self,
        speaker: str,
        message: str,
        timestamp: datetime = None,
        intent: Optional[Intent] = None,
        entities: List[Entity] = None,
        metadata: Dict[str, Any] = None
    ):
        self.id = str(uuid.uuid4())
        self.speaker = speaker
        self.message = message
        self.timestamp = timestamp or datetime.now()
        self.intent = intent
        self.entities = entities or []
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the conversation entry to a dictionary."""
        return {
            "id": self.id,
            "speaker": self.speaker,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "intent": self.intent.value if self.intent else None,
            "entities": [
                {
                    "entity_type": e.entity_type,
                    "value": e.value,
                    "confidence": e.confidence
                } for e in self.entities
            ],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationEntry':
        """Create a conversation entry from a dictionary."""
        entry = cls(
            speaker=data["speaker"],
            message=data["message"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )
        entry.id = data["id"]
        
        # Reconstruct intent if available
        if data.get("intent"):
            try:
                entry.intent = Intent(data["intent"])
            except ValueError:
                entry.intent = None
        
        # Reconstruct entities if available
        if data.get("entities"):
            entry.entities = [
                Entity(
                    entity_type=e["entity_type"],
                    value=e["value"],
                    confidence=e["confidence"]
                ) for e in data["entities"]
            ]
        
        return entry

class AccountState:
    """Represents the state of an account in the ALL-USE system."""
    def __init__(
        self,
        account_id: str,
        account_type: str,
        initial_balance: float,
        current_balance: float,
        pending_reinvestment: float = 0.0,
        is_parent: bool = True,
        parent_id: Optional[str] = None,
        creation_date: datetime = None,
        last_updated: datetime = None
    ):
        self.account_id = account_id
        self.account_type = account_type
        self.initial_balance = initial_balance
        self.current_balance = current_balance
        self.pending_reinvestment = pending_reinvestment
        self.is_parent = is_parent
        self.parent_id = parent_id
        self.creation_date = creation_date or datetime.now()
        self.last_updated = last_updated or datetime.now()
        self.weekly_history = []
        self.trade_history = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the account state to a dictionary."""
        return {
            "account_id": self.account_id,
            "account_type": self.account_type,
            "initial_balance": self.initial_balance,
            "current_balance": self.current_balance,
            "pending_reinvestment": self.pending_reinvestment,
            "is_parent": self.is_parent,
            "parent_id": self.parent_id,
            "creation_date": self.creation_date.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "weekly_history": self.weekly_history,
            "trade_history": self.trade_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AccountState':
        """Create an account state from a dictionary."""
        account = cls(
            account_id=data["account_id"],
            account_type=data["account_type"],
            initial_balance=data["initial_balance"],
            current_balance=data["current_balance"],
            pending_reinvestment=data.get("pending_reinvestment", 0.0),
            is_parent=data.get("is_parent", True),
            parent_id=data.get("parent_id"),
            creation_date=datetime.fromisoformat(data["creation_date"]),
            last_updated=datetime.fromisoformat(data["last_updated"])
        )
        account.weekly_history = data.get("weekly_history", [])
        account.trade_history = data.get("trade_history", [])
        return account

class ProtocolState:
    """Represents the state of the ALL-USE protocol execution."""
    def __init__(
        self,
        user_id: str,
        initial_investment: float,
        start_date: datetime = None,
        last_updated: datetime = None
    ):
        self.user_id = user_id
        self.initial_investment = initial_investment
        self.start_date = start_date or datetime.now()
        self.last_updated = last_updated or datetime.now()
        self.accounts = {}  # Dict[account_id, AccountState]
        self.current_week = 0
        self.current_year = 0
        self.fork_events = []
        self.merge_events = []
        self.reinvestment_events = []
    
    def add_account(self, account: AccountState) -> None:
        """Add an account to the protocol state."""
        self.accounts[account.account_id] = account
        self.last_updated = datetime.now()
    
    def update_account(self, account_id: str, updates: Dict[str, Any]) -> None:
        """Update an account in the protocol state."""
        if account_id not in self.accounts:
            raise ValueError(f"Account {account_id} not found")
        
        account = self.accounts[account_id]
        for key, value in updates.items():
            if hasattr(account, key):
                setattr(account, key, value)
        
        account.last_updated = datetime.now()
        self.last_updated = datetime.now()
    
    def get_account(self, account_id: str) -> AccountState:
        """Get an account from the protocol state."""
        if account_id not in self.accounts:
            raise ValueError(f"Account {account_id} not found")
        return self.accounts[account_id]
    
    def get_all_accounts(self) -> List[AccountState]:
        """Get all accounts in the protocol state."""
        return list(self.accounts.values())
    
    def get_parent_accounts(self) -> List[AccountState]:
        """Get all parent accounts in the protocol state."""
        return [account for account in self.accounts.values() if account.is_parent]
    
    def get_forked_accounts(self) -> List[AccountState]:
        """Get all forked accounts in the protocol state."""
        return [account for account in self.accounts.values() if not account.is_parent]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the protocol state to a dictionary."""
        return {
            "user_id": self.user_id,
            "initial_investment": self.initial_investment,
            "start_date": self.start_date.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "accounts": {
                account_id: account.to_dict() 
                for account_id, account in self.accounts.items()
            },
            "current_week": self.current_week,
            "current_year": self.current_year,
            "fork_events": self.fork_events,
            "merge_events": self.merge_events,
            "reinvestment_events": self.reinvestment_events
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProtocolState':
        """Create a protocol state from a dictionary."""
        protocol_state = cls(
            user_id=data["user_id"],
            initial_investment=data["initial_investment"],
            start_date=datetime.fromisoformat(data["start_date"]),
            last_updated=datetime.fromisoformat(data["last_updated"])
        )
        
        # Reconstruct accounts
        for account_id, account_data in data.get("accounts", {}).items():
            protocol_state.accounts[account_id] = AccountState.from_dict(account_data)
        
        protocol_state.current_week = data.get("current_week", 0)
        protocol_state.current_year = data.get("current_year", 0)
        protocol_state.fork_events = data.get("fork_events", [])
        protocol_state.merge_events = data.get("merge_events", [])
        protocol_state.reinvestment_events = data.get("reinvestment_events", [])
        
        return protocol_state

class UserPreferences:
    """Represents user preferences and settings."""
    def __init__(
        self,
        user_id: str,
        risk_tolerance: str = "moderate",
        communication_frequency: str = "weekly",
        notification_preferences: Dict[str, bool] = None,
        display_preferences: Dict[str, Any] = None,
        created_at: datetime = None,
        last_updated: datetime = None
    ):
        self.user_id = user_id
        self.risk_tolerance = risk_tolerance
        self.communication_frequency = communication_frequency
        self.notification_preferences = notification_preferences or {
            "trade_execution": True,
            "account_updates": True,
            "weekly_summary": True,
            "monthly_summary": True,
            "quarterly_summary": True,
            "annual_summary": True
        }
        self.display_preferences = display_preferences or {
            "currency": "USD",
            "date_format": "MM/DD/YYYY",
            "theme": "light"
        }
        self.created_at = created_at or datetime.now()
        self.last_updated = last_updated or datetime.now()
    
    def update_preferences(self, updates: Dict[str, Any]) -> None:
        """Update user preferences."""
        for key, value in updates.items():
            if key == "notification_preferences" and isinstance(value, dict):
                self.notification_preferences.update(value)
            elif key == "display_preferences" and isinstance(value, dict):
                self.display_preferences.update(value)
            elif hasattr(self, key):
                setattr(self, key, value)
        
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user preferences to a dictionary."""
        return {
            "user_id": self.user_id,
            "risk_tolerance": self.risk_tolerance,
            "communication_frequency": self.communication_frequency,
            "notification_preferences": self.notification_preferences,
            "display_preferences": self.display_preferences,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPreferences':
        """Create user preferences from a dictionary."""
        return cls(
            user_id=data["user_id"],
            risk_tolerance=data.get("risk_tolerance", "moderate"),
            communication_frequency=data.get("communication_frequency", "weekly"),
            notification_preferences=data.get("notification_preferences", {}),
            display_preferences=data.get("display_preferences", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"])
        )

class ConversationMemory:
    """Stores and retrieves conversation history."""
    
    def __init__(self, max_entries: int = 100):
        self.entries = []
        self.max_entries = max_entries
    
    def add_entry(self, entry: ConversationEntry) -> None:
        """Add a conversation entry to memory."""
        self.entries.append(entry)
        
        # Trim if exceeding max entries
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
    
    def get_entries(self, limit: int = None, speaker: str = None) -> List[ConversationEntry]:
        """Get conversation entries, optionally filtered by speaker."""
        filtered = self.entries
        if speaker:
            filtered = [e for e in filtered if e.speaker == speaker]
        
        if limit:
            return filtered[-limit:]
        return filtered
    
    def get_last_entry(self, speaker: str = None) -> Optional[ConversationEntry]:
        """Get the last conversation entry, optionally filtered by speaker."""
        entries = self.get_entries(speaker=speaker)
        if entries:
            return entries[-1]
        return None
    
    def search_by_intent(self, intent: Intent) -> List[ConversationEntry]:
        """Search for conversation entries by intent."""
        return [e for e in self.entries if e.intent == intent]
    
    def search_by_entity(self, entity_type: str) -> List[ConversationEntry]:
        """Search for conversation entries containing a specific entity type."""
        return [
            e for e in self.entries 
            if any(entity.entity_type == entity_type for entity in e.entities)
        ]
    
    def clear(self) -> None:
        """Clear all conversation entries."""
        self.entries = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation memory to a dictionary."""
        return {
            "max_entries": self.max_entries,
            "entries": [entry.to_dict() for entry in self.entries]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMemory':
        """Create conversation memory from a dictionary."""
        memory = cls(max_entries=data.get("max_entries", 100))
        memory.entries = [
            ConversationEntry.from_dict(entry_data) 
            for entry_data in data.get("entries", [])
        ]
        return memory

class ProtocolStateMemory:
    """Maintains the state of ALL-USE protocol execution."""
    
    def __init__(self):
        self.protocol_states = {}  # Dict[user_id, ProtocolState]
    
    def create_protocol_state(self, user_id: str, initial_investment: float) -> ProtocolState:
        """Create a new protocol state for a user."""
        protocol_state = ProtocolState(user_id, initial_investment)
        self.protocol_states[user_id] = protocol_state
        
        # Initialize with default accounts
        gen_acc = AccountState(
            account_id=f"{user_id}_gen_acc",
            account_type="Gen-Acc",
            initial_balance=initial_investment * 0.4,
            current_balance=initial_investment * 0.4
        )
        rev_acc = AccountState(
            account_id=f"{user_id}_rev_acc",
            account_type="Rev-Acc",
            initial_balance=initial_investment * 0.3,
            current_balance=initial_investment * 0.3
        )
        com_acc = AccountState(
            account_id=f"{user_id}_com_acc",
            account_type="Com-Acc",
            initial_balance=initial_investment * 0.3,
            current_balance=initial_investment * 0.3
        )
        
        protocol_state.add_account(gen_acc)
        protocol_state.add_account(rev_acc)
        protocol_state.add_account(com_acc)
        
        return protocol_state
    
    def get_protocol_state(self, user_id: str) -> Optional[ProtocolState]:
        """Get the protocol state for a user."""
        return self.protocol_states.get(user_id)
    
    def update_protocol_state(self, user_id: str, updates: Dict[str, Any]) -> None:
        """Update the protocol state for a user."""
        if user_id not in self.protocol_states:
            raise ValueError(f"Protocol state for user {user_id} not found")
        
        protocol_state = self.protocol_states[user_id]
        for key, value in updates.items():
            if hasattr(protocol_state, key):
                setattr(protocol_state, key, value)
        
        protocol_state.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert protocol state memory to a dictionary."""
        return {
            "protocol_states": {
                user_id: protocol_state.to_dict() 
                for user_id, protocol_state in self.protocol_states.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProtocolStateMemory':
        """Create protocol state memory from a dictionary."""
        memory = cls()
        for user_id, protocol_state_data in data.get("protocol_states", {}).items():
            memory.protocol_states[user_id] = ProtocolState.from_dict(protocol_state_data)
        return memory

class UserPreferencesMemory:
    """Stores user preferences and settings."""
    
    def __init__(self):
        self.preferences = {}  # Dict[user_id, UserPreferences]
    
    def create_user_preferences(self, user_id: str) -> UserPreferences:
        """Create default preferences for a user."""
        preferences = UserPreferences(user_id)
        self.preferences[user_id] = preferences
        return preferences
    
    def get_user_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """Get preferences for a user."""
        return self.preferences.get(user_id)
    
    def update_user_preferences(self, user_id: str, updates: Dict[str, Any]) -> None:
        """Update preferences for a user."""
        if user_id not in self.preferences:
            self.create_user_preferences(user_id)
        
        self.preferences[user_id].update_preferences(updates)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user preferences memory to a dictionary."""
        return {
            "preferences": {
                user_id: preferences.to_dict() 
                for user_id, preferences in self.preferences.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPreferencesMemory':
        """Create user preferences memory from a dictionary."""
        memory = cls()
        for user_id, preferences_data in data.get("preferences", {}).items():
            memory.preferences[user_id] = UserPreferences.from_dict(preferences_data)
        return memory

class MemoryManager:
    """Coordinates between different memory systems."""
    
    def __init__(
        self,
        conversation_memory: ConversationMemory = None,
        protocol_state_memory: ProtocolStateMemory = None,
        user_preferences_memory: UserPreferencesMemory = None,
        persistence_dir: str = None
    ):
        self.conversation_memory = conversation_memory or ConversationMemory()
        self.protocol_state_memory = protocol_state_memory or ProtocolStateMemory()
        self.user_preferences_memory = user_preferences_memory or UserPreferencesMemory()
        self.persistence_dir = persistence_dir
        
        # Create persistence directory if it doesn't exist
        if self.persistence_dir and not os.path.exists(self.persistence_dir):
            os.makedirs(self.persistence_dir)
    
    def process_perception_event(self, event: PerceptionEvent, user_id: str) -> None:
        """Process a perception event and update memory accordingly."""
        if event.event_type == EventType.USER_MESSAGE:
            # Add to conversation memory
            entry = ConversationEntry(
                speaker="user",
                message=event.raw_content,
                timestamp=event.timestamp,
                intent=event.intent,
                entities=event.entities
            )
            self.conversation_memory.add_entry(entry)
        
        elif event.event_type == EventType.ACCOUNT_UPDATE:
            # Update protocol state
            protocol_state = self.protocol_state_memory.get_protocol_state(user_id)
            if protocol_state:
                for account_id, updates in event.processed_content.get("accounts", {}).items():
                    if account_id in protocol_state.accounts:
                        protocol_state.update_account(account_id, updates)
    
    def add_agent_message(self, message: str, user_id: str, metadata: Dict[str, Any] = None) -> None:
        """Add an agent message to conversation memory."""
        entry = ConversationEntry(
            speaker="agent",
            message=message,
            metadata=metadata
        )
        self.conversation_memory.add_entry(entry)
    
    def get_conversation_context(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation context for cognitive processing."""
        entries = self.conversation_memory.get_entries(limit=limit)
        return [
            {
                "speaker": entry.speaker,
                "message": entry.message,
                "timestamp": entry.timestamp.isoformat(),
                "intent": entry.intent.value if entry.intent else None
            } for entry in entries
        ]
    
    def get_protocol_context(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get protocol state context for cognitive processing."""
        protocol_state = self.protocol_state_memory.get_protocol_state(user_id)
        if not protocol_state:
            return None
        
        # Create a simplified context with key information
        return {
            "initial_investment": protocol_state.initial_investment,
            "current_week": protocol_state.current_week,
            "current_year": protocol_state.current_year,
            "accounts": {
                account.account_type: {
                    "initial_balance": account.initial_balance,
                    "current_balance": account.current_balance,
                    "pending_reinvestment": account.pending_reinvestment
                } for account in protocol_state.get_parent_accounts()
            },
            "forked_accounts": len(protocol_state.get_forked_accounts()),
            "total_value": sum(account.current_balance for account in protocol_state.get_all_accounts())
        }
    
    def get_user_context(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user preferences context for cognitive processing."""
        preferences = self.user_preferences_memory.get_user_preferences(user_id)
        if not preferences:
            return None
        
        return {
            "risk_tolerance": preferences.risk_tolerance,
            "communication_frequency": preferences.communication_frequency,
            "notification_preferences": preferences.notification_preferences,
            "display_preferences": preferences.display_preferences
        }
    
    def save_to_disk(self) -> None:
        """Save all memory to disk."""
        if not self.persistence_dir:
            return
        
        # Save conversation memory
        conversation_path = os.path.join(self.persistence_dir, "conversation_memory.json")
        with open(conversation_path, "w") as f:
            json.dump(self.conversation_memory.to_dict(), f, indent=2)
        
        # Save protocol state memory
        protocol_path = os.path.join(self.persistence_dir, "protocol_state_memory.json")
        with open(protocol_path, "w") as f:
            json.dump(self.protocol_state_memory.to_dict(), f, indent=2)
        
        # Save user preferences memory
        preferences_path = os.path.join(self.persistence_dir, "user_preferences_memory.json")
        with open(preferences_path, "w") as f:
            json.dump(self.user_preferences_memory.to_dict(), f, indent=2)
    
    def load_from_disk(self) -> bool:
        """Load all memory from disk. Returns True if successful."""
        if not self.persistence_dir:
            return False
        
        try:
            # Load conversation memory
            conversation_path = os.path.join(self.persistence_dir, "conversation_memory.json")
            if os.path.exists(conversation_path):
                with open(conversation_path, "r") as f:
                    self.conversation_memory = ConversationMemory.from_dict(json.load(f))
            
            # Load protocol state memory
            protocol_path = os.path.join(self.persistence_dir, "protocol_state_memory.json")
            if os.path.exists(protocol_path):
                with open(protocol_path, "r") as f:
                    self.protocol_state_memory = ProtocolStateMemory.from_dict(json.load(f))
            
            # Load user preferences memory
            preferences_path = os.path.join(self.persistence_dir, "user_preferences_memory.json")
            if os.path.exists(preferences_path):
                with open(preferences_path, "r") as f:
                    self.user_preferences_memory = UserPreferencesMemory.from_dict(json.load(f))
            
            return True
        except Exception as e:
            print(f"Error loading memory from disk: {e}")
            return False
