"""
ALL-USE Memory Manager Module

This module implements the memory systems for the ALL-USE agent,
providing conversation memory and protocol state tracking.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('all_use_memory.log')
    ]
)

logger = logging.getLogger('all_use_memory')

class ConversationMemory:
    """
    Memory system for storing and retrieving conversation history.
    
    This class manages the agent's memory of past interactions with the user,
    providing context for current decisions and responses.
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize the conversation memory system.
        
        Args:
            max_history: Maximum number of conversation turns to store
        """
        self.max_history = max_history
        self.history = []
        logger.info("Conversation memory system initialized")
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: The role of the message sender ('user' or 'agent')
            content: The content of the message
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        
        self.history.append(message)
        
        # Trim history if it exceeds max_history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        logger.info(f"Added {role} message to conversation memory")
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the conversation history.
        
        Args:
            limit: Optional limit on the number of messages to retrieve
            
        Returns:
            List of message dictionaries
        """
        if limit is None:
            return self.history
        return self.history[-limit:]
    
    def get_last_user_message(self) -> Optional[Dict[str, Any]]:
        """
        Get the last message from the user.
        
        Returns:
            The last user message or None if no user messages exist
        """
        for message in reversed(self.history):
            if message['role'] == 'user':
                return message
        return None
    
    def get_last_agent_message(self) -> Optional[Dict[str, Any]]:
        """
        Get the last message from the agent.
        
        Returns:
            The last agent message or None if no agent messages exist
        """
        for message in reversed(self.history):
            if message['role'] == 'agent':
                return message
        return None
    
    def search_history(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the conversation history for messages containing the query.
        
        Args:
            query: The search query
            
        Returns:
            List of matching message dictionaries
        """
        return [msg for msg in self.history if query.lower() in msg['content'].lower()]
    
    def clear(self) -> None:
        """Clear the conversation history."""
        self.history = []
        logger.info("Conversation memory cleared")


class ProtocolStateMemory:
    """
    Memory system for tracking the state of the ALL-USE protocol.
    
    This class manages the agent's memory of the current protocol state,
    including account structure, week classification, and decision history.
    """
    
    def __init__(self):
        """Initialize the protocol state memory system."""
        self.state = {
            'accounts': {
                'GEN_ACC': [],  # List of Gen-Acc balances
                'REV_ACC': [],  # List of Rev-Acc balances
                'COM_ACC': []   # List of Com-Acc balances
            },
            'week_classification': None,  # Current week classification (Green, Red, Chop)
            'last_decision': None,        # Last protocol decision
            'decision_history': [],       # History of protocol decisions
            'fork_history': [],           # History of account forks
            'merge_history': [],          # History of account merges
            'reinvestment_history': []    # History of reinvestments
        }
        logger.info("Protocol state memory system initialized")
    
    def initialize_accounts(self, initial_investment: float, gen_acc_pct: float, 
                           rev_acc_pct: float, com_acc_pct: float, 
                           cash_buffer: float = 0.05) -> None:
        """
        Initialize the account structure with an initial investment.
        
        Args:
            initial_investment: The initial investment amount
            gen_acc_pct: Percentage allocation for Gen-Acc
            rev_acc_pct: Percentage allocation for Rev-Acc
            com_acc_pct: Percentage allocation for Com-Acc
            cash_buffer: Cash buffer percentage (default: 5%)
        """
        # Calculate effective allocations after cash buffer
        gen_acc_effective = gen_acc_pct * (1 - cash_buffer)
        rev_acc_effective = rev_acc_pct * (1 - cash_buffer)
        com_acc_effective = com_acc_pct * (1 - cash_buffer)
        
        # Calculate account balances
        gen_acc_balance = initial_investment * gen_acc_effective
        rev_acc_balance = initial_investment * rev_acc_effective
        com_acc_balance = initial_investment * com_acc_effective
        
        # Initialize accounts
        self.state['accounts']['GEN_ACC'] = [gen_acc_balance]
        self.state['accounts']['REV_ACC'] = [rev_acc_balance]
        self.state['accounts']['COM_ACC'] = [com_acc_balance]
        
        logger.info(f"Initialized accounts with ${initial_investment:,.2f}")
    
    def set_week_classification(self, classification: str) -> None:
        """
        Set the current week classification.
        
        Args:
            classification: The week classification ('Green', 'Red', or 'Chop')
        """
        if classification not in ['Green', 'Red', 'Chop']:
            raise ValueError("Week classification must be 'Green', 'Red', or 'Chop'")
        
        self.state['week_classification'] = classification
        logger.info(f"Set week classification to {classification}")
    
    def add_decision(self, decision: Dict[str, Any]) -> None:
        """
        Add a protocol decision to the history.
        
        Args:
            decision: The protocol decision dictionary
        """
        decision_with_timestamp = {
            **decision,
            'timestamp': datetime.now().isoformat()
        }
        
        self.state['last_decision'] = decision_with_timestamp
        self.state['decision_history'].append(decision_with_timestamp)
        
        logger.info(f"Added decision: {decision['action']} - {decision['response_type']}")
    
    def record_fork(self, source_account_index: int, new_gen_acc_amount: float, 
                   com_acc_amount: float) -> None:
        """
        Record an account fork.
        
        Args:
            source_account_index: Index of the source Gen-Acc
            new_gen_acc_amount: Amount allocated to the new Gen-Acc
            com_acc_amount: Amount allocated to Com-Acc
        """
        fork_record = {
            'timestamp': datetime.now().isoformat(),
            'source_account_index': source_account_index,
            'new_gen_acc_amount': new_gen_acc_amount,
            'com_acc_amount': com_acc_amount
        }
        
        # Update account balances
        self.state['accounts']['GEN_ACC'][source_account_index] -= (new_gen_acc_amount + com_acc_amount)
        self.state['accounts']['GEN_ACC'].append(new_gen_acc_amount)
        self.state['accounts']['COM_ACC'][0] += com_acc_amount
        
        self.state['fork_history'].append(fork_record)
        
        logger.info(f"Recorded fork: ${new_gen_acc_amount:,.2f} to new Gen-Acc, ${com_acc_amount:,.2f} to Com-Acc")
    
    def record_merge(self, source_account_index: int, merge_amount: float) -> None:
        """
        Record an account merge.
        
        Args:
            source_account_index: Index of the source account being merged
            merge_amount: Amount being merged into Com-Acc
        """
        merge_record = {
            'timestamp': datetime.now().isoformat(),
            'source_account_index': source_account_index,
            'merge_amount': merge_amount
        }
        
        # Update account balances
        self.state['accounts']['GEN_ACC'].pop(source_account_index)
        self.state['accounts']['COM_ACC'][0] += merge_amount
        
        self.state['merge_history'].append(merge_record)
        
        logger.info(f"Recorded merge: ${merge_amount:,.2f} from Gen-Acc to Com-Acc")
    
    def record_reinvestment(self, account_type: str, account_index: int, 
                           contracts_amount: float, leaps_amount: float) -> None:
        """
        Record a reinvestment.
        
        Args:
            account_type: The account type ('GEN_ACC', 'REV_ACC', or 'COM_ACC')
            account_index: Index of the account being reinvested
            contracts_amount: Amount reinvested in contracts
            leaps_amount: Amount reinvested in LEAPS
        """
        reinvestment_record = {
            'timestamp': datetime.now().isoformat(),
            'account_type': account_type,
            'account_index': account_index,
            'contracts_amount': contracts_amount,
            'leaps_amount': leaps_amount
        }
        
        self.state['reinvestment_history'].append(reinvestment_record)
        
        logger.info(f"Recorded reinvestment for {account_type}: ${contracts_amount:,.2f} to contracts, ${leaps_amount:,.2f} to LEAPS")
    
    def get_account_balances(self) -> Dict[str, List[float]]:
        """
        Get the current account balances.
        
        Returns:
            Dictionary of account balances
        """
        return self.state['accounts']
    
    def get_total_balance(self) -> float:
        """
        Get the total balance across all accounts.
        
        Returns:
            Total balance
        """
        total = 0.0
        for account_type, balances in self.state['accounts'].items():
            total += sum(balances)
        return total
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the full protocol state.
        
        Returns:
            Dictionary containing the full protocol state
        """
        return self.state
    
    def reset(self) -> None:
        """Reset the protocol state."""
        self.__init__()
        logger.info("Protocol state reset")


class UserPreferencesMemory:
    """
    Memory system for tracking user preferences and settings.
    
    This class manages the agent's memory of user preferences,
    settings, and personalization information.
    """
    
    def __init__(self):
        """Initialize the user preferences memory system."""
        self.preferences = {
            'risk_tolerance': None,  # User's risk tolerance level
            'income_priority': None, # User's priority on income vs growth
            'communication_style': None, # User's preferred communication style
            'notification_preferences': {}, # User's notification preferences
            'custom_settings': {}     # Custom user settings
        }
        logger.info("User preferences memory system initialized")
    
    def set_preference(self, key: str, value: Any) -> None:
        """
        Set a user preference.
        
        Args:
            key: The preference key
            value: The preference value
        """
        if key in self.preferences:
            self.preferences[key] = value
        else:
            self.preferences['custom_settings'][key] = value
        
        logger.info(f"Set user preference: {key}")
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """
        Get a user preference.
        
        Args:
            key: The preference key
            default: Default value if preference doesn't exist
            
        Returns:
            The preference value or default
        """
        if key in self.preferences:
            return self.preferences[key]
        return self.preferences['custom_settings'].get(key, default)
    
    def get_all_preferences(self) -> Dict[str, Any]:
        """
        Get all user preferences.
        
        Returns:
            Dictionary of all user preferences
        """
        return self.preferences
    
    def clear(self) -> None:
        """Clear all user preferences."""
        self.__init__()
        logger.info("User preferences cleared")


class MemoryManager:
    """
    Memory manager for the ALL-USE agent.
    
    This class coordinates the different memory systems used by the agent.
    """
    
    def __init__(self):
        """Initialize the memory manager."""
        self.conversation_memory = ConversationMemory()
        self.protocol_state_memory = ProtocolStateMemory()
        self.user_preferences_memory = UserPreferencesMemory()
        logger.info("Memory manager initialized")
    
    def initialize_accounts(self, initial_investment: float, gen_acc_pct: float, 
                           rev_acc_pct: float, com_acc_pct: float, 
                           cash_buffer: float = 0.05) -> None:
        """
        Initialize the account structure with an initial investment.
        
        Args:
            initial_investment: The initial investment amount
            gen_acc_pct: Percentage allocation for Gen-Acc
            rev_acc_pct: Percentage allocation for Rev-Acc
            com_acc_pct: Percentage allocation for Com-Acc
            cash_buffer: Cash buffer percentage (default: 5%)
        """
        self.protocol_state_memory.initialize_accounts(
            initial_investment, gen_acc_pct, rev_acc_pct, com_acc_pct, cash_buffer
        )
    
    def add_user_message(self, content: str) -> None:
        """
        Add a user message to conversation memory.
        
        Args:
            content: The message content
        """
        self.conversation_memory.add_message('user', content)
    
    def add_agent_message(self, content: str) -> None:
        """
        Add an agent message to conversation memory.
        
        Args:
            content: The message content
        """
        self.conversation_memory.add_message('agent', content)
    
    def add_decision(self, decision: Dict[str, Any]) -> None:
        """
        Add a protocol decision to protocol state memory.
        
        Args:
            decision: The protocol decision dictionary
        """
        self.protocol_state_memory.add_decision(decision)
    
    def set_week_classification(self, classification: str) -> None:
        """
        Set the current week classification in protocol state memory.
        
        Args:
            classification: The week classification ('Green', 'Red', or 'Chop')
        """
        self.protocol_state_memory.set_week_classification(classification)
    
    def set_user_preference(self, key: str, value: Any) -> None:
        """
        Set a user preference in user preferences memory.
        
        Args:
            key: The preference key
            value: The preference value
        """
        self.user_preferences_memory.set_preference(key, value)
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the conversation history.
        
        Args:
            limit: Optional limit on the number of messages to retrieve
            
        Returns:
            List of message dictionaries
        """
        return self.conversation_memory.get_history(limit)
    
    def get_protocol_state(self) -> Dict[str, Any]:
        """
        Get the full protocol state.
        
        Returns:
            Dictionary containing the full protocol state
        """
        return self.protocol_state_memory.get_state()
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """
        Get all user preferences.
        
        Returns:
            Dictionary of all user preferences
        """
        return self.user_preferences_memory.get_all_preferences()
    
    def reset_all(self) -> None:
        """Reset all memory systems."""
        self.conversation_memory.clear()
        self.protocol_state_memory.reset()
        self.user_preferences_memory.clear()
        logger.info("All memory systems reset")
