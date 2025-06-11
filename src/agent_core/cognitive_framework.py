"""
ALL-USE Cognitive Framework Module

This module implements the cognitive framework for the ALL-USE agent,
providing decision-making capabilities based on the ALL-USE protocol.
"""

import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protocol_engine.all_use_parameters import ALLUSEParameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('all_use_cognitive.log')
    ]
)

logger = logging.getLogger('all_use_cognitive')

class CognitiveFramework:
    """
    Cognitive framework for the ALL-USE agent.
    
    This class implements the decision-making capabilities of the agent,
    applying the ALL-USE protocol to user inputs and context.
    """
    
    def __init__(self):
        """Initialize the cognitive framework."""
        self.parameters = ALLUSEParameters
        logger.info("Cognitive framework initialized")
    
    def process_input(self, user_input: str, conversation_context: Dict[str, Any], 
                     protocol_state: Dict[str, Any], user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user input and context to extract intent and entities.
        
        Args:
            user_input: The input text from the user
            conversation_context: The current conversation context
            protocol_state: The current protocol state
            user_preferences: The user's preferences
            
        Returns:
            Dict containing processed input information
        """
        logger.info(f"Processing input: {user_input[:50]}...")
        
        # Process input (to be expanded in future phases)
        processed_input = {
            'raw_input': user_input,
            'intent': self._detect_intent(user_input, conversation_context),
            'entities': self._extract_entities(user_input),
            'context': self._determine_context(conversation_context, protocol_state, user_preferences)
        }
        
        return processed_input
    
    def make_decision(self, processed_input: Dict[str, Any], protocol_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make decision based on processed input and protocol state.
        
        Args:
            processed_input: The processed input from process_input
            protocol_state: The current protocol state
            
        Returns:
            Dict containing decision information
        """
        logger.info(f"Making decision for intent: {processed_input['intent']}")
        
        # Decision logic based on intent (to be expanded in future phases)
        intent = processed_input['intent']
        context = processed_input['context']
        decision = {
            'intent': intent,
            'action': None,
            'response_type': None,
            'parameters': {},
            'rationale': ''
        }
        
        if intent == 'greeting':
            decision['action'] = 'respond'
            decision['response_type'] = 'greeting'
            decision['rationale'] = 'Responding to user greeting'
        
        elif intent == 'explain_protocol':
            decision['action'] = 'explain'
            decision['response_type'] = 'protocol_explanation'
            decision['parameters'] = {'detail_level': 'overview'}
            decision['rationale'] = 'User requested protocol explanation'
        
        elif intent == 'setup_accounts':
            decision['action'] = 'setup'
            decision['response_type'] = 'account_setup'
            decision['parameters'] = {
                'initial_investment': processed_input.get('entities', {}).get('amount', 0),
                'allocation': {
                    'GEN_ACC': self.parameters.INITIAL_ALLOCATION['GEN_ACC'],
                    'REV_ACC': self.parameters.INITIAL_ALLOCATION['REV_ACC'],
                    'COM_ACC': self.parameters.INITIAL_ALLOCATION['COM_ACC']
                },
                'cash_buffer': self.parameters.CASH_BUFFER
            }
            decision['rationale'] = 'User requested account setup'
        
        elif intent == 'classify_week':
            decision['action'] = 'classify'
            decision['response_type'] = 'week_classification'
            decision['parameters'] = {'market_data': processed_input.get('entities', {}).get('market_data', {})}
            decision['rationale'] = 'User requested week classification'
        
        elif intent == 'recommend_trades':
            if not context['has_accounts']:
                decision['action'] = 'clarify'
                decision['response_type'] = 'account_setup_required'
                decision['rationale'] = 'Account setup required before trade recommendations'
            elif not context['week_classified']:
                decision['action'] = 'clarify'
                decision['response_type'] = 'week_classification_required'
                decision['rationale'] = 'Week classification required before trade recommendations'
            else:
                decision['action'] = 'recommend'
                decision['response_type'] = 'trade_recommendations'
                decision['parameters'] = {
                    'week_classification': protocol_state['week_classification'],
                    'account_balances': protocol_state['accounts']
                }
                decision['rationale'] = 'Generating trade recommendations based on protocol'
        
        elif intent == 'check_performance':
            if not context['has_accounts']:
                decision['action'] = 'clarify'
                decision['response_type'] = 'account_setup_required'
                decision['rationale'] = 'Account setup required before checking performance'
            else:
                decision['action'] = 'report'
                decision['response_type'] = 'performance_report'
                decision['parameters'] = {'account_balances': protocol_state['accounts']}
                decision['rationale'] = 'Generating performance report'
        
        elif intent == 'fork_account':
            if not context['has_accounts']:
                decision['action'] = 'clarify'
                decision['response_type'] = 'account_setup_required'
                decision['rationale'] = 'Account setup required before forking accounts'
            else:
                # Check if any Gen-Acc has reached the fork threshold
                gen_accounts = protocol_state['accounts']['GEN_ACC']
                fork_candidates = []
                
                for i, balance in enumerate(gen_accounts):
                    if balance >= self.parameters.FORK_THRESHOLD:
                        fork_candidates.append((i, balance))
                
                if fork_candidates:
                    # Sort by balance descending
                    fork_candidates.sort(key=lambda x: x[1], reverse=True)
                    index, balance = fork_candidates[0]
                    
                    decision['action'] = 'fork'
                    decision['response_type'] = 'account_fork'
                    decision['parameters'] = {
                        'account_index': index,
                        'account_balance': balance,
                        'fork_amount': balance / 2,  # 50% of the balance
                        'allocation': {
                            'new_gen_acc': 0.5,  # 50% to new Gen-Acc
                            'com_acc': 0.5       # 50% to Com-Acc
                        }
                    }
                    decision['rationale'] = f'Gen-Acc {index} has reached fork threshold'
                else:
                    decision['action'] = 'clarify'
                    decision['response_type'] = 'fork_threshold_not_reached'
                    decision['rationale'] = 'No Gen-Acc has reached the fork threshold'
        
        else:
            decision['action'] = 'clarify'
            decision['response_type'] = 'clarification'
            decision['rationale'] = 'Intent not recognized, requesting clarification'
        
        return decision
    
    def _detect_intent(self, user_input: str, conversation_context: Dict[str, Any]) -> str:
        """
        Detect the intent of the user input.
        
        Args:
            user_input: The input text from the user
            conversation_context: The current conversation context
            
        Returns:
            String representing the detected intent
        """
        # Simple keyword-based intent detection (to be expanded in future phases)
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return 'greeting'
        
        elif any(word in user_input_lower for word in ['explain', 'tell me about', 'how does', 'what is']):
            if any(word in user_input_lower for word in ['protocol', 'all-use', 'system', 'strategy']):
                return 'explain_protocol'
        
        elif any(word in user_input_lower for word in ['setup', 'create', 'start', 'initialize']):
            if any(word in user_input_lower for word in ['account', 'accounts']):
                return 'setup_accounts'
        
        elif any(word in user_input_lower for word in ['classify', 'what kind of week', 'market condition']):
            return 'classify_week'
        
        elif any(word in user_input_lower for word in ['recommend', 'suggest', 'what should', 'trade']):
            return 'recommend_trades'
        
        elif any(word in user_input_lower for word in ['performance', 'how am i doing', 'results', 'progress']):
            return 'check_performance'
        
        elif any(word in user_input_lower for word in ['fork', 'split', 'create new account']):
            return 'fork_account'
        
        return 'unknown'
    
    def _extract_entities(self, user_input: str) -> Dict[str, Any]:
        """
        Extract entities from the user input.
        
        Args:
            user_input: The input text from the user
            
        Returns:
            Dict containing extracted entities
        """
        # Simple entity extraction (to be expanded in future phases)
        entities = {}
        
        # Extract monetary amounts
        import re
        amount_matches = re.findall(r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:k|K|thousand|million|m|M)?', user_input)
        
        if amount_matches:
            # Process the first match
            amount_str = amount_matches[0].replace(',', '')
            
            # Check for multipliers
            if 'k' in user_input.lower() or 'K' in user_input or 'thousand' in user_input.lower():
                amount = float(amount_str) * 1000
            elif 'm' in user_input.lower() or 'M' in user_input or 'million' in user_input.lower():
                amount = float(amount_str) * 1000000
            else:
                amount = float(amount_str)
            
            entities['amount'] = amount
        
        # Extract week classification
        week_type_matches = re.findall(r'\b(green|red|chop)\b', user_input.lower())
        if week_type_matches:
            entities['week_type'] = week_type_matches[0].capitalize()
        
        return entities
    
    def _determine_context(self, conversation_context: Dict[str, Any], 
                          protocol_state: Dict[str, Any], 
                          user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the current context based on conversation, protocol state, and user preferences.
        
        Args:
            conversation_context: The current conversation context
            protocol_state: The current protocol state
            user_preferences: The user's preferences
            
        Returns:
            Dict containing context information
        """
        # Context determination (to be expanded in future phases)
        context = {
            'conversation_length': conversation_context.get('conversation_length', 0),
            'has_accounts': any(protocol_state.get('accounts', {}).get(acc_type, []) 
                              for acc_type in ['GEN_ACC', 'REV_ACC', 'COM_ACC']),
            'week_classified': protocol_state.get('week_classification') is not None,
            'risk_tolerance': user_preferences.get('risk_tolerance'),
            'income_priority': user_preferences.get('income_priority')
        }
        
        return context


class ContextManager:
    """
    Context manager for the ALL-USE agent.
    
    This class manages the context of the current conversation and protocol state,
    providing awareness of the current situation for decision-making.
    """
    
    def __init__(self):
        """Initialize the context manager."""
        self.current_context = {
            'conversation_length': 0,
            'last_intent': None,
            'last_action': None,
            'has_accounts': False,
            'week_classified': False,
            'pending_decisions': []
        }
        logger.info("Context manager initialized")
    
    def update_context(self, conversation_history: List[Dict[str, Any]], 
                      protocol_state: Dict[str, Any], 
                      user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the current context based on conversation history, protocol state, and user preferences.
        
        Args:
            conversation_history: The conversation history
            protocol_state: The current protocol state
            user_preferences: The user's preferences
            
        Returns:
            Dict containing the updated context
        """
        # Update conversation context
        self.current_context['conversation_length'] = len(conversation_history)
        
        # Update last intent and action if available
        if protocol_state.get('last_decision'):
            self.current_context['last_intent'] = protocol_state['last_decision'].get('intent')
            self.current_context['last_action'] = protocol_state['last_decision'].get('action')
        
        # Update account and week classification status
        self.current_context['has_accounts'] = any(protocol_state.get('accounts', {}).get(acc_type, []) 
                                                for acc_type in ['GEN_ACC', 'REV_ACC', 'COM_ACC'])
        self.current_context['week_classified'] = protocol_state.get('week_classification') is not None
        
        # Update pending decisions
        self._update_pending_decisions(protocol_state)
        
        logger.info("Context updated")
        return self.current_context
    
    def _update_pending_decisions(self, protocol_state: Dict[str, Any]) -> None:
        """
        Update the list of pending decisions based on protocol state.
        
        Args:
            protocol_state: The current protocol state
        """
        pending_decisions = []
        
        # Check if accounts need to be set up
        if not self.current_context['has_accounts']:
            pending_decisions.append({
                'type': 'account_setup',
                'priority': 'high',
                'description': 'Set up ALL-USE account structure'
            })
        
        # Check if week needs to be classified
        elif not self.current_context['week_classified']:
            pending_decisions.append({
                'type': 'week_classification',
                'priority': 'high',
                'description': 'Classify current market week'
            })
        
        # Check if any Gen-Acc has reached fork threshold
        elif self.current_context['has_accounts']:
            gen_accounts = protocol_state.get('accounts', {}).get('GEN_ACC', [])
            
            for i, balance in enumerate(gen_accounts):
                if balance >= ALLUSEParameters.FORK_THRESHOLD:
                    pending_decisions.append({
                        'type': 'account_fork',
                        'priority': 'medium',
                        'description': f'Fork Gen-Acc {i} (${balance:,.2f})',
                        'account_index': i,
                        'account_balance': balance
                    })
        
        self.current_context['pending_decisions'] = pending_decisions
    
    def get_context(self) -> Dict[str, Any]:
        """
        Get the current context.
        
        Returns:
            Dict containing the current context
        """
        return self.current_context


class ErrorHandler:
    """
    Error handler for the ALL-USE agent.
    
    This class manages error handling and recovery mechanisms for the agent.
    """
    
    def __init__(self):
        """Initialize the error handler."""
        self.error_history = []
        logger.info("Error handler initialized")
    
    def handle_error(self, error_type: str, error_message: str, 
                    context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an error and determine recovery action.
        
        Args:
            error_type: The type of error
            error_message: The error message
            context: The current context
            
        Returns:
            Dict containing recovery action
        """
        logger.error(f"Error: {error_type} - {error_message}")
        
        # Record error
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': error_message,
            'context': context
        }
        
        self.error_history.append(error_record)
        
        # Determine recovery action based on error type
        recovery_action = {
            'action': 'clarify',
            'response_type': 'error_recovery',
            'parameters': {
                'error_type': error_type,
                'error_message': error_message
            },
            'rationale': f'Recovering from {error_type} error'
        }
        
        if error_type == 'intent_recognition':
            recovery_action['response_type'] = 'clarification'
            recovery_action['rationale'] = 'Failed to recognize intent, requesting clarification'
        
        elif error_type == 'entity_extraction':
            recovery_action['response_type'] = 'entity_clarification'
            recovery_action['rationale'] = 'Failed to extract entities, requesting clarification'
        
        elif error_type == 'protocol_application':
            recovery_action['response_type'] = 'protocol_clarification'
            recovery_action['rationale'] = 'Failed to apply protocol, requesting clarification'
        
        elif error_type == 'account_operation':
            recovery_action['response_type'] = 'account_clarification'
            recovery_action['rationale'] = 'Failed to perform account operation, requesting clarification'
        
        return recovery_action
    
    def get_error_history(self) -> List[Dict[str, Any]]:
        """
        Get the error history.
        
        Returns:
            List of error records
        """
        return self.error_history
    
    def clear_error_history(self) -> None:
        """Clear the error history."""
        self.error_history = []
        logger.info("Error history cleared")
