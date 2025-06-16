"""
ALL-USE Enhanced Agent Module

This module implements the enhanced ALL-USE agent that integrates all core components
including memory management, cognitive framework, and response generation.
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protocol_engine.all_use_parameters import ALLUSEParameters
from agent_core.memory_manager import MemoryManager
from agent_core.cognitive_framework import CognitiveFramework, ContextManager
from agent_core.response_generator import ResponseGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('all_use_enhanced_agent.log')
    ]
)

logger = logging.getLogger('all_use_enhanced_agent')

class EnhancedALLUSEAgent:
    """
    Enhanced ALL-USE agent that integrates all core components.
    
    This class implements the complete agent architecture with:
    - Perception-cognition-action loop
    - Comprehensive memory systems
    - Advanced cognitive framework
    - Personality-driven response generation
    """
    
    def __init__(self):
        """Initialize the enhanced ALL-USE agent."""
        self.parameters = ALLUSEParameters
        self.memory_manager = MemoryManager()
        self.cognitive_framework = CognitiveFramework()
        self.context_manager = ContextManager()
        self.response_generator = ResponseGenerator()
        
        # Agent state
        self.agent_id = f"alluse_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_start_time = datetime.now()
        self.total_interactions = 0
        
        logger.info(f"Enhanced ALL-USE Agent initialized with ID: {self.agent_id}")
    
    def process_message(self, user_input: str) -> str:
        """
        Process a user message through the complete perception-cognition-action loop.
        
        Args:
            user_input: The input text from the user
            
        Returns:
            String containing the agent's response
        """
        try:
            logger.info(f"Processing message: {user_input[:100]}...")
            self.total_interactions += 1
            
            # PERCEPTION: Process input and update context
            self.memory_manager.add_user_message(user_input)
            
            # Get current state for processing
            conversation_history = self.memory_manager.get_conversation_history()
            protocol_state = self.memory_manager.get_protocol_state()
            user_preferences = self.memory_manager.get_user_preferences()
            
            # Update context
            current_context = self.context_manager.update_context(
                conversation_history, protocol_state, user_preferences
            )
            
            # Process input through cognitive framework
            processed_input = self.cognitive_framework.process_input(
                user_input, current_context, protocol_state, user_preferences
            )
            
            # COGNITION: Make decision based on processed input
            decision = self.cognitive_framework.make_decision(processed_input, protocol_state)
            
            # Store decision in memory
            self.memory_manager.add_decision(decision)
            
            # ACTION: Generate response based on decision
            response = self.response_generator.generate_response(
                decision, protocol_state, current_context
            )
            
            # Store response in memory
            self.memory_manager.add_agent_message(response)
            
            # Execute any protocol actions if needed
            self._execute_protocol_actions(decision)
            
            logger.info(f"Message processed successfully. Response length: {len(response)}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            error_response = (
                "I apologize, but I encountered an error while processing your request. "
                "Please try rephrasing your question or contact support if the issue persists."
            )
            self.memory_manager.add_agent_message(error_response)
            return error_response
    
    def _execute_protocol_actions(self, decision: Dict[str, Any]) -> None:
        """
        Execute protocol actions based on the decision.
        
        Args:
            decision: The decision from the cognitive framework
        """
        try:
            if decision['action'] == 'setup' and decision['response_type'] == 'account_setup':
                self._execute_account_setup(decision['parameters'])
            
            elif decision['action'] == 'classify' and decision['response_type'] == 'week_classification':
                self._execute_week_classification(decision['parameters'])
            
            elif decision['action'] == 'fork' and decision['response_type'] == 'account_fork':
                self._execute_account_fork(decision['parameters'])
            
            elif decision['action'] == 'recommend' and decision['response_type'] == 'trade_recommendations':
                self._execute_trade_recommendations(decision['parameters'])
            
        except Exception as e:
            logger.error(f"Error executing protocol action: {str(e)}")
    
    def _execute_account_setup(self, parameters: Dict[str, Any]) -> None:
        """
        Execute account setup action.
        
        Args:
            parameters: Parameters for account setup
        """
        initial_investment = parameters.get('initial_investment', 0)
        
        if initial_investment > 0:
            # Initialize accounts in memory
            self.memory_manager.initialize_accounts(
                initial_investment,
                self.parameters.INITIAL_ALLOCATION['GEN_ACC'],
                self.parameters.INITIAL_ALLOCATION['REV_ACC'],
                self.parameters.INITIAL_ALLOCATION['COM_ACC'],
                self.parameters.CASH_BUFFER
            )
            
            logger.info(f"Account setup executed for ${initial_investment:,.2f}")
    
    def _execute_week_classification(self, parameters: Dict[str, Any]) -> None:
        """
        Execute week classification action.
        
        Args:
            parameters: Parameters for week classification
        """
        # In Phase 1, this is manual classification
        # Future phases will implement automated classification
        week_type = parameters.get('week_type')
        
        if week_type and week_type in ['Green', 'Red', 'Chop']:
            self.memory_manager.set_week_classification(week_type)
            logger.info(f"Week classification executed: {week_type}")
    
    def _execute_account_fork(self, parameters: Dict[str, Any]) -> None:
        """
        Execute account fork action.
        
        Args:
            parameters: Parameters for account fork
        """
        account_index = parameters.get('account_index', 0)
        fork_amount = parameters.get('fork_amount', 0)
        allocation = parameters.get('allocation', {})
        
        if fork_amount > 0:
            new_gen_acc_amount = fork_amount * allocation.get('new_gen_acc', 0.5)
            com_acc_amount = fork_amount * allocation.get('com_acc', 0.5)
            
            # Record fork in memory
            self.memory_manager.protocol_state_memory.record_fork(
                account_index, new_gen_acc_amount, com_acc_amount
            )
            
            logger.info(f"Account fork executed: ${new_gen_acc_amount:,.2f} to new Gen-Acc, ${com_acc_amount:,.2f} to Com-Acc")
    
    def _execute_trade_recommendations(self, parameters: Dict[str, Any]) -> None:
        """
        Execute trade recommendations action.
        
        Args:
            parameters: Parameters for trade recommendations
        """
        # In Phase 1, this logs the recommendation
        # Future phases will implement actual trade execution
        week_classification = parameters.get('week_classification')
        account_balances = parameters.get('account_balances', {})
        
        logger.info(f"Trade recommendations executed for {week_classification} week")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get the current status of the agent.
        
        Returns:
            Dict containing agent status information
        """
        protocol_state = self.memory_manager.get_protocol_state()
        conversation_history = self.memory_manager.get_conversation_history()
        
        return {
            'agent_id': self.agent_id,
            'session_start_time': self.session_start_time.isoformat(),
            'total_interactions': self.total_interactions,
            'conversation_length': len(conversation_history),
            'has_accounts': any(protocol_state.get('accounts', {}).get(acc_type, []) 
                              for acc_type in ['GEN_ACC', 'REV_ACC', 'COM_ACC']),
            'week_classified': protocol_state.get('week_classification') is not None,
            'total_balance': self.memory_manager.protocol_state_memory.get_total_balance(),
            'account_balances': self.memory_manager.protocol_state_memory.get_account_balances(),
            'last_decision': protocol_state.get('last_decision'),
            'decision_count': len(protocol_state.get('decision_history', []))
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a performance summary of the agent and protocol.
        
        Returns:
            Dict containing performance summary
        """
        protocol_state = self.memory_manager.get_protocol_state()
        account_balances = self.memory_manager.protocol_state_memory.get_account_balances()
        
        # Calculate totals
        gen_acc_total = sum(account_balances.get('GEN_ACC', [0]))
        rev_acc_total = sum(account_balances.get('REV_ACC', [0]))
        com_acc_total = sum(account_balances.get('COM_ACC', [0]))
        total_balance = gen_acc_total + rev_acc_total + com_acc_total
        
        # Count accounts
        gen_acc_count = len(account_balances.get('GEN_ACC', []))
        rev_acc_count = len(account_balances.get('REV_ACC', []))
        com_acc_count = len(account_balances.get('COM_ACC', []))
        
        # Count forks and merges
        fork_count = len(protocol_state.get('fork_history', []))
        merge_count = len(protocol_state.get('merge_history', []))
        
        return {
            'total_balance': total_balance,
            'account_totals': {
                'GEN_ACC': gen_acc_total,
                'REV_ACC': rev_acc_total,
                'COM_ACC': com_acc_total
            },
            'account_counts': {
                'GEN_ACC': gen_acc_count,
                'REV_ACC': rev_acc_count,
                'COM_ACC': com_acc_count
            },
            'fork_count': fork_count,
            'merge_count': merge_count,
            'week_classification': protocol_state.get('week_classification'),
            'decision_count': len(protocol_state.get('decision_history', []))
        }
    
    def reset_session(self) -> None:
        """Reset the agent session, clearing all memory."""
        self.memory_manager = MemoryManager()
        self.context_manager = ContextManager()
        self.session_start_time = datetime.now()
        self.total_interactions = 0
        
        logger.info("Agent session reset")
    
    def export_session_data(self) -> Dict[str, Any]:
        """
        Export all session data for analysis or backup.
        
        Returns:
            Dict containing all session data
        """
        return {
            'agent_id': self.agent_id,
            'session_start_time': self.session_start_time.isoformat(),
            'total_interactions': self.total_interactions,
            'conversation_history': self.memory_manager.get_conversation_history(),
            'protocol_state': self.memory_manager.get_protocol_state(),
            'user_preferences': self.memory_manager.get_user_preferences(),
            'agent_status': self.get_agent_status(),
            'performance_summary': self.get_performance_summary()
        }


class AgentInterface:
    """
    Interface for interacting with the ALL-USE agent.
    
    This class provides a simple interface for users to interact with the agent
    and access its capabilities.
    """
    
    def __init__(self):
        """Initialize the agent interface."""
        self.agent = EnhancedALLUSEAgent()
        logger.info("Agent interface initialized")
    
    def chat(self, message: str) -> str:
        """
        Send a message to the agent and get a response.
        
        Args:
            message: The message to send to the agent
            
        Returns:
            The agent's response
        """
        return self.agent.process_message(message)
    
    def status(self) -> Dict[str, Any]:
        """
        Get the current status of the agent.
        
        Returns:
            Dict containing agent status
        """
        return self.agent.get_agent_status()
    
    def performance(self) -> Dict[str, Any]:
        """
        Get the performance summary of the agent.
        
        Returns:
            Dict containing performance summary
        """
        return self.agent.get_performance_summary()
    
    def reset(self) -> None:
        """Reset the agent session."""
        self.agent.reset_session()
    
    def export(self) -> Dict[str, Any]:
        """
        Export all session data.
        
        Returns:
            Dict containing all session data
        """
        return self.agent.export_session_data()


# Example usage and testing
if __name__ == "__main__":
    # Create agent interface
    interface = AgentInterface()
    
    # Example conversation
    print("ALL-USE Agent Interface")
    print("=" * 50)
    
    # Test greeting
    response = interface.chat("Hello!")
    print(f"User: Hello!")
    print(f"Agent: {response}")
    print()
    
    # Test protocol explanation
    response = interface.chat("Can you explain the ALL-USE protocol?")
    print(f"User: Can you explain the ALL-USE protocol?")
    print(f"Agent: {response}")
    print()
    
    # Test account setup
    response = interface.chat("I want to set up accounts with $300,000")
    print(f"User: I want to set up accounts with $300,000")
    print(f"Agent: {response}")
    print()
    
    # Check status
    status = interface.status()
    print(f"Agent Status: {status}")
    print()
    
    # Check performance
    performance = interface.performance()
    print(f"Performance Summary: {performance}")

