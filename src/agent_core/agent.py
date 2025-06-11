"""
ALL-USE Agent Core Module

This module implements the core agent architecture for the ALL-USE agent,
following the perception-cognition-action loop pattern.
"""

import sys
import os
import logging
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
        logging.FileHandler('all_use_agent.log')
    ]
)

logger = logging.getLogger('all_use_agent')

class ALLUSEAgent:
    """
    Core agent class implementing the ALL-USE protocol.
    
    This class follows the perception-cognition-action loop pattern:
    1. Perception: Process inputs from user and environment
    2. Cognition: Make decisions based on inputs and protocol
    3. Action: Execute decisions and generate outputs
    """
    
    def __init__(self):
        """Initialize the ALL-USE agent."""
        self.parameters = ALLUSEParameters
        self.conversation_memory = []
        self.protocol_state = {
            'accounts': {
                'GEN_ACC': [],
                'REV_ACC': [],
                'COM_ACC': []
            },
            'week_classification': None,
            'last_decision': None,
            'decision_history': []
        }
        self.user_preferences = {}
        logger.info("ALL-USE Agent initialized")
    
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input (perception).
        
        Args:
            user_input: The input text from the user
            
        Returns:
            Dict containing processed input information
        """
        logger.info(f"Processing input: {user_input[:50]}...")
        
        # Store input in conversation memory
        self.conversation_memory.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Process input (to be expanded in future phases)
        processed_input = {
            'raw_input': user_input,
            'intent': self._detect_intent(user_input),
            'entities': self._extract_entities(user_input),
            'context': self._determine_context()
        }
        
        return processed_input
    
    def make_decision(self, processed_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make decision based on processed input (cognition).
        
        Args:
            processed_input: The processed input from process_input
            
        Returns:
            Dict containing decision information
        """
        logger.info(f"Making decision for intent: {processed_input['intent']}")
        
        # Decision logic based on intent (to be expanded in future phases)
        intent = processed_input['intent']
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
                'allocation': self.parameters.INITIAL_ALLOCATION
            }
            decision['rationale'] = 'User requested account setup'
        
        elif intent == 'classify_week':
            decision['action'] = 'classify'
            decision['response_type'] = 'week_classification'
            decision['parameters'] = {'market_data': processed_input.get('entities', {}).get('market_data', {})}
            decision['rationale'] = 'User requested week classification'
        
        else:
            decision['action'] = 'clarify'
            decision['response_type'] = 'clarification'
            decision['rationale'] = 'Intent not recognized, requesting clarification'
        
        # Store decision in history
        self.protocol_state['last_decision'] = decision
        self.protocol_state['decision_history'].append(decision)
        
        return decision
    
    def generate_output(self, decision: Dict[str, Any]) -> str:
        """
        Generate output based on decision (action).
        
        Args:
            decision: The decision from make_decision
            
        Returns:
            String containing the agent's response
        """
        logger.info(f"Generating output for action: {decision['action']}")
        
        # Generate response based on decision (to be expanded in future phases)
        response = ""
        
        if decision['action'] == 'respond' and decision['response_type'] == 'greeting':
            response = self._generate_greeting()
        
        elif decision['action'] == 'explain' and decision['response_type'] == 'protocol_explanation':
            response = self._generate_protocol_explanation(decision['parameters']['detail_level'])
        
        elif decision['action'] == 'setup' and decision['response_type'] == 'account_setup':
            response = self._generate_account_setup(decision['parameters'])
        
        elif decision['action'] == 'classify' and decision['response_type'] == 'week_classification':
            response = self._generate_week_classification(decision['parameters'])
        
        elif decision['action'] == 'clarify':
            response = self._generate_clarification()
        
        # Store response in conversation memory
        self.conversation_memory.append({
            'role': 'agent',
            'content': response,
            'timestamp': datetime.now().isoformat()
        })
        
        return response
    
    def process_message(self, user_input: str) -> str:
        """
        Process a user message through the full perception-cognition-action loop.
        
        Args:
            user_input: The input text from the user
            
        Returns:
            String containing the agent's response
        """
        processed_input = self.process_input(user_input)
        decision = self.make_decision(processed_input)
        response = self.generate_output(decision)
        return response
    
    def _detect_intent(self, user_input: str) -> str:
        """
        Detect the intent of the user input.
        
        Args:
            user_input: The input text from the user
            
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
        
        return entities
    
    def _determine_context(self) -> Dict[str, Any]:
        """
        Determine the current conversation context.
        
        Returns:
            Dict containing context information
        """
        # Simple context determination (to be expanded in future phases)
        context = {
            'conversation_length': len(self.conversation_memory),
            'has_accounts': any(self.protocol_state['accounts'].values()),
            'week_classified': self.protocol_state['week_classification'] is not None
        }
        
        return context
    
    def _generate_greeting(self) -> str:
        """Generate a greeting response."""
        return (
            "Hello! I'm your ALL-USE agent, designed to help you implement the ALL-USE "
            "wealth-building protocol. I can guide you through setting up your three-tiered "
            "account structure, making protocol-driven decisions, and tracking your progress. "
            "How can I assist you today?"
        )
    
    def _generate_protocol_explanation(self, detail_level: str) -> str:
        """
        Generate a protocol explanation.
        
        Args:
            detail_level: The level of detail for the explanation
            
        Returns:
            String containing the explanation
        """
        if detail_level == 'overview':
            return (
                "The ALL-USE protocol is a wealth-building system based on a three-tiered account structure:\n\n"
                "1. Generation Account (Gen-Acc): Weekly premium harvesting using 40-50 delta options on "
                "volatile stocks like TSLA and NVDA, targeting 1.5% weekly returns.\n\n"
                "2. Revenue Account (Rev-Acc): Stable income generation using 30-40 delta options on "
                "market leaders like AAPL, AMZN, and MSFT, targeting 1.0% weekly returns.\n\n"
                "3. Compounding Account (Com-Acc): Long-term geometric growth using 20-30 delta options "
                "on stable market leaders, targeting 0.5% weekly returns.\n\n"
                "The system uses account forking when Gen-Acc reaches a $50K surplus, with reinvestment "
                "following a quarterly schedule (75% to contracts, 25% to LEAPS for Rev-Acc and Com-Acc). "
                "This creates geometric rather than linear growth over time.\n\n"
                "Would you like more details on a specific aspect of the protocol?"
            )
        else:
            # More detailed explanations to be implemented in future phases
            return self._generate_protocol_explanation('overview')
    
    def _generate_account_setup(self, parameters: Dict[str, Any]) -> str:
        """
        Generate an account setup response.
        
        Args:
            parameters: Parameters for account setup
            
        Returns:
            String containing the account setup response
        """
        initial_investment = parameters.get('initial_investment', 0)
        
        if initial_investment <= 0:
            return (
                "I'd be happy to help you set up your ALL-USE account structure. "
                "To get started, I'll need to know your initial investment amount. "
                "What total amount would you like to invest in the ALL-USE system?"
            )
        
        # Calculate account allocations
        gen_acc_allocation = initial_investment * self.parameters.get_effective_allocation('GEN_ACC')
        rev_acc_allocation = initial_investment * self.parameters.get_effective_allocation('REV_ACC')
        com_acc_allocation = initial_investment * self.parameters.get_effective_allocation('COM_ACC')
        
        # Update protocol state
        self.protocol_state['accounts']['GEN_ACC'] = [gen_acc_allocation]
        self.protocol_state['accounts']['REV_ACC'] = [rev_acc_allocation]
        self.protocol_state['accounts']['COM_ACC'] = [com_acc_allocation]
        
        return (
            f"Great! I've set up your ALL-USE account structure with an initial investment of ${initial_investment:,.2f}:\n\n"
            f"1. Generation Account (Gen-Acc): ${gen_acc_allocation:,.2f}\n"
            f"   - Weekly premium harvesting (40-50 delta options)\n"
            f"   - Target stocks: TSLA, NVDA\n"
            f"   - Expected weekly return: 1.5%\n\n"
            f"2. Revenue Account (Rev-Acc): ${rev_acc_allocation:,.2f}\n"
            f"   - Stable income generation (30-40 delta options)\n"
            f"   - Target stocks: AAPL, AMZN, MSFT\n"
            f"   - Expected weekly return: 1.0%\n\n"
            f"3. Compounding Account (Com-Acc): ${com_acc_allocation:,.2f}\n"
            f"   - Long-term geometric growth (20-30 delta options)\n"
            f"   - Target stocks: AAPL, AMZN, MSFT\n"
            f"   - Expected weekly return: 0.5%\n\n"
            f"Each account includes a 5% cash buffer for adjustments and opportunities.\n\n"
            f"Would you like me to explain the next steps for implementing the ALL-USE protocol with these accounts?"
        )
    
    def _generate_week_classification(self, parameters: Dict[str, Any]) -> str:
        """
        Generate a week classification response.
        
        Args:
            parameters: Parameters for week classification
            
        Returns:
            String containing the week classification response
        """
        # Simple week classification (to be expanded in future phases)
        # In Phase 1, this is manual classification based on user input
        return (
            "To classify the current market week as Green, Red, or Chop, I need some information:\n\n"
            "1. Overall market trend (major indices up/down/sideways)\n"
            "2. Volatility levels (VIX and historical volatility)\n"
            "3. Support/resistance levels for key stocks\n\n"
            "In Phase 1, week classification is manual. Please provide this information, "
            "or let me know your assessment of the current week type."
        )
    
    def _generate_clarification(self) -> str:
        """Generate a clarification response."""
        return (
            "I'm not sure I understood your request. As your ALL-USE agent, I can help with:\n\n"
            "- Explaining the ALL-USE protocol and account structure\n"
            "- Setting up your three-tiered account system\n"
            "- Classifying market weeks as Green, Red, or Chop\n"
            "- Recommending protocol-based trading decisions\n"
            "- Tracking your account performance and growth\n\n"
            "Could you please clarify what you'd like assistance with?"
        )


if __name__ == "__main__":
    # Simple CLI for testing
    agent = ALLUSEAgent()
    print("ALL-USE Agent initialized. Type 'exit' to quit.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
        
        response = agent.process_message(user_input)
        print(f"\nALL-USE Agent: {response}")
