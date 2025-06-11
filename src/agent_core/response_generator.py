"""
ALL-USE Response Generator Module

This module implements the response generation capabilities for the ALL-USE agent,
providing personality-aligned responses based on decisions.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protocol_engine.all_use_parameters import ALLUSEParameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('all_use_response.log')
    ]
)

logger = logging.getLogger('all_use_response')

class ResponseGenerator:
    """
    Response generator for the ALL-USE agent.
    
    This class generates natural language responses based on decisions,
    incorporating the agent's personality traits.
    """
    
    def __init__(self):
        """Initialize the response generator."""
        self.parameters = ALLUSEParameters
        self.personality_traits = {
            'methodical': 0.8,  # High methodical trait (0.0-1.0)
            'confident': 0.7,   # High confidence trait (0.0-1.0)
            'educational': 0.8, # High educational trait (0.0-1.0)
            'calm': 0.9,        # Very high calm trait (0.0-1.0)
            'precise': 0.9      # Very high precision trait (0.0-1.0)
        }
        logger.info("Response generator initialized")
    
    def generate_response(self, decision: Dict[str, Any], 
                         protocol_state: Dict[str, Any],
                         context: Dict[str, Any]) -> str:
        """
        Generate a response based on decision, protocol state, and context.
        
        Args:
            decision: The decision from the cognitive framework
            protocol_state: The current protocol state
            context: The current context
            
        Returns:
            String containing the agent's response
        """
        logger.info(f"Generating response for action: {decision['action']}")
        
        # Generate response based on decision
        response = ""
        
        if decision['action'] == 'respond' and decision['response_type'] == 'greeting':
            response = self._generate_greeting()
        
        elif decision['action'] == 'explain' and decision['response_type'] == 'protocol_explanation':
            response = self._generate_protocol_explanation(decision['parameters']['detail_level'])
        
        elif decision['action'] == 'setup' and decision['response_type'] == 'account_setup':
            response = self._generate_account_setup(decision['parameters'], protocol_state)
        
        elif decision['action'] == 'classify' and decision['response_type'] == 'week_classification':
            response = self._generate_week_classification(decision['parameters'])
        
        elif decision['action'] == 'recommend' and decision['response_type'] == 'trade_recommendations':
            response = self._generate_trade_recommendations(decision['parameters'], protocol_state)
        
        elif decision['action'] == 'report' and decision['response_type'] == 'performance_report':
            response = self._generate_performance_report(decision['parameters'], protocol_state)
        
        elif decision['action'] == 'fork' and decision['response_type'] == 'account_fork':
            response = self._generate_account_fork(decision['parameters'], protocol_state)
        
        elif decision['action'] == 'clarify':
            if decision['response_type'] == 'account_setup_required':
                response = self._generate_account_setup_required()
            elif decision['response_type'] == 'week_classification_required':
                response = self._generate_week_classification_required()
            elif decision['response_type'] == 'fork_threshold_not_reached':
                response = self._generate_fork_threshold_not_reached(protocol_state)
            else:
                response = self._generate_clarification()
        
        # Apply personality traits to response
        response = self._apply_personality_traits(response)
        
        return response
    
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
                "1. Generation Account (Gen-Acc): Weekly premium harvesting using "
                f"{self.parameters.GEN_ACC_DELTA_RANGE[0]}-{self.parameters.GEN_ACC_DELTA_RANGE[1]} delta options on "
                f"volatile stocks like {', '.join(self.parameters.TARGET_STOCKS['GEN_ACC'])}, "
                f"targeting {self.parameters.GEN_ACC_WEEKLY_RETURN:.1%} weekly returns.\n\n"
                "2. Revenue Account (Rev-Acc): Stable income generation using "
                f"{self.parameters.REV_ACC_DELTA_RANGE[0]}-{self.parameters.REV_ACC_DELTA_RANGE[1]} delta options on "
                f"market leaders like {', '.join(self.parameters.TARGET_STOCKS['REV_ACC'])}, "
                f"targeting {self.parameters.REV_ACC_WEEKLY_RETURN:.1%} weekly returns.\n\n"
                "3. Compounding Account (Com-Acc): Long-term geometric growth using "
                f"{self.parameters.COM_ACC_DELTA_RANGE[0]}-{self.parameters.COM_ACC_DELTA_RANGE[1]} delta options "
                f"on stable market leaders, targeting {self.parameters.COM_ACC_WEEKLY_RETURN:.1%} weekly returns.\n\n"
                f"The system uses account forking when Gen-Acc reaches a ${self.parameters.FORK_THRESHOLD:,} surplus, "
                "with reinvestment following a quarterly schedule "
                f"({self.parameters.REINVESTMENT_ALLOCATION['CONTRACTS']:.0%} to contracts, "
                f"{self.parameters.REINVESTMENT_ALLOCATION['LEAPS']:.0%} to LEAPS for Rev-Acc and Com-Acc). "
                "This creates geometric rather than linear growth over time.\n\n"
                "Would you like more details on a specific aspect of the protocol?"
            )
        elif detail_level == 'account_structure':
            return (
                "The ALL-USE three-tiered account structure is designed to balance income generation, "
                "stability, and long-term growth:\n\n"
                "1. Generation Account (Gen-Acc):\n"
                f"   - Initial allocation: {self.parameters.INITIAL_ALLOCATION['GEN_ACC']:.0%} of investment\n"
                f"   - Cash buffer: {self.parameters.CASH_BUFFER:.0%}\n"
                f"   - Weekly return target: {self.parameters.GEN_ACC_WEEKLY_RETURN:.1%}\n"
                f"   - Option strategy: {self.parameters.GEN_ACC_DELTA_RANGE[0]}-{self.parameters.GEN_ACC_DELTA_RANGE[1]} delta\n"
                f"   - Target stocks: {', '.join(self.parameters.TARGET_STOCKS['GEN_ACC'])}\n"
                f"   - Entry protocol: {self.parameters.ENTRY_PROTOCOL['GEN_ACC'].replace('_', ' ').title()}\n"
                f"   - Reinvestment: {self.parameters.REINVESTMENT_FREQUENCY['GEN_ACC'].title()}\n"
                f"   - Forking threshold: ${self.parameters.FORK_THRESHOLD:,}\n\n"
                "2. Revenue Account (Rev-Acc):\n"
                f"   - Initial allocation: {self.parameters.INITIAL_ALLOCATION['REV_ACC']:.0%} of investment\n"
                f"   - Cash buffer: {self.parameters.CASH_BUFFER:.0%}\n"
                f"   - Weekly return target: {self.parameters.REV_ACC_WEEKLY_RETURN:.1%}\n"
                f"   - Option strategy: {self.parameters.REV_ACC_DELTA_RANGE[0]}-{self.parameters.REV_ACC_DELTA_RANGE[1]} delta\n"
                f"   - Target stocks: {', '.join(self.parameters.TARGET_STOCKS['REV_ACC'])}\n"
                f"   - Entry protocol: {self.parameters.ENTRY_PROTOCOL['REV_ACC'].replace('_', ' ').title()}\n"
                f"   - Reinvestment: {self.parameters.REINVESTMENT_FREQUENCY['REV_ACC'].title()}\n"
                f"   - Reinvestment allocation: {self.parameters.REINVESTMENT_ALLOCATION['CONTRACTS']:.0%} contracts, "
                f"{self.parameters.REINVESTMENT_ALLOCATION['LEAPS']:.0%} LEAPS\n\n"
                "3. Compounding Account (Com-Acc):\n"
                f"   - Initial allocation: {self.parameters.INITIAL_ALLOCATION['COM_ACC']:.0%} of investment\n"
                f"   - Cash buffer: {self.parameters.CASH_BUFFER:.0%}\n"
                f"   - Weekly return target: {self.parameters.COM_ACC_WEEKLY_RETURN:.1%}\n"
                f"   - Option strategy: {self.parameters.COM_ACC_DELTA_RANGE[0]}-{self.parameters.COM_ACC_DELTA_RANGE[1]} delta\n"
                f"   - Target stocks: {', '.join(self.parameters.TARGET_STOCKS['COM_ACC'])}\n"
                f"   - Reinvestment: {self.parameters.REINVESTMENT_FREQUENCY['COM_ACC'].title()}\n"
                f"   - Reinvestment allocation: {self.parameters.REINVESTMENT_ALLOCATION['CONTRACTS']:.0%} contracts, "
                f"{self.parameters.REINVESTMENT_ALLOCATION['LEAPS']:.0%} LEAPS\n"
                f"   - Merge threshold: ${self.parameters.MERGE_THRESHOLD:,}\n\n"
                "This structure creates a balanced approach to wealth-building, with each account serving a specific purpose "
                "in the overall system."
            )
        elif detail_level == 'forking':
            return (
                "Account forking is a key mechanism in the ALL-USE protocol that creates geometric growth:\n\n"
                f"1. When a Gen-Acc reaches a surplus of ${self.parameters.FORK_THRESHOLD:,}, it triggers a fork.\n\n"
                "2. The fork splits the surplus 50/50:\n"
                "   - 50% creates a new Gen-Acc\n"
                "   - 50% goes to the Com-Acc\n\n"
                "3. The new Gen-Acc follows the same protocol as the original Gen-Acc.\n\n"
                f"4. When a forked account reaches ${self.parameters.MERGE_THRESHOLD:,}, it merges into the parent Com-Acc.\n\n"
                "This process creates a multiplier effect, as each new Gen-Acc generates its own income stream and eventually "
                "reaches the forking threshold itself. Over time, this creates exponential rather than linear growth.\n\n"
                "For example, starting with one Gen-Acc, you might eventually have multiple Gen-Accs all generating income "
                "and feeding the Com-Acc, which compounds over time."
            )
        else:
            # More detailed explanations to be implemented in future phases
            return self._generate_protocol_explanation('overview')
    
    def _generate_account_setup(self, parameters: Dict[str, Any], protocol_state: Dict[str, Any]) -> str:
        """
        Generate an account setup response.
        
        Args:
            parameters: Parameters for account setup
            protocol_state: The current protocol state
            
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
        
        return (
            f"Great! I've set up your ALL-USE account structure with an initial investment of ${initial_investment:,.2f}:\n\n"
            f"1. Generation Account (Gen-Acc): ${gen_acc_allocation:,.2f}\n"
            f"   - Weekly premium harvesting ({self.parameters.GEN_ACC_DELTA_RANGE[0]}-{self.parameters.GEN_ACC_DELTA_RANGE[1]} delta options)\n"
            f"   - Target stocks: {', '.join(self.parameters.TARGET_STOCKS['GEN_ACC'])}\n"
            f"   - Expected weekly return: {self.parameters.GEN_ACC_WEEKLY_RETURN:.1%}\n\n"
            f"2. Revenue Account (Rev-Acc): ${rev_acc_allocation:,.2f}\n"
            f"   - Stable income generation ({self.parameters.REV_ACC_DELTA_RANGE[0]}-{self.parameters.REV_ACC_DELTA_RANGE[1]} delta options)\n"
            f"   - Target stocks: {', '.join(self.parameters.TARGET_STOCKS['REV_ACC'])}\n"
            f"   - Expected weekly return: {self.parameters.REV_ACC_WEEKLY_RETURN:.1%}\n\n"
            f"3. Compounding Account (Com-Acc): ${com_acc_allocation:,.2f}\n"
            f"   - Long-term geometric growth ({self.parameters.COM_ACC_DELTA_RANGE[0]}-{self.parameters.COM_ACC_DELTA_RANGE[1]} delta options)\n"
            f"   - Target stocks: {', '.join(self.parameters.TARGET_STOCKS['COM_ACC'])}\n"
            f"   - Expected weekly return: {self.parameters.COM_ACC_WEEKLY_RETURN:.1%}\n\n"
            f"Each account includes a {self.parameters.CASH_BUFFER:.0%} cash buffer for adjustments and opportunities.\n\n"
            f"The next step is to classify the current market week as Green, Red, or Chop. "
            f"This will determine the specific protocol decisions for each account. "
            f"Would you like me to help you classify the current week?"
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
    
    def _generate_trade_recommendations(self, parameters: Dict[str, Any], protocol_state: Dict[str, Any]) -> str:
        """
        Generate trade recommendations based on protocol.
        
        Args:
            parameters: Parameters for trade recommendations
            protocol_state: The current protocol state
            
        Returns:
            String containing the trade recommendations
        """
        week_classification = parameters.get('week_classification')
        account_balances = parameters.get('account_balances', {})
        
        gen_acc_total = sum(account_balances.get('GEN_ACC', [0]))
        rev_acc_total = sum(account_balances.get('REV_ACC', [0]))
        com_acc_total = sum(account_balances.get('COM_ACC', [0]))
        
        response = f"Based on the current {week_classification} week classification, here are your protocol-driven trade recommendations:\n\n"
        
        if week_classification == 'Green':
            response += (
                "1. Generation Account (Gen-Acc):\n"
                f"   - Total balance: ${gen_acc_total:,.2f}\n"
                f"   - Protocol: Thursday entry with {self.parameters.GEN_ACC_DELTA_RANGE[0]}-{self.parameters.GEN_ACC_DELTA_RANGE[1]} delta puts\n"
                f"   - Target stocks: {', '.join(self.parameters.TARGET_STOCKS['GEN_ACC'])}\n"
                "   - Position sizing: 10-15% of account per position\n\n"
                "2. Revenue Account (Rev-Acc):\n"
                f"   - Total balance: ${rev_acc_total:,.2f}\n"
                f"   - Protocol: Monday-Wednesday entry with {self.parameters.REV_ACC_DELTA_RANGE[0]}-{self.parameters.REV_ACC_DELTA_RANGE[1]} delta puts\n"
                f"   - Target stocks: {', '.join(self.parameters.TARGET_STOCKS['REV_ACC'])}\n"
                "   - Position sizing: 5-10% of account per position\n\n"
                "3. Compounding Account (Com-Acc):\n"
                f"   - Total balance: ${com_acc_total:,.2f}\n"
                "   - Protocol: No action required this week (quarterly schedule)\n"
            )
        elif week_classification == 'Red':
            response += (
                "1. Generation Account (Gen-Acc):\n"
                f"   - Total balance: ${gen_acc_total:,.2f}\n"
                f"   - Protocol: Monday entry with {self.parameters.GEN_ACC_DELTA_RANGE[0]-5}-{self.parameters.GEN_ACC_DELTA_RANGE[1]-5} delta puts (more conservative)\n"
                f"   - Target stocks: {', '.join(self.parameters.TARGET_STOCKS['GEN_ACC'])}\n"
                "   - Position sizing: 5-10% of account per position (reduced size)\n\n"
                "2. Revenue Account (Rev-Acc):\n"
                f"   - Total balance: ${rev_acc_total:,.2f}\n"
                f"   - Protocol: Monday entry with {self.parameters.REV_ACC_DELTA_RANGE[0]-5}-{self.parameters.REV_ACC_DELTA_RANGE[1]-5} delta puts (more conservative)\n"
                f"   - Target stocks: {', '.join(self.parameters.TARGET_STOCKS['REV_ACC'])}\n"
                "   - Position sizing: 3-5% of account per position (reduced size)\n\n"
                "3. Compounding Account (Com-Acc):\n"
                f"   - Total balance: ${com_acc_total:,.2f}\n"
                "   - Protocol: No action required this week (quarterly schedule)\n"
            )
        elif week_classification == 'Chop':
            response += (
                "1. Generation Account (Gen-Acc):\n"
                f"   - Total balance: ${gen_acc_total:,.2f}\n"
                "   - Protocol: No new entries this week (sit out choppy conditions)\n"
                "   - Focus on managing existing positions\n\n"
                "2. Revenue Account (Rev-Acc):\n"
                f"   - Total balance: ${rev_acc_total:,.2f}\n"
                f"   - Protocol: Wednesday entry with {self.parameters.REV_ACC_DELTA_RANGE[0]-10}-{self.parameters.REV_ACC_DELTA_RANGE[1]-10} delta puts (very conservative)\n"
                f"   - Target stocks: {', '.join(self.parameters.TARGET_STOCKS['REV_ACC'])}\n"
                "   - Position sizing: 3-5% of account per position (reduced size)\n\n"
                "3. Compounding Account (Com-Acc):\n"
                f"   - Total balance: ${com_acc_total:,.2f}\n"
                "   - Protocol: No action required this week (quarterly schedule)\n"
            )
        
        response += (
            "\nWould you like me to provide specific trade recommendations for any of these accounts? "
            "I can calculate exact strike prices and position sizes based on current market data."
        )
        
        return response
    
    def _generate_performance_report(self, parameters: Dict[str, Any], protocol_state: Dict[str, Any]) -> str:
        """
        Generate a performance report.
        
        Args:
            parameters: Parameters for performance report
            protocol_state: The current protocol state
            
        Returns:
            String containing the performance report
        """
        account_balances = parameters.get('account_balances', {})
        
        gen_acc_balances = account_balances.get('GEN_ACC', [0])
        rev_acc_balances = account_balances.get('REV_ACC', [0])
        com_acc_balances = account_balances.get('COM_ACC', [0])
        
        gen_acc_total = sum(gen_acc_balances)
        rev_acc_total = sum(rev_acc_balances)
        com_acc_total = sum(com_acc_balances)
        total_portfolio = gen_acc_total + rev_acc_total + com_acc_total
        
        # In a real implementation, we would track historical performance
        # For now, we'll generate a simple report
        
        response = "ALL-USE Performance Report:\n\n"
        
        response += "Account Balances:\n"
        response += f"1. Generation Account (Gen-Acc): ${gen_acc_total:,.2f}\n"
        
        if len(gen_acc_balances) > 1:
            for i, balance in enumerate(gen_acc_balances):
                response += f"   - Gen-Acc-{i+1}: ${balance:,.2f}\n"
        
        response += f"2. Revenue Account (Rev-Acc): ${rev_acc_total:,.2f}\n"
        response += f"3. Compounding Account (Com-Acc): ${com_acc_total:,.2f}\n"
        response += f"Total Portfolio Value: ${total_portfolio:,.2f}\n\n"
        
        # Check for forking opportunities
        fork_opportunities = []
        for i, balance in enumerate(gen_acc_balances):
            if balance >= self.parameters.FORK_THRESHOLD:
                fork_opportunities.append((i, balance))
        
        if fork_opportunities:
            response += "Forking Opportunities:\n"
            for i, balance in fork_opportunities:
                fork_amount = balance / 2
                response += f"- Gen-Acc-{i+1} (${balance:,.2f}) is ready for forking\n"
                response += f"  This would create a new Gen-Acc with ${fork_amount:,.2f} and add ${fork_amount:,.2f} to Com-Acc\n"
            response += "\n"
        
        response += (
            "Would you like a more detailed performance analysis, including historical returns "
            "and projections? In future phases, I'll provide comprehensive performance tracking "
            "and visualization."
        )
        
        return response
    
    def _generate_account_fork(self, parameters: Dict[str, Any], protocol_state: Dict[str, Any]) -> str:
        """
        Generate an account fork response.
        
        Args:
            parameters: Parameters for account fork
            protocol_state: The current protocol state
            
        Returns:
            String containing the account fork response
        """
        account_index = parameters.get('account_index', 0)
        account_balance = parameters.get('account_balance', 0)
        fork_amount = parameters.get('fork_amount', 0)
        
        new_gen_acc_amount = fork_amount * parameters.get('allocation', {}).get('new_gen_acc', 0.5)
        com_acc_amount = fork_amount * parameters.get('allocation', {}).get('com_acc', 0.5)
        
        remaining_balance = account_balance - fork_amount
        
        response = (
            f"I've executed a fork for Gen-Acc-{account_index+1} according to the ALL-USE protocol:\n\n"
            f"Original Gen-Acc-{account_index+1} balance: ${account_balance:,.2f}\n"
            f"Fork amount: ${fork_amount:,.2f}\n\n"
            f"This fork creates:\n"
            f"1. New Gen-Acc-{len(protocol_state['accounts']['GEN_ACC'])+1} with ${new_gen_acc_amount:,.2f}\n"
            f"2. Addition of ${com_acc_amount:,.2f} to Com-Acc\n\n"
            f"Remaining balance in Gen-Acc-{account_index+1}: ${remaining_balance:,.2f}\n\n"
            "The new Gen-Acc will follow the same protocol as the original Gen-Acc, "
            "generating additional income and eventually reaching its own forking threshold. "
            "This fork contributes to the geometric growth pattern of the ALL-USE system."
        )
        
        return response
    
    def _generate_account_setup_required(self) -> str:
        """Generate a response indicating account setup is required."""
        return (
            "Before I can provide recommendations or perform account operations, "
            "we need to set up your ALL-USE account structure. This involves creating "
            "your three-tiered account system with the appropriate allocations.\n\n"
            "To get started, please let me know your initial investment amount, and "
            "I'll help you set up your Gen-Acc, Rev-Acc, and Com-Acc according to the "
            "ALL-USE protocol."
        )
    
    def _generate_week_classification_required(self) -> str:
        """Generate a response indicating week classification is required."""
        return (
            "Before I can provide trade recommendations, we need to classify the current "
            "market week as Green, Red, or Chop. This classification determines the specific "
            "protocol decisions for each account.\n\n"
            "To classify the week, I need information about:\n"
            "1. Overall market trend (major indices up/down/sideways)\n"
            "2. Volatility levels (VIX and historical volatility)\n"
            "3. Support/resistance levels for key stocks\n\n"
            "Please provide this information, or let me know your assessment of the current week type."
        )
    
    def _generate_fork_threshold_not_reached(self, protocol_state: Dict[str, Any]) -> str:
        """
        Generate a response indicating fork threshold has not been reached.
        
        Args:
            protocol_state: The current protocol state
            
        Returns:
            String containing the response
        """
        gen_acc_balances = protocol_state.get('accounts', {}).get('GEN_ACC', [0])
        highest_balance = max(gen_acc_balances) if gen_acc_balances else 0
        remaining_amount = self.parameters.FORK_THRESHOLD - highest_balance
        
        if highest_balance <= 0:
            return (
                "I don't see any Generation Accounts set up yet. Before we can discuss "
                "account forking, we need to set up your ALL-USE account structure. "
                "Would you like to do that now?"
            )
        
        return (
            f"None of your Generation Accounts have reached the forking threshold of ${self.parameters.FORK_THRESHOLD:,} yet. "
            f"Your highest Gen-Acc balance is ${highest_balance:,.2f}, which is ${remaining_amount:,.2f} away from "
            "the forking threshold.\n\n"
            "The ALL-USE protocol specifies that account forking occurs when a Gen-Acc reaches "
            f"a surplus of ${self.parameters.FORK_THRESHOLD:,}. When this happens, 50% of the surplus "
            "creates a new Gen-Acc, and 50% goes to the Com-Acc, contributing to the geometric "
            "growth pattern of the system.\n\n"
            "Would you like me to provide a projection of when your Gen-Acc might reach the forking threshold?"
        )
    
    def _generate_clarification(self) -> str:
        """Generate a clarification response."""
        return (
            "I'm not sure I understood your request. As your ALL-USE agent, I can help with:\n\n"
            "- Explaining the ALL-USE protocol and account structure\n"
            "- Setting up your three-tiered account system\n"
            "- Classifying market weeks as Green, Red, or Chop\n"
            "- Recommending protocol-based trading decisions\n"
            "- Tracking your account performance and growth\n"
            "- Managing account forking and merging\n\n"
            "Could you please clarify what you'd like assistance with?"
        )
    
    def _apply_personality_traits(self, response: str) -> str:
        """
        Apply personality traits to the response.
        
        Args:
            response: The original response
            
        Returns:
            String containing the personality-enhanced response
        """
        # In a more sophisticated implementation, we would use NLP techniques
        # to modify the response based on personality traits
        
        # For now, we'll make simple adjustments
        
        # Add methodical structure if trait is high
        if self.personality_traits['methodical'] > 0.7 and len(response) > 200 and '\n\n' not in response:
            # Add structure to long responses that don't already have it
            sentences = response.split('. ')
            if len(sentences) > 3:
                structured_response = "Let me walk you through this systematically:\n\n"
                for i, sentence in enumerate(sentences[:3]):
                    structured_response += f"{i+1}. {sentence}.\n"
                if len(sentences) > 3:
                    structured_response += "\n" + ". ".join(sentences[3:])
                response = structured_response
        
        # Add confidence markers if trait is high
        if self.personality_traits['confident'] > 0.7:
            confidence_phrases = [
                "Based on the ALL-USE protocol, ",
                "According to our established parameters, ",
                "Following the proven ALL-USE system, "
            ]
            
            # Only add if not already present
            if not any(phrase in response for phrase in confidence_phrases):
                for phrase in confidence_phrases:
                    if not response.startswith(phrase) and len(response) > 50:
                        response = phrase + response[0].lower() + response[1:]
                        break
        
        # Add educational elements if trait is high
        if self.personality_traits['educational'] > 0.8 and len(response) > 100:
            educational_additions = [
                "\n\nThis approach creates a mathematical edge by removing emotion from trading decisions.",
                "\n\nRemember that the ALL-USE system is designed for geometric rather than linear growth.",
                "\n\nThe key to success with ALL-USE is consistent protocol application, not market prediction."
            ]
            
            # Only add if not already present
            if not any(addition in response for addition in educational_additions):
                import random
                response += random.choice(educational_additions)
        
        # Ensure precision in numbers if trait is high
        if self.personality_traits['precise'] > 0.8:
            import re
            # Find approximate percentages and make them more precise
            response = re.sub(r'about (\d+)%', r'\1.0%', response)
            
        return response


class DialogueManager:
    """
    Dialogue manager for the ALL-USE agent.
    
    This class manages the dialogue flow and conversation state.
    """
    
    def __init__(self):
        """Initialize the dialogue manager."""
        self.response_generator = ResponseGenerator()
        self.conversation_state = {
            'current_topic': None,
            'pending_questions': [],
            'unanswered_queries': []
        }
        logger.info("Dialogue manager initialized")
    
    def generate_response(self, decision: Dict[str, Any], 
                         protocol_state: Dict[str, Any],
                         context: Dict[str, Any]) -> str:
        """
        Generate a response using the response generator.
        
        Args:
            decision: The decision from the cognitive framework
            protocol_state: The current protocol state
            context: The current context
            
        Returns:
            String containing the agent's response
        """
        # Update conversation state
        self._update_conversation_state(decision)
        
        # Generate response
        response = self.response_generator.generate_response(
            decision, protocol_state, context
        )
        
        return response
    
    def _update_conversation_state(self, decision: Dict[str, Any]) -> None:
        """
        Update the conversation state based on the decision.
        
        Args:
            decision: The decision from the cognitive framework
        """
        # Update current topic
        if decision['action'] in ['explain', 'setup', 'classify', 'recommend', 'report', 'fork']:
            self.conversation_state['current_topic'] = decision['response_type']
        
        # Track pending questions
        if decision['action'] == 'clarify':
            self.conversation_state['pending_questions'].append({
                'type': decision['response_type'],
                'timestamp': datetime.now().isoformat()
            })
        
        # Track unanswered queries
        if decision['action'] == 'clarify' and decision['response_type'] == 'clarification':
            self.conversation_state['unanswered_queries'].append({
                'intent': decision['intent'],
                'timestamp': datetime.now().isoformat()
            })
        
        logger.info(f"Updated conversation state, current topic: {self.conversation_state['current_topic']}")
    
    def get_conversation_state(self) -> Dict[str, Any]:
        """
        Get the current conversation state.
        
        Returns:
            Dict containing the conversation state
        """
        return self.conversation_state
