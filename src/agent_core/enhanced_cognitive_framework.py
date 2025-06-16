"""
ALL-USE Enhanced Cognitive Framework Module

This module implements enhanced cognitive capabilities for the ALL-USE agent,
including advanced intent detection, entity extraction, and context management.
"""

import logging
import sys
import os
import re
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
        logging.FileHandler('all_use_enhanced_cognitive.log')
    ]
)

logger = logging.getLogger('all_use_enhanced_cognitive')

class EnhancedIntentDetector:
    """
    Enhanced intent detection system for the ALL-USE agent.
    
    This class provides more sophisticated intent detection capabilities
    using pattern matching and context awareness.
    """
    
    def __init__(self):
        """Initialize the enhanced intent detector."""
        self.intent_patterns = {
            'greeting': [
                r'\b(hello|hi|hey|greetings|good\s+(morning|afternoon|evening))\b',
                r'\b(howdy|what\'s\s+up|sup)\b'
            ],
            'explain_protocol': [
                r'\b(explain|tell\s+me\s+about|describe|what\s+is)\b.*\b(protocol|all-use|system|strategy)\b',
                r'\b(how\s+does|how\s+do)\b.*\b(work|function|operate)\b',
                r'\b(overview|summary|introduction)\b.*\b(protocol|system)\b'
            ],
            'explain_accounts': [
                r'\b(explain|tell\s+me\s+about|describe)\b.*\b(account|accounts)\b',
                r'\b(what\s+are|what\s+is)\b.*\b(gen-acc|rev-acc|com-acc|generation|revenue|compounding)\b',
                r'\b(account\s+structure|three-tiered)\b'
            ],
            'explain_forking': [
                r'\b(explain|tell\s+me\s+about|describe)\b.*\b(fork|forking|split)\b',
                r'\b(how\s+does|when\s+does)\b.*\b(fork|forking|account\s+split)\b',
                r'\b(fork\s+threshold|50k|50000)\b'
            ],
            'setup_accounts': [
                r'\b(setup|create|start|initialize|begin)\b.*\b(account|accounts)\b',
                r'\b(set\s+up|get\s+started)\b.*\b(investment|portfolio)\b',
                r'\b(invest|investing)\b.*\b(\$|\d+)\b',
                r'\b(want\s+to\s+start|want\s+to\s+invest|start\s+investing)\b',
                r'\b(initialize\s+my\s+portfolio|setup\s+my\s+accounts)\b'
            ],
            'classify_week': [
                r'\b(classify|classification|what\s+kind\s+of\s+week)\b',
                r'\b(market\s+condition|week\s+type|green|red|chop)\b',
                r'\b(current\s+week|this\s+week)\b.*\b(market|condition)\b'
            ],
            'recommend_trades': [
                r'\b(recommend|suggest|advise)\b.*\b(trade|trades|position)\b',
                r'\b(what\s+should\s+i|should\s+i)\b.*\b(trade|buy|sell)\b',
                r'\b(trading\s+recommendation|trade\s+suggestion)\b'
            ],
            'check_performance': [
                r'\b(performance|how\s+am\s+i\s+doing|results|progress)\b',
                r'\b(show\s+me|check|view)\b.*\b(balance|account|portfolio)\b',
                r'\b(profit|loss|return|growth)\b'
            ],
            'fork_account': [
                r'\b(fork|split|create\s+new)\b.*\b(account)\b',
                r'\b(ready\s+to\s+fork|fork\s+threshold)\b',
                r'\b(50k|50000)\b.*\b(surplus|threshold)\b'
            ],
            'reinvest': [
                r'\b(reinvest|reinvestment|quarterly)\b',
                r'\b(contracts|leaps)\b.*\b(reinvest|allocation)\b',
                r'\b(75%|25%)\b.*\b(split|allocation)\b'
            ],
            'week_green': [
                r'\b(green\s+week|bullish|uptrend|market\s+up)\b',
                r'\b(positive\s+momentum|strong\s+market)\b'
            ],
            'week_red': [
                r'\b(red\s+week|bearish|downtrend|market\s+down)\b',
                r'\b(negative\s+momentum|weak\s+market)\b'
            ],
            'week_chop': [
                r'\b(chop|choppy|sideways|range-bound)\b',
                r'\b(volatile|uncertain|mixed\s+signals)\b'
            ],
            'help': [
                r'\b(help|assistance|support|guide)\b',
                r'\b(what\s+can\s+you\s+do|capabilities|features)\b',
                r'\b(how\s+to|instructions)\b'
            ],
            'status': [
                r'\b(status|current\s+state|where\s+am\s+i)\b',
                r'\b(show\s+me\s+my|my\s+current)\b.*\b(account|balance|position)\b'
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for intent, patterns in self.intent_patterns.items():
            self.compiled_patterns[intent] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        
        logger.info("Enhanced intent detector initialized")
    
    def detect_intent(self, user_input: str, conversation_context: Dict[str, Any]) -> str:
        """
        Detect the intent of the user input using enhanced pattern matching.
        
        Args:
            user_input: The input text from the user
            conversation_context: The current conversation context
            
        Returns:
            String representing the detected intent
        """
        user_input_clean = user_input.strip().lower()
        
        # Score each intent based on pattern matches
        intent_scores = {}
        
        for intent, patterns in self.compiled_patterns.items():
            score = 0
            for pattern in patterns:
                matches = pattern.findall(user_input_clean)
                score += len(matches)
            
            if score > 0:
                intent_scores[intent] = score
        
        # Apply context-based adjustments
        intent_scores = self._apply_context_adjustments(intent_scores, conversation_context)
        
        # Return the highest scoring intent
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            logger.info(f"Detected intent: {best_intent} (score: {intent_scores[best_intent]})")
            return best_intent
        
        logger.info("No intent detected, returning 'unknown'")
        return 'unknown'
    
    def _apply_context_adjustments(self, intent_scores: Dict[str, int], 
                                 conversation_context: Dict[str, Any]) -> Dict[str, int]:
        """
        Apply context-based adjustments to intent scores.
        
        Args:
            intent_scores: Current intent scores
            conversation_context: The conversation context
            
        Returns:
            Adjusted intent scores
        """
        # Boost setup-related intents if no accounts exist
        if not conversation_context.get('has_accounts', False):
            if 'setup_accounts' in intent_scores:
                intent_scores['setup_accounts'] *= 1.5
        
        # Boost classification intents if accounts exist but week not classified
        if (conversation_context.get('has_accounts', False) and 
            not conversation_context.get('week_classified', False)):
            for intent in ['classify_week', 'week_green', 'week_red', 'week_chop']:
                if intent in intent_scores:
                    intent_scores[intent] *= 1.3
        
        # Boost trade recommendation intents if both accounts and week classification exist
        if (conversation_context.get('has_accounts', False) and 
            conversation_context.get('week_classified', False)):
            if 'recommend_trades' in intent_scores:
                intent_scores['recommend_trades'] *= 1.2
        
        return intent_scores


class EnhancedEntityExtractor:
    """
    Enhanced entity extraction system for the ALL-USE agent.
    
    This class provides sophisticated entity extraction capabilities
    for various types of information relevant to the ALL-USE protocol.
    """
    
    def __init__(self):
        """Initialize the enhanced entity extractor."""
        self.patterns = {
            'monetary_amount': [
                r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:k|K|thousand)',
                r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:m|M|million)',
                r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)',
                r'(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*dollars?'
            ],
            'percentage': [
                r'(\d+(?:\.\d+)?)\s*%',
                r'(\d+(?:\.\d+)?)\s*percent'
            ],
            'week_type': [
                r'\b(green|red|chop|choppy)\b'
            ],
            'account_type': [
                r'\b(gen-acc|generation\s+account|gen\s+acc)\b',
                r'\b(rev-acc|revenue\s+account|rev\s+acc)\b',
                r'\b(com-acc|compounding\s+account|com\s+acc)\b'
            ],
            'stock_symbol': [
                r'\b(TSLA|NVDA|AAPL|AMZN|MSFT|GOOGL|META|NFLX)\b'
            ],
            'delta_range': [
                r'(\d+)-(\d+)\s*delta',
                r'delta\s+(\d+)-(\d+)',
                r'(\d+)\s*to\s*(\d+)\s*delta'
            ],
            'time_period': [
                r'\b(weekly|monthly|quarterly|annually|daily)\b',
                r'\b(week|month|quarter|year|day)\b'
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for entity_type, patterns in self.patterns.items():
            self.compiled_patterns[entity_type] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        
        logger.info("Enhanced entity extractor initialized")
    
    def extract_entities(self, user_input: str) -> Dict[str, Any]:
        """
        Extract entities from the user input using enhanced pattern matching.
        
        Args:
            user_input: The input text from the user
            
        Returns:
            Dict containing extracted entities
        """
        entities = {}
        
        # Extract monetary amounts
        amount = self._extract_monetary_amount(user_input)
        if amount:
            entities['amount'] = amount
        
        # Extract percentages
        percentages = self._extract_percentages(user_input)
        if percentages:
            entities['percentages'] = percentages
        
        # Extract week type
        week_type = self._extract_week_type(user_input)
        if week_type:
            entities['week_type'] = week_type
        
        # Extract account types
        account_types = self._extract_account_types(user_input)
        if account_types:
            entities['account_types'] = account_types
        
        # Extract stock symbols
        stock_symbols = self._extract_stock_symbols(user_input)
        if stock_symbols:
            entities['stock_symbols'] = stock_symbols
        
        # Extract delta ranges
        delta_ranges = self._extract_delta_ranges(user_input)
        if delta_ranges:
            entities['delta_ranges'] = delta_ranges
        
        # Extract time periods
        time_periods = self._extract_time_periods(user_input)
        if time_periods:
            entities['time_periods'] = time_periods
        
        logger.info(f"Extracted entities: {list(entities.keys())}")
        return entities
    
    def _extract_monetary_amount(self, text: str) -> Optional[float]:
        """Extract monetary amount from text."""
        for pattern in self.compiled_patterns['monetary_amount']:
            match = pattern.search(text)
            if match:
                amount_str = match.group(1).replace(',', '')
                amount = float(amount_str)
                
                # Check for multipliers
                if 'k' in match.group(0).lower() or 'thousand' in match.group(0).lower():
                    amount *= 1000
                elif 'm' in match.group(0).lower() or 'million' in match.group(0).lower():
                    amount *= 1000000
                
                return amount
        return None
    
    def _extract_percentages(self, text: str) -> List[float]:
        """Extract percentages from text."""
        percentages = []
        for pattern in self.compiled_patterns['percentage']:
            matches = pattern.findall(text)
            for match in matches:
                percentages.append(float(match))
        return percentages
    
    def _extract_week_type(self, text: str) -> Optional[str]:
        """Extract week type from text."""
        for pattern in self.compiled_patterns['week_type']:
            match = pattern.search(text)
            if match:
                week_type = match.group(1).lower()
                if week_type in ['choppy']:
                    return 'Chop'
                return week_type.capitalize()
        return None
    
    def _extract_account_types(self, text: str) -> List[str]:
        """Extract account types from text."""
        account_types = []
        for pattern in self.compiled_patterns['account_type']:
            matches = pattern.findall(text)
            for match in matches:
                if 'gen' in match.lower():
                    account_types.append('GEN_ACC')
                elif 'rev' in match.lower():
                    account_types.append('REV_ACC')
                elif 'com' in match.lower():
                    account_types.append('COM_ACC')
        return list(set(account_types))  # Remove duplicates
    
    def _extract_stock_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text."""
        symbols = []
        for pattern in self.compiled_patterns['stock_symbol']:
            matches = pattern.findall(text)
            symbols.extend(matches)
        return list(set(symbols))  # Remove duplicates
    
    def _extract_delta_ranges(self, text: str) -> List[Tuple[int, int]]:
        """Extract delta ranges from text."""
        ranges = []
        for pattern in self.compiled_patterns['delta_range']:
            matches = pattern.findall(text)
            for match in matches:
                if len(match) == 2:
                    ranges.append((int(match[0]), int(match[1])))
        return ranges
    
    def _extract_time_periods(self, text: str) -> List[str]:
        """Extract time periods from text."""
        periods = []
        for pattern in self.compiled_patterns['time_period']:
            matches = pattern.findall(text)
            periods.extend(matches)
        return list(set(periods))  # Remove duplicates


class EnhancedContextManager:
    """
    Enhanced context manager for the ALL-USE agent.
    
    This class provides sophisticated context management capabilities
    including conversation flow tracking and decision history analysis.
    """
    
    def __init__(self):
        """Initialize the enhanced context manager."""
        self.current_context = {
            'conversation_length': 0,
            'last_intent': None,
            'last_action': None,
            'has_accounts': False,
            'week_classified': False,
            'pending_decisions': [],
            'conversation_flow': [],
            'user_expertise_level': 'beginner',
            'preferred_detail_level': 'overview',
            'recent_topics': []
        }
        logger.info("Enhanced context manager initialized")
    
    def update_context(self, conversation_history: List[Dict[str, Any]], 
                      protocol_state: Dict[str, Any], 
                      user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the current context with enhanced analysis.
        
        Args:
            conversation_history: The conversation history
            protocol_state: The current protocol state
            user_preferences: The user's preferences
            
        Returns:
            Dict containing the updated context
        """
        # Update basic context
        self.current_context['conversation_length'] = len(conversation_history)
        
        # Update last intent and action if available
        if protocol_state.get('last_decision'):
            self.current_context['last_intent'] = protocol_state['last_decision'].get('intent')
            self.current_context['last_action'] = protocol_state['last_decision'].get('action')
        
        # Update account and week classification status
        self.current_context['has_accounts'] = any(protocol_state.get('accounts', {}).get(acc_type, []) 
                                                for acc_type in ['GEN_ACC', 'REV_ACC', 'COM_ACC'])
        self.current_context['week_classified'] = protocol_state.get('week_classification') is not None
        
        # Analyze conversation flow
        self._analyze_conversation_flow(conversation_history, protocol_state)
        
        # Determine user expertise level
        self._determine_user_expertise(conversation_history, protocol_state)
        
        # Update recent topics
        self._update_recent_topics(conversation_history)
        
        # Update pending decisions
        self._update_pending_decisions(protocol_state)
        
        logger.info("Enhanced context updated")
        return self.current_context
    
    def _analyze_conversation_flow(self, conversation_history: List[Dict[str, Any]], 
                                 protocol_state: Dict[str, Any]) -> None:
        """
        Analyze the conversation flow to understand user journey.
        
        Args:
            conversation_history: The conversation history
            protocol_state: The current protocol state
        """
        flow_stages = []
        
        # Analyze decision history to determine flow
        decision_history = protocol_state.get('decision_history', [])
        
        for decision in decision_history:
            intent = decision.get('intent')
            action = decision.get('action')
            
            if intent == 'greeting':
                flow_stages.append('introduction')
            elif intent == 'explain_protocol':
                flow_stages.append('learning')
            elif intent == 'setup_accounts':
                flow_stages.append('setup')
            elif intent in ['classify_week', 'week_green', 'week_red', 'week_chop']:
                flow_stages.append('classification')
            elif intent == 'recommend_trades':
                flow_stages.append('trading')
            elif intent == 'check_performance':
                flow_stages.append('monitoring')
        
        self.current_context['conversation_flow'] = flow_stages
    
    def _determine_user_expertise(self, conversation_history: List[Dict[str, Any]], 
                                protocol_state: Dict[str, Any]) -> None:
        """
        Determine user expertise level based on conversation patterns.
        
        Args:
            conversation_history: The conversation history
            protocol_state: The current protocol state
        """
        expertise_indicators = {
            'beginner': 0,
            'intermediate': 0,
            'advanced': 0
        }
        
        # Analyze user messages for expertise indicators
        for message in conversation_history:
            if message['role'] == 'user':
                content = message['content'].lower()
                
                # Beginner indicators
                if any(word in content for word in ['explain', 'what is', 'how does', 'help', 'new to']):
                    expertise_indicators['beginner'] += 1
                
                # Intermediate indicators
                if any(word in content for word in ['delta', 'options', 'premium', 'volatility']):
                    expertise_indicators['intermediate'] += 1
                
                # Advanced indicators
                if any(word in content for word in ['atr', 'theta', 'gamma', 'vega', 'implied volatility']):
                    expertise_indicators['advanced'] += 1
        
        # Determine expertise level
        max_score = max(expertise_indicators.values())
        if max_score > 0:
            for level, score in expertise_indicators.items():
                if score == max_score:
                    self.current_context['user_expertise_level'] = level
                    break
    
    def _update_recent_topics(self, conversation_history: List[Dict[str, Any]]) -> None:
        """
        Update recent topics based on conversation history.
        
        Args:
            conversation_history: The conversation history
        """
        topics = []
        
        # Analyze last 5 messages for topics
        recent_messages = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
        
        for message in recent_messages:
            if message['role'] == 'user':
                content = message['content'].lower()
                
                if any(word in content for word in ['account', 'setup', 'initialize']):
                    topics.append('account_setup')
                elif any(word in content for word in ['fork', 'split', 'threshold']):
                    topics.append('forking')
                elif any(word in content for word in ['week', 'classify', 'green', 'red', 'chop']):
                    topics.append('week_classification')
                elif any(word in content for word in ['trade', 'recommend', 'position']):
                    topics.append('trading')
                elif any(word in content for word in ['performance', 'balance', 'profit']):
                    topics.append('performance')
        
        self.current_context['recent_topics'] = list(set(topics))  # Remove duplicates
    
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
            fork_threshold = ALLUSEParameters.FORK_THRESHOLD
            
            for i, balance in enumerate(gen_accounts):
                if balance >= fork_threshold:
                    pending_decisions.append({
                        'type': 'account_fork',
                        'priority': 'medium',
                        'description': f'Gen-Acc {i} ready for forking (${balance:,.2f})'
                    })
        
        self.current_context['pending_decisions'] = pending_decisions
    
    def get_context_summary(self) -> str:
        """
        Get a human-readable summary of the current context.
        
        Returns:
            String containing context summary
        """
        summary_parts = []
        
        # Conversation stage
        if 'setup' in self.current_context['conversation_flow']:
            summary_parts.append("User has completed account setup")
        elif 'learning' in self.current_context['conversation_flow']:
            summary_parts.append("User is learning about the protocol")
        else:
            summary_parts.append("User is in initial conversation stage")
        
        # Account status
        if self.current_context['has_accounts']:
            summary_parts.append("Accounts are configured")
        else:
            summary_parts.append("Accounts need to be set up")
        
        # Week classification status
        if self.current_context['week_classified']:
            summary_parts.append("Week has been classified")
        else:
            summary_parts.append("Week classification pending")
        
        # User expertise
        expertise = self.current_context['user_expertise_level']
        summary_parts.append(f"User expertise level: {expertise}")
        
        # Pending decisions
        pending_count = len(self.current_context['pending_decisions'])
        if pending_count > 0:
            summary_parts.append(f"{pending_count} pending decision(s)")
        
        return "; ".join(summary_parts)


class EnhancedCognitiveFramework:
    """
    Enhanced cognitive framework that integrates all enhanced components.
    
    This class provides the complete enhanced cognitive capabilities
    for the ALL-USE agent.
    """
    
    def __init__(self):
        """Initialize the enhanced cognitive framework."""
        self.parameters = ALLUSEParameters
        self.intent_detector = EnhancedIntentDetector()
        self.entity_extractor = EnhancedEntityExtractor()
        self.context_manager = EnhancedContextManager()
        logger.info("Enhanced cognitive framework initialized")
    
    def process_input(self, user_input: str, conversation_context: Dict[str, Any], 
                     protocol_state: Dict[str, Any], user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user input using enhanced cognitive capabilities.
        
        Args:
            user_input: The input text from the user
            conversation_context: The current conversation context
            protocol_state: The current protocol state
            user_preferences: The user's preferences
            
        Returns:
            Dict containing processed input information
        """
        logger.info(f"Processing input with enhanced framework: {user_input[:50]}...")
        
        # Detect intent using enhanced detector
        intent = self.intent_detector.detect_intent(user_input, conversation_context)
        
        # Extract entities using enhanced extractor
        entities = self.entity_extractor.extract_entities(user_input)
        
        # Update context using enhanced manager
        enhanced_context = self.context_manager.update_context(
            conversation_context.get('conversation_history', []),
            protocol_state,
            user_preferences
        )
        
        # Create processed input with enhanced information
        processed_input = {
            'raw_input': user_input,
            'intent': intent,
            'entities': entities,
            'context': enhanced_context,
            'context_summary': self.context_manager.get_context_summary()
        }
        
        logger.info(f"Enhanced processing complete. Intent: {intent}, Entities: {list(entities.keys())}")
        return processed_input
    
    def make_decision(self, processed_input: Dict[str, Any], protocol_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make enhanced decision based on processed input and protocol state.
        
        Args:
            processed_input: The processed input from process_input
            protocol_state: The current protocol state
            
        Returns:
            Dict containing decision information
        """
        intent = processed_input['intent']
        entities = processed_input['entities']
        context = processed_input['context']
        
        logger.info(f"Making enhanced decision for intent: {intent}")
        
        # Enhanced decision logic with more sophisticated handling
        decision = {
            'intent': intent,
            'action': None,
            'response_type': None,
            'parameters': {},
            'rationale': '',
            'confidence': 0.0,
            'context_used': context
        }
        
        # Handle different intents with enhanced logic
        if intent == 'greeting':
            decision.update({
                'action': 'respond',
                'response_type': 'greeting',
                'confidence': 0.9,
                'rationale': 'Responding to user greeting'
            })
        
        elif intent in ['explain_protocol', 'explain_accounts', 'explain_forking']:
            detail_level = self._determine_detail_level(context, entities)
            decision.update({
                'action': 'explain',
                'response_type': intent,
                'parameters': {'detail_level': detail_level},
                'confidence': 0.8,
                'rationale': f'User requested {intent.replace("_", " ")}'
            })
        
        elif intent == 'setup_accounts':
            amount = entities.get('amount', 0)
            decision.update({
                'action': 'setup',
                'response_type': 'account_setup',
                'parameters': {
                    'initial_investment': amount,
                    'allocation': {
                        'GEN_ACC': self.parameters.INITIAL_ALLOCATION['GEN_ACC'],
                        'REV_ACC': self.parameters.INITIAL_ALLOCATION['REV_ACC'],
                        'COM_ACC': self.parameters.INITIAL_ALLOCATION['COM_ACC']
                    },
                    'cash_buffer': self.parameters.CASH_BUFFER
                },
                'confidence': 0.9 if amount > 0 else 0.6,
                'rationale': 'User requested account setup'
            })
        
        elif intent in ['classify_week', 'week_green', 'week_red', 'week_chop']:
            week_type = entities.get('week_type')
            if intent.startswith('week_'):
                week_type = intent.split('_')[1].capitalize()
                if week_type == 'Chop':
                    week_type = 'Chop'
            
            decision.update({
                'action': 'classify',
                'response_type': 'week_classification',
                'parameters': {
                    'week_type': week_type,
                    'market_data': entities.get('market_data', {})
                },
                'confidence': 0.8 if week_type else 0.5,
                'rationale': 'User provided or requested week classification'
            })
        
        elif intent == 'recommend_trades':
            if not context['has_accounts']:
                decision.update({
                    'action': 'clarify',
                    'response_type': 'account_setup_required',
                    'confidence': 0.9,
                    'rationale': 'Account setup required before trade recommendations'
                })
            elif not context['week_classified']:
                decision.update({
                    'action': 'clarify',
                    'response_type': 'week_classification_required',
                    'confidence': 0.9,
                    'rationale': 'Week classification required before trade recommendations'
                })
            else:
                decision.update({
                    'action': 'recommend',
                    'response_type': 'trade_recommendations',
                    'parameters': {
                        'week_classification': protocol_state['week_classification'],
                        'account_balances': protocol_state['accounts'],
                        'account_types': entities.get('account_types', ['GEN_ACC', 'REV_ACC', 'COM_ACC'])
                    },
                    'confidence': 0.8,
                    'rationale': 'Generating trade recommendations based on protocol'
                })
        
        elif intent == 'check_performance':
            if not context['has_accounts']:
                decision.update({
                    'action': 'clarify',
                    'response_type': 'account_setup_required',
                    'confidence': 0.9,
                    'rationale': 'Account setup required before checking performance'
                })
            else:
                decision.update({
                    'action': 'report',
                    'response_type': 'performance_report',
                    'parameters': {
                        'account_balances': protocol_state['accounts'],
                        'detail_level': self._determine_detail_level(context, entities)
                    },
                    'confidence': 0.8,
                    'rationale': 'Generating performance report'
                })
        
        elif intent == 'help':
            decision.update({
                'action': 'help',
                'response_type': 'help_menu',
                'parameters': {
                    'context': context,
                    'expertise_level': context.get('user_expertise_level', 'beginner')
                },
                'confidence': 0.9,
                'rationale': 'User requested help or assistance'
            })
        
        elif intent == 'status':
            decision.update({
                'action': 'status',
                'response_type': 'status_report',
                'parameters': {
                    'context': context,
                    'protocol_state': protocol_state
                },
                'confidence': 0.9,
                'rationale': 'User requested current status'
            })
        
        else:
            decision.update({
                'action': 'clarify',
                'response_type': 'clarification',
                'confidence': 0.3,
                'rationale': 'Intent not recognized, requesting clarification'
            })
        
        logger.info(f"Enhanced decision made: {decision['action']} - {decision['response_type']} (confidence: {decision['confidence']})")
        return decision
    
    def _determine_detail_level(self, context: Dict[str, Any], entities: Dict[str, Any]) -> str:
        """
        Determine the appropriate detail level for responses.
        
        Args:
            context: The current context
            entities: Extracted entities
            
        Returns:
            String representing the detail level
        """
        expertise_level = context.get('user_expertise_level', 'beginner')
        
        # Check if user explicitly requested detail level
        if 'detail' in entities.get('time_periods', []):
            return 'detailed'
        elif 'overview' in entities.get('time_periods', []):
            return 'overview'
        
        # Determine based on expertise level
        if expertise_level == 'beginner':
            return 'overview'
        elif expertise_level == 'intermediate':
            return 'detailed'
        else:  # advanced
            return 'technical'

