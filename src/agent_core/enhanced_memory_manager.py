"""
ALL-USE Enhanced Memory Manager Module

This module implements enhanced memory management capabilities for the ALL-USE agent,
including advanced conversation tracking, protocol state management, and user profiling.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('all_use_enhanced_memory.log')
    ]
)

logger = logging.getLogger('all_use_enhanced_memory')

class EnhancedConversationMemory:
    """
    Enhanced conversation memory system with advanced tracking capabilities.
    
    This class provides sophisticated conversation memory management including
    topic tracking, sentiment analysis, and conversation pattern recognition.
    """
    
    def __init__(self, max_history: int = 200):
        """
        Initialize the enhanced conversation memory system.
        
        Args:
            max_history: Maximum number of conversation turns to store
        """
        self.max_history = max_history
        self.history = []
        self.topic_history = defaultdict(list)
        self.conversation_patterns = {
            'question_count': 0,
            'explanation_requests': 0,
            'setup_attempts': 0,
            'clarification_requests': 0
        }
        self.session_metrics = {
            'start_time': datetime.now(),
            'total_messages': 0,
            'user_messages': 0,
            'agent_messages': 0,
            'average_response_length': 0
        }
        logger.info("Enhanced conversation memory system initialized")
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a message to the conversation history with enhanced metadata.
        
        Args:
            role: The role of the message sender ('user' or 'agent')
            content: The content of the message
            metadata: Optional metadata about the message
        """
        timestamp = datetime.now()
        
        message = {
            'role': role,
            'content': content,
            'timestamp': timestamp.isoformat(),
            'message_id': f"{role}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}",
            'length': len(content),
            'metadata': metadata or {}
        }
        
        # Add enhanced metadata
        if role == 'user':
            message['metadata'].update({
                'word_count': len(content.split()),
                'has_question': '?' in content,
                'has_numbers': any(char.isdigit() for char in content),
                'sentiment': self._analyze_sentiment(content)
            })
        elif role == 'agent':
            message['metadata'].update({
                'response_time': self._calculate_response_time(),
                'explanation_type': self._classify_explanation_type(content)
            })
        
        self.history.append(message)
        
        # Update conversation patterns
        self._update_conversation_patterns(message)
        
        # Update session metrics
        self._update_session_metrics(message)
        
        # Extract and track topics
        self._extract_and_track_topics(message)
        
        # Trim history if it exceeds max_history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        logger.info(f"Added {role} message to enhanced conversation memory")
    
    def get_conversation_summary(self, last_n_messages: int = 10) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the conversation.
        
        Args:
            last_n_messages: Number of recent messages to analyze
            
        Returns:
            Dict containing conversation summary
        """
        recent_messages = self.history[-last_n_messages:] if len(self.history) > last_n_messages else self.history
        
        summary = {
            'total_messages': len(self.history),
            'recent_messages_count': len(recent_messages),
            'conversation_duration': self._calculate_conversation_duration(),
            'dominant_topics': self._get_dominant_topics(),
            'conversation_patterns': self.conversation_patterns.copy(),
            'session_metrics': self.session_metrics.copy(),
            'user_engagement_level': self._assess_user_engagement(),
            'conversation_stage': self._determine_conversation_stage()
        }
        
        return summary
    
    def search_by_topic(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search conversation history by topic.
        
        Args:
            topic: The topic to search for
            limit: Maximum number of results to return
            
        Returns:
            List of messages related to the topic
        """
        topic_messages = self.topic_history.get(topic.lower(), [])
        return topic_messages[-limit:] if len(topic_messages) > limit else topic_messages
    
    def get_conversation_flow(self) -> List[Dict[str, Any]]:
        """
        Get the conversation flow with topic transitions.
        
        Returns:
            List of conversation flow stages
        """
        flow = []
        current_topic = None
        topic_start = None
        
        for message in self.history:
            message_topics = message['metadata'].get('topics', [])
            
            if message_topics:
                new_topic = message_topics[0]  # Primary topic
                
                if new_topic != current_topic:
                    if current_topic:
                        flow.append({
                            'topic': current_topic,
                            'start_time': topic_start,
                            'end_time': message['timestamp'],
                            'duration': self._calculate_duration(topic_start, message['timestamp']),
                            'message_count': len([m for m in self.history 
                                                if m['timestamp'] >= topic_start and 
                                                   m['timestamp'] < message['timestamp']])
                        })
                    
                    current_topic = new_topic
                    topic_start = message['timestamp']
        
        # Add the last topic if exists
        if current_topic and topic_start:
            flow.append({
                'topic': current_topic,
                'start_time': topic_start,
                'end_time': self.history[-1]['timestamp'],
                'duration': self._calculate_duration(topic_start, self.history[-1]['timestamp']),
                'message_count': len([m for m in self.history if m['timestamp'] >= topic_start])
            })
        
        return flow
    
    def _analyze_sentiment(self, content: str) -> str:
        """
        Analyze sentiment of user message (simple implementation).
        
        Args:
            content: The message content
            
        Returns:
            String representing sentiment
        """
        positive_words = ['good', 'great', 'excellent', 'perfect', 'amazing', 'love', 'like', 'thanks']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'confused', 'frustrated']
        
        content_lower = content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _calculate_response_time(self) -> float:
        """
        Calculate response time for agent messages.
        
        Returns:
            Response time in seconds
        """
        if len(self.history) < 1:
            return 0.0
        
        last_message = self.history[-1]
        if last_message['role'] == 'user':
            # Time since last user message
            return (datetime.now() - datetime.fromisoformat(last_message['timestamp'])).total_seconds()
        
        return 0.0
    
    def _classify_explanation_type(self, content: str) -> str:
        """
        Classify the type of explanation provided by the agent.
        
        Args:
            content: The agent response content
            
        Returns:
            String representing explanation type
        """
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['protocol', 'system', 'strategy']):
            return 'protocol_explanation'
        elif any(word in content_lower for word in ['account', 'gen-acc', 'rev-acc', 'com-acc']):
            return 'account_explanation'
        elif any(word in content_lower for word in ['fork', 'split', 'threshold']):
            return 'forking_explanation'
        elif any(word in content_lower for word in ['trade', 'recommend', 'position']):
            return 'trading_guidance'
        elif any(word in content_lower for word in ['performance', 'balance', 'profit']):
            return 'performance_report'
        else:
            return 'general_response'
    
    def _update_conversation_patterns(self, message: Dict[str, Any]) -> None:
        """
        Update conversation patterns based on the message.
        
        Args:
            message: The message to analyze
        """
        if message['role'] == 'user':
            if message['metadata'].get('has_question'):
                self.conversation_patterns['question_count'] += 1
            
            content_lower = message['content'].lower()
            if any(word in content_lower for word in ['explain', 'tell me', 'what is']):
                self.conversation_patterns['explanation_requests'] += 1
            elif any(word in content_lower for word in ['setup', 'create', 'start']):
                self.conversation_patterns['setup_attempts'] += 1
            elif any(word in content_lower for word in ['clarify', 'confused', 'understand']):
                self.conversation_patterns['clarification_requests'] += 1
    
    def _update_session_metrics(self, message: Dict[str, Any]) -> None:
        """
        Update session metrics based on the message.
        
        Args:
            message: The message to analyze
        """
        self.session_metrics['total_messages'] += 1
        
        if message['role'] == 'user':
            self.session_metrics['user_messages'] += 1
        else:
            self.session_metrics['agent_messages'] += 1
        
        # Update average response length
        total_length = sum(msg['length'] for msg in self.history)
        self.session_metrics['average_response_length'] = total_length / len(self.history)
    
    def _extract_and_track_topics(self, message: Dict[str, Any]) -> None:
        """
        Extract and track topics from the message.
        
        Args:
            message: The message to analyze
        """
        content_lower = message['content'].lower()
        topics = []
        
        # Define topic keywords
        topic_keywords = {
            'account_setup': ['setup', 'create', 'initialize', 'account'],
            'protocol_explanation': ['protocol', 'explain', 'system', 'strategy'],
            'forking': ['fork', 'split', 'threshold', '50k'],
            'week_classification': ['week', 'classify', 'green', 'red', 'chop'],
            'trading': ['trade', 'recommend', 'position', 'buy', 'sell'],
            'performance': ['performance', 'balance', 'profit', 'loss'],
            'reinvestment': ['reinvest', 'quarterly', 'contracts', 'leaps']
        }
        
        # Extract topics based on keywords
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)
                self.topic_history[topic].append(message)
        
        # Add topics to message metadata
        message['metadata']['topics'] = topics
    
    def _calculate_conversation_duration(self) -> float:
        """
        Calculate the total conversation duration in minutes.
        
        Returns:
            Duration in minutes
        """
        if not self.history:
            return 0.0
        
        start_time = datetime.fromisoformat(self.history[0]['timestamp'])
        end_time = datetime.fromisoformat(self.history[-1]['timestamp'])
        
        return (end_time - start_time).total_seconds() / 60.0
    
    def _get_dominant_topics(self, top_n: int = 3) -> List[Tuple[str, int]]:
        """
        Get the most discussed topics in the conversation.
        
        Args:
            top_n: Number of top topics to return
            
        Returns:
            List of tuples (topic, message_count)
        """
        topic_counts = {topic: len(messages) for topic, messages in self.topic_history.items()}
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_topics[:top_n]
    
    def _assess_user_engagement(self) -> str:
        """
        Assess the user's engagement level based on conversation patterns.
        
        Returns:
            String representing engagement level
        """
        if not self.history:
            return 'unknown'
        
        user_messages = [msg for msg in self.history if msg['role'] == 'user']
        
        if not user_messages:
            return 'low'
        
        # Calculate engagement metrics
        avg_message_length = sum(msg['length'] for msg in user_messages) / len(user_messages)
        question_ratio = self.conversation_patterns['question_count'] / len(user_messages)
        
        if avg_message_length > 50 and question_ratio > 0.3:
            return 'high'
        elif avg_message_length > 20 and question_ratio > 0.1:
            return 'medium'
        else:
            return 'low'
    
    def _determine_conversation_stage(self) -> str:
        """
        Determine the current stage of the conversation.
        
        Returns:
            String representing conversation stage
        """
        if not self.history:
            return 'initial'
        
        # Analyze recent topics
        recent_topics = []
        for message in self.history[-5:]:  # Last 5 messages
            recent_topics.extend(message['metadata'].get('topics', []))
        
        if 'trading' in recent_topics or 'performance' in recent_topics:
            return 'active_management'
        elif 'account_setup' in recent_topics:
            return 'setup'
        elif 'protocol_explanation' in recent_topics:
            return 'learning'
        else:
            return 'exploration'
    
    def _calculate_duration(self, start_time: str, end_time: str) -> float:
        """
        Calculate duration between two timestamps in minutes.
        
        Args:
            start_time: Start timestamp (ISO format)
            end_time: End timestamp (ISO format)
            
        Returns:
            Duration in minutes
        """
        start = datetime.fromisoformat(start_time)
        end = datetime.fromisoformat(end_time)
        
        return (end - start).total_seconds() / 60.0


class EnhancedProtocolStateMemory:
    """
    Enhanced protocol state memory with advanced tracking and analytics.
    
    This class provides sophisticated protocol state management including
    performance tracking, decision analysis, and predictive insights.
    """
    
    def __init__(self):
        """Initialize the enhanced protocol state memory system."""
        self.state = {
            'accounts': {
                'GEN_ACC': [],
                'REV_ACC': [],
                'COM_ACC': []
            },
            'week_classification': None,
            'last_decision': None,
            'decision_history': [],
            'fork_history': [],
            'merge_history': [],
            'reinvestment_history': [],
            'performance_history': [],
            'risk_metrics': {},
            'protocol_adherence': {
                'total_decisions': 0,
                'protocol_compliant': 0,
                'adherence_rate': 1.0
            }
        }
        
        # Enhanced tracking
        self.account_performance = defaultdict(list)
        self.weekly_returns = defaultdict(list)
        self.decision_patterns = defaultdict(int)
        
        logger.info("Enhanced protocol state memory system initialized")
    
    def record_performance_snapshot(self) -> None:
        """
        Record a performance snapshot of all accounts.
        """
        timestamp = datetime.now().isoformat()
        
        # Calculate current balances
        gen_acc_total = sum(self.state['accounts']['GEN_ACC'])
        rev_acc_total = sum(self.state['accounts']['REV_ACC'])
        com_acc_total = sum(self.state['accounts']['COM_ACC'])
        total_balance = gen_acc_total + rev_acc_total + com_acc_total
        
        snapshot = {
            'timestamp': timestamp,
            'total_balance': total_balance,
            'account_balances': {
                'GEN_ACC': gen_acc_total,
                'REV_ACC': rev_acc_total,
                'COM_ACC': com_acc_total
            },
            'account_counts': {
                'GEN_ACC': len(self.state['accounts']['GEN_ACC']),
                'REV_ACC': len(self.state['accounts']['REV_ACC']),
                'COM_ACC': len(self.state['accounts']['COM_ACC'])
            },
            'week_classification': self.state['week_classification']
        }
        
        self.state['performance_history'].append(snapshot)
        logger.info(f"Performance snapshot recorded: ${total_balance:,.2f}")
    
    def calculate_performance_metrics(self, period_days: int = 30) -> Dict[str, Any]:
        """
        Calculate performance metrics for a given period.
        
        Args:
            period_days: Number of days to analyze
            
        Returns:
            Dict containing performance metrics
        """
        cutoff_date = datetime.now() - timedelta(days=period_days)
        
        # Filter performance history for the period
        period_snapshots = [
            snapshot for snapshot in self.state['performance_history']
            if datetime.fromisoformat(snapshot['timestamp']) >= cutoff_date
        ]
        
        if len(period_snapshots) < 2:
            return {'error': 'Insufficient data for performance calculation'}
        
        # Calculate metrics
        start_balance = period_snapshots[0]['total_balance']
        end_balance = period_snapshots[-1]['total_balance']
        
        total_return = (end_balance - start_balance) / start_balance if start_balance > 0 else 0
        annualized_return = (1 + total_return) ** (365 / period_days) - 1
        
        # Calculate volatility (simplified)
        daily_returns = []
        for i in range(1, len(period_snapshots)):
            prev_balance = period_snapshots[i-1]['total_balance']
            curr_balance = period_snapshots[i]['total_balance']
            daily_return = (curr_balance - prev_balance) / prev_balance if prev_balance > 0 else 0
            daily_returns.append(daily_return)
        
        volatility = self._calculate_volatility(daily_returns) if daily_returns else 0
        
        metrics = {
            'period_days': period_days,
            'start_balance': start_balance,
            'end_balance': end_balance,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': annualized_return / volatility if volatility > 0 else 0,
            'max_balance': max(snapshot['total_balance'] for snapshot in period_snapshots),
            'min_balance': min(snapshot['total_balance'] for snapshot in period_snapshots),
            'snapshots_count': len(period_snapshots)
        }
        
        return metrics
    
    def analyze_decision_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in decision making.
        
        Returns:
            Dict containing decision pattern analysis
        """
        if not self.state['decision_history']:
            return {'error': 'No decision history available'}
        
        # Count decision types
        decision_types = defaultdict(int)
        intent_distribution = defaultdict(int)
        confidence_scores = []
        
        for decision in self.state['decision_history']:
            decision_types[decision.get('action', 'unknown')] += 1
            intent_distribution[decision.get('intent', 'unknown')] += 1
            
            if 'confidence' in decision:
                confidence_scores.append(decision['confidence'])
        
        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        analysis = {
            'total_decisions': len(self.state['decision_history']),
            'decision_types': dict(decision_types),
            'intent_distribution': dict(intent_distribution),
            'average_confidence': avg_confidence,
            'high_confidence_decisions': len([c for c in confidence_scores if c > 0.8]),
            'low_confidence_decisions': len([c for c in confidence_scores if c < 0.5])
        }
        
        return analysis
    
    def predict_next_fork(self) -> Dict[str, Any]:
        """
        Predict when the next account fork might occur.
        
        Returns:
            Dict containing fork prediction
        """
        gen_accounts = self.state['accounts']['GEN_ACC']
        
        if not gen_accounts:
            return {'error': 'No Gen-Acc accounts available'}
        
        # Find account closest to fork threshold
        fork_threshold = 100000  # $100K as per updated parameters
        closest_account = None
        closest_balance = 0
        closest_index = -1
        
        for i, balance in enumerate(gen_accounts):
            if balance > closest_balance and balance < fork_threshold:
                closest_balance = balance
                closest_account = balance
                closest_index = i
        
        if closest_account is None:
            # Check if any account already exceeds threshold
            for i, balance in enumerate(gen_accounts):
                if balance >= fork_threshold:
                    return {
                        'ready_for_fork': True,
                        'account_index': i,
                        'current_balance': balance,
                        'surplus': balance - fork_threshold
                    }
            
            return {'error': 'No accounts approaching fork threshold'}
        
        # Estimate time to fork based on recent performance
        amount_needed = fork_threshold - closest_balance
        
        # Simple prediction based on average weekly return
        weekly_return_rate = 0.015  # 1.5% for Gen-Acc
        estimated_weeks = amount_needed / (closest_balance * weekly_return_rate)
        
        prediction = {
            'account_index': closest_index,
            'current_balance': closest_balance,
            'amount_needed': amount_needed,
            'estimated_weeks_to_fork': estimated_weeks,
            'estimated_date': (datetime.now() + timedelta(weeks=estimated_weeks)).isoformat()
        }
        
        return prediction
    
    def get_risk_assessment(self) -> Dict[str, Any]:
        """
        Get current risk assessment of the portfolio.
        
        Returns:
            Dict containing risk assessment
        """
        total_balance = self.get_total_balance()
        
        if total_balance == 0:
            return {'error': 'No portfolio balance available'}
        
        # Calculate account concentration
        gen_acc_total = sum(self.state['accounts']['GEN_ACC'])
        rev_acc_total = sum(self.state['accounts']['REV_ACC'])
        com_acc_total = sum(self.state['accounts']['COM_ACC'])
        
        concentration = {
            'GEN_ACC': gen_acc_total / total_balance,
            'REV_ACC': rev_acc_total / total_balance,
            'COM_ACC': com_acc_total / total_balance
        }
        
        # Assess risk level based on concentration
        risk_level = 'low'
        if concentration['GEN_ACC'] > 0.6:  # More than 60% in high-risk Gen-Acc
            risk_level = 'high'
        elif concentration['GEN_ACC'] > 0.4:  # More than 40% in Gen-Acc
            risk_level = 'medium'
        
        # Calculate diversification score
        diversification_score = 1 - max(concentration.values())
        
        assessment = {
            'total_balance': total_balance,
            'concentration': concentration,
            'risk_level': risk_level,
            'diversification_score': diversification_score,
            'account_counts': {
                'GEN_ACC': len(self.state['accounts']['GEN_ACC']),
                'REV_ACC': len(self.state['accounts']['REV_ACC']),
                'COM_ACC': len(self.state['accounts']['COM_ACC'])
            },
            'recommendations': self._generate_risk_recommendations(concentration, risk_level)
        }
        
        return assessment
    
    def _calculate_volatility(self, returns: List[float]) -> float:
        """
        Calculate volatility of returns.
        
        Args:
            returns: List of return values
            
        Returns:
            Volatility (standard deviation)
        """
        if len(returns) < 2:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        
        return variance ** 0.5
    
    def _generate_risk_recommendations(self, concentration: Dict[str, float], risk_level: str) -> List[str]:
        """
        Generate risk management recommendations.
        
        Args:
            concentration: Account concentration ratios
            risk_level: Current risk level
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if risk_level == 'high':
            recommendations.append("Consider rebalancing to reduce Gen-Acc concentration")
            recommendations.append("Increase allocation to Rev-Acc and Com-Acc for stability")
        
        if concentration['GEN_ACC'] > 0.5:
            recommendations.append("Gen-Acc concentration is high - monitor closely")
        
        if concentration['COM_ACC'] < 0.2:
            recommendations.append("Consider increasing Com-Acc allocation for long-term growth")
        
        if len(self.state['accounts']['GEN_ACC']) > 5:
            recommendations.append("Multiple Gen-Acc accounts - consider consolidation strategy")
        
        return recommendations


class EnhancedUserPreferencesMemory:
    """
    Enhanced user preferences memory with learning capabilities.
    
    This class provides sophisticated user preference tracking including
    behavioral analysis, preference learning, and personalization.
    """
    
    def __init__(self):
        """Initialize the enhanced user preferences memory system."""
        self.preferences = {
            'risk_tolerance': None,
            'income_priority': None,
            'communication_style': None,
            'notification_preferences': {},
            'custom_settings': {},
            'learning_preferences': {
                'detail_level': 'overview',
                'explanation_style': 'educational',
                'preferred_topics': []
            },
            'behavioral_patterns': {
                'question_frequency': 0,
                'setup_speed': 'normal',
                'decision_confidence': 'medium',
                'engagement_level': 'medium'
            }
        }
        
        # Learning system
        self.interaction_history = []
        self.preference_confidence = defaultdict(float)
        
        logger.info("Enhanced user preferences memory system initialized")
    
    def learn_from_interaction(self, interaction_data: Dict[str, Any]) -> None:
        """
        Learn user preferences from interaction data.
        
        Args:
            interaction_data: Data about the user interaction
        """
        self.interaction_history.append({
            'timestamp': datetime.now().isoformat(),
            'data': interaction_data
        })
        
        # Update behavioral patterns
        self._update_behavioral_patterns(interaction_data)
        
        # Infer preferences
        self._infer_preferences(interaction_data)
        
        logger.info("User preferences updated from interaction")
    
    def get_personalized_settings(self) -> Dict[str, Any]:
        """
        Get personalized settings based on learned preferences.
        
        Returns:
            Dict containing personalized settings
        """
        settings = {
            'response_style': self._determine_response_style(),
            'detail_level': self._determine_detail_level(),
            'explanation_approach': self._determine_explanation_approach(),
            'notification_frequency': self._determine_notification_frequency(),
            'risk_guidance_level': self._determine_risk_guidance_level()
        }
        
        return settings
    
    def _update_behavioral_patterns(self, interaction_data: Dict[str, Any]) -> None:
        """
        Update behavioral patterns based on interaction data.
        
        Args:
            interaction_data: Data about the user interaction
        """
        # Update question frequency
        if interaction_data.get('has_question'):
            self.preferences['behavioral_patterns']['question_frequency'] += 1
        
        # Update setup speed
        if interaction_data.get('intent') == 'setup_accounts':
            setup_time = interaction_data.get('response_time', 0)
            if setup_time < 30:  # Quick setup
                self.preferences['behavioral_patterns']['setup_speed'] = 'fast'
            elif setup_time > 120:  # Slow setup
                self.preferences['behavioral_patterns']['setup_speed'] = 'deliberate'
        
        # Update decision confidence
        confidence = interaction_data.get('confidence', 0.5)
        if confidence > 0.8:
            self.preferences['behavioral_patterns']['decision_confidence'] = 'high'
        elif confidence < 0.4:
            self.preferences['behavioral_patterns']['decision_confidence'] = 'low'
    
    def _infer_preferences(self, interaction_data: Dict[str, Any]) -> None:
        """
        Infer user preferences from interaction data.
        
        Args:
            interaction_data: Data about the user interaction
        """
        # Infer risk tolerance
        if interaction_data.get('intent') == 'setup_accounts':
            amount = interaction_data.get('amount', 0)
            if amount > 500000:
                self.preferences['risk_tolerance'] = 'high'
                self.preference_confidence['risk_tolerance'] += 0.3
            elif amount > 100000:
                self.preferences['risk_tolerance'] = 'medium'
                self.preference_confidence['risk_tolerance'] += 0.2
            else:
                self.preferences['risk_tolerance'] = 'conservative'
                self.preference_confidence['risk_tolerance'] += 0.1
        
        # Infer communication style
        message_length = interaction_data.get('message_length', 0)
        if message_length > 100:
            self.preferences['communication_style'] = 'detailed'
            self.preference_confidence['communication_style'] += 0.1
        elif message_length < 20:
            self.preferences['communication_style'] = 'concise'
            self.preference_confidence['communication_style'] += 0.1
        
        # Infer learning preferences
        if interaction_data.get('intent') in ['explain_protocol', 'explain_accounts']:
            self.preferences['learning_preferences']['preferred_topics'].append(
                interaction_data.get('intent')
            )
    
    def _determine_response_style(self) -> str:
        """
        Determine the appropriate response style for the user.
        
        Returns:
            String representing response style
        """
        communication_style = self.preferences.get('communication_style')
        behavioral_patterns = self.preferences.get('behavioral_patterns', {})
        
        if communication_style == 'detailed' or behavioral_patterns.get('question_frequency', 0) > 5:
            return 'comprehensive'
        elif communication_style == 'concise':
            return 'brief'
        else:
            return 'balanced'
    
    def _determine_detail_level(self) -> str:
        """
        Determine the appropriate detail level for explanations.
        
        Returns:
            String representing detail level
        """
        learning_prefs = self.preferences.get('learning_preferences', {})
        behavioral_patterns = self.preferences.get('behavioral_patterns', {})
        
        if behavioral_patterns.get('question_frequency', 0) > 10:
            return 'detailed'
        elif learning_prefs.get('detail_level') == 'overview':
            return 'overview'
        else:
            return 'standard'
    
    def _determine_explanation_approach(self) -> str:
        """
        Determine the appropriate explanation approach.
        
        Returns:
            String representing explanation approach
        """
        risk_tolerance = self.preferences.get('risk_tolerance')
        
        if risk_tolerance == 'conservative':
            return 'cautious'
        elif risk_tolerance == 'high':
            return 'confident'
        else:
            return 'balanced'
    
    def _determine_notification_frequency(self) -> str:
        """
        Determine the appropriate notification frequency.
        
        Returns:
            String representing notification frequency
        """
        engagement_level = self.preferences.get('behavioral_patterns', {}).get('engagement_level', 'medium')
        
        if engagement_level == 'high':
            return 'frequent'
        elif engagement_level == 'low':
            return 'minimal'
        else:
            return 'moderate'
    
    def _determine_risk_guidance_level(self) -> str:
        """
        Determine the appropriate level of risk guidance.
        
        Returns:
            String representing risk guidance level
        """
        risk_tolerance = self.preferences.get('risk_tolerance')
        decision_confidence = self.preferences.get('behavioral_patterns', {}).get('decision_confidence', 'medium')
        
        if risk_tolerance == 'conservative' or decision_confidence == 'low':
            return 'high'
        elif risk_tolerance == 'high' and decision_confidence == 'high':
            return 'minimal'
        else:
            return 'moderate'


class EnhancedMemoryManager:
    """
    Enhanced memory manager that coordinates all enhanced memory systems.
    
    This class provides comprehensive memory management with advanced
    analytics, learning capabilities, and performance optimization.
    """
    
    def __init__(self):
        """Initialize the enhanced memory manager."""
        self.conversation_memory = EnhancedConversationMemory()
        self.protocol_state_memory = EnhancedProtocolStateMemory()
        self.user_preferences_memory = EnhancedUserPreferencesMemory()
        
        # Cross-system analytics
        self.session_analytics = {
            'start_time': datetime.now(),
            'total_interactions': 0,
            'performance_snapshots': 0,
            'preference_updates': 0
        }
        
        logger.info("Enhanced memory manager initialized")
    
    def process_interaction(self, user_input: str, agent_response: str, 
                          decision_data: Dict[str, Any]) -> None:
        """
        Process a complete interaction through all memory systems.
        
        Args:
            user_input: The user's input
            agent_response: The agent's response
            decision_data: Data about the decision made
        """
        # Add messages to conversation memory
        self.conversation_memory.add_message('user', user_input, {
            'intent': decision_data.get('intent'),
            'entities': decision_data.get('entities', {}),
            'confidence': decision_data.get('confidence', 0.5)
        })
        
        self.conversation_memory.add_message('agent', agent_response, {
            'action': decision_data.get('action'),
            'response_type': decision_data.get('response_type')
        })
        
        # Update protocol state if needed
        if decision_data.get('action') in ['setup', 'fork', 'classify']:
            self.protocol_state_memory.add_decision(decision_data)
        
        # Learn user preferences
        interaction_data = {
            'intent': decision_data.get('intent'),
            'confidence': decision_data.get('confidence', 0.5),
            'message_length': len(user_input),
            'has_question': '?' in user_input,
            'amount': decision_data.get('parameters', {}).get('initial_investment', 0)
        }
        
        self.user_preferences_memory.learn_from_interaction(interaction_data)
        
        # Update session analytics
        self.session_analytics['total_interactions'] += 1
        
        logger.info("Interaction processed through enhanced memory systems")
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status across all memory systems.
        
        Returns:
            Dict containing comprehensive status
        """
        conversation_summary = self.conversation_memory.get_conversation_summary()
        protocol_metrics = self.protocol_state_memory.calculate_performance_metrics()
        user_settings = self.user_preferences_memory.get_personalized_settings()
        
        status = {
            'session_analytics': self.session_analytics.copy(),
            'conversation_summary': conversation_summary,
            'protocol_metrics': protocol_metrics,
            'user_settings': user_settings,
            'memory_health': self._assess_memory_health()
        }
        
        return status
    
    def _assess_memory_health(self) -> Dict[str, Any]:
        """
        Assess the health and performance of memory systems.
        
        Returns:
            Dict containing memory health assessment
        """
        health = {
            'conversation_memory': {
                'message_count': len(self.conversation_memory.history),
                'topic_diversity': len(self.conversation_memory.topic_history),
                'status': 'healthy'
            },
            'protocol_memory': {
                'decision_count': len(self.protocol_state_memory.state['decision_history']),
                'performance_snapshots': len(self.protocol_state_memory.state['performance_history']),
                'status': 'healthy'
            },
            'preferences_memory': {
                'interaction_count': len(self.user_preferences_memory.interaction_history),
                'confidence_level': sum(self.user_preferences_memory.preference_confidence.values()) / 
                                  max(len(self.user_preferences_memory.preference_confidence), 1),
                'status': 'healthy'
            }
        }
        
        return health

