"""
ALL-USE Enhanced Agent Core Tests

This module contains comprehensive tests for the enhanced ALL-USE agent core components
including enhanced cognitive framework, enhanced memory manager, and integration tests.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agent_core.enhanced_cognitive_framework import (
    EnhancedIntentDetector, EnhancedEntityExtractor, EnhancedContextManager, EnhancedCognitiveFramework
)
from src.agent_core.enhanced_memory_manager import (
    EnhancedConversationMemory, EnhancedProtocolStateMemory, 
    EnhancedUserPreferencesMemory, EnhancedMemoryManager
)
from src.agent_core.enhanced_agent import EnhancedALLUSEAgent, AgentInterface
from src.protocol_engine.all_use_parameters import ALLUSEParameters

class TestEnhancedIntentDetector(unittest.TestCase):
    """Test cases for EnhancedIntentDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = EnhancedIntentDetector()
    
    def test_greeting_detection(self):
        """Test greeting intent detection."""
        test_cases = [
            "Hello there!",
            "Hi, how are you?",
            "Good morning",
            "Hey what's up"
        ]
        
        for test_input in test_cases:
            intent = self.detector.detect_intent(test_input, {})
            self.assertEqual(intent, 'greeting', f"Failed for input: {test_input}")
    
    def test_protocol_explanation_detection(self):
        """Test protocol explanation intent detection."""
        test_cases = [
            "Explain the ALL-USE protocol",
            "Tell me about the system",
            "How does the protocol work?",
            "What is the ALL-USE strategy?"
        ]
        
        for test_input in test_cases:
            intent = self.detector.detect_intent(test_input, {})
            self.assertEqual(intent, 'explain_protocol', f"Failed for input: {test_input}")
    
    def test_account_setup_detection(self):
        """Test account setup intent detection."""
        test_cases = [
            "Setup my accounts",
            "Create accounts with $300,000",
            "I want to start investing",
            "Initialize my portfolio"
        ]
        
        for test_input in test_cases:
            intent = self.detector.detect_intent(test_input, {})
            self.assertEqual(intent, 'setup_accounts', f"Failed for input: {test_input}")
    
    def test_context_adjustments(self):
        """Test context-based intent adjustments."""
        # Test setup boost when no accounts exist
        context = {'has_accounts': False}
        intent = self.detector.detect_intent("I want to invest", context)
        self.assertEqual(intent, 'setup_accounts')
        
        # Test classification boost when accounts exist but week not classified
        context = {'has_accounts': True, 'week_classified': False}
        intent = self.detector.detect_intent("What kind of week is this?", context)
        self.assertEqual(intent, 'classify_week')
    
    def test_week_type_detection(self):
        """Test week type intent detection."""
        test_cases = [
            ("This is a green week", 'classify_week'),
            ("Market looks red today", 'classify_week'),
            ("Very choppy market", 'week_chop')  # This correctly detects as week_chop
        ]
        
        for test_input, expected_intent in test_cases:
            intent = self.detector.detect_intent(test_input, {})
            self.assertEqual(intent, expected_intent, f"Failed for input: {test_input}")


class TestEnhancedEntityExtractor(unittest.TestCase):
    """Test cases for EnhancedEntityExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = EnhancedEntityExtractor()
    
    def test_monetary_amount_extraction(self):
        """Test monetary amount extraction."""
        test_cases = [
            ("I have $300,000 to invest", 300000),
            ("I want to invest 500k", 500000),
            ("I have 2 million dollars", 2000000),
            ("$50,000 is my budget", 50000)
        ]
        
        for test_input, expected_amount in test_cases:
            entities = self.extractor.extract_entities(test_input)
            self.assertEqual(entities.get('amount'), expected_amount, f"Failed for input: {test_input}")
    
    def test_percentage_extraction(self):
        """Test percentage extraction."""
        test_cases = [
            "I want 75% in contracts",
            "25% allocation to LEAPS",
            "1.5 percent weekly return"
        ]
        
        for test_input in test_cases:
            entities = self.extractor.extract_entities(test_input)
            self.assertIn('percentages', entities, f"Failed for input: {test_input}")
            self.assertGreater(len(entities['percentages']), 0)
    
    def test_week_type_extraction(self):
        """Test week type extraction."""
        test_cases = [
            ("This is a green week", "Green"),
            ("Market is red", "Red"),
            ("Very choppy conditions", "Chop")
        ]
        
        for test_input, expected_type in test_cases:
            entities = self.extractor.extract_entities(test_input)
            self.assertEqual(entities.get('week_type'), expected_type, f"Failed for input: {test_input}")
    
    def test_account_type_extraction(self):
        """Test account type extraction."""
        test_cases = [
            "Tell me about Gen-Acc",
            "Revenue account performance",
            "Compounding account strategy"
        ]
        
        for test_input in test_cases:
            entities = self.extractor.extract_entities(test_input)
            self.assertIn('account_types', entities, f"Failed for input: {test_input}")
            self.assertGreater(len(entities['account_types']), 0)
    
    def test_stock_symbol_extraction(self):
        """Test stock symbol extraction."""
        test_input = "I want to trade TSLA and NVDA options"
        entities = self.extractor.extract_entities(test_input)
        
        self.assertIn('stock_symbols', entities)
        self.assertIn('TSLA', entities['stock_symbols'])
        self.assertIn('NVDA', entities['stock_symbols'])
    
    def test_delta_range_extraction(self):
        """Test delta range extraction."""
        test_cases = [
            "40-50 delta options",
            "delta 30-40 range",
            "20 to 30 delta"
        ]
        
        for test_input in test_cases:
            entities = self.extractor.extract_entities(test_input)
            self.assertIn('delta_ranges', entities, f"Failed for input: {test_input}")
            self.assertGreater(len(entities['delta_ranges']), 0)


class TestEnhancedContextManager(unittest.TestCase):
    """Test cases for EnhancedContextManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.context_manager = EnhancedContextManager()
    
    def test_context_initialization(self):
        """Test context manager initialization."""
        context = self.context_manager.current_context
        
        required_keys = [
            'conversation_length', 'last_intent', 'last_action',
            'has_accounts', 'week_classified', 'pending_decisions',
            'conversation_flow', 'user_expertise_level', 'recent_topics'
        ]
        
        for key in required_keys:
            self.assertIn(key, context)
    
    def test_context_update(self):
        """Test context update functionality."""
        conversation_history = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'agent', 'content': 'Hi there!'}
        ]
        
        protocol_state = {
            'accounts': {'GEN_ACC': [100000], 'REV_ACC': [75000], 'COM_ACC': [75000]},
            'week_classification': 'Green',
            'last_decision': {'intent': 'greeting', 'action': 'respond'}
        }
        
        user_preferences = {'risk_tolerance': 'medium'}
        
        updated_context = self.context_manager.update_context(
            conversation_history, protocol_state, user_preferences
        )
        
        self.assertEqual(updated_context['conversation_length'], 2)
        self.assertTrue(updated_context['has_accounts'])
        self.assertTrue(updated_context['week_classified'])
        self.assertEqual(updated_context['last_intent'], 'greeting')
    
    def test_user_expertise_determination(self):
        """Test user expertise level determination."""
        # Test beginner indicators
        conversation_history = [
            {'role': 'user', 'content': 'Can you explain what this system does?'},
            {'role': 'user', 'content': 'I am new to options trading'}
        ]
        
        self.context_manager.update_context(conversation_history, {}, {})
        self.assertEqual(self.context_manager.current_context['user_expertise_level'], 'beginner')
    
    def test_pending_decisions_update(self):
        """Test pending decisions update."""
        # Test account setup required
        protocol_state = {'accounts': {'GEN_ACC': [], 'REV_ACC': [], 'COM_ACC': []}}
        
        self.context_manager.update_context([], protocol_state, {})
        pending = self.context_manager.current_context['pending_decisions']
        
        self.assertGreater(len(pending), 0)
        self.assertEqual(pending[0]['type'], 'account_setup')
    
    def test_context_summary(self):
        """Test context summary generation."""
        # Setup context with accounts
        self.context_manager.current_context.update({
            'has_accounts': True,
            'week_classified': True,
            'conversation_flow': ['setup', 'classification'],
            'user_expertise_level': 'intermediate'
        })
        
        summary = self.context_manager.get_context_summary()
        
        self.assertIsInstance(summary, str)
        self.assertIn('intermediate', summary)


class TestEnhancedConversationMemory(unittest.TestCase):
    """Test cases for EnhancedConversationMemory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.memory = EnhancedConversationMemory(max_history=10)
    
    def test_message_addition_with_metadata(self):
        """Test adding messages with enhanced metadata."""
        self.memory.add_message('user', 'Hello, can you help me?', {'intent': 'greeting'})
        
        self.assertEqual(len(self.memory.history), 1)
        message = self.memory.history[0]
        
        self.assertEqual(message['role'], 'user')
        self.assertEqual(message['content'], 'Hello, can you help me?')
        self.assertIn('metadata', message)
        self.assertIn('word_count', message['metadata'])
        self.assertIn('has_question', message['metadata'])
        self.assertTrue(message['metadata']['has_question'])
    
    def test_conversation_summary(self):
        """Test conversation summary generation."""
        # Add some messages
        self.memory.add_message('user', 'Hello')
        self.memory.add_message('agent', 'Hi there!')
        self.memory.add_message('user', 'Explain the protocol')
        
        summary = self.memory.get_conversation_summary()
        
        required_keys = [
            'total_messages', 'conversation_duration', 'dominant_topics',
            'conversation_patterns', 'user_engagement_level'
        ]
        
        for key in required_keys:
            self.assertIn(key, summary)
        
        self.assertEqual(summary['total_messages'], 3)
    
    def test_topic_tracking(self):
        """Test topic extraction and tracking."""
        self.memory.add_message('user', 'I want to setup my accounts')
        self.memory.add_message('user', 'Tell me about forking')
        
        # Check if topics were extracted
        account_messages = self.memory.search_by_topic('account_setup')
        forking_messages = self.memory.search_by_topic('forking')
        
        self.assertGreater(len(account_messages), 0)
        self.assertGreater(len(forking_messages), 0)
    
    def test_conversation_flow(self):
        """Test conversation flow tracking."""
        # Add messages with different topics
        self.memory.add_message('user', 'Setup my accounts')
        self.memory.add_message('agent', 'I will help you setup accounts')
        self.memory.add_message('user', 'What about forking?')
        
        flow = self.memory.get_conversation_flow()
        
        self.assertIsInstance(flow, list)
        if flow:  # If topics were detected
            self.assertIn('topic', flow[0])
            self.assertIn('start_time', flow[0])
    
    def test_sentiment_analysis(self):
        """Test basic sentiment analysis."""
        # Test positive sentiment
        positive_sentiment = self.memory._analyze_sentiment("This is great! I love it!")
        self.assertEqual(positive_sentiment, 'positive')
        
        # Test negative sentiment
        negative_sentiment = self.memory._analyze_sentiment("This is terrible and confusing")
        self.assertEqual(negative_sentiment, 'negative')
        
        # Test neutral sentiment
        neutral_sentiment = self.memory._analyze_sentiment("Please explain the protocol")
        self.assertEqual(neutral_sentiment, 'neutral')


class TestEnhancedProtocolStateMemory(unittest.TestCase):
    """Test cases for EnhancedProtocolStateMemory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.memory = EnhancedProtocolStateMemory()
    
    def test_performance_snapshot(self):
        """Test performance snapshot recording."""
        # Initialize accounts
        self.memory.initialize_accounts(300000, 0.4, 0.3, 0.3, 0.05)
        
        # Record snapshot
        self.memory.record_performance_snapshot()
        
        self.assertEqual(len(self.memory.state['performance_history']), 1)
        snapshot = self.memory.state['performance_history'][0]
        
        required_keys = ['timestamp', 'total_balance', 'account_balances', 'account_counts']
        for key in required_keys:
            self.assertIn(key, snapshot)
    
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation."""
        # Initialize accounts
        self.memory.initialize_accounts(300000, 0.4, 0.3, 0.3, 0.05)
        
        # Record initial snapshot
        self.memory.record_performance_snapshot()
        
        # Simulate growth and record another snapshot
        self.memory.state['accounts']['GEN_ACC'][0] *= 1.1  # 10% growth
        self.memory.record_performance_snapshot()
        
        metrics = self.memory.calculate_performance_metrics(period_days=1)
        
        if 'error' not in metrics:
            self.assertIn('total_return', metrics)
            self.assertIn('start_balance', metrics)
            self.assertIn('end_balance', metrics)
    
    def test_decision_pattern_analysis(self):
        """Test decision pattern analysis."""
        # Add some decisions
        decisions = [
            {'action': 'setup', 'intent': 'setup_accounts', 'confidence': 0.9},
            {'action': 'explain', 'intent': 'explain_protocol', 'confidence': 0.8},
            {'action': 'recommend', 'intent': 'recommend_trades', 'confidence': 0.7}
        ]
        
        for decision in decisions:
            self.memory.state['decision_history'].append(decision)
        
        analysis = self.memory.analyze_decision_patterns()
        
        self.assertEqual(analysis['total_decisions'], 3)
        self.assertIn('decision_types', analysis)
        self.assertIn('average_confidence', analysis)
        self.assertAlmostEqual(analysis['average_confidence'], 0.8, places=1)
    
    def test_fork_prediction(self):
        """Test fork prediction functionality."""
        # Initialize accounts with one close to fork threshold
        self.memory.state['accounts']['GEN_ACC'] = [95000]  # Close to 100K threshold
        
        prediction = self.memory.predict_next_fork()
        
        if 'error' not in prediction:
            self.assertIn('account_index', prediction)
            self.assertIn('current_balance', prediction)
            self.assertIn('amount_needed', prediction)
            self.assertIn('estimated_weeks_to_fork', prediction)
    
    def test_risk_assessment(self):
        """Test risk assessment functionality."""
        # Initialize accounts with high Gen-Acc concentration
        self.memory.state['accounts'] = {
            'GEN_ACC': [200000],  # High concentration
            'REV_ACC': [50000],
            'COM_ACC': [50000]
        }
        
        assessment = self.memory.get_risk_assessment()
        
        required_keys = ['total_balance', 'concentration', 'risk_level', 'diversification_score']
        for key in required_keys:
            self.assertIn(key, assessment)
        
        # Should detect high risk due to Gen-Acc concentration
        self.assertEqual(assessment['risk_level'], 'high')


class TestEnhancedUserPreferencesMemory(unittest.TestCase):
    """Test cases for EnhancedUserPreferencesMemory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.memory = EnhancedUserPreferencesMemory()
    
    def test_preference_learning(self):
        """Test preference learning from interactions."""
        # Simulate high-amount setup interaction
        interaction_data = {
            'intent': 'setup_accounts',
            'amount': 500000,
            'confidence': 0.9,
            'message_length': 50
        }
        
        self.memory.learn_from_interaction(interaction_data)
        
        # Should infer high risk tolerance
        self.assertEqual(self.memory.preferences['risk_tolerance'], 'high')
    
    def test_behavioral_pattern_updates(self):
        """Test behavioral pattern updates."""
        # Simulate quick setup
        interaction_data = {
            'intent': 'setup_accounts',
            'response_time': 20,  # Quick response
            'confidence': 0.9
        }
        
        self.memory.learn_from_interaction(interaction_data)
        
        self.assertEqual(self.memory.preferences['behavioral_patterns']['setup_speed'], 'fast')
    
    def test_personalized_settings(self):
        """Test personalized settings generation."""
        # Set some preferences
        self.memory.preferences['risk_tolerance'] = 'high'
        self.memory.preferences['communication_style'] = 'detailed'
        
        settings = self.memory.get_personalized_settings()
        
        required_keys = [
            'response_style', 'detail_level', 'explanation_approach',
            'notification_frequency', 'risk_guidance_level'
        ]
        
        for key in required_keys:
            self.assertIn(key, settings)


class TestEnhancedMemoryManager(unittest.TestCase):
    """Test cases for EnhancedMemoryManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = EnhancedMemoryManager()
    
    def test_interaction_processing(self):
        """Test complete interaction processing."""
        user_input = "I want to setup accounts with $300,000"
        agent_response = "I'll help you setup your accounts"
        decision_data = {
            'intent': 'setup_accounts',
            'action': 'setup',
            'confidence': 0.9,
            'parameters': {'initial_investment': 300000}
        }
        
        self.manager.process_interaction(user_input, agent_response, decision_data)
        
        # Check that all memory systems were updated
        self.assertEqual(len(self.manager.conversation_memory.history), 2)
        self.assertEqual(self.manager.session_analytics['total_interactions'], 1)
    
    def test_comprehensive_status(self):
        """Test comprehensive status generation."""
        # Process some interactions first
        self.manager.process_interaction(
            "Hello", "Hi there!", 
            {'intent': 'greeting', 'action': 'respond', 'confidence': 0.9}
        )
        
        status = self.manager.get_comprehensive_status()
        
        required_keys = [
            'session_analytics', 'conversation_summary', 
            'protocol_metrics', 'user_settings', 'memory_health'
        ]
        
        for key in required_keys:
            self.assertIn(key, status)
    
    def test_memory_health_assessment(self):
        """Test memory health assessment."""
        health = self.manager._assess_memory_health()
        
        required_systems = ['conversation_memory', 'protocol_memory', 'preferences_memory']
        
        for system in required_systems:
            self.assertIn(system, health)
            self.assertIn('status', health[system])


class TestEnhancedCognitiveFramework(unittest.TestCase):
    """Test cases for EnhancedCognitiveFramework class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.framework = EnhancedCognitiveFramework()
    
    def test_enhanced_input_processing(self):
        """Test enhanced input processing."""
        user_input = "I want to setup accounts with $300,000"
        conversation_context = {'conversation_history': []}
        protocol_state = {'accounts': {'GEN_ACC': [], 'REV_ACC': [], 'COM_ACC': []}}
        user_preferences = {}
        
        processed_input = self.framework.process_input(
            user_input, conversation_context, protocol_state, user_preferences
        )
        
        required_keys = ['raw_input', 'intent', 'entities', 'context', 'context_summary']
        for key in required_keys:
            self.assertIn(key, processed_input)
        
        self.assertEqual(processed_input['intent'], 'setup_accounts')
        self.assertIn('amount', processed_input['entities'])
        self.assertEqual(processed_input['entities']['amount'], 300000)
    
    def test_enhanced_decision_making(self):
        """Test enhanced decision making."""
        processed_input = {
            'intent': 'setup_accounts',
            'entities': {'amount': 300000},
            'context': {'has_accounts': False, 'user_expertise_level': 'beginner'}
        }
        
        protocol_state = {'accounts': {'GEN_ACC': [], 'REV_ACC': [], 'COM_ACC': []}}
        
        decision = self.framework.make_decision(processed_input, protocol_state)
        
        required_keys = ['intent', 'action', 'response_type', 'parameters', 'rationale', 'confidence']
        for key in required_keys:
            self.assertIn(key, decision)
        
        self.assertEqual(decision['action'], 'setup')
        self.assertEqual(decision['response_type'], 'account_setup')
        self.assertGreater(decision['confidence'], 0.5)
    
    def test_detail_level_determination(self):
        """Test detail level determination."""
        # Test beginner level
        context = {'user_expertise_level': 'beginner'}
        detail_level = self.framework._determine_detail_level(context, {})
        self.assertEqual(detail_level, 'overview')
        
        # Test advanced level
        context = {'user_expertise_level': 'advanced'}
        detail_level = self.framework._determine_detail_level(context, {})
        self.assertEqual(detail_level, 'technical')


class TestEnhancedAgentIntegration(unittest.TestCase):
    """Integration tests for the enhanced agent system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.interface = AgentInterface()
    
    def test_enhanced_conversation_flow(self):
        """Test enhanced conversation flow."""
        # Step 1: Greeting with enhanced processing
        response1 = self.interface.chat('Hello!')
        self.assertIn('ALL-USE agent', response1)
        
        # Step 2: Protocol explanation with enhanced detail
        response2 = self.interface.chat('Can you explain the ALL-USE protocol in detail?')
        self.assertIn('Generation Account', response2)
        
        # Step 3: Account setup with amount extraction
        response3 = self.interface.chat('I want to setup accounts with $500,000')
        self.assertIn('$500,000', response3)
        
        # Verify enhanced status
        status = self.interface.status()
        self.assertTrue(status['has_accounts'])
        self.assertGreater(status['total_balance'], 0)
    
    def test_enhanced_entity_extraction_integration(self):
        """Test enhanced entity extraction in full conversation."""
        # Test with complex input containing multiple entities
        response = self.interface.chat(
            'I want to setup accounts with $750,000 and focus on TSLA and NVDA with 40-50 delta options'
        )
        
        # Should handle the complex input appropriately
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    def test_enhanced_context_awareness(self):
        """Test enhanced context awareness."""
        # Setup accounts first
        self.interface.chat('Setup accounts with $300,000')
        
        # Now ask for recommendations - should recognize context
        response = self.interface.chat('What trades do you recommend?')
        
        # Should recognize that week classification is needed first
        self.assertIn('week', response.lower())
    
    def test_enhanced_memory_persistence(self):
        """Test enhanced memory persistence across interactions."""
        # Have a conversation
        self.interface.chat('Hello!')
        self.interface.chat('Explain the protocol')
        self.interface.chat('Setup accounts with $400,000')
        
        # Get comprehensive status
        status = self.interface.status()
        
        # Should have conversation history and context
        self.assertGreater(status['conversation_length'], 0)
        self.assertTrue(status['has_accounts'])
    
    def test_error_handling_enhancement(self):
        """Test enhanced error handling."""
        # Test with malformed input
        response = self.interface.chat('')
        self.assertIsInstance(response, str)
        
        # Test with very complex input
        complex_input = 'a' * 1000 + ' setup accounts'
        response = self.interface.chat(complex_input)
        self.assertIsInstance(response, str)


class TestPerformanceAndScalability(unittest.TestCase):
    """Test performance and scalability of enhanced components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.interface = AgentInterface()
    
    def test_large_conversation_handling(self):
        """Test handling of large conversations."""
        # Simulate a long conversation
        for i in range(50):
            response = self.interface.chat(f'Message {i}: Tell me about the protocol')
            self.assertIsInstance(response, str)
        
        # Should still function properly
        status = self.interface.status()
        self.assertEqual(status['conversation_length'], 100)  # 50 user + 50 agent messages
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        # Create memory manager with large history
        memory = EnhancedConversationMemory(max_history=1000)
        
        # Add many messages
        for i in range(1500):
            memory.add_message('user', f'Message {i}')
        
        # Should respect max_history limit
        self.assertEqual(len(memory.history), 1000)
    
    def test_pattern_recognition_performance(self):
        """Test pattern recognition performance."""
        detector = EnhancedIntentDetector()
        
        # Test with many different inputs
        test_inputs = [
            'Hello there!',
            'Explain the protocol',
            'Setup my accounts',
            'What trades do you recommend?',
            'Check my performance'
        ] * 100  # 500 total inputs
        
        # Should handle all inputs efficiently
        for test_input in test_inputs:
            intent = detector.detect_intent(test_input, {})
            self.assertIsInstance(intent, str)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestEnhancedIntentDetector,
        TestEnhancedEntityExtractor,
        TestEnhancedContextManager,
        TestEnhancedConversationMemory,
        TestEnhancedProtocolStateMemory,
        TestEnhancedUserPreferencesMemory,
        TestEnhancedMemoryManager,
        TestEnhancedCognitiveFramework,
        TestEnhancedAgentIntegration,
        TestPerformanceAndScalability
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")

