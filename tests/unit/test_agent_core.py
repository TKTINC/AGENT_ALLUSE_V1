"""
Unit Tests for Enhanced Agent Core Components

This module contains comprehensive unit tests for the enhanced agent core,
including cognitive framework, memory management, and response generation.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from tests.utils.test_utilities import (
    MockDataGenerator, MockServices, TestAssertions, TestFixtures
)


class TestEnhancedAgent:
    """Test cases for the Enhanced ALL-USE Agent."""
    
    def setup_method(self):
        """Set up test fixtures before each test."""
        self.test_data = TestFixtures.setup_test_environment()
        self.mock_services = {
            'market_data': MockServices.create_mock_market_data_service(),
            'brokerage': MockServices.create_mock_brokerage_service(),
            'database': MockServices.create_mock_database()
        }
    
    def test_agent_initialization(self):
        """Test agent initialization with various configurations."""
        from src.agent_core.enhanced_agent import EnhancedALLUSEAgent
        
        # Test default initialization
        agent = EnhancedALLUSEAgent()
        assert hasattr(agent, 'memory_manager')
        assert hasattr(agent, 'cognitive_framework')
        assert hasattr(agent, 'response_generator')
        
        # Test that components are properly initialized
        assert agent.memory_manager is not None
        assert agent.cognitive_framework is not None
        assert agent.response_generator is not None
    
    def test_agent_response_time(self):
        """Test that agent responses meet performance requirements."""
        agent = TestFixtures.create_test_agent()
        
        def process_message():
            return agent.process_message("Hello, I want to set up my accounts")
        
        # Assert response time is under 100ms
        response = TestAssertions.assert_response_time(process_message, 100)
        assert response is not None
        assert 'response' in response
    
    def test_agent_memory_usage(self):
        """Test that agent memory usage is within acceptable limits."""
        agent = TestFixtures.create_test_agent()
        
        def process_multiple_messages():
            responses = []
            for i in range(10):
                response = agent.process_message(f"Test message {i}")
                responses.append(response)
            return responses
        
        # Assert memory usage is under 50MB for processing 10 messages
        responses = TestAssertions.assert_memory_usage(process_multiple_messages, 50)
        assert len(responses) == 10
    
    def test_conversation_flow(self):
        """Test complete conversation flow from greeting to account setup."""
        agent = TestFixtures.create_test_agent()
        
        # Test greeting
        greeting_response = agent.process_message("Hello")
        assert greeting_response['intent'] == 'greeting'
        assert 'welcome' in greeting_response['response'].lower()
        
        # Test account inquiry
        inquiry_response = agent.process_message("I want to learn about account setup")
        assert inquiry_response['intent'] == 'account_inquiry'
        
        # Test account setup
        setup_response = agent.process_message("Set up my accounts with $50,000")
        assert setup_response['intent'] == 'setup_accounts'
        assert setup_response['entities']['amount'] == 50000
    
    @pytest.mark.parametrize("message,expected_intent", [
        ("Hello there", "greeting"),
        ("Hi, how are you?", "greeting"),
        ("Tell me about my accounts", "account_inquiry"),
        ("What's my balance?", "account_inquiry"),
        ("Set up accounts with $25000", "setup_accounts"),
        ("I want to start trading", "setup_accounts"),
        ("How is the market today?", "market_analysis"),
        ("What's the risk level?", "risk_assessment"),
        ("Show me my performance", "performance_inquiry")
    ])
    def test_intent_detection(self, message, expected_intent):
        """Test intent detection accuracy across various message types."""
        agent = TestFixtures.create_test_agent()
        response = agent.process_message(message)
        assert response['intent'] == expected_intent
    
    def test_entity_extraction(self):
        """Test entity extraction from user messages."""
        agent = TestFixtures.create_test_agent()
        
        # Test amount extraction
        response = agent.process_message("I have $75,000 to invest")
        assert 'amount' in response['entities']
        assert response['entities']['amount'] == 75000
        
        # Test percentage extraction
        response = agent.process_message("I want 25% risk level")
        assert 'percentage' in response['entities']
        assert response['entities']['percentage'] == 25
        
        # Test stock symbol extraction
        response = agent.process_message("What about AAPL and TSLA?")
        assert 'stocks' in response['entities']
        assert 'AAPL' in response['entities']['stocks']
        assert 'TSLA' in response['entities']['stocks']
    
    def test_context_management(self):
        """Test context management across conversation turns."""
        agent = TestFixtures.create_test_agent()
        
        # Establish context
        agent.process_message("I want to set up accounts with $100,000")
        
        # Test context retention
        response = agent.process_message("What's the next step?")
        assert response['context_maintained'] is True
        
        # Test context evolution
        agent.process_message("Actually, let's start with $50,000")
        context = agent.memory_manager.get_conversation_context()
        assert context['current_amount'] == 50000


class TestCognitiveFramework:
    """Test cases for the Enhanced Cognitive Framework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from src.agent_core.enhanced_cognitive_framework import EnhancedCognitiveFramework
        self.cognitive_framework = EnhancedCognitiveFramework()
    
    def test_intent_detection_accuracy(self):
        """Test intent detection accuracy with various inputs."""
        test_cases = [
            ("Hello, how are you?", "greeting"),
            ("What's my account balance?", "account_inquiry"),
            ("Set up my trading accounts", "setup_accounts"),
            ("How is SPY performing?", "market_analysis"),
            ("What's my risk exposure?", "risk_assessment"),
            ("Show me my returns", "performance_inquiry"),
            ("I want to adjust my positions", "position_adjustment"),
            ("What's the market outlook?", "market_outlook"),
            ("Help me understand options", "education_request"),
            ("I'm worried about losses", "risk_concern")
        ]
        
        for message, expected_intent in test_cases:
            detected_intent = self.cognitive_framework.detect_intent(message)
            assert detected_intent == expected_intent, f"Failed for message: '{message}'"
    
    def test_entity_extraction_comprehensive(self):
        """Test comprehensive entity extraction capabilities."""
        # Test amount extraction
        entities = self.cognitive_framework.extract_entities("I have $125,000 to invest")
        assert entities['amount'] == 125000
        
        # Test percentage extraction
        entities = self.cognitive_framework.extract_entities("Set risk to 15%")
        assert entities['percentage'] == 15
        
        # Test delta extraction
        entities = self.cognitive_framework.extract_entities("Use 30 delta options")
        assert entities['delta'] == 30
        
        # Test multiple entities
        entities = self.cognitive_framework.extract_entities("Invest $50,000 in AAPL with 25% risk")
        assert entities['amount'] == 50000
        assert entities['percentage'] == 25
        assert 'AAPL' in entities['stocks']
    
    def test_context_analysis(self):
        """Test context analysis and scoring."""
        conversation_history = MockDataGenerator.generate_conversation_history(5)
        
        context_score = self.cognitive_framework.analyze_context(
            "What should I do next?", 
            conversation_history
        )
        
        assert 0 <= context_score <= 1
        assert isinstance(context_score, float)
    
    def test_decision_making(self):
        """Test decision-making capabilities."""
        context = {
            'user_intent': 'setup_accounts',
            'entities': {'amount': 75000},
            'conversation_history': [],
            'user_expertise': 'beginner'
        }
        
        decision = self.cognitive_framework.make_decision(context)
        
        assert 'action' in decision
        assert 'explanation' in decision
        assert 'confidence' in decision
        assert 0 <= decision['confidence'] <= 1


class TestMemoryManager:
    """Test cases for the Enhanced Memory Manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from src.agent_core.enhanced_memory_manager import EnhancedMemoryManager
        self.memory_manager = EnhancedMemoryManager()
    
    def test_conversation_memory(self):
        """Test conversation memory functionality."""
        # Add conversation entries
        self.memory_manager.add_conversation_entry(
            "Hello", "Hi there! Welcome to ALL-USE.", "greeting"
        )
        self.memory_manager.add_conversation_entry(
            "Set up accounts", "I'll help you set up your accounts.", "setup_accounts"
        )
        
        # Test retrieval
        history = self.memory_manager.get_conversation_history(limit=2)
        assert len(history) == 2
        assert history[0]['intent'] == 'greeting'
        assert history[1]['intent'] == 'setup_accounts'
    
    def test_protocol_state_memory(self):
        """Test protocol state memory functionality."""
        # Update protocol state
        self.memory_manager.update_protocol_state('account_setup', {
            'total_amount': 100000,
            'account_type': 'GEN_ACC',
            'setup_complete': False
        })
        
        # Test retrieval
        state = self.memory_manager.get_protocol_state('account_setup')
        assert state['total_amount'] == 100000
        assert state['account_type'] == 'GEN_ACC'
        assert state['setup_complete'] is False
    
    def test_user_preferences_memory(self):
        """Test user preferences memory functionality."""
        # Update preferences
        self.memory_manager.update_user_preference('risk_tolerance', 'moderate')
        self.memory_manager.update_user_preference('communication_style', 'detailed')
        
        # Test retrieval
        preferences = self.memory_manager.get_user_preferences()
        assert preferences['risk_tolerance'] == 'moderate'
        assert preferences['communication_style'] == 'detailed'
    
    def test_memory_analytics(self):
        """Test memory analytics and insights."""
        # Add some conversation data
        for i in range(10):
            self.memory_manager.add_conversation_entry(
                f"Message {i}", f"Response {i}", "test_intent"
            )
        
        # Get analytics
        analytics = self.memory_manager.get_memory_analytics()
        
        assert 'conversation_count' in analytics
        assert 'intent_distribution' in analytics
        assert 'memory_usage' in analytics
        assert analytics['conversation_count'] == 10
    
    def test_memory_limits(self):
        """Test memory limit enforcement."""
        # Set a small memory limit
        self.memory_manager.config['memory_limit'] = 5
        
        # Add more entries than the limit
        for i in range(10):
            self.memory_manager.add_conversation_entry(
                f"Message {i}", f"Response {i}", "test_intent"
            )
        
        # Check that only the most recent entries are kept
        history = self.memory_manager.get_conversation_history()
        assert len(history) <= 5


class TestResponseGeneration:
    """Test cases for response generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from src.agent_core.response_generator import ResponseGenerator
        self.response_generator = ResponseGenerator()
    
    def test_response_generation_quality(self):
        """Test response generation quality and consistency."""
        context = {
            'intent': 'greeting',
            'entities': {},
            'user_message': "Hello there!",
            'conversation_history': []
        }
        
        response = self.response_generator.generate_response(context)
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert 'welcome' in response.lower() or 'hello' in response.lower()
    
    def test_response_personalization(self):
        """Test response personalization based on user preferences."""
        context = {
            'intent': 'account_inquiry',
            'entities': {},
            'user_message': "Tell me about my accounts",
            'user_preferences': {'communication_style': 'concise'}
        }
        
        concise_response = self.response_generator.generate_response(context)
        
        context['user_preferences']['communication_style'] = 'detailed'
        detailed_response = self.response_generator.generate_response(context)
        
        # Detailed response should be longer
        assert len(detailed_response) > len(concise_response)
    
    def test_response_consistency(self):
        """Test response consistency for similar inputs."""
        context = {
            'intent': 'greeting',
            'entities': {},
            'user_message': "Hi",
            'conversation_history': []
        }
        
        responses = []
        for _ in range(5):
            response = self.response_generator.generate_response(context)
            responses.append(response)
        
        # All responses should be greetings
        for response in responses:
            assert any(word in response.lower() for word in ['hello', 'hi', 'welcome', 'greetings'])


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests for agent components."""
    
    def test_agent_response_benchmark(self, benchmark):
        """Benchmark agent response time."""
        agent = TestFixtures.create_test_agent()
        
        def process_message():
            return agent.process_message("Hello, set up my accounts with $50,000")
        
        result = benchmark(process_message)
        assert result is not None
    
    def test_memory_operations_benchmark(self, benchmark):
        """Benchmark memory operations."""
        from src.agent_core.enhanced_memory_manager import EnhancedMemoryManager
        memory_manager = EnhancedMemoryManager()
        
        def memory_operations():
            for i in range(100):
                memory_manager.add_conversation_entry(
                    f"Message {i}", f"Response {i}", "test_intent"
                )
            return memory_manager.get_conversation_history()
        
        result = benchmark(memory_operations)
        assert len(result) > 0
    
    def test_cognitive_processing_benchmark(self, benchmark):
        """Benchmark cognitive processing."""
        from src.agent_core.enhanced_cognitive_framework import EnhancedCognitiveFramework
        cognitive_framework = EnhancedCognitiveFramework()
        
        def cognitive_processing():
            message = "I want to set up my trading accounts with $100,000 and moderate risk"
            intent = cognitive_framework.detect_intent(message)
            entities = cognitive_framework.extract_entities(message)
            return intent, entities
        
        result = benchmark(cognitive_processing)
        assert result[0] is not None  # intent
        assert result[1] is not None  # entities


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])

