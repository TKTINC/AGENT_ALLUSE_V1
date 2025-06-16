"""
ALL-USE Agent Core Tests

This module contains comprehensive tests for the ALL-USE agent core components.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agent_core.enhanced_agent import EnhancedALLUSEAgent, AgentInterface
from src.agent_core.memory_manager import MemoryManager, ConversationMemory, ProtocolStateMemory, UserPreferencesMemory
from src.agent_core.cognitive_framework import CognitiveFramework, ContextManager
from src.agent_core.response_generator import ResponseGenerator
from src.protocol_engine.all_use_parameters import ALLUSEParameters

class TestConversationMemory(unittest.TestCase):
    """Test cases for ConversationMemory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.memory = ConversationMemory(max_history=5)
    
    def test_add_message(self):
        """Test adding messages to conversation memory."""
        self.memory.add_message('user', 'Hello')
        self.memory.add_message('agent', 'Hi there!')
        
        history = self.memory.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]['role'], 'user')
        self.assertEqual(history[0]['content'], 'Hello')
        self.assertEqual(history[1]['role'], 'agent')
        self.assertEqual(history[1]['content'], 'Hi there!')
    
    def test_max_history_limit(self):
        """Test that history is limited to max_history."""
        for i in range(10):
            self.memory.add_message('user', f'Message {i}')
        
        history = self.memory.get_history()
        self.assertEqual(len(history), 5)
        self.assertEqual(history[0]['content'], 'Message 5')
        self.assertEqual(history[-1]['content'], 'Message 9')
    
    def test_get_last_user_message(self):
        """Test getting the last user message."""
        self.memory.add_message('user', 'First user message')
        self.memory.add_message('agent', 'Agent response')
        self.memory.add_message('user', 'Second user message')
        
        last_user_msg = self.memory.get_last_user_message()
        self.assertEqual(last_user_msg['content'], 'Second user message')
    
    def test_search_history(self):
        """Test searching conversation history."""
        self.memory.add_message('user', 'Tell me about accounts')
        self.memory.add_message('agent', 'Accounts are important')
        self.memory.add_message('user', 'What about forking?')
        
        results = self.memory.search_history('accounts')
        self.assertEqual(len(results), 2)


class TestProtocolStateMemory(unittest.TestCase):
    """Test cases for ProtocolStateMemory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.memory = ProtocolStateMemory()
    
    def test_initialize_accounts(self):
        """Test account initialization."""
        self.memory.initialize_accounts(300000, 0.4, 0.3, 0.3, 0.05)
        
        balances = self.memory.get_account_balances()
        self.assertEqual(len(balances['GEN_ACC']), 1)
        self.assertEqual(len(balances['REV_ACC']), 1)
        self.assertEqual(len(balances['COM_ACC']), 1)
        
        # Check effective allocations (after 5% cash buffer)
        gen_acc_expected = 300000 * 0.4 * 0.95
        rev_acc_expected = 300000 * 0.3 * 0.95
        com_acc_expected = 300000 * 0.3 * 0.95
        
        self.assertAlmostEqual(balances['GEN_ACC'][0], gen_acc_expected, places=2)
        self.assertAlmostEqual(balances['REV_ACC'][0], rev_acc_expected, places=2)
        self.assertAlmostEqual(balances['COM_ACC'][0], com_acc_expected, places=2)
    
    def test_set_week_classification(self):
        """Test setting week classification."""
        self.memory.set_week_classification('Green')
        state = self.memory.get_state()
        self.assertEqual(state['week_classification'], 'Green')
        
        # Test invalid classification
        with self.assertRaises(ValueError):
            self.memory.set_week_classification('Invalid')
    
    def test_record_fork(self):
        """Test recording account fork."""
        # Initialize accounts first
        self.memory.initialize_accounts(300000, 0.4, 0.3, 0.3, 0.05)
        
        # Simulate Gen-Acc growth to fork threshold
        initial_gen_balance = self.memory.get_account_balances()['GEN_ACC'][0]
        self.memory.state['accounts']['GEN_ACC'][0] = initial_gen_balance + 50000
        
        # Record fork
        self.memory.record_fork(0, 25000, 25000)
        
        balances = self.memory.get_account_balances()
        self.assertEqual(len(balances['GEN_ACC']), 2)  # Original + new forked account
        self.assertEqual(balances['GEN_ACC'][1], 25000)  # New Gen-Acc
        
        fork_history = self.memory.get_state()['fork_history']
        self.assertEqual(len(fork_history), 1)
    
    def test_get_total_balance(self):
        """Test getting total balance across all accounts."""
        self.memory.initialize_accounts(300000, 0.4, 0.3, 0.3, 0.05)
        
        total = self.memory.get_total_balance()
        expected_total = 300000 * 0.95  # After 5% cash buffer
        self.assertAlmostEqual(total, expected_total, places=2)


class TestCognitiveFramework(unittest.TestCase):
    """Test cases for CognitiveFramework class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.framework = CognitiveFramework()
    
    def test_detect_intent_greeting(self):
        """Test intent detection for greetings."""
        intent = self.framework._detect_intent('Hello there!', {})
        self.assertEqual(intent, 'greeting')
        
        intent = self.framework._detect_intent('Hi', {})
        self.assertEqual(intent, 'greeting')
    
    def test_detect_intent_explain_protocol(self):
        """Test intent detection for protocol explanation."""
        intent = self.framework._detect_intent('Explain the ALL-USE protocol', {})
        self.assertEqual(intent, 'explain_protocol')
        
        intent = self.framework._detect_intent('Tell me about the system', {})
        self.assertEqual(intent, 'explain_protocol')
    
    def test_detect_intent_setup_accounts(self):
        """Test intent detection for account setup."""
        intent = self.framework._detect_intent('Setup my accounts', {})
        self.assertEqual(intent, 'setup_accounts')
        
        intent = self.framework._detect_intent('Create accounts', {})
        self.assertEqual(intent, 'setup_accounts')
    
    def test_extract_entities_amount(self):
        """Test entity extraction for monetary amounts."""
        entities = self.framework._extract_entities('I have $300,000 to invest')
        self.assertEqual(entities['amount'], 300000)
        
        entities = self.framework._extract_entities('I want to invest 500k')
        self.assertEqual(entities['amount'], 500000)
        
        entities = self.framework._extract_entities('I have 2 million dollars')
        self.assertEqual(entities['amount'], 2000000)
    
    def test_extract_entities_week_type(self):
        """Test entity extraction for week types."""
        entities = self.framework._extract_entities('This is a green week')
        self.assertEqual(entities['week_type'], 'Green')
        
        entities = self.framework._extract_entities('Market looks red')
        self.assertEqual(entities['week_type'], 'Red')


class TestResponseGenerator(unittest.TestCase):
    """Test cases for ResponseGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = ResponseGenerator()
    
    def test_generate_greeting(self):
        """Test greeting response generation."""
        response = self.generator._generate_greeting()
        self.assertIn('ALL-USE agent', response)
        self.assertIn('three-tiered', response)
    
    def test_generate_protocol_explanation(self):
        """Test protocol explanation generation."""
        response = self.generator._generate_protocol_explanation('overview')
        self.assertIn('Generation Account', response)
        self.assertIn('Revenue Account', response)
        self.assertIn('Compounding Account', response)
        self.assertIn('1.5%', response)
        self.assertIn('1.0%', response)
        self.assertIn('0.5%', response)


class TestEnhancedALLUSEAgent(unittest.TestCase):
    """Test cases for EnhancedALLUSEAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = EnhancedALLUSEAgent()
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertIsNotNone(self.agent.agent_id)
        self.assertIsNotNone(self.agent.memory_manager)
        self.assertIsNotNone(self.agent.cognitive_framework)
        self.assertIsNotNone(self.agent.response_generator)
        self.assertEqual(self.agent.total_interactions, 0)
    
    def test_process_message_greeting(self):
        """Test processing a greeting message."""
        response = self.agent.process_message('Hello!')
        
        self.assertIn('ALL-USE agent', response)
        self.assertEqual(self.agent.total_interactions, 1)
        
        # Check that message was stored in memory
        history = self.agent.memory_manager.get_conversation_history()
        self.assertEqual(len(history), 2)  # User message + agent response
        self.assertEqual(history[0]['role'], 'user')
        self.assertEqual(history[1]['role'], 'agent')
    
    def test_process_message_account_setup(self):
        """Test processing an account setup message."""
        response = self.agent.process_message('Setup accounts with $300,000')
        
        # Check that accounts were created
        status = self.agent.get_agent_status()
        self.assertTrue(status['has_accounts'])
        self.assertGreater(status['total_balance'], 0)
    
    def test_get_agent_status(self):
        """Test getting agent status."""
        status = self.agent.get_agent_status()
        
        required_keys = [
            'agent_id', 'session_start_time', 'total_interactions',
            'conversation_length', 'has_accounts', 'week_classified',
            'total_balance', 'account_balances'
        ]
        
        for key in required_keys:
            self.assertIn(key, status)
    
    def test_get_performance_summary(self):
        """Test getting performance summary."""
        # Setup accounts first
        self.agent.process_message('Setup accounts with $300,000')
        
        performance = self.agent.get_performance_summary()
        
        required_keys = [
            'total_balance', 'account_totals', 'account_counts',
            'fork_count', 'merge_count'
        ]
        
        for key in required_keys:
            self.assertIn(key, performance)
        
        self.assertGreater(performance['total_balance'], 0)
    
    def test_reset_session(self):
        """Test resetting agent session."""
        # Process some messages first
        self.agent.process_message('Hello!')
        self.agent.process_message('Setup accounts with $300,000')
        
        # Verify state before reset
        self.assertGreater(self.agent.total_interactions, 0)
        self.assertTrue(self.agent.get_agent_status()['has_accounts'])
        
        # Reset session
        self.agent.reset_session()
        
        # Verify state after reset
        self.assertEqual(self.agent.total_interactions, 0)
        self.assertFalse(self.agent.get_agent_status()['has_accounts'])


class TestAgentInterface(unittest.TestCase):
    """Test cases for AgentInterface class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.interface = AgentInterface()
    
    def test_chat(self):
        """Test chat functionality."""
        response = self.interface.chat('Hello!')
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    def test_status(self):
        """Test status functionality."""
        status = self.interface.status()
        self.assertIsInstance(status, dict)
        self.assertIn('agent_id', status)
    
    def test_performance(self):
        """Test performance functionality."""
        performance = self.interface.performance()
        self.assertIsInstance(performance, dict)
        self.assertIn('total_balance', performance)
    
    def test_reset(self):
        """Test reset functionality."""
        # Chat first to create some state
        self.interface.chat('Hello!')
        initial_interactions = self.interface.status()['total_interactions']
        
        # Reset
        self.interface.reset()
        
        # Verify reset
        post_reset_interactions = self.interface.status()['total_interactions']
        self.assertEqual(post_reset_interactions, 0)
    
    def test_export(self):
        """Test export functionality."""
        # Create some state first
        self.interface.chat('Hello!')
        
        export_data = self.interface.export()
        self.assertIsInstance(export_data, dict)
        
        required_keys = [
            'agent_id', 'session_start_time', 'total_interactions',
            'conversation_history', 'protocol_state', 'user_preferences'
        ]
        
        for key in required_keys:
            self.assertIn(key, export_data)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete agent system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.interface = AgentInterface()
    
    def test_complete_workflow(self):
        """Test a complete workflow from greeting to account setup."""
        # Step 1: Greeting
        response1 = self.interface.chat('Hello!')
        self.assertIn('ALL-USE agent', response1)
        
        # Step 2: Protocol explanation
        response2 = self.interface.chat('Explain the ALL-USE protocol')
        self.assertIn('Generation Account', response2)
        self.assertIn('Revenue Account', response2)
        self.assertIn('Compounding Account', response2)
        
        # Step 3: Account setup
        response3 = self.interface.chat('Setup accounts with $300,000')
        self.assertIn('$300,000', response3)
        
        # Verify accounts were created
        status = self.interface.status()
        self.assertTrue(status['has_accounts'])
        self.assertAlmostEqual(status['total_balance'], 285000, places=0)  # 300k * 0.95 (cash buffer)
        
        # Step 4: Check performance
        performance = self.interface.performance()
        self.assertEqual(performance['account_counts']['GEN_ACC'], 1)
        self.assertEqual(performance['account_counts']['REV_ACC'], 1)
        self.assertEqual(performance['account_counts']['COM_ACC'], 1)
    
    def test_error_handling(self):
        """Test error handling in the agent."""
        # Test with empty input
        response = self.interface.chat('')
        self.assertIsInstance(response, str)
        
        # Test with very long input
        long_input = 'a' * 10000
        response = self.interface.chat(long_input)
        self.assertIsInstance(response, str)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)

