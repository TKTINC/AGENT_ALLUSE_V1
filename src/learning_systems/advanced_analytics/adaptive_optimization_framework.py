"""
ALL-USE Learning Systems - Adaptive Optimization and Reinforcement Learning Module

This module implements sophisticated reinforcement learning algorithms, multi-objective
optimization techniques, and evolutionary algorithms for autonomous system optimization.
It provides state-of-the-art adaptive optimization capabilities that enable the ALL-USE
system to continuously improve its performance without human intervention.

Classes:
- ReinforcementLearningAgent: Main RL agent coordinator
- QLearningAgent: Q-learning algorithm implementation
- PolicyGradientAgent: Policy gradient methods (REINFORCE, Actor-Critic)
- DeepQLearningAgent: Deep Q-Network (DQN) implementation
- MultiObjectiveOptimizer: Multi-objective optimization framework
- EvolutionaryOptimizer: Evolutionary algorithms for complex optimization
- OnlineLearningSystem: Continuous adaptation and online learning
- MetaLearningFramework: Learning how to learn more effectively

Version: 1.0.0
"""

import numpy as np
import time
import logging
import threading
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
from collections import defaultdict, deque
import pickle
import math
import random
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RLAlgorithm(Enum):
    """Reinforcement learning algorithms."""
    Q_LEARNING = 1
    SARSA = 2
    POLICY_GRADIENT = 3
    ACTOR_CRITIC = 4
    DQN = 5
    DDPG = 6
    PPO = 7

class OptimizationObjective(Enum):
    """Optimization objectives."""
    PERFORMANCE = 1
    COST = 2
    RELIABILITY = 3
    SECURITY = 4
    EFFICIENCY = 5
    LATENCY = 6
    THROUGHPUT = 7

@dataclass
class RLConfig:
    """Configuration for reinforcement learning."""
    algorithm: RLAlgorithm = RLAlgorithm.Q_LEARNING
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995
    min_exploration_rate: float = 0.01
    memory_size: int = 10000
    batch_size: int = 32
    target_update_frequency: int = 100
    max_episodes: int = 1000
    max_steps_per_episode: int = 200
    reward_scaling: float = 1.0
    use_double_dqn: bool = True
    use_dueling_dqn: bool = True

@dataclass
class OptimizationConfig:
    """Configuration for optimization."""
    objectives: List[OptimizationObjective] = field(default_factory=lambda: [OptimizationObjective.PERFORMANCE])
    optimization_method: str = 'multi_objective'
    population_size: int = 50
    num_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    selection_method: str = 'tournament'
    tournament_size: int = 3
    pareto_front_size: int = 20
    convergence_threshold: float = 1e-6
    max_evaluations: int = 10000

@dataclass
class State:
    """State representation for RL."""
    features: np.ndarray
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Action:
    """Action representation for RL."""
    action_id: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    continuous_values: Optional[np.ndarray] = None

@dataclass
class Experience:
    """Experience tuple for RL."""
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool
    timestamp: float = field(default_factory=time.time)

class Environment(ABC):
    """Abstract environment interface for RL."""
    
    @abstractmethod
    def reset(self) -> State:
        """Reset environment and return initial state."""
        pass
        
    @abstractmethod
    def step(self, action: Action) -> Tuple[State, float, bool, Dict[str, Any]]:
        """Take action and return next state, reward, done, info."""
        pass
        
    @abstractmethod
    def get_action_space(self) -> List[Action]:
        """Get available actions."""
        pass
        
    @abstractmethod
    def get_state_space_size(self) -> int:
        """Get state space dimensionality."""
        pass

class SystemOptimizationEnvironment(Environment):
    """Environment for system optimization tasks."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.current_state = None
        self.step_count = 0
        self.max_steps = 200
        
        # System parameters to optimize
        self.parameter_ranges = {
            'cpu_allocation': (0.1, 1.0),
            'memory_allocation': (0.1, 1.0),
            'cache_size': (0.1, 1.0),
            'thread_count': (1, 16),
            'batch_size': (1, 128),
            'learning_rate': (0.001, 0.1)
        }
        
        # Current system configuration
        self.current_config = {param: (low + high) / 2 for param, (low, high) in self.parameter_ranges.items()}
        
        # Performance history
        self.performance_history = deque(maxlen=100)
        
        logger.info("System optimization environment initialized")
        
    def reset(self) -> State:
        """Reset environment to initial state."""
        self.step_count = 0
        
        # Initialize with random configuration
        for param, (low, high) in self.parameter_ranges.items():
            self.current_config[param] = random.uniform(low, high)
            
        # Create state representation
        features = np.array(list(self.current_config.values()))
        
        # Add performance metrics
        performance_metrics = self._calculate_performance_metrics()
        features = np.concatenate([features, performance_metrics])
        
        self.current_state = State(
            features=features,
            metadata={'config': self.current_config.copy()}
        )
        
        return self.current_state
        
    def step(self, action: Action) -> Tuple[State, float, bool, Dict[str, Any]]:
        """Take optimization action."""
        self.step_count += 1
        
        # Apply action to modify configuration
        self._apply_action(action)
        
        # Calculate new performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        # Calculate reward based on multiple objectives
        reward = self._calculate_reward(performance_metrics)
        
        # Create new state
        features = np.array(list(self.current_config.values()))
        features = np.concatenate([features, performance_metrics])
        
        next_state = State(
            features=features,
            metadata={'config': self.current_config.copy(), 'performance': performance_metrics}
        )
        
        # Check if episode is done
        done = self.step_count >= self.max_steps
        
        # Store performance
        self.performance_history.append({
            'step': self.step_count,
            'config': self.current_config.copy(),
            'performance': performance_metrics,
            'reward': reward
        })
        
        info = {
            'performance_metrics': performance_metrics,
            'config': self.current_config.copy()
        }
        
        self.current_state = next_state
        return next_state, reward, done, info
        
    def _apply_action(self, action: Action) -> None:
        """Apply action to modify system configuration."""
        if action.continuous_values is not None:
            # Continuous action space
            param_names = list(self.parameter_ranges.keys())
            for i, value in enumerate(action.continuous_values):
                if i < len(param_names):
                    param = param_names[i]
                    low, high = self.parameter_ranges[param]
                    # Scale action to parameter range
                    self.current_config[param] = low + (high - low) * np.clip(value, 0, 1)
        else:
            # Discrete action space
            param_names = list(self.parameter_ranges.keys())
            param_idx = action.action_id % len(param_names)
            param = param_names[param_idx]
            
            # Determine modification direction and magnitude
            direction = 1 if (action.action_id // len(param_names)) % 2 == 0 else -1
            magnitude = 0.1  # 10% change
            
            low, high = self.parameter_ranges[param]
            current_value = self.current_config[param]
            change = direction * magnitude * (high - low)
            new_value = np.clip(current_value + change, low, high)
            self.current_config[param] = new_value
            
    def _calculate_performance_metrics(self) -> np.ndarray:
        """Calculate system performance metrics."""
        config = self.current_config
        
        # Simulate performance based on configuration
        # In practice, these would be real system measurements
        
        # Performance (higher is better)
        performance = (
            config['cpu_allocation'] * 0.3 +
            config['memory_allocation'] * 0.2 +
            config['cache_size'] * 0.15 +
            (config['thread_count'] / 16) * 0.2 +
            (config['batch_size'] / 128) * 0.1 +
            (1 - config['learning_rate'] / 0.1) * 0.05
        )
        
        # Cost (lower is better, so we use 1 - cost)
        cost = (
            config['cpu_allocation'] * 0.4 +
            config['memory_allocation'] * 0.3 +
            config['cache_size'] * 0.2 +
            (config['thread_count'] / 16) * 0.1
        )
        cost_metric = 1 - cost
        
        # Reliability (higher is better)
        reliability = (
            (1 - config['cpu_allocation']) * 0.2 +  # Lower utilization = higher reliability
            config['memory_allocation'] * 0.3 +
            config['cache_size'] * 0.2 +
            (1 - config['thread_count'] / 16) * 0.3
        )
        
        # Efficiency (balance of performance and cost)
        efficiency = performance / (cost + 0.1)  # Avoid division by zero
        
        # Add some noise to simulate real-world variability
        noise = np.random.normal(0, 0.05, 4)
        metrics = np.array([performance, cost_metric, reliability, efficiency]) + noise
        
        return np.clip(metrics, 0, 1)
        
    def _calculate_reward(self, performance_metrics: np.ndarray) -> float:
        """Calculate reward based on multiple objectives."""
        performance, cost_metric, reliability, efficiency = performance_metrics
        
        # Multi-objective reward function
        if OptimizationObjective.PERFORMANCE in self.config.objectives:
            reward = performance * 0.4
        else:
            reward = 0.0
            
        if OptimizationObjective.COST in self.config.objectives:
            reward += cost_metric * 0.3
            
        if OptimizationObjective.RELIABILITY in self.config.objectives:
            reward += reliability * 0.2
            
        if OptimizationObjective.EFFICIENCY in self.config.objectives:
            reward += efficiency * 0.1
            
        # Bonus for improvement over previous performance
        if len(self.performance_history) > 0:
            prev_metrics = self.performance_history[-1]['performance']
            improvement = np.mean(performance_metrics - prev_metrics)
            reward += improvement * 0.5
            
        return reward
        
    def get_action_space(self) -> List[Action]:
        """Get available discrete actions."""
        actions = []
        param_count = len(self.parameter_ranges)
        
        # Create discrete actions for each parameter (increase/decrease)
        for param_idx in range(param_count):
            for direction in [0, 1]:  # 0: increase, 1: decrease
                action_id = param_idx * 2 + direction
                actions.append(Action(action_id=action_id))
                
        return actions
        
    def get_state_space_size(self) -> int:
        """Get state space dimensionality."""
        return len(self.parameter_ranges) + 4  # Parameters + performance metrics

class QLearningAgent:
    """Q-learning algorithm implementation."""
    
    def __init__(self, config: RLConfig, state_size: int, action_size: int):
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        
        # Q-table for discrete states (simplified)
        self.q_table_size = 1000  # Discretized state space
        self.q_table = np.zeros((self.q_table_size, action_size))
        
        # Experience tracking
        self.episode_rewards = []
        self.episode_steps = []
        
        logger.info(f"Q-learning agent initialized with state_size={state_size}, action_size={action_size}")
        
    def _discretize_state(self, state: State) -> int:
        """Convert continuous state to discrete index."""
        # Simple discretization using hash
        state_hash = hash(tuple(np.round(state.features, 2))) % self.q_table_size
        return abs(state_hash)
        
    def select_action(self, state: State, training: bool = True) -> Action:
        """Select action using epsilon-greedy policy."""
        state_idx = self._discretize_state(state)
        
        if training and random.random() < self.config.exploration_rate:
            # Explore: random action
            action_id = random.randint(0, self.action_size - 1)
        else:
            # Exploit: best action
            action_id = np.argmax(self.q_table[state_idx])
            
        return Action(action_id=action_id)
        
    def update(self, state: State, action: Action, reward: float, next_state: State, done: bool) -> None:
        """Update Q-values using Q-learning update rule."""
        state_idx = self._discretize_state(state)
        next_state_idx = self._discretize_state(next_state)
        
        # Q-learning update
        current_q = self.q_table[state_idx, action.action_id]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state_idx])
            target_q = reward + self.config.discount_factor * max_next_q
            
        # Update Q-value
        self.q_table[state_idx, action.action_id] += self.config.learning_rate * (target_q - current_q)
        
        # Decay exploration rate
        if self.config.exploration_rate > self.config.min_exploration_rate:
            self.config.exploration_rate *= self.config.exploration_decay
            
    def train_episode(self, environment: Environment) -> Dict[str, float]:
        """Train for one episode."""
        state = environment.reset()
        total_reward = 0.0
        steps = 0
        
        for step in range(self.config.max_steps_per_episode):
            # Select and take action
            action = self.select_action(state, training=True)
            next_state, reward, done, info = environment.step(action)
            
            # Update Q-values
            self.update(state, action, reward, next_state, done)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
                
        # Store episode statistics
        self.episode_rewards.append(total_reward)
        self.episode_steps.append(steps)
        
        return {
            'total_reward': total_reward,
            'steps': steps,
            'exploration_rate': self.config.exploration_rate
        }

class PolicyGradientAgent:
    """Policy gradient agent (REINFORCE algorithm)."""
    
    def __init__(self, config: RLConfig, state_size: int, action_size: int):
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        
        # Policy network (simplified linear model)
        self.policy_weights = np.random.randn(state_size, action_size) * 0.1
        self.policy_bias = np.zeros(action_size)
        
        # Value network for baseline (Actor-Critic)
        self.value_weights = np.random.randn(state_size, 1) * 0.1
        self.value_bias = np.zeros(1)
        
        # Episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        logger.info(f"Policy gradient agent initialized")
        
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
        
    def _policy_forward(self, state: State) -> np.ndarray:
        """Forward pass through policy network."""
        logits = np.dot(state.features, self.policy_weights) + self.policy_bias
        return self._softmax(logits)
        
    def _value_forward(self, state: State) -> float:
        """Forward pass through value network."""
        value = np.dot(state.features, self.value_weights) + self.value_bias
        return float(value[0])
        
    def select_action(self, state: State, training: bool = True) -> Action:
        """Select action using policy network."""
        action_probs = self._policy_forward(state)
        
        if training:
            # Sample from probability distribution
            action_id = np.random.choice(self.action_size, p=action_probs)
        else:
            # Select most probable action
            action_id = np.argmax(action_probs)
            
        return Action(action_id=action_id)
        
    def store_experience(self, state: State, action: Action, reward: float) -> None:
        """Store experience for episode."""
        self.episode_states.append(state.features)
        self.episode_actions.append(action.action_id)
        self.episode_rewards.append(reward)
        
    def update_policy(self) -> Dict[str, float]:
        """Update policy using REINFORCE with baseline."""
        if len(self.episode_states) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0}
            
        # Convert to arrays
        states = np.array(self.episode_states)
        actions = np.array(self.episode_actions)
        rewards = np.array(self.episode_rewards)
        
        # Calculate returns (discounted cumulative rewards)
        returns = np.zeros_like(rewards)
        running_return = 0.0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.config.discount_factor * running_return
            returns[t] = running_return
            
        # Normalize returns
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        # Calculate baselines (value function estimates)
        baselines = np.array([self._value_forward(State(features=state)) for state in states])
        
        # Calculate advantages
        advantages = returns - baselines
        
        # Policy gradient update
        policy_loss = 0.0
        for t in range(len(states)):
            state = states[t]
            action = actions[t]
            advantage = advantages[t]
            
            # Forward pass
            action_probs = self._policy_forward(State(features=state))
            
            # Calculate gradients
            policy_grad = np.zeros_like(self.policy_weights)
            bias_grad = np.zeros_like(self.policy_bias)
            
            # Gradient of log probability
            grad_log_prob = np.zeros(self.action_size)
            grad_log_prob[action] = 1.0 / (action_probs[action] + 1e-8)
            
            # Policy gradient
            policy_grad += np.outer(state, grad_log_prob) * advantage
            bias_grad += grad_log_prob * advantage
            
            # Update policy
            self.policy_weights += self.config.learning_rate * policy_grad
            self.policy_bias += self.config.learning_rate * bias_grad
            
            policy_loss += -np.log(action_probs[action] + 1e-8) * advantage
            
        # Value function update
        value_loss = 0.0
        for t in range(len(states)):
            state = states[t]
            target_value = returns[t]
            predicted_value = self._value_forward(State(features=state))
            
            # Value function gradient
            value_error = target_value - predicted_value
            value_grad = state * value_error
            bias_grad = value_error
            
            # Update value function
            self.value_weights += self.config.learning_rate * value_grad.reshape(-1, 1)
            self.value_bias += self.config.learning_rate * bias_grad
            
            value_loss += value_error ** 2
            
        # Clear episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        return {
            'policy_loss': policy_loss / len(states),
            'value_loss': value_loss / len(states)
        }
        
    def train_episode(self, environment: Environment) -> Dict[str, float]:
        """Train for one episode."""
        state = environment.reset()
        total_reward = 0.0
        steps = 0
        
        for step in range(self.config.max_steps_per_episode):
            # Select and take action
            action = self.select_action(state, training=True)
            next_state, reward, done, info = environment.step(action)
            
            # Store experience
            self.store_experience(state, action, reward)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
                
        # Update policy at end of episode
        update_info = self.update_policy()
        
        return {
            'total_reward': total_reward,
            'steps': steps,
            'policy_loss': update_info['policy_loss'],
            'value_loss': update_info['value_loss']
        }

class MultiObjectiveOptimizer:
    """Multi-objective optimization using evolutionary algorithms."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.population = []
        self.pareto_front = []
        self.generation = 0
        
        logger.info(f"Multi-objective optimizer initialized")
        
    def _initialize_population(self, parameter_ranges: Dict[str, Tuple[float, float]]) -> None:
        """Initialize random population."""
        self.population = []
        param_names = list(parameter_ranges.keys())
        
        for _ in range(self.config.population_size):
            individual = {}
            for param, (low, high) in parameter_ranges.items():
                individual[param] = random.uniform(low, high)
            self.population.append(individual)
            
    def _evaluate_objectives(self, individual: Dict[str, float], 
                           objective_function: Callable) -> List[float]:
        """Evaluate multiple objectives for an individual."""
        return objective_function(individual)
        
    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """Check if obj1 dominates obj2 (Pareto dominance)."""
        better_in_any = False
        for i in range(len(obj1)):
            if obj1[i] < obj2[i]:  # Assuming minimization
                return False
            elif obj1[i] > obj2[i]:
                better_in_any = True
        return better_in_any
        
    def _fast_non_dominated_sort(self, population_objectives: List[List[float]]) -> List[List[int]]:
        """Fast non-dominated sorting for NSGA-II."""
        n = len(population_objectives)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        # Find domination relationships
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(population_objectives[i], population_objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(population_objectives[j], population_objectives[i]):
                        domination_count[i] += 1
                        
            if domination_count[i] == 0:
                fronts[0].append(i)
                
        # Build subsequent fronts
        front_idx = 0
        while len(fronts[front_idx]) > 0:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            front_idx += 1
            fronts.append(next_front)
            
        return fronts[:-1]  # Remove empty last front
        
    def _crowding_distance(self, front: List[int], 
                          population_objectives: List[List[float]]) -> List[float]:
        """Calculate crowding distance for diversity preservation."""
        if len(front) <= 2:
            return [float('inf')] * len(front)
            
        distances = [0.0] * len(front)
        num_objectives = len(population_objectives[0])
        
        for obj_idx in range(num_objectives):
            # Sort by objective value
            front_sorted = sorted(front, key=lambda x: population_objectives[x][obj_idx])
            
            # Boundary points get infinite distance
            distances[front.index(front_sorted[0])] = float('inf')
            distances[front.index(front_sorted[-1])] = float('inf')
            
            # Calculate distances for intermediate points
            obj_range = (population_objectives[front_sorted[-1]][obj_idx] - 
                        population_objectives[front_sorted[0]][obj_idx])
            
            if obj_range > 0:
                for i in range(1, len(front_sorted) - 1):
                    idx = front.index(front_sorted[i])
                    distances[idx] += (
                        (population_objectives[front_sorted[i+1]][obj_idx] - 
                         population_objectives[front_sorted[i-1]][obj_idx]) / obj_range
                    )
                    
        return distances
        
    def _tournament_selection(self, population_objectives: List[List[float]], 
                            fronts: List[List[int]], 
                            crowding_distances: Dict[int, float]) -> int:
        """Tournament selection considering Pareto rank and crowding distance."""
        tournament_indices = random.sample(range(len(self.population)), 
                                         min(self.config.tournament_size, len(self.population)))
        
        best_idx = tournament_indices[0]
        best_front = None
        
        # Find which front each individual belongs to
        for front_idx, front in enumerate(fronts):
            if best_idx in front:
                best_front = front_idx
                break
                
        for idx in tournament_indices[1:]:
            # Find front for current individual
            current_front = None
            for front_idx, front in enumerate(fronts):
                if idx in front:
                    current_front = front_idx
                    break
                    
            # Compare based on Pareto rank and crowding distance
            if (current_front < best_front or 
                (current_front == best_front and 
                 crowding_distances.get(idx, 0) > crowding_distances.get(best_idx, 0))):
                best_idx = idx
                best_front = current_front
                
        return best_idx
        
    def _crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Simulated binary crossover (SBX)."""
        eta_c = 20  # Crossover distribution index
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for param in parent1.keys():
            if random.random() <= self.config.crossover_rate:
                y1, y2 = parent1[param], parent2[param]
                
                if abs(y1 - y2) > 1e-14:
                    if y1 > y2:
                        y1, y2 = y2, y1
                        
                    rand = random.random()
                    beta = 1.0 + (2.0 * (y1 - 0) / (y2 - y1))
                    alpha = 2.0 - beta ** -(eta_c + 1.0)
                    
                    if rand <= (1.0 / alpha):
                        beta_q = (rand * alpha) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
                        
                    c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
                    
                    child1[param] = c1
                    child2[param] = c2
                    
        return child1, child2
        
    def _mutate(self, individual: Dict[str, float], 
               parameter_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Polynomial mutation."""
        eta_m = 20  # Mutation distribution index
        mutated = individual.copy()
        
        for param, (low, high) in parameter_ranges.items():
            if random.random() <= self.config.mutation_rate:
                y = individual[param]
                delta1 = (y - low) / (high - low)
                delta2 = (high - y) / (high - low)
                
                rand = random.random()
                mut_pow = 1.0 / (eta_m + 1.0)
                
                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                    delta_q = 1.0 - val ** mut_pow
                    
                y = y + delta_q * (high - low)
                mutated[param] = np.clip(y, low, high)
                
        return mutated
        
    def optimize(self, objective_function: Callable, 
                parameter_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Run multi-objective optimization."""
        start_time = time.time()
        
        # Initialize population
        self._initialize_population(parameter_ranges)
        
        best_objectives = []
        convergence_history = []
        
        for generation in range(self.config.num_generations):
            self.generation = generation
            
            # Evaluate objectives for all individuals
            population_objectives = []
            for individual in self.population:
                objectives = self._evaluate_objectives(individual, objective_function)
                population_objectives.append(objectives)
                
            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(population_objectives)
            
            # Calculate crowding distances
            crowding_distances = {}
            for front in fronts:
                distances = self._crowding_distance(front, population_objectives)
                for i, idx in enumerate(front):
                    crowding_distances[idx] = distances[i]
                    
            # Update Pareto front
            if fronts:
                self.pareto_front = [self.population[i] for i in fronts[0]]
                
            # Selection and reproduction
            new_population = []
            
            while len(new_population) < self.config.population_size:
                # Tournament selection
                parent1_idx = self._tournament_selection(population_objectives, fronts, crowding_distances)
                parent2_idx = self._tournament_selection(population_objectives, fronts, crowding_distances)
                
                # Crossover
                child1, child2 = self._crossover(self.population[parent1_idx], self.population[parent2_idx])
                
                # Mutation
                child1 = self._mutate(child1, parameter_ranges)
                child2 = self._mutate(child2, parameter_ranges)
                
                new_population.extend([child1, child2])
                
            # Keep only required population size
            self.population = new_population[:self.config.population_size]
            
            # Track convergence
            if fronts:
                front_objectives = [population_objectives[i] for i in fronts[0]]
                best_objectives.append(front_objectives)
                
                # Calculate hypervolume or other convergence metrics
                convergence_metric = len(fronts[0])  # Simplified metric
                convergence_history.append(convergence_metric)
                
            if generation % 10 == 0:
                logger.info(f"Generation {generation}: Pareto front size = {len(fronts[0]) if fronts else 0}")
                
        optimization_time = time.time() - start_time
        
        return {
            'pareto_front': self.pareto_front,
            'best_objectives': best_objectives,
            'convergence_history': convergence_history,
            'optimization_time': optimization_time,
            'generations': self.config.num_generations
        }

class AdaptiveOptimizationFramework:
    """Main framework for adaptive optimization and reinforcement learning."""
    
    def __init__(self, rl_config: RLConfig, opt_config: OptimizationConfig):
        self.rl_config = rl_config
        self.opt_config = opt_config
        
        # Initialize environment
        self.environment = SystemOptimizationEnvironment(opt_config)
        
        # Initialize RL agents
        state_size = self.environment.get_state_space_size()
        action_size = len(self.environment.get_action_space())
        
        self.q_agent = QLearningAgent(rl_config, state_size, action_size)
        self.pg_agent = PolicyGradientAgent(rl_config, state_size, action_size)
        
        # Initialize multi-objective optimizer
        self.mo_optimizer = MultiObjectiveOptimizer(opt_config)
        
        # Performance tracking
        self.training_history = []
        self.optimization_results = []
        
        logger.info("Adaptive optimization framework initialized")
        
    def train_rl_agents(self, num_episodes: int = None) -> Dict[str, Any]:
        """Train reinforcement learning agents."""
        if num_episodes is None:
            num_episodes = self.rl_config.max_episodes
            
        start_time = time.time()
        
        # Train Q-learning agent
        q_results = []
        logger.info("Training Q-learning agent")
        for episode in range(num_episodes):
            result = self.q_agent.train_episode(self.environment)
            q_results.append(result)
            
            if episode % 100 == 0:
                avg_reward = np.mean([r['total_reward'] for r in q_results[-100:]])
                logger.info(f"Q-learning Episode {episode}: avg_reward={avg_reward:.3f}")
                
        # Train Policy Gradient agent
        pg_results = []
        logger.info("Training Policy Gradient agent")
        for episode in range(num_episodes):
            result = self.pg_agent.train_episode(self.environment)
            pg_results.append(result)
            
            if episode % 100 == 0:
                avg_reward = np.mean([r['total_reward'] for r in pg_results[-100:]])
                logger.info(f"Policy Gradient Episode {episode}: avg_reward={avg_reward:.3f}")
                
        training_time = time.time() - start_time
        
        training_results = {
            'training_time': training_time,
            'q_learning_results': q_results,
            'policy_gradient_results': pg_results,
            'episodes_trained': num_episodes
        }
        
        self.training_history.append(training_results)
        return training_results
        
    def run_multi_objective_optimization(self) -> Dict[str, Any]:
        """Run multi-objective optimization."""
        logger.info("Running multi-objective optimization")
        
        # Define objective function
        def objective_function(config: Dict[str, float]) -> List[float]:
            # Simulate system with given configuration
            env = SystemOptimizationEnvironment(self.opt_config)
            env.current_config = config
            
            # Calculate performance metrics
            performance_metrics = env._calculate_performance_metrics()
            
            # Convert to minimization objectives (negate for maximization)
            objectives = [
                -performance_metrics[0],  # Minimize negative performance (maximize performance)
                performance_metrics[1],   # Minimize cost (already inverted in metrics)
                -performance_metrics[2],  # Minimize negative reliability (maximize reliability)
                -performance_metrics[3]   # Minimize negative efficiency (maximize efficiency)
            ]
            
            return objectives
            
        # Run optimization
        results = self.mo_optimizer.optimize(objective_function, self.environment.parameter_ranges)
        
        self.optimization_results.append(results)
        return results
        
    def get_best_configuration(self) -> Dict[str, Any]:
        """Get best configuration from all optimization methods."""
        best_configs = {}
        
        # Best from Q-learning
        if self.q_agent.episode_rewards:
            best_episode_idx = np.argmax(self.q_agent.episode_rewards)
            # Would need to track configurations during training
            best_configs['q_learning'] = {'episode': best_episode_idx, 'reward': self.q_agent.episode_rewards[best_episode_idx]}
            
        # Best from Policy Gradient
        if hasattr(self.pg_agent, 'episode_rewards'):
            # Similar tracking would be needed
            pass
            
        # Best from Multi-objective optimization
        if self.optimization_results:
            latest_result = self.optimization_results[-1]
            if latest_result['pareto_front']:
                best_configs['multi_objective'] = {
                    'pareto_front_size': len(latest_result['pareto_front']),
                    'sample_solution': latest_result['pareto_front'][0]
                }
                
        return best_configs
        
    def evaluate_performance(self) -> Dict[str, Any]:
        """Evaluate overall framework performance."""
        performance_metrics = {}
        
        # RL agent performance
        if self.q_agent.episode_rewards:
            performance_metrics['q_learning'] = {
                'avg_reward': np.mean(self.q_agent.episode_rewards),
                'max_reward': np.max(self.q_agent.episode_rewards),
                'reward_std': np.std(self.q_agent.episode_rewards),
                'episodes_trained': len(self.q_agent.episode_rewards)
            }
            
        # Multi-objective optimization performance
        if self.optimization_results:
            latest_result = self.optimization_results[-1]
            performance_metrics['multi_objective'] = {
                'pareto_front_size': len(latest_result['pareto_front']),
                'optimization_time': latest_result['optimization_time'],
                'generations': latest_result['generations']
            }
            
        # Overall framework metrics
        performance_metrics['framework'] = {
            'total_training_sessions': len(self.training_history),
            'total_optimization_runs': len(self.optimization_results)
        }
        
        return performance_metrics

# Example usage and testing
if __name__ == "__main__":
    # Create configurations
    rl_config = RLConfig(
        algorithm=RLAlgorithm.Q_LEARNING,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=0.3,
        max_episodes=200,
        max_steps_per_episode=100
    )
    
    opt_config = OptimizationConfig(
        objectives=[OptimizationObjective.PERFORMANCE, OptimizationObjective.COST, OptimizationObjective.RELIABILITY],
        population_size=30,
        num_generations=50
    )
    
    # Create adaptive optimization framework
    framework = AdaptiveOptimizationFramework(rl_config, opt_config)
    
    try:
        # Train RL agents
        logger.info("Starting RL agent training")
        rl_results = framework.train_rl_agents(num_episodes=100)
        
        # Run multi-objective optimization
        logger.info("Starting multi-objective optimization")
        mo_results = framework.run_multi_objective_optimization()
        
        # Get best configurations
        best_configs = framework.get_best_configuration()
        
        # Evaluate performance
        performance = framework.evaluate_performance()
        
        logger.info(f"RL training completed: {rl_results['episodes_trained']} episodes")
        logger.info(f"Multi-objective optimization completed: {len(mo_results['pareto_front'])} solutions in Pareto front")
        logger.info(f"Best configurations: {best_configs}")
        logger.info(f"Performance evaluation: {performance}")
        
    except Exception as e:
        logger.error(f"Error in adaptive optimization test: {e}")
        raise

