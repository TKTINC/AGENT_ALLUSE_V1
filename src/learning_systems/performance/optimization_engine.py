"""
WS5-P5: Optimization Engine
Advanced optimization engine for autonomous performance enhancement.

This module provides comprehensive optimization capabilities including:
- Multi-algorithm parameter optimization
- Resource allocation optimization
- Performance-driven decision making
- Automated configuration management
- Intelligent optimization strategy selection
"""

import time
import threading
import json
import logging
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import os
from abc import ABC, abstractmethod
from scipy.optimize import minimize, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationParameter:
    """Represents a parameter to be optimized."""
    name: str
    current_value: float
    min_value: float
    max_value: float
    parameter_type: str  # 'continuous', 'discrete', 'categorical'
    step_size: Optional[float] = None
    categories: Optional[List[str]] = None
    importance: float = 1.0
    
    def validate_value(self, value: float) -> bool:
        """Validate if value is within parameter bounds."""
        return self.min_value <= value <= self.max_value
    
    def normalize_value(self, value: float) -> float:
        """Normalize value to [0, 1] range."""
        return (value - self.min_value) / (self.max_value - self.min_value)
    
    def denormalize_value(self, normalized_value: float) -> float:
        """Denormalize value from [0, 1] range."""
        return self.min_value + normalized_value * (self.max_value - self.min_value)

@dataclass
class OptimizationResult:
    """Represents the result of an optimization operation."""
    optimization_id: str
    algorithm: str
    parameters: Dict[str, float]
    objective_value: float
    improvement: float
    execution_time: float
    iterations: int
    success: bool
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'optimization_id': self.optimization_id,
            'algorithm': self.algorithm,
            'parameters': self.parameters,
            'objective_value': self.objective_value,
            'improvement': self.improvement,
            'execution_time': self.execution_time,
            'iterations': self.iterations,
            'success': self.success,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata or {}
        }

class OptimizationAlgorithm(ABC):
    """Abstract base class for optimization algorithms."""
    
    @abstractmethod
    def optimize(self, 
                 objective_function: Callable[[Dict[str, float]], float],
                 parameters: Dict[str, OptimizationParameter],
                 max_iterations: int = 100) -> OptimizationResult:
        """Perform optimization using the algorithm."""
        pass
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Get the name of the optimization algorithm."""
        pass

class GeneticAlgorithmOptimizer(OptimizationAlgorithm):
    """Genetic Algorithm optimization implementation."""
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        """Initialize genetic algorithm optimizer."""
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
    def get_algorithm_name(self) -> str:
        return "genetic_algorithm"
    
    def optimize(self, 
                 objective_function: Callable[[Dict[str, float]], float],
                 parameters: Dict[str, OptimizationParameter],
                 max_iterations: int = 100) -> OptimizationResult:
        """Perform genetic algorithm optimization."""
        start_time = time.time()
        optimization_id = f"ga_{int(time.time())}"
        
        # Initialize population
        population = self._initialize_population(parameters)
        best_individual = None
        best_fitness = float('-inf')
        
        for generation in range(max_iterations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                try:
                    fitness = objective_function(individual)
                    fitness_scores.append(fitness)
                    
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_individual = individual.copy()
                except Exception as e:
                    fitness_scores.append(float('-inf'))
                    logger.warning(f"Error evaluating individual: {e}")
            
            # Selection, crossover, and mutation
            population = self._evolve_population(population, fitness_scores, parameters)
        
        execution_time = time.time() - start_time
        
        # Calculate improvement
        initial_params = {name: param.current_value for name, param in parameters.items()}
        initial_fitness = objective_function(initial_params)
        improvement = ((best_fitness - initial_fitness) / abs(initial_fitness)) * 100 if initial_fitness != 0 else 0
        
        return OptimizationResult(
            optimization_id=optimization_id,
            algorithm=self.get_algorithm_name(),
            parameters=best_individual or initial_params,
            objective_value=best_fitness,
            improvement=improvement,
            execution_time=execution_time,
            iterations=max_iterations,
            success=best_individual is not None,
            timestamp=datetime.now()
        )
    
    def _initialize_population(self, parameters: Dict[str, OptimizationParameter]) -> List[Dict[str, float]]:
        """Initialize random population."""
        population = []
        for _ in range(self.population_size):
            individual = {}
            for name, param in parameters.items():
                if param.parameter_type == 'continuous':
                    individual[name] = random.uniform(param.min_value, param.max_value)
                elif param.parameter_type == 'discrete':
                    individual[name] = random.randint(int(param.min_value), int(param.max_value))
            population.append(individual)
        return population
    
    def _evolve_population(self, population: List[Dict[str, float]], 
                          fitness_scores: List[float],
                          parameters: Dict[str, OptimizationParameter]) -> List[Dict[str, float]]:
        """Evolve population through selection, crossover, and mutation."""
        # Selection (tournament selection)
        selected = self._tournament_selection(population, fitness_scores)
        
        # Crossover
        offspring = []
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]
            
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2, parameters)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1.copy(), parent2.copy()])
        
        # Mutation
        for individual in offspring:
            if random.random() < self.mutation_rate:
                self._mutate(individual, parameters)
        
        return offspring[:self.population_size]
    
    def _tournament_selection(self, population: List[Dict[str, float]], 
                             fitness_scores: List[float], 
                             tournament_size: int = 3) -> List[Dict[str, float]]:
        """Tournament selection for genetic algorithm."""
        selected = []
        for _ in range(self.population_size):
            tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
            winner_index = max(tournament_indices, key=lambda i: fitness_scores[i])
            selected.append(population[winner_index].copy())
        return selected
    
    def _crossover(self, parent1: Dict[str, float], parent2: Dict[str, float],
                   parameters: Dict[str, OptimizationParameter]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Single-point crossover for genetic algorithm."""
        child1, child2 = parent1.copy(), parent2.copy()
        
        param_names = list(parameters.keys())
        crossover_point = random.randint(1, len(param_names) - 1)
        
        for i in range(crossover_point, len(param_names)):
            param_name = param_names[i]
            child1[param_name], child2[param_name] = child2[param_name], child1[param_name]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, float], parameters: Dict[str, OptimizationParameter]):
        """Mutation for genetic algorithm."""
        for name, param in parameters.items():
            if random.random() < 0.1:  # 10% chance to mutate each parameter
                if param.parameter_type == 'continuous':
                    mutation_strength = (param.max_value - param.min_value) * 0.1
                    individual[name] += random.gauss(0, mutation_strength)
                    individual[name] = max(param.min_value, min(param.max_value, individual[name]))
                elif param.parameter_type == 'discrete':
                    individual[name] = random.randint(int(param.min_value), int(param.max_value))

class BayesianOptimizer(OptimizationAlgorithm):
    """Bayesian Optimization using Gaussian Processes."""
    
    def __init__(self, acquisition_function: str = 'expected_improvement', exploration_weight: float = 0.1):
        """Initialize Bayesian optimizer."""
        self.acquisition_function = acquisition_function
        self.exploration_weight = exploration_weight
        self.gp = GaussianProcessRegressor(kernel=Matern(length_scale=1.0, nu=2.5), alpha=1e-6)
        
    def get_algorithm_name(self) -> str:
        return "bayesian_optimization"
    
    def optimize(self, 
                 objective_function: Callable[[Dict[str, float]], float],
                 parameters: Dict[str, OptimizationParameter],
                 max_iterations: int = 100) -> OptimizationResult:
        """Perform Bayesian optimization."""
        start_time = time.time()
        optimization_id = f"bo_{int(time.time())}"
        
        # Initialize with random samples
        X_samples = []
        y_samples = []
        best_params = None
        best_value = float('-inf')
        
        # Initial random sampling
        n_initial = min(10, max_iterations // 2)
        for _ in range(n_initial):
            params = self._sample_random_parameters(parameters)
            try:
                value = objective_function(params)
                X_samples.append(self._params_to_array(params, parameters))
                y_samples.append(value)
                
                if value > best_value:
                    best_value = value
                    best_params = params.copy()
            except Exception as e:
                logger.warning(f"Error in initial sampling: {e}")
        
        # Bayesian optimization loop
        for iteration in range(n_initial, max_iterations):
            if len(X_samples) < 2:
                break
                
            # Fit Gaussian Process
            X_array = np.array(X_samples)
            y_array = np.array(y_samples)
            
            try:
                self.gp.fit(X_array, y_array)
                
                # Find next point to evaluate
                next_params = self._acquire_next_point(parameters)
                next_value = objective_function(next_params)
                
                X_samples.append(self._params_to_array(next_params, parameters))
                y_samples.append(next_value)
                
                if next_value > best_value:
                    best_value = next_value
                    best_params = next_params.copy()
                    
            except Exception as e:
                logger.warning(f"Error in Bayesian optimization iteration {iteration}: {e}")
                # Fall back to random sampling
                params = self._sample_random_parameters(parameters)
                try:
                    value = objective_function(params)
                    X_samples.append(self._params_to_array(params, parameters))
                    y_samples.append(value)
                    
                    if value > best_value:
                        best_value = value
                        best_params = params.copy()
                except Exception:
                    pass
        
        execution_time = time.time() - start_time
        
        # Calculate improvement
        initial_params = {name: param.current_value for name, param in parameters.items()}
        try:
            initial_value = objective_function(initial_params)
            improvement = ((best_value - initial_value) / abs(initial_value)) * 100 if initial_value != 0 else 0
        except:
            improvement = 0
        
        return OptimizationResult(
            optimization_id=optimization_id,
            algorithm=self.get_algorithm_name(),
            parameters=best_params or initial_params,
            objective_value=best_value,
            improvement=improvement,
            execution_time=execution_time,
            iterations=max_iterations,
            success=best_params is not None,
            timestamp=datetime.now()
        )
    
    def _sample_random_parameters(self, parameters: Dict[str, OptimizationParameter]) -> Dict[str, float]:
        """Sample random parameters within bounds."""
        params = {}
        for name, param in parameters.items():
            if param.parameter_type == 'continuous':
                params[name] = random.uniform(param.min_value, param.max_value)
            elif param.parameter_type == 'discrete':
                params[name] = random.randint(int(param.min_value), int(param.max_value))
        return params
    
    def _params_to_array(self, params: Dict[str, float], parameters: Dict[str, OptimizationParameter]) -> np.ndarray:
        """Convert parameter dictionary to normalized array."""
        array = []
        for name in sorted(parameters.keys()):
            param = parameters[name]
            normalized = param.normalize_value(params[name])
            array.append(normalized)
        return np.array(array)
    
    def _array_to_params(self, array: np.ndarray, parameters: Dict[str, OptimizationParameter]) -> Dict[str, float]:
        """Convert normalized array to parameter dictionary."""
        params = {}
        for i, name in enumerate(sorted(parameters.keys())):
            param = parameters[name]
            value = param.denormalize_value(array[i])
            if param.parameter_type == 'discrete':
                value = round(value)
            params[name] = value
        return params
    
    def _acquire_next_point(self, parameters: Dict[str, OptimizationParameter]) -> Dict[str, float]:
        """Acquire next point using acquisition function."""
        # Simple random search for acquisition function optimization
        best_acquisition = float('-inf')
        best_params = None
        
        for _ in range(1000):  # Random search iterations
            candidate_array = np.random.random(len(parameters))
            candidate_params = self._array_to_params(candidate_array, parameters)
            
            try:
                acquisition_value = self._expected_improvement(candidate_array)
                if acquisition_value > best_acquisition:
                    best_acquisition = acquisition_value
                    best_params = candidate_params
            except Exception:
                continue
        
        return best_params or self._sample_random_parameters(parameters)
    
    def _expected_improvement(self, x: np.ndarray) -> float:
        """Calculate expected improvement acquisition function."""
        try:
            x = x.reshape(1, -1)
            mu, sigma = self.gp.predict(x, return_std=True)
            
            if sigma[0] == 0:
                return 0
            
            # Current best value
            y_best = np.max(self.gp.y_train_) if hasattr(self.gp, 'y_train_') else 0
            
            # Expected improvement calculation
            z = (mu[0] - y_best - self.exploration_weight) / sigma[0]
            ei = (mu[0] - y_best - self.exploration_weight) * self._normal_cdf(z) + sigma[0] * self._normal_pdf(z)
            
            return ei
        except Exception:
            return 0
    
    def _normal_cdf(self, x: float) -> float:
        """Standard normal cumulative distribution function."""
        return 0.5 * (1 + np.tanh(x / np.sqrt(2)))
    
    def _normal_pdf(self, x: float) -> float:
        """Standard normal probability density function."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

class ParticleSwarmOptimizer(OptimizationAlgorithm):
    """Particle Swarm Optimization implementation."""
    
    def __init__(self, swarm_size: int = 30, inertia: float = 0.7, cognitive: float = 1.5, social: float = 1.5):
        """Initialize particle swarm optimizer."""
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        
    def get_algorithm_name(self) -> str:
        return "particle_swarm_optimization"
    
    def optimize(self, 
                 objective_function: Callable[[Dict[str, float]], float],
                 parameters: Dict[str, OptimizationParameter],
                 max_iterations: int = 100) -> OptimizationResult:
        """Perform particle swarm optimization."""
        start_time = time.time()
        optimization_id = f"pso_{int(time.time())}"
        
        # Initialize swarm
        particles = self._initialize_swarm(parameters)
        velocities = self._initialize_velocities(parameters)
        personal_best = [p.copy() for p in particles]
        personal_best_fitness = [float('-inf')] * self.swarm_size
        global_best = None
        global_best_fitness = float('-inf')
        
        for iteration in range(max_iterations):
            # Evaluate particles
            for i, particle in enumerate(particles):
                try:
                    fitness = objective_function(particle)
                    
                    # Update personal best
                    if fitness > personal_best_fitness[i]:
                        personal_best_fitness[i] = fitness
                        personal_best[i] = particle.copy()
                    
                    # Update global best
                    if fitness > global_best_fitness:
                        global_best_fitness = fitness
                        global_best = particle.copy()
                        
                except Exception as e:
                    logger.warning(f"Error evaluating particle {i}: {e}")
            
            # Update velocities and positions
            for i in range(self.swarm_size):
                self._update_particle(i, particles, velocities, personal_best, global_best, parameters)
        
        execution_time = time.time() - start_time
        
        # Calculate improvement
        initial_params = {name: param.current_value for name, param in parameters.items()}
        try:
            initial_fitness = objective_function(initial_params)
            improvement = ((global_best_fitness - initial_fitness) / abs(initial_fitness)) * 100 if initial_fitness != 0 else 0
        except:
            improvement = 0
        
        return OptimizationResult(
            optimization_id=optimization_id,
            algorithm=self.get_algorithm_name(),
            parameters=global_best or initial_params,
            objective_value=global_best_fitness,
            improvement=improvement,
            execution_time=execution_time,
            iterations=max_iterations,
            success=global_best is not None,
            timestamp=datetime.now()
        )
    
    def _initialize_swarm(self, parameters: Dict[str, OptimizationParameter]) -> List[Dict[str, float]]:
        """Initialize particle swarm."""
        swarm = []
        for _ in range(self.swarm_size):
            particle = {}
            for name, param in parameters.items():
                if param.parameter_type == 'continuous':
                    particle[name] = random.uniform(param.min_value, param.max_value)
                elif param.parameter_type == 'discrete':
                    particle[name] = random.randint(int(param.min_value), int(param.max_value))
            swarm.append(particle)
        return swarm
    
    def _initialize_velocities(self, parameters: Dict[str, OptimizationParameter]) -> List[Dict[str, float]]:
        """Initialize particle velocities."""
        velocities = []
        for _ in range(self.swarm_size):
            velocity = {}
            for name, param in parameters.items():
                max_velocity = (param.max_value - param.min_value) * 0.1
                velocity[name] = random.uniform(-max_velocity, max_velocity)
            velocities.append(velocity)
        return velocities
    
    def _update_particle(self, i: int, particles: List[Dict[str, float]], 
                        velocities: List[Dict[str, float]],
                        personal_best: List[Dict[str, float]],
                        global_best: Dict[str, float],
                        parameters: Dict[str, OptimizationParameter]):
        """Update particle velocity and position."""
        if global_best is None:
            return
            
        for name, param in parameters.items():
            # Update velocity
            r1, r2 = random.random(), random.random()
            
            cognitive_component = self.cognitive * r1 * (personal_best[i][name] - particles[i][name])
            social_component = self.social * r2 * (global_best[name] - particles[i][name])
            
            velocities[i][name] = (self.inertia * velocities[i][name] + 
                                  cognitive_component + social_component)
            
            # Limit velocity
            max_velocity = (param.max_value - param.min_value) * 0.2
            velocities[i][name] = max(-max_velocity, min(max_velocity, velocities[i][name]))
            
            # Update position
            particles[i][name] += velocities[i][name]
            
            # Enforce bounds
            particles[i][name] = max(param.min_value, min(param.max_value, particles[i][name]))
            
            # Handle discrete parameters
            if param.parameter_type == 'discrete':
                particles[i][name] = round(particles[i][name])

class SimulatedAnnealingOptimizer(OptimizationAlgorithm):
    """Simulated Annealing optimization implementation."""
    
    def __init__(self, initial_temperature: float = 100.0, cooling_rate: float = 0.95, min_temperature: float = 0.01):
        """Initialize simulated annealing optimizer."""
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        
    def get_algorithm_name(self) -> str:
        return "simulated_annealing"
    
    def optimize(self, 
                 objective_function: Callable[[Dict[str, float]], float],
                 parameters: Dict[str, OptimizationParameter],
                 max_iterations: int = 100) -> OptimizationResult:
        """Perform simulated annealing optimization."""
        start_time = time.time()
        optimization_id = f"sa_{int(time.time())}"
        
        # Initialize with current parameters
        current_params = {name: param.current_value for name, param in parameters.items()}
        try:
            current_fitness = objective_function(current_params)
        except Exception:
            current_fitness = float('-inf')
        
        best_params = current_params.copy()
        best_fitness = current_fitness
        temperature = self.initial_temperature
        
        for iteration in range(max_iterations):
            # Generate neighbor solution
            neighbor_params = self._generate_neighbor(current_params, parameters, temperature)
            
            try:
                neighbor_fitness = objective_function(neighbor_params)
                
                # Accept or reject neighbor
                if self._accept_solution(current_fitness, neighbor_fitness, temperature):
                    current_params = neighbor_params
                    current_fitness = neighbor_fitness
                    
                    # Update best solution
                    if neighbor_fitness > best_fitness:
                        best_fitness = neighbor_fitness
                        best_params = neighbor_params.copy()
                
            except Exception as e:
                logger.warning(f"Error evaluating neighbor solution: {e}")
            
            # Cool down
            temperature = max(self.min_temperature, temperature * self.cooling_rate)
        
        execution_time = time.time() - start_time
        
        # Calculate improvement
        initial_params = {name: param.current_value for name, param in parameters.items()}
        try:
            initial_fitness = objective_function(initial_params)
            improvement = ((best_fitness - initial_fitness) / abs(initial_fitness)) * 100 if initial_fitness != 0 else 0
        except:
            improvement = 0
        
        return OptimizationResult(
            optimization_id=optimization_id,
            algorithm=self.get_algorithm_name(),
            parameters=best_params,
            objective_value=best_fitness,
            improvement=improvement,
            execution_time=execution_time,
            iterations=max_iterations,
            success=True,
            timestamp=datetime.now()
        )
    
    def _generate_neighbor(self, current_params: Dict[str, float], 
                          parameters: Dict[str, OptimizationParameter],
                          temperature: float) -> Dict[str, float]:
        """Generate neighbor solution."""
        neighbor = current_params.copy()
        
        # Randomly select parameter to modify
        param_name = random.choice(list(parameters.keys()))
        param = parameters[param_name]
        
        if param.parameter_type == 'continuous':
            # Gaussian perturbation scaled by temperature
            perturbation_scale = (param.max_value - param.min_value) * 0.1 * (temperature / self.initial_temperature)
            perturbation = random.gauss(0, perturbation_scale)
            neighbor[param_name] = max(param.min_value, min(param.max_value, current_params[param_name] + perturbation))
        elif param.parameter_type == 'discrete':
            # Random discrete step
            step = random.choice([-1, 1])
            neighbor[param_name] = max(param.min_value, min(param.max_value, current_params[param_name] + step))
        
        return neighbor
    
    def _accept_solution(self, current_fitness: float, neighbor_fitness: float, temperature: float) -> bool:
        """Determine whether to accept neighbor solution."""
        if neighbor_fitness > current_fitness:
            return True
        
        if temperature <= 0:
            return False
        
        # Metropolis criterion
        probability = np.exp((neighbor_fitness - current_fitness) / temperature)
        return random.random() < probability

class ParameterOptimizer:
    """Manages parameter optimization using multiple algorithms."""
    
    def __init__(self):
        """Initialize parameter optimizer."""
        self.algorithms = {
            'genetic_algorithm': GeneticAlgorithmOptimizer(),
            'bayesian_optimization': BayesianOptimizer(),
            'particle_swarm': ParticleSwarmOptimizer(),
            'simulated_annealing': SimulatedAnnealingOptimizer()
        }
        self.optimization_history = deque(maxlen=1000)
        
    def add_algorithm(self, name: str, algorithm: OptimizationAlgorithm):
        """Add custom optimization algorithm."""
        self.algorithms[name] = algorithm
        logger.info(f"Added optimization algorithm: {name}")
    
    def optimize_parameters(self, 
                           objective_function: Callable[[Dict[str, float]], float],
                           parameters: Dict[str, OptimizationParameter],
                           algorithm: str = 'auto',
                           max_iterations: int = 100) -> OptimizationResult:
        """Optimize parameters using specified algorithm."""
        if algorithm == 'auto':
            algorithm = self._select_best_algorithm(parameters)
        
        if algorithm not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        optimizer = self.algorithms[algorithm]
        result = optimizer.optimize(objective_function, parameters, max_iterations)
        
        # Store result in history
        self.optimization_history.append(result)
        
        logger.info(f"Optimization completed: {algorithm}, improvement: {result.improvement:.2f}%")
        return result
    
    def _select_best_algorithm(self, parameters: Dict[str, OptimizationParameter]) -> str:
        """Automatically select best algorithm based on problem characteristics."""
        # Simple heuristics for algorithm selection
        num_params = len(parameters)
        has_discrete = any(p.parameter_type == 'discrete' for p in parameters.values())
        
        if num_params <= 5:
            return 'bayesian_optimization'  # Good for low-dimensional problems
        elif has_discrete:
            return 'genetic_algorithm'  # Handles discrete parameters well
        elif num_params <= 20:
            return 'particle_swarm'  # Good for medium-dimensional problems
        else:
            return 'simulated_annealing'  # Scalable to high dimensions
    
    def get_optimization_history(self, algorithm: str = None) -> List[OptimizationResult]:
        """Get optimization history, optionally filtered by algorithm."""
        if algorithm is None:
            return list(self.optimization_history)
        return [r for r in self.optimization_history if r.algorithm == algorithm]
    
    def get_best_result(self, algorithm: str = None) -> Optional[OptimizationResult]:
        """Get best optimization result."""
        history = self.get_optimization_history(algorithm)
        if not history:
            return None
        return max(history, key=lambda r: r.objective_value)

class ResourceOptimizer:
    """Optimizes resource allocation and utilization."""
    
    def __init__(self):
        """Initialize resource optimizer."""
        self.resource_allocations = {}
        self.allocation_history = deque(maxlen=500)
        
    def optimize_cpu_allocation(self, processes: List[Dict[str, Any]], total_cpu: float) -> Dict[str, float]:
        """Optimize CPU allocation across processes."""
        if not processes:
            return {}
        
        # Simple priority-based allocation
        total_priority = sum(p.get('priority', 1.0) for p in processes)
        allocations = {}
        
        for process in processes:
            priority = process.get('priority', 1.0)
            min_cpu = process.get('min_cpu', 0.1)
            max_cpu = process.get('max_cpu', total_cpu)
            
            # Proportional allocation based on priority
            base_allocation = (priority / total_priority) * total_cpu
            allocation = max(min_cpu, min(max_cpu, base_allocation))
            
            allocations[process['name']] = allocation
        
        # Normalize to ensure total doesn't exceed available CPU
        total_allocated = sum(allocations.values())
        if total_allocated > total_cpu:
            scale_factor = total_cpu / total_allocated
            allocations = {name: alloc * scale_factor for name, alloc in allocations.items()}
        
        self.allocation_history.append({
            'timestamp': datetime.now(),
            'type': 'cpu',
            'allocations': allocations.copy()
        })
        
        return allocations
    
    def optimize_memory_allocation(self, processes: List[Dict[str, Any]], total_memory: float) -> Dict[str, float]:
        """Optimize memory allocation across processes."""
        if not processes:
            return {}
        
        allocations = {}
        remaining_memory = total_memory
        
        # First pass: allocate minimum required memory
        for process in processes:
            min_memory = process.get('min_memory', 0.1)
            allocations[process['name']] = min_memory
            remaining_memory -= min_memory
        
        # Second pass: distribute remaining memory based on priority and max limits
        if remaining_memory > 0:
            total_priority = sum(p.get('priority', 1.0) for p in processes)
            
            for process in processes:
                priority = process.get('priority', 1.0)
                max_memory = process.get('max_memory', total_memory)
                current_allocation = allocations[process['name']]
                
                # Additional allocation based on priority
                additional = (priority / total_priority) * remaining_memory
                new_allocation = min(max_memory, current_allocation + additional)
                allocations[process['name']] = new_allocation
        
        self.allocation_history.append({
            'timestamp': datetime.now(),
            'type': 'memory',
            'allocations': allocations.copy()
        })
        
        return allocations
    
    def optimize_network_bandwidth(self, connections: List[Dict[str, Any]], total_bandwidth: float) -> Dict[str, float]:
        """Optimize network bandwidth allocation."""
        if not connections:
            return {}
        
        allocations = {}
        
        # Quality of Service (QoS) based allocation
        high_priority = [c for c in connections if c.get('qos', 'normal') == 'high']
        normal_priority = [c for c in connections if c.get('qos', 'normal') == 'normal']
        low_priority = [c for c in connections if c.get('qos', 'normal') == 'low']
        
        remaining_bandwidth = total_bandwidth
        
        # Allocate to high priority first
        for conn in high_priority:
            min_bandwidth = conn.get('min_bandwidth', 0.1)
            max_bandwidth = conn.get('max_bandwidth', total_bandwidth * 0.5)
            allocation = min(max_bandwidth, remaining_bandwidth * 0.3)  # Up to 30% for high priority
            allocations[conn['name']] = max(min_bandwidth, allocation)
            remaining_bandwidth -= allocations[conn['name']]
        
        # Allocate to normal priority
        if normal_priority and remaining_bandwidth > 0:
            per_connection = remaining_bandwidth * 0.7 / len(normal_priority)
            for conn in normal_priority:
                min_bandwidth = conn.get('min_bandwidth', 0.1)
                max_bandwidth = conn.get('max_bandwidth', total_bandwidth * 0.3)
                allocation = min(max_bandwidth, per_connection)
                allocations[conn['name']] = max(min_bandwidth, allocation)
                remaining_bandwidth -= allocations[conn['name']]
        
        # Allocate remaining to low priority
        if low_priority and remaining_bandwidth > 0:
            per_connection = remaining_bandwidth / len(low_priority)
            for conn in low_priority:
                min_bandwidth = conn.get('min_bandwidth', 0.05)
                allocations[conn['name']] = max(min_bandwidth, per_connection)
        
        self.allocation_history.append({
            'timestamp': datetime.now(),
            'type': 'network',
            'allocations': allocations.copy()
        })
        
        return allocations

class ConfigurationManager:
    """Manages automated configuration optimization."""
    
    def __init__(self, config_file: str = "optimized_config.json"):
        """Initialize configuration manager."""
        self.config_file = config_file
        self.current_config = {}
        self.config_history = deque(maxlen=100)
        self.load_config()
        
    def load_config(self):
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.current_config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
            else:
                self.current_config = self._get_default_config()
                self.save_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.current_config = self._get_default_config()
    
    def save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.current_config, f, indent=2)
            logger.info(f"Saved configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'performance': {
                'max_threads': 4,
                'cache_size_mb': 256,
                'batch_size': 32,
                'timeout_seconds': 30
            },
            'optimization': {
                'algorithm': 'auto',
                'max_iterations': 100,
                'convergence_threshold': 0.001
            },
            'monitoring': {
                'collection_interval': 0.1,
                'analysis_interval': 60.0,
                'report_interval': 300.0
            },
            'resources': {
                'cpu_limit_percent': 80,
                'memory_limit_mb': 2048,
                'disk_limit_gb': 10
            }
        }
    
    def optimize_configuration(self, performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Optimize configuration based on performance metrics."""
        optimized_config = self.current_config.copy()
        
        # CPU optimization
        cpu_usage = performance_metrics.get('cpu_usage_percent', 50)
        if cpu_usage > 90:
            # Reduce thread count if CPU is overloaded
            current_threads = optimized_config['performance']['max_threads']
            optimized_config['performance']['max_threads'] = max(1, current_threads - 1)
        elif cpu_usage < 30:
            # Increase thread count if CPU is underutilized
            current_threads = optimized_config['performance']['max_threads']
            optimized_config['performance']['max_threads'] = min(16, current_threads + 1)
        
        # Memory optimization
        memory_usage = performance_metrics.get('memory_usage_percent', 50)
        if memory_usage > 85:
            # Reduce cache size if memory is high
            current_cache = optimized_config['performance']['cache_size_mb']
            optimized_config['performance']['cache_size_mb'] = max(64, int(current_cache * 0.8))
        elif memory_usage < 40:
            # Increase cache size if memory is available
            current_cache = optimized_config['performance']['cache_size_mb']
            optimized_config['performance']['cache_size_mb'] = min(1024, int(current_cache * 1.2))
        
        # Batch size optimization
        throughput = performance_metrics.get('throughput', 100)
        if throughput < 50:
            # Reduce batch size for better responsiveness
            current_batch = optimized_config['performance']['batch_size']
            optimized_config['performance']['batch_size'] = max(8, int(current_batch * 0.8))
        elif throughput > 200:
            # Increase batch size for better efficiency
            current_batch = optimized_config['performance']['batch_size']
            optimized_config['performance']['batch_size'] = min(128, int(current_batch * 1.2))
        
        # Store configuration change
        if optimized_config != self.current_config:
            self.config_history.append({
                'timestamp': datetime.now(),
                'old_config': self.current_config.copy(),
                'new_config': optimized_config.copy(),
                'metrics': performance_metrics.copy()
            })
            
            self.current_config = optimized_config
            self.save_config()
            logger.info("Configuration optimized based on performance metrics")
        
        return optimized_config
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self.current_config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set_config_value(self, key_path: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config = self.current_config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        self.save_config()

class OptimizationEngine:
    """Main optimization engine orchestrating all optimization components."""
    
    def __init__(self):
        """Initialize optimization engine."""
        self.parameter_optimizer = ParameterOptimizer()
        self.resource_optimizer = ResourceOptimizer()
        self.config_manager = ConfigurationManager()
        
        self.optimization_queue = deque()
        self.is_running = False
        self.optimization_thread = None
        self.optimization_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'total_improvement': 0.0,
            'average_improvement': 0.0
        }
        
        logger.info("Optimization engine initialized")
    
    def start_optimization_engine(self):
        """Start the optimization engine."""
        if self.is_running:
            logger.warning("Optimization engine already running")
            return
        
        self.is_running = True
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        logger.info("Optimization engine started")
    
    def stop_optimization_engine(self):
        """Stop the optimization engine."""
        self.is_running = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
        logger.info("Optimization engine stopped")
    
    def queue_parameter_optimization(self, 
                                   objective_function: Callable[[Dict[str, float]], float],
                                   parameters: Dict[str, OptimizationParameter],
                                   algorithm: str = 'auto',
                                   priority: int = 1):
        """Queue parameter optimization task."""
        task = {
            'type': 'parameter_optimization',
            'objective_function': objective_function,
            'parameters': parameters,
            'algorithm': algorithm,
            'priority': priority,
            'timestamp': datetime.now()
        }
        self.optimization_queue.append(task)
        logger.info(f"Queued parameter optimization task with priority {priority}")
    
    def queue_resource_optimization(self, 
                                  resource_type: str,
                                  resource_data: Dict[str, Any],
                                  priority: int = 2):
        """Queue resource optimization task."""
        task = {
            'type': 'resource_optimization',
            'resource_type': resource_type,
            'resource_data': resource_data,
            'priority': priority,
            'timestamp': datetime.now()
        }
        self.optimization_queue.append(task)
        logger.info(f"Queued {resource_type} optimization task with priority {priority}")
    
    def queue_configuration_optimization(self, 
                                       performance_metrics: Dict[str, float],
                                       priority: int = 3):
        """Queue configuration optimization task."""
        task = {
            'type': 'configuration_optimization',
            'performance_metrics': performance_metrics,
            'priority': priority,
            'timestamp': datetime.now()
        }
        self.optimization_queue.append(task)
        logger.info(f"Queued configuration optimization task with priority {priority}")
    
    def _optimization_loop(self):
        """Main optimization loop."""
        while self.is_running:
            try:
                if self.optimization_queue:
                    # Sort queue by priority (lower number = higher priority)
                    sorted_queue = sorted(self.optimization_queue, key=lambda x: x['priority'])
                    task = sorted_queue[0]
                    self.optimization_queue.remove(task)
                    
                    # Execute optimization task
                    self._execute_optimization_task(task)
                else:
                    time.sleep(1)  # Wait for tasks
                    
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(5)  # Prevent tight error loop
    
    def _execute_optimization_task(self, task: Dict[str, Any]):
        """Execute optimization task."""
        try:
            self.optimization_stats['total_optimizations'] += 1
            
            if task['type'] == 'parameter_optimization':
                result = self.parameter_optimizer.optimize_parameters(
                    task['objective_function'],
                    task['parameters'],
                    task['algorithm']
                )
                
                if result.success:
                    self.optimization_stats['successful_optimizations'] += 1
                    self.optimization_stats['total_improvement'] += result.improvement
                    self.optimization_stats['average_improvement'] = (
                        self.optimization_stats['total_improvement'] / 
                        self.optimization_stats['successful_optimizations']
                    )
                    logger.info(f"Parameter optimization completed: {result.improvement:.2f}% improvement")
                
            elif task['type'] == 'resource_optimization':
                resource_type = task['resource_type']
                resource_data = task['resource_data']
                
                if resource_type == 'cpu':
                    allocations = self.resource_optimizer.optimize_cpu_allocation(
                        resource_data['processes'], 
                        resource_data['total_cpu']
                    )
                elif resource_type == 'memory':
                    allocations = self.resource_optimizer.optimize_memory_allocation(
                        resource_data['processes'], 
                        resource_data['total_memory']
                    )
                elif resource_type == 'network':
                    allocations = self.resource_optimizer.optimize_network_bandwidth(
                        resource_data['connections'], 
                        resource_data['total_bandwidth']
                    )
                
                logger.info(f"Resource optimization completed: {resource_type}")
                
            elif task['type'] == 'configuration_optimization':
                optimized_config = self.config_manager.optimize_configuration(
                    task['performance_metrics']
                )
                logger.info("Configuration optimization completed")
                
        except Exception as e:
            logger.error(f"Error executing optimization task: {e}")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization engine statistics."""
        return {
            'is_running': self.is_running,
            'queue_size': len(self.optimization_queue),
            'stats': self.optimization_stats.copy(),
            'parameter_history_size': len(self.parameter_optimizer.optimization_history),
            'resource_history_size': len(self.resource_optimizer.allocation_history),
            'config_history_size': len(self.config_manager.config_history)
        }
    
    def run_comprehensive_optimization(self, performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Run comprehensive optimization across all components."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'parameter_optimization': None,
            'resource_optimization': None,
            'configuration_optimization': None
        }
        
        try:
            # Configuration optimization
            optimized_config = self.config_manager.optimize_configuration(performance_metrics)
            results['configuration_optimization'] = {
                'status': 'completed',
                'config': optimized_config
            }
            
            # Resource optimization (example)
            if 'cpu_usage_percent' in performance_metrics:
                cpu_processes = [
                    {'name': 'learning_system', 'priority': 3, 'min_cpu': 0.5, 'max_cpu': 4.0},
                    {'name': 'monitoring', 'priority': 2, 'min_cpu': 0.1, 'max_cpu': 1.0},
                    {'name': 'optimization', 'priority': 1, 'min_cpu': 0.2, 'max_cpu': 2.0}
                ]
                cpu_allocations = self.resource_optimizer.optimize_cpu_allocation(cpu_processes, 8.0)
                results['resource_optimization'] = {
                    'status': 'completed',
                    'cpu_allocations': cpu_allocations
                }
            
        except Exception as e:
            logger.error(f"Error in comprehensive optimization: {e}")
            results['error'] = str(e)
        
        return results

# Example usage and testing
if __name__ == "__main__":
    # Create optimization engine
    engine = OptimizationEngine()
    
    # Example objective function
    def example_objective(params):
        # Simulate a performance metric based on parameters
        x = params.get('learning_rate', 0.01)
        y = params.get('batch_size', 32)
        z = params.get('hidden_units', 64)
        
        # Simulate some complex performance function
        return -(x - 0.1)**2 - (y - 50)**2/1000 - (z - 100)**2/10000 + 100
    
    # Define parameters to optimize
    parameters = {
        'learning_rate': OptimizationParameter('learning_rate', 0.01, 0.001, 0.5, 'continuous'),
        'batch_size': OptimizationParameter('batch_size', 32, 8, 128, 'discrete'),
        'hidden_units': OptimizationParameter('hidden_units', 64, 32, 256, 'discrete')
    }
    
    # Test parameter optimization
    print("Testing parameter optimization...")
    result = engine.parameter_optimizer.optimize_parameters(
        example_objective, 
        parameters, 
        algorithm='genetic_algorithm',
        max_iterations=50
    )
    
    print(f"Optimization result: {result.improvement:.2f}% improvement")
    print(f"Best parameters: {result.parameters}")
    
    # Test resource optimization
    print("\nTesting resource optimization...")
    processes = [
        {'name': 'process1', 'priority': 3, 'min_cpu': 0.5, 'max_cpu': 2.0},
        {'name': 'process2', 'priority': 1, 'min_cpu': 0.2, 'max_cpu': 1.0},
        {'name': 'process3', 'priority': 2, 'min_cpu': 0.3, 'max_cpu': 1.5}
    ]
    
    cpu_allocations = engine.resource_optimizer.optimize_cpu_allocation(processes, 4.0)
    print(f"CPU allocations: {cpu_allocations}")
    
    # Test configuration optimization
    print("\nTesting configuration optimization...")
    performance_metrics = {
        'cpu_usage_percent': 85,
        'memory_usage_percent': 70,
        'throughput': 150
    }
    
    optimized_config = engine.config_manager.optimize_configuration(performance_metrics)
    print(f"Optimized configuration: {optimized_config}")
    
    # Get optimization stats
    stats = engine.get_optimization_stats()
    print(f"\nOptimization stats: {stats}")

