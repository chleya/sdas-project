"""
ES Trainer - Evolution Strategies for SDAS Structure Pool Optimization
使用进化策略优化结构池的核心参数
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass
import copy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments'))

from sdas import SDASAgent, Config, run_episode
from digital_petri_dish import DigitalPetriDish


@dataclass
class ESConfig:
    """ES 训练配置"""
    population_size: int = 50          # 种群大小
    sigma: float = 0.1                 # 扰动标准差
    learning_rate: float = 0.05        # 学习率
    n_generations: int = 100           # 迭代代数
    elite_ratio: float = 0.2           # 精英比例
    n_eval_episodes: int = 3           # 每个个体评估的回合数
    max_steps_per_episode: int = 100   # 每回合最大步数
    
    # 参数边界
    param_bounds = {
        'decay_rate': (0.01, 0.3),
        'create_threshold': (0.2, 0.8),
        'max_structures': (8, 64),
        'utility_lr': (0.01, 0.3),
    }


class StructurePoolParams:
    """
    结构池的可优化参数
    这些参数将用 ES 来优化
    """
    
    def __init__(self, 
                 decay_rate: float = 0.08,
                 create_threshold: float = 0.42,
                 max_structures: int = 32,
                 utility_lr: float = 0.1):
        self.decay_rate = decay_rate
        self.create_threshold = create_threshold
        self.max_structures = max_structures
        self.utility_lr = utility_lr
    
    def to_vector(self) -> np.ndarray:
        """转换为向量表示"""
        return np.array([
            self.decay_rate,
            self.create_threshold,
            self.max_structures / 64.0,  # 归一化到 0-1
            self.utility_lr,
        ])
    
    @classmethod
    def from_vector(cls, vector: np.ndarray) -> 'StructurePoolParams':
        """从向量恢复参数"""
        return cls(
            decay_rate=np.clip(vector[0], 0.01, 0.3),
            create_threshold=np.clip(vector[1], 0.2, 0.8),
            max_structures=int(np.clip(vector[2] * 64, 8, 64)),
            utility_lr=np.clip(vector[3], 0.01, 0.3),
        )
    
    def to_config(self) -> Config:
        """转换为 SDAS Config"""
        config = Config()
        config.decay_rate = self.decay_rate
        config.create_threshold = self.create_threshold
        config.max_structures = self.max_structures
        return config
    
    def __repr__(self):
        return f"StructurePoolParams(decay={self.decay_rate:.3f}, threshold={self.create_threshold:.3f}, max_struct={self.max_structures}, utility_lr={self.utility_lr:.3f})"


class ESTrainer:
    """
    进化策略训练器
    使用 OpenAI ES 的简化版本
    """
    
    def __init__(self, es_config: ESConfig = None, env_config: Dict = None):
        self.es_config = es_config or ESConfig()
        self.env_config = env_config or {
            'width': 15,
            'height': 15,
            'n_obstacles': 25,
            'n_energy': 5
        }
        
        # 当前最佳参数
        self.current_params = StructurePoolParams()
        self.best_fitness = -float('inf')
        self.fitness_history = []
        
    def evaluate_fitness(self, params: StructurePoolParams, seed: int = None) -> float:
        """
        评估一组参数的适应度
        在多个回合上平均表现
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        total_reward = 0
        total_steps = 0
        
        for _ in range(self.es_config.n_eval_episodes):
            # 创建环境
            env = DigitalPetriDish(**self.env_config)
            
            # 创建智能体，使用当前参数
            config = params.to_config()
            agent = SDASAgent(config)
            
            # 运行回合
            result = run_episode(env, agent, max_steps=self.es_config.max_steps_per_episode)
            
            total_reward += result['total_reward']
            total_steps += result['steps']
        
        # 适应度 = 平均奖励 + 存活步数奖励（鼓励探索）
        avg_reward = total_reward / self.es_config.n_eval_episodes
        avg_steps = total_steps / self.es_config.n_eval_episodes
        
        # 综合适应度
        fitness = avg_reward + 0.01 * avg_steps
        
        return fitness
    
    def train_generation(self) -> Tuple[StructurePoolParams, float]:
        """
        训练一代
        返回 (最佳参数, 最佳适应度)
        """
        population = []
        fitnesses = []
        
        print(f"\n--- Generation ---")
        print(f"Current params: {self.current_params}")
        
        # 生成种群
        for i in range(self.es_config.population_size):
            # 添加扰动
            noise = np.random.randn(4) * self.es_config.sigma
            perturbed_vector = self.current_params.to_vector() + noise
            perturbed_params = StructurePoolParams.from_vector(perturbed_vector)
            
            # 评估
            fitness = self.evaluate_fitness(perturbed_params, seed=i)
            
            population.append((perturbed_params, noise))
            fitnesses.append(fitness)
            
            if i % 10 == 0:
                print(f"  Eval {i}/{self.es_config.population_size}: fitness={fitness:.3f}")
        
        # 计算加权更新（基于适应度排名）
        fitnesses = np.array(fitnesses)
        
        # 使用适应度排名而非原始值（更稳定）
        ranks = np.argsort(np.argsort(-fitnesses))  # 排名，0=最好
        weights = np.maximum(0, np.log(self.es_config.population_size / 2 + 1) - np.log(ranks + 1))
        weights = weights / np.sum(weights) - 1.0 / self.es_config.population_size
        
        # 更新参数
        current_vector = self.current_params.to_vector()
        update = np.zeros(4)
        for i, (_, noise) in enumerate(population):
            update += weights[i] * noise
        
        new_vector = current_vector + self.es_config.learning_rate * update
        self.current_params = StructurePoolParams.from_vector(new_vector)
        
        # 记录最佳
        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        
        self.fitness_history.append({
            'mean': np.mean(fitnesses),
            'max': np.max(fitnesses),
            'min': np.min(fitnesses),
            'best_params': population[best_idx][0]
        })
        
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            print(f"  *** New best fitness: {best_fitness:.3f} ***")
        
        print(f"  Fitness stats: mean={np.mean(fitnesses):.3f}, max={np.max(fitnesses):.3f}, min={np.min(fitnesses):.3f}")
        print(f"  Updated params: {self.current_params}")
        
        return self.current_params, best_fitness
    
    def train(self, n_generations: int = None) -> StructurePoolParams:
        """
        完整训练
        """
        n_gen = n_generations or self.es_config.n_generations
        
        print("=" * 60)
        print("Starting ES Training for SDAS Structure Pool")
        print("=" * 60)
        print(f"Population size: {self.es_config.population_size}")
        print(f"Sigma: {self.es_config.sigma}")
        print(f"Learning rate: {self.es_config.learning_rate}")
        print(f"Generations: {n_gen}")
        print(f"Eval episodes per individual: {self.es_config.n_eval_episodes}")
        print("=" * 60)
        
        for gen in range(n_gen):
            print(f"\n{'='*60}")
            print(f"Generation {gen + 1}/{n_gen}")
            print('='*60)
            
            params, fitness = self.train_generation()
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Best fitness achieved: {self.best_fitness:.3f}")
        print(f"Final parameters: {self.current_params}")
        
        return self.current_params
    
    def plot_history(self):
        """绘制训练历史"""
        try:
            import matplotlib.pyplot as plt
            
            generations = range(len(self.fitness_history))
            means = [h['mean'] for h in self.fitness_history]
            maxs = [h['max'] for h in self.fitness_history]
            mins = [h['min'] for h in self.fitness_history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(generations, means, label='Mean Fitness', alpha=0.7)
            plt.plot(generations, maxs, label='Max Fitness', linewidth=2)
            plt.fill_between(generations, mins, maxs, alpha=0.2)
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('ES Training Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('es_training_history.png', dpi=150, bbox_inches='tight')
            print("\nTraining history saved to es_training_history.png")
        except ImportError:
            print("\nmatplotlib not available, skipping plot")


def quick_test():
    """快速测试 ES 训练器"""
    print("Quick ES Trainer Test")
    print("=" * 50)
    
    # 小规模配置用于测试
    es_config = ESConfig(
        population_size=10,
        n_generations=3,
        n_eval_episodes=2,
        max_steps_per_episode=50
    )
    
    trainer = ESTrainer(es_config)
    best_params = trainer.train(n_generations=3)
    
    print("\n" + "=" * 50)
    print("Test Complete!")
    print(f"Best params: {best_params}")
    
    return best_params


if __name__ == "__main__":
    quick_test()
