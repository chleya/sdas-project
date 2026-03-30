"""
ES Trainer for Low-Rank Structure Pool
使用进化策略优化低秩结构池的原型参数
"""

import numpy as np
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments'))

from structure_pool_lora import LowRankStructurePool, LowRankStructure
from sdas import SDASAgent, Config, run_episode, Encoder, WorldModel, ActionPolicy
from digital_petri_dish import DigitalPetriDish


@dataclass
class LoRAESConfig:
    """低秩 ES 训练配置"""
    population_size: int = 20          # 种群大小（比标准 ES 小，因为参数量更大）
    sigma: float = 0.05                # 扰动标准差（更小，避免破坏低秩结构）
    learning_rate: float = 0.03        # 学习率
    n_generations: int = 50            # 迭代代数
    n_eval_episodes: int = 2           # 每个个体评估的回合数
    max_steps_per_episode: int = 100   # 每回合最大步数
    rank: int = 2                      # 低秩维度（关键：rank 越小压缩比越高）


class LoRAESTrainer:
    """
    低秩结构池的进化策略训练器
    优化所有低秩参数：base_vector + 所有结构的 (B, a)
    """
    
    def __init__(self, es_config: LoRAESConfig = None, env_config: Dict = None):
        self.es_config = es_config or LoRAESConfig()
        self.env_config = env_config or {
            'width': 15,
            'height': 15,
            'n_obstacles': 25,
            'n_energy': 5
        }
        
        # 创建参考结构池（用于确定参数维度）
        self.ref_pool = LowRankStructurePool(
            vector_dim=64,  # 与 Encoder 输出维度一致
            rank=self.es_config.rank,
            max_structures=16
        )
        
        # 初始化一些结构以确定参数维度
        for _ in range(5):
            obs = np.random.randn(64)
            self.ref_pool.observe(obs, f"init_{_}")
        
        # 获取初始参数
        self.current_params = self.ref_pool.get_all_lora_params()
        self.best_fitness = -float('inf')
        self.fitness_history = []
        
        print(f"LoRA ES Trainer initialized")
        print(f"  Vector dim: 64")
        print(f"  Rank: {self.es_config.rank}")
        print(f"  Initial structures: {len(self.ref_pool.structures)}")
        print(f"  Total parameters: {len(self.current_params)}")
    
    def create_agent_with_params(self, params: np.ndarray) -> SDASAgent:
        """
        使用给定的低秩参数创建智能体
        """
        # 创建标准智能体
        config = Config()
        agent = SDASAgent(config)
        
        # 创建新的低秩结构池并设置参数
        pool = LowRankStructurePool(
            vector_dim=64,
            rank=self.es_config.rank,
            max_structures=16
        )
        
        # 复制参考结构池的结构，然后设置参数
        pool.structures = []
        for s in self.ref_pool.structures:
            new_struct = LowRankStructure(
                id=s.id,
                base_vector=pool.base_vector,
                B=s.B.copy(),
                a=s.a.copy(),
                label=s.label,
                age=s.age,
                utility=s.utility
            )
            pool.structures.append(new_struct)
        pool.next_id = self.ref_pool.next_id
        
        # 设置 ES 优化的参数
        pool.set_all_lora_params(params)
        
        # 替换智能体的结构池
        agent.structure_pool = pool
        
        return agent
    
    def evaluate_fitness(self, params: np.ndarray, seed: int = None) -> float:
        """评估一组参数的适应度"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        total_reward = 0
        total_steps = 0
        
        for _ in range(self.es_config.n_eval_episodes):
            env = DigitalPetriDish(**self.env_config)
            agent = self.create_agent_with_params(params)
            
            result = run_episode(env, agent, max_steps=self.es_config.max_steps_per_episode)
            
            total_reward += result['total_reward']
            total_steps += result['steps']
        
        avg_reward = total_reward / self.es_config.n_eval_episodes
        avg_steps = total_steps / self.es_config.n_eval_episodes
        
        # 综合适应度
        fitness = avg_reward + 0.01 * avg_steps
        
        return fitness
    
    def train_generation(self) -> Tuple[np.ndarray, float]:
        """训练一代"""
        population = []
        fitnesses = []
        
        print(f"\n--- Generation ---")
        print(f"Current best fitness: {self.best_fitness:.3f}")
        
        # 生成种群
        for i in range(self.es_config.population_size):
            # 添加扰动
            noise = np.random.randn(len(self.current_params)) * self.es_config.sigma
            perturbed_params = self.current_params + noise
            
            # 评估
            fitness = self.evaluate_fitness(perturbed_params, seed=i)
            
            population.append((perturbed_params, noise))
            fitnesses.append(fitness)
            
            if i % 5 == 0:
                print(f"  Eval {i}/{self.es_config.population_size}: fitness={fitness:.3f}")
        
        # 计算加权更新
        fitnesses = np.array(fitnesses)
        ranks = np.argsort(np.argsort(-fitnesses))
        weights = np.maximum(0, np.log(self.es_config.population_size / 2 + 1) - np.log(ranks + 1))
        weights = weights / np.sum(weights) - 1.0 / self.es_config.population_size
        
        # 更新参数
        update = np.zeros(len(self.current_params))
        for i, (_, noise) in enumerate(population):
            update += weights[i] * noise
        
        self.current_params = self.current_params + self.es_config.learning_rate * update
        
        # 记录最佳
        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        
        self.fitness_history.append({
            'mean': np.mean(fitnesses),
            'max': np.max(fitnesses),
            'min': np.min(fitnesses)
        })
        
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            print(f"  *** New best fitness: {best_fitness:.3f} ***")
        
        print(f"  Fitness: mean={np.mean(fitnesses):.3f}, max={np.max(fitnesses):.3f}")
        
        return self.current_params, best_fitness
    
    def train(self, n_generations: int = None) -> np.ndarray:
        """完整训练"""
        n_gen = n_generations or self.es_config.n_generations
        
        print("=" * 60)
        print("LoRA ES Training for SDAS Structure Pool")
        print("=" * 60)
        print(f"Population: {self.es_config.population_size}")
        print(f"Sigma: {self.es_config.sigma}")
        print(f"Learning rate: {self.es_config.learning_rate}")
        print(f"Rank: {self.es_config.rank}")
        print(f"Generations: {n_gen}")
        print("=" * 60)
        
        for gen in range(n_gen):
            print(f"\n{'='*60}")
            print(f"Generation {gen + 1}/{n_gen}")
            print('='*60)
            
            params, fitness = self.train_generation()
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Best fitness: {self.best_fitness:.3f}")
        
        return self.current_params
    
    def save_best_params(self, filepath: str = "lora_best_params.npy"):
        """保存最佳参数"""
        np.save(filepath, self.current_params)
        print(f"Best parameters saved to {filepath}")
    
    def load_params(self, filepath: str) -> np.ndarray:
        """加载参数"""
        self.current_params = np.load(filepath)
        print(f"Parameters loaded from {filepath}")
        return self.current_params


def quick_test():
    """快速测试 LoRA ES 训练器"""
    print("Quick LoRA ES Trainer Test")
    print("=" * 50)
    
    es_config = LoRAESConfig(
        population_size=5,
        n_generations=2,
        n_eval_episodes=1,
        max_steps_per_episode=30,
        rank=2
    )
    
    trainer = LoRAESTrainer(es_config)
    best_params = trainer.train(n_generations=2)
    
    print("\n" + "=" * 50)
    print("Test Complete!")
    print(f"Best fitness: {trainer.best_fitness:.3f}")
    
    return best_params


if __name__ == "__main__":
    quick_test()
