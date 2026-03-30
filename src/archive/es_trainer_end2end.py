"""
End-to-End ES Trainer for SDAS
端到端全参数优化 (方案 C)

优化所有可训练参数：
1. Encoder weights and biases
2. World Model weights and biases
3. Structure Pool (可选：同时优化参数和原型)
4. (可选) Action Policy parameters
"""

import numpy as np
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments'))

from sdas import SDASAgent, Config, run_episode, Encoder, WorldModel, ActionPolicy
from digital_petri_dish import DigitalPetriDish


@dataclass
class End2EndESConfig:
    """端到端 ES 配置"""
    population_size: int = 15          # 种群大小（较小，因为参数量大）
    sigma: float = 0.02                # 扰动标准差（更小，避免破坏训练）
    learning_rate: float = 0.02        # 学习率
    n_generations: int = 30            # 迭代代数
    n_eval_episodes: int = 2           # 每个个体评估回合数
    max_steps_per_episode: int = 100   # 每回合最大步数
    
    # 优化选项
    optimize_encoder: bool = True
    optimize_world_model: bool = True
    optimize_structure_pool_hparams: bool = True


class End2EndESTrainer:
    """
    端到端全参数 ES 训练器
    """
    
    def __init__(self, es_config: End2EndESConfig = None, env_config: Dict = None):
        self.es_config = es_config or End2EndESConfig()
        self.env_config = env_config or {
            'width': 15,
            'height': 15,
            'n_obstacles': 25,
            'n_energy': 5
        }
        
        # 创建参考智能体以确定参数维度
        self.ref_agent = SDASAgent(Config())
        
        # 获取初始参数
        self.current_params = self._agent_to_params(self.ref_agent)
        self.best_fitness = -float('inf')
        self.fitness_history = []
        
        print("End-to-End ES Trainer initialized")
        print(f"  Total parameters: {len(self.current_params)}")
        print(f"  Optimize Encoder: {self.es_config.optimize_encoder}")
        print(f"  Optimize World Model: {self.es_config.optimize_world_model}")
        print(f"  Optimize Pool HParams: {self.es_config.optimize_structure_pool_hparams}")
    
    def _agent_to_params(self, agent: SDASAgent) -> np.ndarray:
        """将智能体的所有参数展平为向量"""
        params = []
        
        if self.es_config.optimize_encoder:
            params.append(agent.encoder.W.flatten())
            params.append(agent.encoder.b.flatten())
        
        if self.es_config.optimize_world_model:
            params.append(agent.world_model.fc1.flatten())
            params.append(agent.world_model.fc2.flatten())
            params.append(agent.world_model.fc_reward.flatten())
        
        if self.es_config.optimize_structure_pool_hparams:
            pool_config = np.array([
                agent.config.decay_rate,
                agent.config.create_threshold,
                agent.config.max_structures / 64.0  # 归一化
            ])
            params.append(pool_config)
        
        return np.concatenate(params)
    
    def _params_to_agent(self, params: np.ndarray) -> SDASAgent:
        """从参数向量重建智能体"""
        agent = SDASAgent(Config())
        idx = 0
        
        if self.es_config.optimize_encoder:
            W_size = agent.encoder.W.size
            agent.encoder.W = params[idx:idx + W_size].reshape(agent.encoder.W.shape)
            idx += W_size
            
            b_size = agent.encoder.b.size
            agent.encoder.b = params[idx:idx + b_size].reshape(agent.encoder.b.shape)
            idx += b_size
        
        if self.es_config.optimize_world_model:
            fc1_size = agent.world_model.fc1.size
            agent.world_model.fc1 = params[idx:idx + fc1_size].reshape(agent.world_model.fc1.shape)
            idx += fc1_size
            
            fc2_size = agent.world_model.fc2.size
            agent.world_model.fc2 = params[idx:idx + fc2_size].reshape(agent.world_model.fc2.shape)
            idx += fc2_size
            
            fc_reward_size = agent.world_model.fc_reward.size
            agent.world_model.fc_reward = params[idx:idx + fc_reward_size].reshape(agent.world_model.fc_reward.shape)
            idx += fc_reward_size
        
        if self.es_config.optimize_structure_pool_hparams:
            agent.config.decay_rate = np.clip(params[idx], 0.01, 0.3)
            agent.config.create_threshold = np.clip(params[idx+1], 0.2, 0.8)
            agent.config.max_structures = int(np.clip(params[idx+2] * 64, 8, 64))
        
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
            agent = self._params_to_agent(params)
            
            result = run_episode(env, agent, max_steps=self.es_config.max_steps_per_episode)
            
            total_reward += result['total_reward']
            total_steps += result['steps']
        
        avg_reward = total_reward / self.es_config.n_eval_episodes
        avg_steps = total_steps / self.es_config.n_eval_episodes
        
        # 综合适应度：奖励 + 步数奖励（鼓励探索）
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
            noise = np.random.randn(len(self.current_params)) * self.es_config.sigma
            perturbed_params = self.current_params + noise
            
            fitness = self.evaluate_fitness(perturbed_params, seed=i)
            
            population.append((perturbed_params, noise))
            fitnesses.append(fitness)
            
            if i % 3 == 0:
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
        print("End-to-End ES Training for SDAS")
        print("=" * 60)
        print(f"Population: {self.es_config.population_size}")
        print(f"Sigma: {self.es_config.sigma}")
        print(f"Learning rate: {self.es_config.learning_rate}")
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
    
    def save_best_agent(self, filepath: str = "end2end_best_agent.npy"):
        """保存最佳智能体参数"""
        np.save(filepath, self.current_params)
        print(f"Best agent saved to {filepath}")
    
    def load_best_agent(self, filepath: str) -> SDASAgent:
        """加载最佳智能体"""
        params = np.load(filepath)
        self.current_params = params
        return self._params_to_agent(params)


def quick_test():
    """快速测试端到端 ES 训练器"""
    print("Quick End-to-End ES Trainer Test")
    print("=" * 50)
    
    es_config = End2EndESConfig(
        population_size=5,
        n_generations=2,
        n_eval_episodes=1,
        max_steps_per_episode=30
    )
    
    trainer = End2EndESTrainer(es_config)
    best_params = trainer.train(n_generations=2)
    
    print("\n" + "=" * 50)
    print("Test Complete!")
    print(f"Best fitness: {trainer.best_fitness:.3f}")
    
    return best_params


if __name__ == "__main__":
    quick_test()
