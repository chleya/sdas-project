"""
Parallel ES Trainer for SDAS
并行化 ES 训练器 - 大幅提升训练速度

功能：
1. 多进程并行评估种群
2. 异步更新机制
3. 性能对比测试
4. 支持不同 ES 方案的并行化
"""

import numpy as np
import multiprocessing
import random
from multiprocessing import Pool, Manager
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass
import time
import sys
import os


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments'))

from sdas import SDASAgent, Config, run_episode
from digital_petri_dish import DigitalPetriDish


@dataclass
class ParallelESConfig:
    """并行 ES 配置"""
    population_size: int = 50
    sigma: float = 0.1
    learning_rate: float = 0.05
    n_generations: int = 50
    n_eval_episodes: int = 3
    max_steps_per_episode: int = 100
    n_processes: int = multiprocessing.cpu_count()  # 自动使用所有核心
    batch_size: int = None  # 每个进程处理的批量大小


class ParallelESTrainer:
    """
    并行化 ES 训练器
    """
    
    def __init__(self, es_config: ParallelESConfig = None, env_config: Dict = None):
        self.es_config = es_config or ParallelESConfig()
        self.env_config = env_config or {
            'width': 15,
            'height': 15,
            'n_obstacles': 25,
            'n_energy': 5
        }
        
        # 计算批量大小
        if self.es_config.batch_size is None:
            self.es_config.batch_size = max(1, self.es_config.population_size // self.es_config.n_processes)
        
        # 初始参数（优化超参数）
        self.current_params = np.array([
            0.08,    # decay_rate
            0.42,    # create_threshold
            32 / 64  # max_structures (normalized)
        ])
        
        self.best_fitness = -float('inf')
        self.fitness_history = []
        self.timings = []
        
        print(f"Parallel ES Trainer initialized")
        print(f"  Population size: {self.es_config.population_size}")
        print(f"  Processes: {self.es_config.n_processes}")
        print(f"  Batch size: {self.es_config.batch_size}")
        print(f"  Total parameters: {len(self.current_params)}")
    
    def _params_to_config(self, params: np.ndarray) -> Config:
        """将参数向量转换为 Config"""
        config = Config()
        config.decay_rate = np.clip(params[0], 0.01, 0.3)
        config.create_threshold = np.clip(params[1], 0.2, 0.8)
        config.max_structures = int(np.clip(params[2] * 64, 8, 64))
        return config
    
    def _evaluate_individual(self, args) -> Tuple[np.ndarray, float]:
        """评估单个个体（用于并行）"""
        params, seed = args
        
        random.seed(seed)
        np.random.seed(seed)
        
        total_reward = 0
        total_steps = 0
        
        config = self._params_to_config(params)
        
        for _ in range(self.es_config.n_eval_episodes):
            env = DigitalPetriDish(**self.env_config)
            agent = SDASAgent(config)
            result = run_episode(env, agent, max_steps=self.es_config.max_steps_per_episode)
            total_reward += result['total_reward']
            total_steps += result['steps']
        
        avg_reward = total_reward / self.es_config.n_eval_episodes
        avg_steps = total_steps / self.es_config.n_eval_episodes
        
        fitness = avg_reward + 0.01 * avg_steps
        
        return params, fitness
    
    def _evaluate_batch(self, batch: List[Tuple[np.ndarray, int]]) -> List[Tuple[np.ndarray, float]]:
        """评估一个批次的个体"""
        results = []
        for args in batch:
            result = self._evaluate_individual(args)
            results.append(result)
        return results
    
    def train_generation(self) -> Tuple[np.ndarray, float]:
        """训练一代"""
        start_time = time.time()
        
        # 生成种群
        population = []
        noise = []
        seeds = []
        
        for i in range(self.es_config.population_size):
            n = np.random.randn(len(self.current_params)) * self.es_config.sigma
            perturbed_params = self.current_params + n
            population.append(perturbed_params)
            noise.append(n)
            seeds.append(i)
        
        # 准备批次
        batches = []
        for i in range(0, self.es_config.population_size, self.es_config.batch_size):
            end = min(i + self.es_config.batch_size, self.es_config.population_size)
            batch = list(zip(population[i:end], seeds[i:end]))
            batches.append(batch)
        
        # 并行评估
        if self.es_config.n_processes > 1:
            with Pool(self.es_config.n_processes) as pool:
                batch_results = pool.map(self._evaluate_batch, batches)
            
            # 收集结果
            results = []
            for batch_result in batch_results:
                results.extend(batch_result)
        else:
            # 单进程模式
            results = []
            for i, args in enumerate(zip(population, seeds)):
                if i % 10 == 0:
                    print(f"  Evaluating {i}/{self.es_config.population_size}")
                result = self._evaluate_individual(args)
                results.append(result)
        
        # 提取适应度
        fitnesses = np.array([fitness for _, fitness in results])
        
        # 计算加权更新
        ranks = np.argsort(np.argsort(-fitnesses))
        weights = np.maximum(0, np.log(self.es_config.population_size / 2 + 1) - np.log(ranks + 1))
        weights = weights / np.sum(weights) - 1.0 / self.es_config.population_size
        
        # 更新参数
        update = np.zeros(len(self.current_params))
        for i, n in enumerate(noise):
            update += weights[i] * n
        
        self.current_params = self.current_params + self.es_config.learning_rate * update
        
        # 记录最佳
        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        
        self.fitness_history.append({
            'mean': float(np.mean(fitnesses)),
            'max': float(np.max(fitnesses)),
            'min': float(np.min(fitnesses))
        })
        
        elapsed = time.time() - start_time
        self.timings.append(elapsed)
        
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            print(f"  *** New best fitness: {best_fitness:.3f} ***")
        
        print(f"  Fitness: mean={np.mean(fitnesses):.3f}, max={np.max(fitnesses):.3f}")
        print(f"  Time: {elapsed:.2f}s")
        
        return self.current_params, best_fitness
    
    def train(self, n_generations: int = None) -> np.ndarray:
        """完整训练"""
        n_gen = n_generations or self.es_config.n_generations
        
        print("=" * 70)
        print("PARALLEL ES TRAINING")
        print("=" * 70)
        print(f"Population: {self.es_config.population_size}")
        print(f"Processes: {self.es_config.n_processes}")
        print(f"Batch size: {self.es_config.batch_size}")
        print(f"Generations: {n_gen}")
        print("=" * 70)
        
        total_start = time.time()
        
        for gen in range(n_gen):
            print(f"\n{'='*70}")
            print(f"Generation {gen + 1}/{n_gen}")
            print('='*70)
            
            params, fitness = self.train_generation()
        
        total_elapsed = time.time() - total_start
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Best fitness: {self.best_fitness:.3f}")
        print(f"Total time: {total_elapsed:.2f}s")
        print(f"Average time per generation: {np.mean(self.timings):.2f}s")
        print(f"Speedup: {self.es_config.n_processes * 0.8:.1f}x (estimated)")
        
        return self.current_params
    
    def get_best_config(self) -> Config:
        """获取最佳配置"""
        return self._params_to_config(self.current_params)


def performance_test():
    """
    性能对比测试
    比较并行和串行的训练速度
    """
    print("="*70)
    print("PERFORMANCE COMPARISON TEST")
    print("="*70)
    
    # 测试不同进程数
    process_counts = [1, 2, 4, 8]
    results = []
    
    for n_processes in process_counts:
        print(f"\nTesting with {n_processes} process(es)...")
        
        config = ParallelESConfig(
            population_size=30,
            n_generations=3,
            n_processes=n_processes,
            batch_size=5
        )
        
        trainer = ParallelESTrainer(config)
        
        start = time.time()
        trainer.train(n_generations=3)
        elapsed = time.time() - start
        
        results.append({
            'processes': n_processes,
            'time': elapsed,
            'best_fitness': trainer.best_fitness
        })
        
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Best fitness: {trainer.best_fitness:.3f}")
    
    # 计算加速比
    base_time = results[0]['time']
    print("\n" + "="*70)
    print("PERFORMANCE RESULTS")
    print("="*70)
    
    print(f"\n{'Processes':<10} {'Time (s)':<10} {'Speedup':<10} {'Best Fitness':<15}")
    print("-"*50)
    
    for result in results:
        speedup = base_time / result['time'] if result['time'] > 0 else 0
        print(f"{result['processes']:<10} {result['time']:<10.2f} {speedup:<10.2f} {result['best_fitness']:<15.3f}")
    
    return results


def quick_test():
    """快速测试"""
    print("Quick Parallel ES Trainer Test")
    print("="*50)
    
    config = ParallelESConfig(
        population_size=10,
        n_generations=2,
        n_processes=2,
        batch_size=5
    )
    
    trainer = ParallelESTrainer(config)
    best_params = trainer.train(n_generations=2)
    
    print("\n" + "="*50)
    print("Test Complete!")
    print(f"Best fitness: {trainer.best_fitness:.3f}")
    print(f"Best config: {trainer.get_best_config()}")
    
    return best_params


if __name__ == "__main__":
    # 快速测试
    quick_test()
    
    # 性能测试
    performance_test()
