#!/usr/bin/env python3
"""
并行化性能测试 - 大规模训练
"""

import time
from src.es_trainer_parallel import ParallelESTrainer, ParallelESConfig


def large_scale_performance_test():
    """
    大规模性能测试
    使用更大的种群和更多的世代数
    """
    print("="*70)
    print("LARGE SCALE PERFORMANCE TEST")
    print("="*70)
    
    # 测试不同进程数
    process_counts = [1, 2, 4]
    results = []
    
    for n_processes in process_counts:
        print(f"\nTesting with {n_processes} process(es)...")
        
        config = ParallelESConfig(
            population_size=100,  # 更大的种群
            n_generations=10,      # 更多的世代
            n_processes=n_processes,
            batch_size=10
        )
        
        trainer = ParallelESTrainer(config)
        
        start = time.time()
        trainer.train(n_generations=10)
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
    print("LARGE SCALE PERFORMANCE RESULTS")
    print("="*70)
    
    print(f"\n{'Processes':<10} {'Time (s)':<10} {'Speedup':<10} {'Best Fitness':<15}")
    print("-"*50)
    
    for result in results:
        speedup = base_time / result['time'] if result['time'] > 0 else 0
        print(f"{result['processes']:<10} {result['time']:<10.2f} {speedup:<10.2f} {result['best_fitness']:<15.3f}")
    
    return results


if __name__ == "__main__":
    large_scale_performance_test()
