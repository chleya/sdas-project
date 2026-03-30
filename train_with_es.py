"""
使用进化策略训练 SDAS 结构池参数
完整训练脚本
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))

from es_trainer import ESTrainer, ESConfig, StructurePoolParams
from sdas import SDASAgent, Config, run_episode
from digital_petri_dish import DigitalPetriDish
import json


def train_with_es():
    """使用 ES 训练最佳参数"""
    
    print("=" * 70)
    print("SDAS Structure Pool Optimization with Evolution Strategies")
    print("=" * 70)
    
    # 配置 ES 训练
    es_config = ESConfig(
        population_size=30,          # 种群大小
        sigma=0.15,                  # 扰动幅度
        learning_rate=0.08,          # 学习率
        n_generations=20,            # 训练代数
        n_eval_episodes=3,           # 每个个体评估回合数
        max_steps_per_episode=100    # 每回合最大步数
    )
    
    # 环境配置
    env_config = {
        'width': 15,
        'height': 15,
        'n_obstacles': 25,
        'n_energy': 5
    }
    
    # 创建训练器
    trainer = ESTrainer(es_config, env_config)
    
    # 训练
    best_params = trainer.train()
    
    # 保存结果
    results = {
        'best_params': {
            'decay_rate': best_params.decay_rate,
            'create_threshold': best_params.create_threshold,
            'max_structures': best_params.max_structures,
            'utility_lr': best_params.utility_lr,
        },
        'best_fitness': trainer.best_fitness,
        'fitness_history': [
            {
                'generation': i,
                'mean': h['mean'],
                'max': h['max'],
                'min': h['min']
            }
            for i, h in enumerate(trainer.fitness_history)
        ]
    }
    
    with open('es_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Results saved to es_training_results.json")
    print("=" * 70)
    
    # 尝试绘制训练曲线
    trainer.plot_history()
    
    return best_params


def compare_performance(optimized_params: StructurePoolParams, n_episodes: int = 10):
    """
    对比优化前后的性能
    """
    print("\n" + "=" * 70)
    print("Performance Comparison: Default vs Optimized")
    print("=" * 70)
    
    env_config = {
        'width': 15,
        'height': 15,
        'n_obstacles': 25,
        'n_energy': 5
    }
    
    # 默认参数
    default_config = Config()
    
    # 优化后的参数
    optimized_config = optimized_params.to_config()
    
    print("\nDefault Parameters:")
    print(f"  decay_rate: {default_config.decay_rate}")
    print(f"  create_threshold: {default_config.create_threshold}")
    print(f"  max_structures: {default_config.max_structures}")
    
    print("\nOptimized Parameters:")
    print(f"  decay_rate: {optimized_config.decay_rate}")
    print(f"  create_threshold: {optimized_config.create_threshold}")
    print(f"  max_structures: {optimized_config.max_structures}")
    
    # 测试默认参数
    print(f"\n--- Testing Default Params ({n_episodes} episodes) ---")
    default_rewards = []
    default_steps = []
    
    for ep in range(n_episodes):
        env = DigitalPetriDish(**env_config)
        agent = SDASAgent(default_config)
        result = run_episode(env, agent, max_steps=100)
        default_rewards.append(result['total_reward'])
        default_steps.append(result['steps'])
        print(f"  Episode {ep+1}: reward={result['total_reward']:.3f}, steps={result['steps']}")
    
    # 测试优化参数
    print(f"\n--- Testing Optimized Params ({n_episodes} episodes) ---")
    optimized_rewards = []
    optimized_steps = []
    
    for ep in range(n_episodes):
        env = DigitalPetriDish(**env_config)
        agent = SDASAgent(optimized_config)
        result = run_episode(env, agent, max_steps=100)
        optimized_rewards.append(result['total_reward'])
        optimized_steps.append(result['steps'])
        print(f"  Episode {ep+1}: reward={result['total_reward']:.3f}, steps={result['steps']}")
    
    # 统计对比
    print("\n" + "=" * 70)
    print("Comparison Results")
    print("=" * 70)
    
    import numpy as np
    
    print(f"\nAverage Reward:")
    print(f"  Default:   {np.mean(default_rewards):.3f} ± {np.std(default_rewards):.3f}")
    print(f"  Optimized: {np.mean(optimized_rewards):.3f} ± {np.std(optimized_rewards):.3f}")
    print(f"  Improvement: {np.mean(optimized_rewards) - np.mean(default_rewards):.3f}")
    
    print(f"\nAverage Steps:")
    print(f"  Default:   {np.mean(default_steps):.1f} ± {np.std(default_steps):.1f}")
    print(f"  Optimized: {np.mean(optimized_steps):.1f} ± {np.std(optimized_steps):.1f}")
    
    print(f"\nSuccess Rate (positive reward):")
    default_success = sum(1 for r in default_rewards if r > 0) / n_episodes
    optimized_success = sum(1 for r in optimized_rewards if r > 0) / n_episodes
    print(f"  Default:   {default_success:.1%}")
    print(f"  Optimized: {optimized_success:.1%}")
    
    return {
        'default_rewards': default_rewards,
        'optimized_rewards': optimized_rewards,
        'default_steps': default_steps,
        'optimized_steps': optimized_steps
    }


if __name__ == "__main__":
    # 询问用户选择
    print("SDAS Evolution Strategies Training")
    print("=" * 70)
    print("1. Quick test (fast, for debugging)")
    print("2. Full training (20 generations, takes a few minutes)")
    print("3. Compare only (use saved optimized params)")
    print("=" * 70)
    
    choice = input("\nSelect option (1/2/3): ").strip()
    
    if choice == "1":
        # 快速测试
        from es_trainer import quick_test
        quick_test()
        
    elif choice == "2":
        # 完整训练
        best_params = train_with_es()
        
        # 询问是否进行对比测试
        compare = input("\nRun performance comparison? (y/n): ").strip().lower()
        if compare == 'y':
            compare_performance(best_params)
            
    elif choice == "3":
        # 仅对比
        if os.path.exists('es_training_results.json'):
            with open('es_training_results.json', 'r') as f:
                results = json.load(f)
            params = StructurePoolParams(**results['best_params'])
            compare_performance(params)
        else:
            print("No saved results found. Please run training first.")
    else:
        print("Invalid choice. Exiting.")
