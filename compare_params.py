"""
快速对比默认参数 vs 优化参数的性能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))

from sdas import SDASAgent, Config, run_episode
from digital_petri_dish import DigitalPetriDish
import numpy as np


def test_params(decay_rate, create_threshold, max_structures, n_episodes=5, label="Test"):
    """测试一组参数"""
    config = Config()
    config.decay_rate = decay_rate
    config.create_threshold = create_threshold
    config.max_structures = max_structures
    
    env_config = {
        'width': 15,
        'height': 15,
        'n_obstacles': 25,
        'n_energy': 5
    }
    
    rewards = []
    steps = []
    
    print(f"\n{label} Params: decay={decay_rate:.3f}, threshold={create_threshold:.3f}, max_struct={max_structures}")
    
    for ep in range(n_episodes):
        env = DigitalPetriDish(**env_config)
        agent = SDASAgent(config)
        result = run_episode(env, agent, max_steps=100)
        rewards.append(result['total_reward'])
        steps.append(result['steps'])
        print(f"  Ep {ep+1}: reward={result['total_reward']:.3f}, steps={result['steps']}")
    
    return rewards, steps


if __name__ == "__main__":
    print("=" * 70)
    print("SDAS Parameter Comparison")
    print("=" * 70)
    
    # 默认参数
    default_rewards, default_steps = test_params(
        decay_rate=0.08,
        create_threshold=0.42,
        max_structures=32,
        n_episodes=5,
        label="DEFAULT"
    )
    
    # 手动设置一组"优化"参数（基于 ES 训练的经验）
    # 通常较低的 decay_rate 和较高的 threshold 表现更好
    optimized_rewards, optimized_steps = test_params(
        decay_rate=0.05,
        create_threshold=0.50,
        max_structures=24,
        n_episodes=5,
        label="OPTIMIZED"
    )
    
    # 统计对比
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nAverage Reward:")
    print(f"  Default:   {np.mean(default_rewards):.3f} ± {np.std(default_rewards):.3f}")
    print(f"  Optimized: {np.mean(optimized_rewards):.3f} ± {np.std(optimized_rewards):.3f}")
    improvement = np.mean(optimized_rewards) - np.mean(default_rewards)
    print(f"  Improvement: {improvement:+.3f} ({improvement/abs(np.mean(default_rewards))*100:+.1f}%)")
    
    print(f"\nAverage Steps:")
    print(f"  Default:   {np.mean(default_steps):.1f}")
    print(f"  Optimized: {np.mean(optimized_steps):.1f}")
    
    print(f"\nSuccess Rate (reward > 0):")
    default_success = sum(1 for r in default_rewards if r > 0) / len(default_rewards)
    opt_success = sum(1 for r in optimized_rewards if r > 0) / len(optimized_rewards)
    print(f"  Default:   {default_success:.1%}")
    print(f"  Optimized: {opt_success:.1%}")
    
    print("\n" + "=" * 70)
    print("To run full ES training, use: python train_with_es.py")
    print("=" * 70)
