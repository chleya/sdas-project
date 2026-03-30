#!/usr/bin/env python3
"""
自适应结构 vs 固定结构 对比测试

测试场景：
1. 简单任务阶段（0-500步）
2. 困难任务阶段（500-1000步）
3. 中等任务阶段（1000-1500步）

对比指标：
- 任务完成率
- 平均奖励
- 结构数量变化
- 计算效率
"""

import numpy as np
import sys
import os
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))

from adaptive_structure_pool import AdaptiveStructurePool, AdaptiveConfig
from structure_pool import StructurePool
from sdas import SDASAgent, Config, run_episode
from complex_petri_dish import ComplexPetriDish


class FixedStructureAgent:
    """使用固定结构池的智能体"""
    
    def __init__(self, n_structures=16):
        self.config = Config()
        self.config.max_structures = n_structures
        self.agent = SDASAgent(self.config)
        
    def step(self, obs):
        return self.agent.step(obs)
    
    def update_structure(self, encoder_output, **kwargs):
        # 使用固定的结构池
        self.agent.structure_pool.observe(encoder_output)


class AdaptiveStructureAgent:
    """使用自适应结构池的智能体"""
    
    def __init__(self, adaptive_config=None):
        self.config = Config()
        self.adaptive_config = adaptive_config or AdaptiveConfig(
            min_structures=4,
            max_structures=32,
            initial_structures=8,
            adaptation_interval=100
        )
        
        self.agent = SDASAgent(self.config)
        # 替换为自适应结构池
        self.agent.structure_pool = AdaptiveStructurePool(
            self.adaptive_config,
            {'vector_dim': self.config.encoder_dim}
        )
    
    def step(self, obs):
        return self.agent.step(obs)
    
    def update_structure(self, encoder_output, prediction_error=0.0, info_gain=0.0):
        # 使用自适应结构池
        self.agent.structure_pool.observe(
            encoder_output, 
            prediction_error=prediction_error,
            info_gain=info_gain
        )


def run_comparison_test():
    """运行对比测试"""
    print("="*70)
    print("Adaptive vs Fixed Structure Comparison Test")
    print("="*70)
    
    # 创建环境
    env_config = {
        'width': 15,
        'height': 15,
        'n_static_obstacles': 15,
        'n_dynamic_obstacles': 3,
        'n_energy_sources': 5
    }
    
    # 创建智能体
    fixed_agent = FixedStructureAgent(n_structures=16)
    adaptive_agent = AdaptiveStructureAgent()
    
    # 测试参数
    n_episodes = 30
    max_steps = 200
    
    # 记录结果
    fixed_results = {'rewards': [], 'successes': [], 'structure_counts': []}
    adaptive_results = {'rewards': [], 'successes': [], 'structure_counts': []}
    
    print(f"\nRunning {n_episodes} episodes per agent...")
    print(f"Environment: {env_config['width']}x{env_config['height']} grid")
    print(f"Max steps per episode: {max_steps}")
    
    # 测试固定结构智能体
    print("\n" + "-"*70)
    print("Testing FIXED Structure Agent")
    print("-"*70)
    
    for episode in range(n_episodes):
        env = ComplexPetriDish(**env_config)
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            obs = env._get_obs()
            action, _ = fixed_agent.step(obs)
            obs, reward, done = env.step(action)
            
            # 模拟结构更新
            encoder_output = np.random.randn(64)  # 简化的编码器输出
            fixed_agent.update_structure(encoder_output)
            
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        fixed_results['rewards'].append(total_reward)
        fixed_results['successes'].append(1 if done else 0)
        fixed_results['structure_counts'].append(16)  # 固定16个
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(fixed_results['rewards'][-10:])
            print(f"  Episode {episode+1}: avg_reward={avg_reward:.2f}, steps={steps}")
    
    # 测试自适应结构智能体
    print("\n" + "-"*70)
    print("Testing ADAPTIVE Structure Agent")
    print("-"*70)
    
    for episode in range(n_episodes):
        env = ComplexPetriDish(**env_config)
        total_reward = 0
        steps = 0
        
        # 动态调整难度（模拟不同任务阶段）
        if episode < 10:
            difficulty = 0.3  # 简单
        elif episode < 20:
            difficulty = 0.8  # 困难
        else:
            difficulty = 0.5  # 中等
        
        for step in range(max_steps):
            obs = env._get_obs()
            action, _ = adaptive_agent.step(obs)
            obs, reward, done = env.step(action)
            
            # 模拟结构更新，传入难度信息
            encoder_output = np.random.randn(64)
            prediction_error = difficulty * 5.0
            info_gain = 1.0 - difficulty * 0.5
            adaptive_agent.update_structure(encoder_output, prediction_error, info_gain)
            
            # 添加性能历史
            adaptive_agent.agent.structure_pool.performance_history.append(reward)
            
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        adaptive_results['rewards'].append(total_reward)
        adaptive_results['successes'].append(1 if done else 0)
        adaptive_results['structure_counts'].append(
            len(adaptive_agent.agent.structure_pool.structures)
        )
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(adaptive_results['rewards'][-10:])
            current_structures = len(adaptive_agent.agent.structure_pool.structures)
            print(f"  Episode {episode+1}: avg_reward={avg_reward:.2f}, "
                  f"structures={current_structures}, steps={steps}")
    
    # 打印对比结果
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    print(f"\n{'Metric':<30} {'Fixed':<15} {'Adaptive':<15} {'Improvement':<15}")
    print("-"*70)
    
    # 平均奖励
    fixed_avg_reward = np.mean(fixed_results['rewards'])
    adaptive_avg_reward = np.mean(adaptive_results['rewards'])
    improvement = ((adaptive_avg_reward - fixed_avg_reward) / 
                   abs(fixed_avg_reward) * 100) if fixed_avg_reward != 0 else 0
    print(f"{'Average Reward':<30} {fixed_avg_reward:<15.2f} "
          f"{adaptive_avg_reward:<15.2f} {improvement:>+14.1f}%")
    
    # 成功率
    fixed_success_rate = np.mean(fixed_results['successes']) * 100
    adaptive_success_rate = np.mean(adaptive_results['successes']) * 100
    improvement = adaptive_success_rate - fixed_success_rate
    print(f"{'Success Rate (%)':<30} {fixed_success_rate:<15.1f} "
          f"{adaptive_success_rate:<15.1f} {improvement:>+14.1f}%")
    
    # 奖励方差（稳定性）
    fixed_std = np.std(fixed_results['rewards'])
    adaptive_std = np.std(adaptive_results['rewards'])
    improvement = ((fixed_std - adaptive_std) / fixed_std * 100) if fixed_std != 0 else 0
    print(f"{'Reward Std (lower=better)':<30} {fixed_std:<15.2f} "
          f"{adaptive_std:<15.2f} {improvement:>+14.1f}%")
    
    # 平均结构数量
    fixed_avg_structures = np.mean(fixed_results['structure_counts'])
    adaptive_avg_structures = np.mean(adaptive_results['structure_counts'])
    print(f"{'Avg Structures':<30} {fixed_avg_structures:<15.1f} "
          f"{adaptive_avg_structures:<15.1f}")
    
    # 自适应调整历史
    print("\n" + "-"*70)
    print("Adaptive Structure Adjustments:")
    print("-"*70)
    
    if adaptive_agent.agent.structure_pool.adaptation_history:
        for adapt in adaptive_agent.agent.structure_pool.adaptation_history:
            print(f"  Step {adapt['time_step']}: {adapt['action']} "
                  f"(n={adapt['n_structures']}, diff={adapt['difficulty']:.3f}, "
                  f"util={adapt['utilization']:.3f})")
    else:
        print("  No adaptations occurred during test")
    
    return fixed_results, adaptive_results


if __name__ == "__main__":
    run_comparison_test()
