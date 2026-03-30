#!/usr/bin/env python3
"""
Baseline对比测试
测试三个agent：
1. 纯随机（RandomAgent）
2. SDAS（有结构池）
3. TabularQ（无结构的Q-learning）
"""

import numpy as np
import random
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sdas import SDASAgent, Config, run_episode
from digital_petri_dish import DigitalPetriDish


class RandomAgent:
    """纯随机Agent - 下界基准"""
    def __init__(self, n_actions=5):
        self.n_actions = n_actions
    
    def step(self, obs):
        return random.randint(0, self.n_actions - 1), {}
    
    def update_structure(self, reward):
        pass
    
    def reset(self):
        pass


class TabularQAgent:
    """无结构的Q-table Agent - 对比SDAS的结构池是否有价值"""
    def __init__(self, n_actions=5, learning_rate=0.1, epsilon=0.1):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.epsilon = epsilon
        self.q = {}  # Q-table: {state_key: [q_values]}
        self.last_state = None
        self.last_action = None
    
    def _get_state_key(self, obs):
        """将观测转换为状态键（离散化）"""
        # 简化：将连续值离散化为几个区间
        agent_x = int(obs.get('agent_x', 0) / 3)  # 离散化为3x3的网格
        agent_y = int(obs.get('agent_y', 0) / 3)
        energy_dist = int(obs.get('energy_dist', 1.0) * 3)  # 0, 1, 2, 3
        obstacle_nearby = 1 if obs.get('obstacle_nearby', False) else 0
        
        return f"{agent_x}_{agent_y}_{energy_dist}_{obstacle_nearby}"
    
    def step(self, obs):
        state_key = self._get_state_key(obs)
        
        # 初始化Q值
        if state_key not in self.q:
            self.q[state_key] = np.zeros(self.n_actions)
        
        # epsilon-greedy
        if random.random() < self.epsilon:
            action = random.randint(0, self.n_actions - 1)
        else:
            action = int(np.argmax(self.q[state_key]))
        
        self.last_state = state_key
        self.last_action = action
        
        return action, {'state_key': state_key}
    
    def update_structure(self, reward):
        """Q-learning更新"""
        if self.last_state is not None and self.last_action is not None:
            # 简单Q-learning更新
            current_q = self.q[self.last_state][self.last_action]
            # 简化：假设下一状态Q值为0（无bootstrapping）
            self.q[self.last_state][self.last_action] += self.lr * (reward - current_q)
    
    def reset(self):
        self.last_state = None
        self.last_action = None


def test_single_agent(agent_class, agent_name, env, n_episodes=50):
    """测试单个agent"""
    print(f"\n{'='*60}")
    print(f"Testing {agent_name}")
    print('='*60)
    
    if agent_class == SDASAgent:
        agent = agent_class(Config())
    else:
        agent = agent_class()
    
    rewards = []
    steps_list = []
    
    for ep in range(n_episodes):
        obs = env._reset()
        agent.reset()
        
        total_reward = 0
        steps = 0
        max_steps = 200
        
        for step in range(max_steps):
            action, info = agent.step(obs)
            obs, reward, done = env.step(action)
            agent.update_structure(reward)
            
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        rewards.append(total_reward)
        steps_list.append(steps)
        
        if (ep + 1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            print(f"  Episode {ep+1}/{n_episodes}: Avg Reward (last 10) = {avg_reward:.2f}")
    
    return {
        'name': agent_name,
        'rewards': rewards,
        'steps': steps_list,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_steps': np.mean(steps_list)
    }


def plot_learning_curves(results, save_path='learning_curves.png'):
    """绘制学习曲线"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 学习曲线（原始奖励）
        ax = axes[0, 0]
        for result in results:
            ax.plot(result['rewards'], label=result['name'], alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Learning Curves (Raw)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 平滑学习曲线（移动平均）
        ax = axes[0, 1]
        window = 10
        for result in results:
            smoothed = np.convolve(result['rewards'], np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=result['name'], linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward (Smoothed)')
        ax.set_title(f'Learning Curves (Moving Avg, window={window})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 步数曲线
        ax = axes[1, 0]
        for result in results:
            ax.plot(result['steps'], label=result['name'], alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Steps per Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 平均奖励对比（柱状图）
        ax = axes[1, 1]
        names = [r['name'] for r in results]
        means = [r['mean_reward'] for r in results]
        stds = [r['std_reward'] for r in results]
        
        x = np.arange(len(names))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.set_ylabel('Mean Reward')
        ax.set_title('Average Performance Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上添加数值标签
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.2f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nLearning curves saved to {save_path}")
        plt.close()
        
    except ImportError:
        print("matplotlib not available, skipping plot")


def main():
    """主测试函数"""
    print("="*70)
    print("SDAS Baseline Comparison Test")
    print("="*70)
    print("\nTesting three agents:")
    print("  1. RandomAgent - Pure random baseline")
    print("  2. TabularQAgent - Q-learning without structure pool")
    print("  3. SDASAgent - Structure-driven agent with competitive pool")
    print()
    
    # 创建环境
    env = DigitalPetriDish(width=15, height=15, n_obstacles=25, n_energy=5)
    n_episodes = 50
    
    # 测试三个agent
    results = []
    
    results.append(test_single_agent(RandomAgent, "Random", env, n_episodes))
    results.append(test_single_agent(TabularQAgent, "TabularQ", env, n_episodes))
    results.append(test_single_agent(SDASAgent, "SDAS", env, n_episodes))
    
    # 打印总结
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Agent':<15} {'Mean Reward':<15} {'Std Reward':<15} {'Mean Steps':<15}")
    print("-"*70)
    
    for result in results:
        print(f"{result['name']:<15} {result['mean_reward']:<15.2f} "
              f"{result['std_reward']:<15.2f} {result['mean_steps']:<15.1f}")
    
    # 计算性能提升
    random_mean = results[0]['mean_reward']
    tabular_mean = results[1]['mean_reward']
    sdas_mean = results[2]['mean_reward']
    
    print("\n" + "="*70)
    print("PERFORMANCE IMPROVEMENTS")
    print("="*70)
    
    if random_mean != 0:
        tabular_improvement = (tabular_mean - random_mean) / abs(random_mean) * 100
        sdas_improvement = (sdas_mean - random_mean) / abs(random_mean) * 100
        print(f"TabularQ vs Random: {tabular_improvement:+.1f}%")
        print(f"SDAS vs Random: {sdas_improvement:+.1f}%")
    
    if tabular_mean != 0:
        sdas_vs_tabular = (sdas_mean - tabular_mean) / abs(tabular_mean) * 100
        print(f"SDAS vs TabularQ: {sdas_vs_tabular:+.1f}%")
    
    # 绘制学习曲线
    plot_learning_curves(results)
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = main()
