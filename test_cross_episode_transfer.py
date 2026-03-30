#!/usr/bin/env python3
"""
跨episode结构复用实验

测试环境改变后，有结构池的SDAS是否比没有结构池的TabularQ恢复更快。

实验设计：
- Episode 1-20: 普通迷宫（基准）
- Episode 21-40: 迷宫布局改变（障碍物重置）
- Episode 41-60: 再次改变

测量指标：
- 每个阶段前5个episode的平均奖励（恢复速度）
- 结构池在环境变化时的利用率
"""

import numpy as np
import random
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sdas import SDASAgent, Config, run_episode
from digital_petri_dish import DigitalPetriDish


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


def test_cross_episode_transfer():
    """跨episode结构复用测试"""
    print("="*80)
    print("Cross-Episode Structure Transfer Experiment")
    print("="*80)
    print("\nExperimental Design:")
    print("  Phase 1 (Ep 1-20): Normal maze (baseline)")
    print("  Phase 2 (Ep 21-40): Maze layout changed (obstacles reset)")
    print("  Phase 3 (Ep 41-60): Maze layout changed again")
    print()
    
    # 环境配置
    width, height = 15, 15
    n_obstacles = 25
    n_energy = 5
    
    # 创建两个环境：一个用于SDAS，一个用于TabularQ
    env_sdas = DigitalPetriDish(width=width, height=height, n_obstacles=n_obstacles, n_energy=n_energy)
    env_tabular = DigitalPetriDish(width=width, height=height, n_obstacles=n_obstacles, n_energy=n_energy)
    
    # 初始化智能体
    sdas_agent = SDASAgent(Config())
    tabular_agent = TabularQAgent()
    
    # 存储结果
    results = {
        'sdas': {'rewards': [], 'phases': []},
        'tabular': {'rewards': [], 'phases': []}
    }
    
    # 结构池状态记录
    structure_stats = []
    
    total_episodes = 60
    max_steps = 200
    
    for episode in range(1, total_episodes + 1):
        # 确定当前阶段
        if episode <= 20:
            phase = 1
        elif episode <= 40:
            phase = 2
        else:
            phase = 3
        
        # 在阶段切换时重置环境
        if episode == 21 or episode == 41:
            print(f"\n{'='*60}")
            print(f"PHASE {phase} START: Resetting environment layout")
            print('='*60)
            # 重置环境（生成新的障碍物布局）
            env_sdas = DigitalPetriDish(width=width, height=height, n_obstacles=n_obstacles, n_energy=n_energy)
            env_tabular = DigitalPetriDish(width=width, height=height, n_obstacles=n_obstacles, n_energy=n_energy)
        
        # 测试SDAS
        print(f"\nEpisode {episode:3d}/{total_episodes} (Phase {phase})")
        
        # 运行SDAS
        obs = env_sdas._reset()
        sdas_agent.reset()
        total_reward_sdas = 0
        steps_sdas = 0
        
        for step in range(max_steps):
            action, info = sdas_agent.step(obs)
            obs, reward, done = env_sdas.step(action)
            sdas_agent.update_structure(reward)
            total_reward_sdas += reward
            steps_sdas += 1
            if done:
                break
        
        # 运行TabularQ
        obs = env_tabular._reset()
        tabular_agent.reset()
        total_reward_tabular = 0
        steps_tabular = 0
        
        for step in range(max_steps):
            action, info = tabular_agent.step(obs)
            obs, reward, done = env_tabular.step(action)
            tabular_agent.update_structure(reward)
            total_reward_tabular += reward
            steps_tabular += 1
            if done:
                break
        
        # 记录结果
        results['sdas']['rewards'].append(total_reward_sdas)
        results['sdas']['phases'].append(phase)
        results['tabular']['rewards'].append(total_reward_tabular)
        results['tabular']['phases'].append(phase)
        
        # 记录结构池状态
        if episode in [20, 21, 40, 41, 60]:
            structure_state = sdas_agent.structure_pool.get_state()
            structure_stats.append({
                'episode': episode,
                'phase': phase,
                'structure_count': structure_state['structure_count'],
                'structures': structure_state['structures']
            })
        
        # 每10个episode打印一次进度
        if episode % 10 == 0:
            avg_sdas = np.mean(results['sdas']['rewards'][-10:])
            avg_tabular = np.mean(results['tabular']['rewards'][-10:])
            print(f"  Average Reward (last 10): SDAS={avg_sdas:5.2f}, TabularQ={avg_tabular:5.2f}")
            print(f"  SDAS Structure Count: {len(sdas_agent.structure_pool.structures)}")
    
    # 分析结果
    analyze_results(results, structure_stats)
    
    return results, structure_stats

def analyze_results(results, structure_stats):
    """分析实验结果"""
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS ANALYSIS")
    print("="*80)
    
    # 按阶段分析
    phases = [1, 2, 3]
    phase_starts = [0, 20, 40]  # 0-based索引
    phase_ends = [20, 40, 60]
    
    print("\nPhase Performance:")
    print("-"*60)
    print(f"{'Phase':<10} {'Agent':<10} {'Full Phase Avg':<15} {'First 5 Avg':<15} {'Last 5 Avg':<15}")
    print("-"*60)
    
    for i, phase in enumerate(phases, 1):
        start = phase_starts[i-1]
        end = phase_ends[i-1]
        
        # 全阶段平均
        sdas_full = np.mean(results['sdas']['rewards'][start:end])
        tabular_full = np.mean(results['tabular']['rewards'][start:end])
        
        # 前5个episode平均（恢复速度）
        sdas_first5 = np.mean(results['sdas']['rewards'][start:start+5])
        tabular_first5 = np.mean(results['tabular']['rewards'][start:start+5])
        
        # 后5个episode平均（稳定性能）
        sdas_last5 = np.mean(results['sdas']['rewards'][end-5:end])
        tabular_last5 = np.mean(results['tabular']['rewards'][end-5:end])
        
        print(f"{phase:<10} {'SDAS':<10} {sdas_full:<15.2f} {sdas_first5:<15.2f} {sdas_last5:<15.2f}")
        print(f"{phase:<10} {'TabularQ':<10} {tabular_full:<15.2f} {tabular_first5:<15.2f} {tabular_last5:<15.2f}")
        print()
    
    # 计算恢复速度差异
    print("Recovery Speed Analysis:")
    print("-"*60)
    
    # Phase 2（环境第一次改变）
    sdas_phase2_first5 = np.mean(results['sdas']['rewards'][20:25])
    tabular_phase2_first5 = np.mean(results['tabular']['rewards'][20:25])
    recovery_diff_phase2 = sdas_phase2_first5 - tabular_phase2_first5
    
    # Phase 3（环境第二次改变）
    sdas_phase3_first5 = np.mean(results['sdas']['rewards'][40:45])
    tabular_phase3_first5 = np.mean(results['tabular']['rewards'][40:45])
    recovery_diff_phase3 = sdas_phase3_first5 - tabular_phase3_first5
    
    print(f"Phase 2 (First Change) Recovery Difference: SDAS - TabularQ = {recovery_diff_phase2:5.2f}")
    print(f"Phase 3 (Second Change) Recovery Difference: SDAS - TabularQ = {recovery_diff_phase3:5.2f}")
    print()
    
    # 结构池分析
    print("Structure Pool Analysis:")
    print("-"*60)
    for stat in structure_stats:
        print(f"Episode {stat['episode']} (Phase {stat['phase']}): {stat['structure_count']} structures")
        # 打印前3个结构的效用
        if stat['structures']:
            top_structures = sorted(stat['structures'], key=lambda x: x['utility'], reverse=True)[:3]
            for s in top_structures:
                print(f"  - Structure {s['id']}: utility={s['utility']:.3f}, label={s['label']}")
    
    # 绘制结果
    plot_results(results)

def plot_results(results):
    """绘制实验结果"""
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 学习曲线
        episodes = np.arange(1, 61)
        ax1.plot(episodes, results['sdas']['rewards'], label='SDAS', linewidth=2)
        ax1.plot(episodes, results['tabular']['rewards'], label='TabularQ', linewidth=2)
        
        # 绘制阶段分隔线
        ax1.axvline(x=20.5, color='gray', linestyle='--', alpha=0.7)
        ax1.axvline(x=40.5, color='gray', linestyle='--', alpha=0.7)
        ax1.text(10, ax1.get_ylim()[1] * 0.95, 'Phase 1', ha='center')
        ax1.text(30, ax1.get_ylim()[1] * 0.95, 'Phase 2', ha='center')
        ax1.text(50, ax1.get_ylim()[1] * 0.95, 'Phase 3', ha='center')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Learning Curves (Cross-Episode Transfer)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 每个阶段的平均奖励
        phases = [1, 2, 3]
        phase_names = ['Phase 1 (Normal)', 'Phase 2 (Change 1)', 'Phase 3 (Change 2)']
        
        sdas_phase_avgs = []
        tabular_phase_avgs = []
        sdas_first5_avgs = []
        tabular_first5_avgs = []
        
        phase_starts = [0, 20, 40]
        phase_ends = [20, 40, 60]
        
        for start, end in zip(phase_starts, phase_ends):
            sdas_phase_avgs.append(np.mean(results['sdas']['rewards'][start:end]))
            tabular_phase_avgs.append(np.mean(results['tabular']['rewards'][start:end]))
            sdas_first5_avgs.append(np.mean(results['sdas']['rewards'][start:start+5]))
            tabular_first5_avgs.append(np.mean(results['tabular']['rewards'][start:start+5]))
        
        x = np.arange(len(phases))
        width = 0.35
        
        ax2.bar(x - width/2, sdas_first5_avgs, width, label='SDAS (First 5)', alpha=0.7)
        ax2.bar(x + width/2, tabular_first5_avgs, width, label='TabularQ (First 5)', alpha=0.7)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(phase_names, rotation=15, ha='right')
        ax2.set_ylabel('Average Reward (First 5 Episodes)')
        ax2.set_title('Recovery Speed Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('cross_episode_transfer.png', dpi=150, bbox_inches='tight')
        print(f"\nResults saved to cross_episode_transfer.png")
        plt.close()
        
    except ImportError:
        print("matplotlib not available, skipping plot")

def main():
    """主函数"""
    results, structure_stats = test_cross_episode_transfer()
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    
    return results


if __name__ == "__main__":
    main()
