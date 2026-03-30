#!/usr/bin/env python3
"""
修复版跨episode结构复用实验

解决了三个关键问题：
1. 两个agent跑完全相同的地图（共享随机种子）
2. 运行多个随机种子，结果有统计意义
3. 添加TabularQ重置对照组，确保记忆状态对等

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
    def __init__(self, n_actions=5, learning_rate=0.1, epsilon=0.1, reset_on_episode=False):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.epsilon = epsilon
        self.reset_on_episode = reset_on_episode
        self.q = {}  # Q-table: {state_key: [q_values]}
        self.last_state = None
        self.last_action = None
    
    def _get_state_key(self, obs):
        """将观测转换为状态键（离散化）"""
        # 注意：agent_x和agent_y是归一化到[0,1]的
        agent_x = int(obs.get('agent_x', 0) * 5)  # 离散化为5个区间
        agent_y = int(obs.get('agent_y', 0) * 5)
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
        """重置智能体状态"""
        if self.reset_on_episode:
            # 每个episode重置Q-table（对照组）
            self.q = {}
        self.last_state = None
        self.last_action = None


def run_experiment(seed):
    """运行单次实验"""
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Running Experiment with Seed: {seed}")
    print('='*60)
    
    # 环境配置
    width, height = 15, 15
    n_obstacles = 25
    n_energy = 5
    
    # 创建三个固定地图（不同阶段）
    env_phase1 = DigitalPetriDish(width=width, height=height, n_obstacles=n_obstacles, n_energy=n_energy)
    env_phase2 = DigitalPetriDish(width=width, height=height, n_obstacles=n_obstacles, n_energy=n_energy)  # 新地图
    env_phase3 = DigitalPetriDish(width=width, height=height, n_obstacles=n_obstacles, n_energy=n_energy)  # 恢复地图
    
    # 初始化智能体
    sdas_agent = SDASAgent(Config())
    tabular_agent = TabularQAgent()  # 常规TabularQ（有记忆）
    tabular_reset_agent = TabularQAgent(reset_on_episode=True)  # 对照组（无记忆）
    
    # 存储结果
    results = {
        'sdas': {'rewards': [], 'phases': []},
        'tabular': {'rewards': [], 'phases': []},
        'tabular_reset': {'rewards': [], 'phases': []}
    }
    
    # 结构池状态记录（只记录第一次种子）
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
        
        # 在阶段切换时选择固定环境
        if episode == 21 or episode == 41:
            print(f"\n{'='*60}")
            print(f"PHASE {phase} START: Changing environment layout")
            print('='*60)
        
        # 根据阶段选择环境
        if phase == 1:
            env = env_phase1
        elif phase == 2:
            env = env_phase2
        else:  # phase == 3
            env = env_phase3
        
        # 测试SDAS
        print(f"\nEpisode {episode:3d}/{total_episodes} (Phase {phase})")
        
        # 运行SDAS
        obs = env._reset()
        sdas_agent.reset()
        total_reward_sdas = 0
        
        for step in range(max_steps):
            action, info = sdas_agent.step(obs)
            obs, reward, done = env.step(action)
            sdas_agent.update_structure(reward)
            total_reward_sdas += reward
            if done:
                break
        
        # 重新重置环境，确保TabularQ跑相同的地图
        obs = env._reset()
        tabular_agent.reset()
        total_reward_tabular = 0
        
        for step in range(max_steps):
            action, info = tabular_agent.step(obs)
            obs, reward, done = env.step(action)
            tabular_agent.update_structure(reward)
            total_reward_tabular += reward
            if done:
                break
        
        # 重新重置环境，确保TabularQ重置版跑相同的地图
        obs = env._reset()
        tabular_reset_agent.reset()
        total_reward_tabular_reset = 0
        
        for step in range(max_steps):
            action, info = tabular_reset_agent.step(obs)
            obs, reward, done = env.step(action)
            tabular_reset_agent.update_structure(reward)
            total_reward_tabular_reset += reward
            if done:
                break
        
        # 记录结果
        results['sdas']['rewards'].append(total_reward_sdas)
        results['sdas']['phases'].append(phase)
        results['tabular']['rewards'].append(total_reward_tabular)
        results['tabular']['phases'].append(phase)
        results['tabular_reset']['rewards'].append(total_reward_tabular_reset)
        results['tabular_reset']['phases'].append(phase)
        
        # 记录结构池状态（只记录第一个种子）
        if seed == 0 and episode in [20, 21, 40, 41, 60]:
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
            avg_tabular_reset = np.mean(results['tabular_reset']['rewards'][-10:])
            print(f"  Average Reward (last 10):")
            print(f"    SDAS: {avg_sdas:5.2f}")
            print(f"    TabularQ: {avg_tabular:5.2f}")
            print(f"    TabularQ (Reset): {avg_tabular_reset:5.2f}")
            print(f"  SDAS Structure Count: {len(sdas_agent.structure_pool.structures)}")
    
    return results, structure_stats

def analyze_results(all_results, structure_stats):
    """分析实验结果"""
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS ANALYSIS")
    print("="*80)
    
    # 计算所有种子的平均值和标准差
    agents = ['sdas', 'tabular', 'tabular_reset']
    agent_names = {'sdas': 'SDAS', 'tabular': 'TabularQ', 'tabular_reset': 'TabularQ (Reset)'}
    
    # 按阶段分析
    phases = [1, 2, 3]
    phase_starts = [0, 20, 40]  # 0-based索引
    phase_ends = [20, 40, 60]
    
    print("\nPhase Performance (Mean ± Std):")
    print("-"*80)
    print(f"{'Phase':<10} {'Agent':<20} {'Full Phase':<15} {'First 5':<15} {'Last 5':<15}")
    print("-"*80)
    
    phase_results = {agent: {phase: {'full': [], 'first5': [], 'last5': []} for phase in phases} for agent in agents}
    
    for results in all_results:
        for agent in agents:
            for i, phase in enumerate(phases, 1):
                start = phase_starts[i-1]
                end = phase_ends[i-1]
                
                # 全阶段平均
                full_avg = np.mean(results[agent]['rewards'][start:end])
                phase_results[agent][phase]['full'].append(full_avg)
                
                # 前5个episode平均（恢复速度）
                first5_avg = np.mean(results[agent]['rewards'][start:start+5])
                phase_results[agent][phase]['first5'].append(first5_avg)
                
                # 后5个episode平均（稳定性能）
                last5_avg = np.mean(results[agent]['rewards'][end-5:end])
                phase_results[agent][phase]['last5'].append(last5_avg)
    
    # 打印结果
    for i, phase in enumerate(phases, 1):
        for agent in agents:
            full_mean = np.mean(phase_results[agent][phase]['full'])
            full_std = np.std(phase_results[agent][phase]['full'])
            first5_mean = np.mean(phase_results[agent][phase]['first5'])
            first5_std = np.std(phase_results[agent][phase]['first5'])
            last5_mean = np.mean(phase_results[agent][phase]['last5'])
            last5_std = np.std(phase_results[agent][phase]['last5'])
            
            print(f"{phase:<10} {agent_names[agent]:<20} {full_mean:5.2f}±{full_std:3.2f} {first5_mean:5.2f}±{first5_std:3.2f} {last5_mean:5.2f}±{last5_std:3.2f}")
        print()
    
    # 计算恢复速度差异
    print("Recovery Speed Analysis:")
    print("-"*80)
    
    # Phase 2（环境第一次改变）
    sdas_phase2_first5 = np.mean([r['sdas']['rewards'][20:25] for r in all_results])
    tabular_phase2_first5 = np.mean([r['tabular']['rewards'][20:25] for r in all_results])
    tabular_reset_phase2_first5 = np.mean([r['tabular_reset']['rewards'][20:25] for r in all_results])
    
    recovery_diff_phase2 = sdas_phase2_first5 - tabular_phase2_first5
    recovery_diff_phase2_reset = sdas_phase2_first5 - tabular_reset_phase2_first5
    
    # Phase 3（环境第二次改变）
    sdas_phase3_first5 = np.mean([r['sdas']['rewards'][40:45] for r in all_results])
    tabular_phase3_first5 = np.mean([r['tabular']['rewards'][40:45] for r in all_results])
    tabular_reset_phase3_first5 = np.mean([r['tabular_reset']['rewards'][40:45] for r in all_results])
    
    recovery_diff_phase3 = sdas_phase3_first5 - tabular_phase3_first5
    recovery_diff_phase3_reset = sdas_phase3_first5 - tabular_reset_phase3_first5
    
    print(f"Phase 2 (First Change) Recovery Difference:")
    print(f"  SDAS - TabularQ: {recovery_diff_phase2:5.2f}")
    print(f"  SDAS - TabularQ (Reset): {recovery_diff_phase2_reset:5.2f}")
    print()
    print(f"Phase 3 (Second Change) Recovery Difference:")
    print(f"  SDAS - TabularQ: {recovery_diff_phase3:5.2f}")
    print(f"  SDAS - TabularQ (Reset): {recovery_diff_phase3_reset:5.2f}")
    print()
    
    # 结构池分析（只显示第一个种子）
    if structure_stats:
        print("Structure Pool Analysis (Seed 0):")
        print("-"*80)
        for stat in structure_stats:
            print(f"Episode {stat['episode']} (Phase {stat['phase']}): {stat['structure_count']} structures")
            # 打印前3个结构的效用
            if stat['structures']:
                top_structures = sorted(stat['structures'], key=lambda x: x['utility'], reverse=True)[:3]
                for s in top_structures:
                    print(f"  - Structure {s['id']}: utility={s['utility']:.3f}, label={s['label']}")
    
    # 绘制结果
    plot_results(all_results)

def plot_results(all_results):
    """绘制实验结果"""
    try:
        import matplotlib.pyplot as plt
        
        # 计算平均值
        avg_results = {
            'sdas': np.mean([r['sdas']['rewards'] for r in all_results], axis=0),
            'tabular': np.mean([r['tabular']['rewards'] for r in all_results], axis=0),
            'tabular_reset': np.mean([r['tabular_reset']['rewards'] for r in all_results], axis=0)
        }
        
        # 计算标准差
        std_results = {
            'sdas': np.std([r['sdas']['rewards'] for r in all_results], axis=0),
            'tabular': np.std([r['tabular']['rewards'] for r in all_results], axis=0),
            'tabular_reset': np.std([r['tabular_reset']['rewards'] for r in all_results], axis=0)
        }
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 学习曲线
        episodes = np.arange(1, 61)
        ax1.plot(episodes, avg_results['sdas'], label='SDAS', linewidth=2)
        ax1.fill_between(episodes, avg_results['sdas'] - std_results['sdas'], 
                         avg_results['sdas'] + std_results['sdas'], alpha=0.2)
        
        ax1.plot(episodes, avg_results['tabular'], label='TabularQ', linewidth=2)
        ax1.fill_between(episodes, avg_results['tabular'] - std_results['tabular'], 
                         avg_results['tabular'] + std_results['tabular'], alpha=0.2)
        
        ax1.plot(episodes, avg_results['tabular_reset'], label='TabularQ (Reset)', linewidth=2, linestyle='--')
        ax1.fill_between(episodes, avg_results['tabular_reset'] - std_results['tabular_reset'], 
                         avg_results['tabular_reset'] + std_results['tabular_reset'], alpha=0.2)
        
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
        tabular_reset_phase_avgs = []
        sdas_first5_avgs = []
        tabular_first5_avgs = []
        tabular_reset_first5_avgs = []
        
        phase_starts = [0, 20, 40]
        phase_ends = [20, 40, 60]
        
        for start, end in zip(phase_starts, phase_ends):
            sdas_phase_avgs.append(np.mean(avg_results['sdas'][start:end]))
            tabular_phase_avgs.append(np.mean(avg_results['tabular'][start:end]))
            tabular_reset_phase_avgs.append(np.mean(avg_results['tabular_reset'][start:end]))
            
            sdas_first5_avgs.append(np.mean(avg_results['sdas'][start:start+5]))
            tabular_first5_avgs.append(np.mean(avg_results['tabular'][start:start+5]))
            tabular_reset_first5_avgs.append(np.mean(avg_results['tabular_reset'][start:start+5]))
        
        x = np.arange(len(phases))
        width = 0.25
        
        ax2.bar(x - width, sdas_first5_avgs, width, label='SDAS (First 5)', alpha=0.7)
        ax2.bar(x, tabular_first5_avgs, width, label='TabularQ (First 5)', alpha=0.7)
        ax2.bar(x + width, tabular_reset_first5_avgs, width, label='TabularQ Reset (First 5)', alpha=0.7)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(phase_names, rotation=15, ha='right')
        ax2.set_ylabel('Average Reward (First 5 Episodes)')
        ax2.set_title('Recovery Speed Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('cross_episode_transfer_fixed.png', dpi=150, bbox_inches='tight')
        print(f"\nResults saved to cross_episode_transfer_fixed.png")
        plt.close()
        
    except ImportError:
        print("matplotlib not available, skipping plot")

def main():
    """主函数"""
    print("="*80)
    print("Cross-Episode Structure Transfer Experiment (Fixed)")
    print("="*80)
    print("\nExperimental Design:")
    print("  Phase 1 (Ep 1-20): Normal maze (baseline)")
    print("  Phase 2 (Ep 21-40): Maze layout changed (obstacles reset)")
    print("  Phase 3 (Ep 41-60): Maze layout changed again")
    print()
    print("Fixed Issues:")
    print("  1. Both agents run on the same map (shared random seed)")
    print("  2. Multiple seeds for statistical significance")
    print("  3. Added TabularQ reset control group")
    print()
    
    N_SEEDS = 5  # 运行5个种子（可以根据需要增加）
    all_results = []
    structure_stats = []
    
    for seed in range(N_SEEDS):
        results, stats = run_experiment(seed)
        all_results.append(results)
        if seed == 0:
            structure_stats = stats
    
    # 分析结果
    analyze_results(all_results, structure_stats)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    
    return all_results


if __name__ == "__main__":
    main()
