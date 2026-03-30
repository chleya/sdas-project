"""
SDAS Visualization Module
可视化分析模块 - 理解结构池的工作机制

功能：
1. 结构池状态可视化（结构关系图、激活热力图）
2. 训练过程曲线（适应度、参数变化）
3. 智能体行为轨迹可视化
4. 结构相似度网络图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from typing import List, Dict, Tuple, Optional
import os


class StructurePoolVisualizer:
    """结构池可视化器"""
    
    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

class AdaptiveStructureVisualizer(StructurePoolVisualizer):
    """自适应结构可视化器"""
    
    def visualize_adaptive_dynamics(self, structure_pool, episode: int = 0, 
                                  save: bool = True, show: bool = False):
        """
        可视化自适应结构池的动态变化
        包括：结构数量变化、阈值调整、策略决策
        """
        if not hasattr(structure_pool, 'adaptive_strategy'):
            print("Not an adaptive structure pool")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Adaptive Structure Dynamics - Episode {episode}', fontsize=16, fontweight='bold')
        
        # 1. 结构数量和阈值变化
        ax1 = axes[0, 0]
        
        # 模拟结构数量变化（实际应该从历史记录中获取）
        steps = range(100)
        structure_counts = [min(16, max(4, 8 + int(5 * np.sin(t/10)))) for t in steps]
        
        # 模拟阈值变化
        thresholds = [0.3 + 0.2 * np.sin(t/15) for t in steps]
        
        ax1.plot(steps, structure_counts, label='Structure Count', linewidth=2, color='blue')
        ax1.set_xlabel('Step', fontsize=12)
        ax1.set_ylabel('Structure Count', fontsize=12, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(steps, thresholds, label='Threshold', linewidth=2, color='red')
        ax1_twin.set_ylabel('Threshold', fontsize=12, color='red')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        
        ax1.set_title('Structure Count & Threshold Dynamics', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 2. 结构统计（包括效用和年龄）
        ax2 = axes[0, 1]
        structures = structure_pool.structures
        if structures:
            utilities = [s.utility for s in structures]
            ages = [s.age for s in structures]
            vigors = [s.vigor() for s in structures]
            
            ax2.bar(range(len(structures)), utilities, label='Utility', alpha=0.7, color='skyblue')
            ax2.set_xlabel('Structure ID', fontsize=12)
            ax2.set_ylabel('Utility', fontsize=12)
            ax2.set_title('Structure Utilities', fontsize=12)
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 自适应策略决策空间
        ax3 = axes[1, 0]
        
        # 模拟决策空间
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)
        
        # 模拟决策函数
        Z = np.exp(-((X-0.5)**2 + (Y-0.5)**2)/0.1)
        
        im = ax3.imshow(Z, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
        ax3.set_xlabel('Trend', fontsize=12)
        ax3.set_ylabel('Current Utilization', fontsize=12)
        ax3.set_title('Adaptive Decision Space', fontsize=12)
        plt.colorbar(im, ax=ax3, label='Action Strength')
        
        # 4. 结构生命周期
        ax4 = axes[1, 1]
        if structures:
            creation_order = sorted(structures, key=lambda s: s.id)
            y_pos = np.arange(len(creation_order))
            
            colors = plt.cm.RdYlGn([s.utility for s in creation_order])
            
            ax4.barh(y_pos, [s.age for s in creation_order], 
                    color=colors, alpha=0.7, edgecolor='black')
            
            for i, s in enumerate(creation_order):
                ax4.text(s.age + 1, i, f"{s.label[:15]}", 
                        va='center', fontsize=8)
            
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels([f"S{s.id}" for s in creation_order])
            ax4.set_xlabel('Age (steps)', fontsize=12)
            ax4.set_title('Structure Lifetimes', fontsize=12)
            ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.save_dir, f'adaptive_dynamics_ep{episode}.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_low_rank_decomposition(self, structure, episode: int = 0, 
                                       save: bool = True, show: bool = False):
        """
        可视化低秩结构的分解
        """
        if not hasattr(structure, 'B') or not hasattr(structure, 'a'):
            print("Not a low-rank structure")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Low-Rank Decomposition - Structure {structure.id}', fontsize=16, fontweight='bold')
        
        # 1. 基向量
        ax1 = axes[0]
        base_vector = getattr(structure, 'base_vector', np.zeros_like(structure.a))
        ax1.bar(range(len(base_vector)), base_vector)
        ax1.set_xlabel('Dimension', fontsize=12)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.set_title('Base Vector', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. 分解矩阵 B
        ax2 = axes[1]
        im = ax2.imshow(structure.B, cmap='viridis')
        ax2.set_xlabel('Output Dimension', fontsize=12)
        ax2.set_ylabel('Rank Dimension', fontsize=12)
        ax2.set_title('Decomposition Matrix B', fontsize=12)
        plt.colorbar(im, ax=ax2)
        
        # 3. 分解向量 a
        ax3 = axes[2]
        ax3.bar(range(len(structure.a)), structure.a)
        ax3.set_xlabel('Rank Dimension', fontsize=12)
        ax3.set_ylabel('Value', fontsize=12)
        ax3.set_title('Decomposition Vector a', fontsize=12)
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.save_dir, f'low_rank_decomp_s{structure.id}_ep{episode}.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_structure_transfer(self, source_pool, target_pool, 
                                   transferred_structures: list, 
                                   episode: int = 0, save: bool = True, show: bool = False):
        """
        可视化结构迁移过程
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Structure Transfer - Episode {episode}', fontsize=16, fontweight='bold')
        
        # 1. 源结构池
        ax1 = axes[0]
        source_structures = source_pool.structures
        if source_structures:
            ages = [s.age for s in source_structures]
            utilities = [s.utility for s in source_structures]
            
            scatter = ax1.scatter(ages, utilities, c=[s.vigor() for s in source_structures], 
                                s=200, cmap='RdYlGn', alpha=0.7, edgecolors='black')
            
            # 标记被迁移的结构
            transferred_ids = {s.id for s in transferred_structures}
            for s in source_structures:
                if s.id in transferred_ids:
                    ax1.scatter(s.age, s.utility, s=300, edgecolor='red', 
                               facecolor='none', linewidth=2)
            
            ax1.set_xlabel('Age', fontsize=12)
            ax1.set_ylabel('Utility', fontsize=12)
            ax1.set_title('Source Structure Pool', fontsize=12)
            ax1.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax1, label='Vigor')
        
        # 2. 目标结构池
        ax2 = axes[1]
        target_structures = target_pool.structures
        if target_structures:
            ages = [s.age for s in target_structures]
            utilities = [s.utility for s in target_structures]
            
            # 区分迁移的结构和原有结构
            transferred_ids = {s.id for s in transferred_structures}
            colors = []
            sizes = []
            for s in target_structures:
                if s.id in transferred_ids:
                    colors.append('red')
                    sizes.append(250)
                else:
                    colors.append('blue')
                    sizes.append(150)
            
            ax2.scatter(ages, utilities, c=colors, s=sizes, alpha=0.7, edgecolors='black')
            
            ax2.set_xlabel('Age', fontsize=12)
            ax2.set_ylabel('Utility', fontsize=12)
            ax2.set_title('Target Structure Pool', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', edgecolor='black', label='Transferred'),
                Patch(facecolor='blue', edgecolor='black', label='Original')
            ]
            ax2.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.save_dir, f'structure_transfer_ep{episode}.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_multi_environment_comparison(self, environments: dict, 
                                             structure_pools: dict, 
                                             episode: int = 0, save: bool = True, show: bool = False):
        """
        可视化多环境比较
        """
        n_envs = len(environments)
        fig, axes = plt.subplots(1, n_envs, figsize=(5*n_envs, 6))
        fig.suptitle(f'Multi-Environment Comparison - Episode {episode}', fontsize=16, fontweight='bold')
        
        for i, (env_name, env) in enumerate(environments.items()):
            ax = axes[i]
            pool = structure_pools[env_name]
            structures = pool.structures
            
            if structures:
                utilities = [s.utility for s in structures]
                ages = [s.age for s in structures]
                vigors = [s.vigor() for s in structures]
                
                scatter = ax.scatter(ages, utilities, c=vigors, 
                                   s=150, cmap='RdYlGn', alpha=0.7, edgecolors='black')
                
                ax.set_xlabel('Age', fontsize=12)
                ax.set_ylabel('Utility', fontsize=12)
                ax.set_title(f'{env_name}\n({len(structures)} structures)', fontsize=12)
                ax.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax, label='Vigor')
            else:
                ax.set_title(f'{env_name}\n(No structures)', fontsize=12)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.save_dir, f'multi_env_comparison_ep{episode}.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_structure_pool(self, structure_pool, episode: int = 0, 
                                 save: bool = True, show: bool = False):
        """
        可视化结构池状态
        包括：结构活力、年龄、效用值的关系
        """
        structures = structure_pool.structures
        if len(structures) == 0:
            print("No structures to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Structure Pool State - Episode {episode}', fontsize=16, fontweight='bold')
        
        # 1. 结构活力 vs 年龄散点图
        ax1 = axes[0, 0]
        ages = [s.age for s in structures]
        utilities = [s.utility for s in structures]
        vigors = [s.vigor() for s in structures]
        labels = [s.label[:10] for s in structures]
        
        scatter = ax1.scatter(ages, utilities, c=vigors, s=200, 
                             cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=1)
        for i, label in enumerate(labels):
            ax1.annotate(label, (ages[i], utilities[i]), 
                        fontsize=8, ha='center', va='bottom')
        
        ax1.set_xlabel('Age (steps)', fontsize=12)
        ax1.set_ylabel('Utility', fontsize=12)
        ax1.set_title('Structure Age vs Utility (color = vigor)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Vigor')
        
        # 2. 结构相似度热力图
        ax2 = axes[0, 1]
        n = len(structures)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if hasattr(structures[i], 'get_prototype'):
                    # 低秩结构池
                    p1 = structures[i].get_prototype()
                    p2 = structures[j].get_prototype()
                else:
                    # 标准结构池
                    p1 = structures[i].prototype
                    p2 = structures[j].prototype
                
                similarity_matrix[i, j] = self._cosine_similarity(p1, p2)
        
        im = ax2.imshow(similarity_matrix, cmap='coolwarm', vmin=0, vmax=1)
        ax2.set_xticks(range(n))
        ax2.set_yticks(range(n))
        ax2.set_xticklabels([f"S{i}" for i in range(n)], fontsize=8)
        ax2.set_yticklabels([f"S{i}" for i in range(n)], fontsize=8)
        ax2.set_title('Structure Similarity Matrix', fontsize=12)
        plt.colorbar(im, ax=ax2, label='Cosine Similarity')
        
        # 3. 结构统计柱状图
        ax3 = axes[1, 0]
        x_pos = np.arange(len(structures))
        width = 0.25
        
        bars1 = ax3.bar(x_pos - width, [s.utility for s in structures], 
                       width, label='Utility', alpha=0.8, color='skyblue')
        bars2 = ax3.bar(x_pos, [s.vigor() for s in structures], 
                       width, label='Vigor', alpha=0.8, color='lightgreen')
        bars3 = ax3.bar(x_pos + width, 
                       [s.age / max([s.age for s in structures] + [1]) for s in structures], 
                       width, label='Age (normalized)', alpha=0.8, color='salmon')
        
        ax3.set_xlabel('Structure ID', fontsize=12)
        ax3.set_ylabel('Value', fontsize=12)
        ax3.set_title('Structure Statistics', fontsize=12)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f"{s.id}\n{s.label[:8]}" for s in structures], fontsize=8)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 结构生命周期时间线
        ax4 = axes[1, 1]
        creation_order = sorted(structures, key=lambda s: s.id)
        y_pos = np.arange(len(creation_order))
        
        colors = plt.cm.RdYlGn([s.utility for s in creation_order])
        
        ax4.barh(y_pos, [s.age for s in creation_order], 
                color=colors, alpha=0.7, edgecolor='black')
        
        for i, s in enumerate(creation_order):
            ax4.text(s.age + 1, i, f"{s.label[:15]} (U:{s.utility:.2f})", 
                    va='center', fontsize=8)
        
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([f"S{s.id}" for s in creation_order])
        ax4.set_xlabel('Age (steps)', fontsize=12)
        ax4.set_title('Structure Lifetime Timeline', fontsize=12)
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.save_dir, f'structure_pool_ep{episode}.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_training_history(self, fitness_history: List[Dict], 
                                   save: bool = True, show: bool = False):
        """
        可视化训练历史
        """
        if not fitness_history:
            print("No training history to visualize")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('ES Training History', fontsize=16, fontweight='bold')
        
        generations = range(len(fitness_history))
        means = [h['mean'] for h in fitness_history]
        maxs = [h['max'] for h in fitness_history]
        mins = [h['min'] for h in fitness_history]
        
        # 1. 适应度曲线
        ax1 = axes[0]
        ax1.plot(generations, means, label='Mean Fitness', linewidth=2, color='blue')
        ax1.plot(generations, maxs, label='Max Fitness', linewidth=2, color='green')
        ax1.fill_between(generations, mins, maxs, alpha=0.2, color='gray', label='Range')
        
        ax1.set_xlabel('Generation', fontsize=12)
        ax1.set_ylabel('Fitness', fontsize=12)
        ax1.set_title('Fitness Over Generations', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 收敛分析
        ax2 = axes[1]
        window = min(5, len(means) // 5 + 1)
        if window > 1:
            moving_avg = np.convolve(means, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(means)), moving_avg, 
                    label=f'Moving Average (window={window})', linewidth=2, color='red')
        
        # 标注最佳代数
        best_gen = np.argmax(maxs)
        ax2.axvline(best_gen, color='green', linestyle='--', alpha=0.5, 
                   label=f'Best Gen: {best_gen}')
        ax2.scatter([best_gen], [maxs[best_gen]], color='green', s=100, zorder=5)
        
        ax2.set_xlabel('Generation', fontsize=12)
        ax2.set_ylabel('Fitness', fontsize=12)
        ax2.set_title('Convergence Analysis', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.save_dir, 'training_history.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_agent_trajectory(self, env, agent, max_steps: int = 100,
                                   save: bool = True, show: bool = False):
        """
        可视化智能体在环境中的轨迹
        """
        from digital_petri_dish import DigitalPetriDish
        
        obs = env._reset()
        agent.reset()
        
        trajectory = [(env.agent_pos.x, env.agent_pos.y)]
        rewards = []
        structure_events = []
        
        for step in range(max_steps):
            action, info = agent.step(obs)
            obs, reward, done = env.step(action)
            
            trajectory.append((env.agent_pos.x, env.agent_pos.y))
            rewards.append(reward)
            structure_events.append(info.get('structure_event', 'unknown'))
            
            if done:
                break
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Agent Trajectory and Behavior', fontsize=16, fontweight='bold')
        
        # 1. 环境地图和轨迹
        ax1 = axes[0]
        
        # 绘制网格
        for x in range(env.width + 1):
            ax1.axvline(x, color='gray', alpha=0.3, linewidth=0.5)
        for y in range(env.height + 1):
            ax1.axhline(y, color='gray', alpha=0.3, linewidth=0.5)
        
        # 绘制障碍物
        for obs_pos in env.obstacles:
            rect = plt.Rectangle((obs_pos.x, obs_pos.y), 1, 1, 
                                facecolor='black', edgecolor='black', alpha=0.7)
            ax1.add_patch(rect)
        
        # 绘制能量
        for energy_pos in env.energy_positions:
            circle = Circle((energy_pos.x + 0.5, energy_pos.y + 0.5), 0.3, 
                          facecolor='gold', edgecolor='orange', linewidth=2)
            ax1.add_patch(circle)
        
        # 绘制轨迹
        traj_x = [p[0] + 0.5 for p in trajectory]
        traj_y = [p[1] + 0.5 for p in trajectory]
        
        # 使用渐变色表示时间
        points = np.array([traj_x, traj_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, cmap='viridis', linewidths=2)
        lc.set_array(np.linspace(0, 1, len(segments)))
        ax1.add_collection(lc)
        
        # 标记起点和终点
        ax1.scatter([traj_x[0]], [traj_y[0]], color='green', s=200, 
                   marker='o', label='Start', zorder=5)
        ax1.scatter([traj_x[-1]], [traj_y[-1]], color='red', s=200, 
                   marker='X', label='End', zorder=5)
        
        ax1.set_xlim(0, env.width)
        ax1.set_ylim(0, env.height)
        ax1.set_aspect('equal')
        ax1.set_xlabel('X', fontsize=12)
        ax1.set_ylabel('Y', fontsize=12)
        ax1.set_title(f'Trajectory (Steps: {len(trajectory)-1})', fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. 累积奖励和结构事件
        ax2 = axes[1]
        
        cumsum_rewards = np.cumsum(rewards)
        ax2.plot(range(len(cumsum_rewards)), cumsum_rewards, 
                linewidth=2, color='blue', label='Cumulative Reward')
        ax2.set_xlabel('Step', fontsize=12)
        ax2.set_ylabel('Cumulative Reward', fontsize=12, color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        # 在顶部显示结构事件
        event_types = list(set(structure_events))
        event_colors = {event: plt.cm.Set3(i / len(event_types)) 
                       for i, event in enumerate(event_types)}
        
        for i, event in enumerate(structure_events):
            if i < len(cumsum_rewards):
                ax2.axvline(i, color=event_colors[event], alpha=0.3, linewidth=1)
        
        # 添加图例
        legend_patches = [mpatches.Patch(color=color, label=event) 
                         for event, color in event_colors.items()]
        ax2.legend(handles=legend_patches, loc='upper left', fontsize=8)
        
        ax2.set_title('Reward Accumulation & Structure Events', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.save_dir, 'agent_trajectory.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_structure_network(self, structure_pool, 
                                   save: bool = True, show: bool = False):
        """
        可视化结构相似度网络图
        """
        structures = structure_pool.structures
        if len(structures) < 2:
            print("Need at least 2 structures for network visualization")
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 计算结构位置（使用相似度进行降维布局）
        n = len(structures)
        
        # 计算相似度矩阵
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if hasattr(structures[i], 'get_prototype'):
                    p1 = structures[i].get_prototype()
                    p2 = structures[j].get_prototype()
                else:
                    p1 = structures[i].prototype
                    p2 = structures[j].prototype
                similarity_matrix[i, j] = self._cosine_similarity(p1, p2)
        
        # 使用简单的圆形布局
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        positions = [(np.cos(a), np.sin(a)) for a in angles]
        
        # 绘制边（相似度连接）
        for i in range(n):
            for j in range(i+1, n):
                sim = similarity_matrix[i, j]
                if sim > 0.5:  # 只显示相似度高的连接
                    x1, y1 = positions[i]
                    x2, y2 = positions[j]
                    ax.plot([x1, x2], [y1, y2], 'gray', 
                           alpha=sim*0.5, linewidth=sim*3)
        
        # 绘制节点
        for i, (x, y) in enumerate(positions):
            s = structures[i]
            size = 1000 + s.utility * 1000
            color = plt.cm.RdYlGn(s.vigor())
            
            circle = Circle((x, y), 0.15, facecolor=color, 
                          edgecolor='black', linewidth=2, zorder=5)
            ax.add_patch(circle)
            
            # 标签
            ax.text(x, y, f"S{s.id}", ha='center', va='center', 
                   fontsize=10, fontweight='bold', zorder=6)
            ax.text(x, y-0.25, s.label[:12], ha='center', va='top', 
                   fontsize=8, zorder=6)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Structure Similarity Network\n(node size = utility, color = vigor, line = similarity)', 
                    fontsize=14, fontweight='bold')
        
        # 添加图例
        legend_elements = [
            mpatches.Patch(color=plt.cm.RdYlGn(1.0), label='High Vigor'),
            mpatches.Patch(color=plt.cm.RdYlGn(0.5), label='Medium Vigor'),
            mpatches.Patch(color=plt.cm.RdYlGn(0.0), label='Low Vigor')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        if save:
            filepath = os.path.join(self.save_dir, 'structure_network.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        a = a.flatten()
        b = b.flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)


def demo_visualization():
    """
    演示可视化功能
    """
    print("="*70)
    print("SDAS Visualization Demo")
    print("="*70)
    
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
    
    from sdas import SDASAgent, Config
    from digital_petri_dish import DigitalPetriDish
    
    # 创建可视化器
    viz = StructurePoolVisualizer(save_dir="visualizations")
    
    # 创建环境和智能体
    print("\n1. Creating environment and agent...")
    env = DigitalPetriDish(width=15, height=15, n_obstacles=20, n_energy=5)
    agent = SDASAgent(Config())
    
    # 运行一个回合
    print("2. Running episode...")
    obs = env._reset()
    agent.reset()
    
    for step in range(50):
        action, info = agent.step(obs)
        obs, reward, done = env.step(action)
        if done:
            break
    
    print(f"   Episode completed: {step+1} steps")
    print(f"   Total structures: {len(agent.structure_pool.structures)}")
    
    # 1. 可视化结构池状态
    print("\n3. Generating structure pool visualization...")
    viz.visualize_structure_pool(agent.structure_pool, episode=0, save=True, show=False)
    
    # 2. 可视化结构网络
    print("4. Generating structure network visualization...")
    viz.visualize_structure_network(agent.structure_pool, save=True, show=False)
    
    # 3. 可视化智能体轨迹
    print("5. Generating trajectory visualization...")
    viz.visualize_agent_trajectory(env, agent, max_steps=50, save=True, show=False)
    
    # 模拟训练历史
    print("6. Generating training history visualization...")
    fake_history = [
        {'mean': -2.0 + i*0.05, 'max': -1.5 + i*0.08, 'min': -2.5 + i*0.03}
        for i in range(20)
    ]
    viz.visualize_training_history(fake_history, save=True, show=False)
    
    print("\n" + "="*70)
    print("All visualizations saved to 'visualizations/' directory")
    print("="*70)
    print("\nGenerated files:")
    for f in os.listdir("visualizations"):
        print(f"  - {f}")


if __name__ == "__main__":
    demo_visualization()
