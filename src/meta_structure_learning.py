#!/usr/bin/env python3
"""
Meta Structure Learning - 元结构学习系统

让SDAS能够学习如何更好地组织结构，优化结构演化策略
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import pickle
import os
import random


@dataclass
class StructureMemory:
    """结构记忆单元"""
    structure_id: int
    prototype: np.ndarray
    utility_history: List[float]
    context: Dict
    timestamp: int
    performance: float  # 结构在环境中的表现


class MetaLearningSystem:
    """元学习系统"""
    
    def __init__(self, memory_capacity: int = 1000):
        self.memory_capacity = memory_capacity
        self.memory = []
        self.memory_index = 0
        self.learning_rate = 0.1
        self.experience_buffer = []
        self.evolution_strategy = self._default_evolution_strategy()
    
    def _default_evolution_strategy(self) -> Dict:
        """默认演化策略"""
        return {
            'selection_rate': 0.2,  # 选择前20%的结构
            'mutation_rate': 0.3,  # 30%的变异概率
            'crossover_rate': 0.5,  # 50%的交叉概率
            'mutation_strength': 0.1,  # 变异强度
            'diversity_weight': 0.3,  # 多样性权重
            'utility_weight': 0.7,  # 效用权重
            'age_penalty': 0.1  # 年龄惩罚
        }
    
    def store_structure(self, structure, context: Dict, performance: float, timestamp: int):
        """存储结构到记忆"""
        try:
            # 确保结构有id属性
            if not hasattr(structure, 'id'):
                structure.id = getattr(structure, 'structure_id', id(structure))
            
            # 确保结构有prototype或get_prototype方法
            if hasattr(structure, 'prototype'):
                prototype = structure.prototype.copy()
            elif hasattr(structure, 'get_prototype'):
                prototype = structure.get_prototype().copy()
            else:
                # 如果没有原型，使用结构的哈希值作为特征
                prototype = np.array([hash(str(structure)) % 1000])
            
            # 确保结构有效用属性
            utility = getattr(structure, 'utility', 0.0)
            
            memory_item = StructureMemory(
                structure_id=structure.id,
                prototype=prototype,
                utility_history=[utility],
                context=context,
                timestamp=timestamp,
                performance=performance
            )
            
            if len(self.memory) >= self.memory_capacity:
                self.memory[self.memory_index] = memory_item
                self.memory_index = (self.memory_index + 1) % self.memory_capacity
            else:
                self.memory.append(memory_item)
            
            print(f"Stored structure {structure.id} with utility {utility} to memory")
        except Exception as e:
            print(f"Error storing structure: {e}")
    
    def retrieve_similar_structures(self, current_context: Dict, k: int = 5) -> List[StructureMemory]:
        """检索相似结构"""
        if not self.memory:
            return []
        
        similarities = []
        for item in self.memory:
            try:
                similarity = self._context_similarity(item.context, current_context)
                # 使用structure_id作为唯一排序键，确保不会比较StructureMemory对象
                similarities.append((-similarity, item.timestamp, item.structure_id, item))  # 使用负相似度以便按升序排序
            except:
                continue
        
        if similarities:
            similarities.sort()
            return [item for _, _, _, item in similarities[:k]]
        else:
            return []
    
    def _context_similarity(self, context1: Dict, context2: Dict) -> float:
        """计算上下文相似度"""
        keys = set(context1.keys()) & set(context2.keys())
        if not keys:
            return 0.0
        
        similarity = 0.0
        for key in keys:
            if isinstance(context1[key], (int, float)):
                # 数值型特征相似度
                max_val = max(abs(context1[key]), abs(context2[key]), 1.0)
                similarity += 1.0 - abs(context1[key] - context2[key]) / max_val
            elif isinstance(context1[key], str):
                # 字符串型特征相似度
                similarity += 1.0 if context1[key] == context2[key] else 0.0
            elif isinstance(context1[key], bool):
                # 布尔型特征相似度
                similarity += 1.0 if context1[key] == context2[key] else 0.0
        
        return similarity / len(keys)
    
    def learn_from_experience(self, experience: Dict):
        """从经验中学习"""
        self.experience_buffer.append(experience)
        
        # 当经验积累到一定程度时，更新演化策略
        if len(self.experience_buffer) >= 20:  # 减少经验缓冲区大小，更快更新策略
            self._update_evolution_strategy()
            self.experience_buffer = []
    
    def _update_evolution_strategy(self):
        """更新演化策略"""
        # 分析经验数据
        if not self.experience_buffer:
            return
        
        success_rate = np.mean([exp['success'] for exp in self.experience_buffer])
        avg_reward = np.mean([exp['reward'] for exp in self.experience_buffer])
        
        print(f"Updating evolution strategy: success_rate={success_rate:.2f}, avg_reward={avg_reward:.2f}")
        
        # 简化的策略调整
        if avg_reward < -1.0:
            # 奖励很差，增加探索
            self.evolution_strategy['mutation_rate'] = min(0.5, self.evolution_strategy['mutation_rate'] + 0.05)
            self.evolution_strategy['diversity_weight'] = min(0.6, self.evolution_strategy['diversity_weight'] + 0.05)
        elif avg_reward > 1.0:
            # 奖励很好，增加利用
            self.evolution_strategy['mutation_rate'] = max(0.1, self.evolution_strategy['mutation_rate'] - 0.03)
            self.evolution_strategy['utility_weight'] = min(0.9, self.evolution_strategy['utility_weight'] + 0.05)
        else:
            # 中等奖励，平衡探索和利用
            self.evolution_strategy['mutation_rate'] = max(0.15, min(0.4, self.evolution_strategy['mutation_rate'] + 0.02))
        
        print(f"Updated strategy: {self.evolution_strategy}")
    
    def generate_evolution_plan(self, current_structures: List, current_context: Dict) -> Dict:
        """生成演化计划"""
        # 检索相似上下文的结构
        similar_structures = self.retrieve_similar_structures(current_context, k=10)
        
        # 分析相似结构的性能
        if similar_structures:
            avg_performance = np.mean([s.performance for s in similar_structures])
            best_performance = max([s.performance for s in similar_structures])
        else:
            avg_performance = 0
            best_performance = 0
        
        # 基于分析结果生成演化计划
        plan = {
            'selection_rate': self.evolution_strategy['selection_rate'],
            'mutation_rate': self.evolution_strategy['mutation_rate'],
            'crossover_rate': self.evolution_strategy['crossover_rate'],
            'mutation_strength': self.evolution_strategy['mutation_strength'],
            'target_diversity': max(0.3, min(0.8, 0.5 + (0.5 - avg_performance) * 0.3)),
            'focus_on_exploration': avg_performance < 0.3,
            'focus_on_exploitation': avg_performance > 0.7
        }
        
        return plan
    
    def evaluate_structure_candidates(self, candidates: List, context: Dict) -> List[Tuple[any, float]]:
        """评估结构候选者"""
        evaluations = []
        
        for candidate in candidates:
            # 计算候选结构与当前上下文的匹配度
            match_score = self._evaluate_structure_context_match(candidate, context)
            
            # 考虑结构的多样性贡献
            diversity_score = self._calculate_diversity_contribution(candidate, candidates)
            
            # 综合评分
            total_score = (
                self.evolution_strategy['utility_weight'] * match_score +
                self.evolution_strategy['diversity_weight'] * diversity_score
            )
            
            evaluations.append((candidate, total_score))
        
        evaluations.sort(key=lambda x: x[1], reverse=True)
        return evaluations
    
    def _evaluate_structure_context_match(self, structure, context: Dict) -> float:
        """评估结构与上下文的匹配度"""
        try:
            # 基于结构的效用和性能计算匹配度
            utility = getattr(structure, 'utility', 0.0)
            
            # 简化的匹配度计算 - 主要基于效用
            match_score = utility
            
            # 考虑结构网络信息（如果可用）
            if hasattr(structure, 'network_centrality'):
                # 网络中心度高的结构更有价值
                match_score *= (1.0 + structure.network_centrality * 0.3)
            
            return min(1.0, max(0.0, match_score))
        except Exception as e:
            print(f"Error evaluating structure: {e}")
            return 0.5  # 默认为中等匹配度
    
    def _calculate_diversity_contribution(self, structure, candidates: List) -> float:
        """计算结构的多样性贡献"""
        if not candidates:
            return 1.0
        
        # 计算与其他候选结构的平均距离
        distances = []
        for other in candidates:
            if other != structure:
                try:
                    if hasattr(structure, 'prototype') and hasattr(other, 'prototype'):
                        p1 = structure.prototype
                        p2 = other.prototype
                    else:
                        p1 = structure.get_prototype()
                        p2 = other.get_prototype()
                    
                    distance = np.linalg.norm(p1 - p2)
                    distances.append(distance)
                except:
                    pass
        
        if distances:
            return np.mean(distances)
        else:
            return 1.0
    
    def save(self, path: str):
        """保存元学习系统状态"""
        data = {
            'memory': self.memory,
            'memory_index': self.memory_index,
            'evolution_strategy': self.evolution_strategy,
            'experience_buffer': self.experience_buffer
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """加载元学习系统状态"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.memory = data['memory']
        self.memory_index = data['memory_index']
        self.evolution_strategy = data['evolution_strategy']
        self.experience_buffer = data['experience_buffer']


class MetaEnhancedSDASAgent:
    """元学习增强的SDAS智能体"""
    
    def __init__(self, config=None):
        # 延迟导入以避免循环依赖
        from enhanced_sdas import EnhancedSDASAgent
        self.base_agent = EnhancedSDASAgent(config)
        self.meta_learner = MetaLearningSystem()
        self.episode_count = 0
        self.step_count = 0
    
    def step(self, obs: dict) -> Tuple[int, dict]:
        """智能体一步"""
        # 获取当前上下文
        context = self._extract_context(obs)
        
        # 生成演化计划
        evolution_plan = self.meta_learner.generate_evolution_plan(
            self.base_agent.structure_pool.structures,
            context
        )
        
        # 应用演化计划到结构池
        try:
            # 调整结构池的演化参数
            if hasattr(self.base_agent.structure_pool, 'selection_rate'):
                self.base_agent.structure_pool.selection_rate = evolution_plan['selection_rate']
            if hasattr(self.base_agent.structure_pool, 'mutation_rate'):
                self.base_agent.structure_pool.mutation_rate = evolution_plan['mutation_rate']
            if hasattr(self.base_agent.structure_pool, 'crossover_rate'):
                self.base_agent.structure_pool.crossover_rate = evolution_plan['crossover_rate']
            if hasattr(self.base_agent.structure_pool, 'mutation_strength'):
                self.base_agent.structure_pool.mutation_strength = evolution_plan['mutation_strength']
        except Exception as e:
            print(f"Error applying evolution plan: {e}")
        
        # 评估当前结构池
        structure_evaluations = []
        try:
            structure_evaluations = self.meta_learner.evaluate_structure_candidates(
                self.base_agent.structure_pool.structures,
                context
            )
        except Exception as e:
            print(f"Error evaluating structures: {e}")
        
        # 基于元学习结果调整结构网络
        if structure_evaluations and hasattr(self.base_agent, 'structure_network'):
            try:
                # 为结构添加网络中心度属性
                for structure, score in structure_evaluations[:5]:  # 只处理前5个最高分的结构
                    if hasattr(self.base_agent.structure_network, 'get_structure_centrality'):
                        centrality = self.base_agent.structure_network.get_structure_centrality(structure.id)
                        structure.network_centrality = centrality
            except Exception as e:
                print(f"Error adjusting structure network: {e}")
        
        # 执行基础智能体的步骤
        action, info = self.base_agent.step(obs)
        
        # 记录信息
        self.step_count += 1
        
        # 添加元学习相关信息
        info['evolution_plan'] = evolution_plan
        info['meta_learning_active'] = True
        info['structure_evaluations'] = [(s.id, round(score, 3)) for s, score in structure_evaluations[:5]]
        info['meta_memory_size'] = len(self.meta_learner.memory)
        
        return action, info
    
    def update(self, reward: float, done: bool):
        """更新智能体"""
        # 更新基础智能体
        self.base_agent.update_structure(reward)
        
        # 当回合结束时，学习经验
        if done:
            self.episode_count += 1
            
            # 提取经验
            experience = self._extract_experience(reward)
            
            # 存储结构到元学习系统
            try:
                for structure in self.base_agent.structure_pool.structures:
                    context = self._extract_context({})
                    self.meta_learner.store_structure(
                        structure,
                        context,
                        reward,
                        self.episode_count
                    )
                print(f"Stored {len(self.base_agent.structure_pool.structures)} structures to meta memory")
            except Exception as e:
                print(f"Error storing structures: {e}")
            
            # 从经验中学习
            self.meta_learner.learn_from_experience(experience)
    
    def _extract_context(self, obs: dict) -> Dict:
        """提取上下文信息"""
        context = {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'n_structures': len(self.base_agent.structure_pool.structures),
            'env_type': 'complex_petri_dish'
        }
        
        # 从观测中提取上下文
        if obs:
            context.update({
                'obstacle_nearby': obs.get('obstacle_nearby', False),
                'energy_dist': obs.get('energy_dist', 1.0),
                'familiarity': obs.get('familiarity', 0.0)
            })
        
        return context
    
    def _extract_experience(self, reward: float) -> Dict:
        """提取经验"""
        # 计算结构池多样性
        diversity_score = self._calculate_structure_diversity()
        
        experience = {
            'reward': reward,
            'success': reward > 0,
            'n_structures': len(self.base_agent.structure_pool.structures),
            'diversity_score': diversity_score,
            'episode': self.episode_count
        }
        
        return experience
    
    def _calculate_structure_diversity(self) -> float:
        """计算结构池多样性"""
        structures = self.base_agent.structure_pool.structures
        if len(structures) < 2:
            return 0.0
        
        # 计算所有结构对之间的平均距离
        distances = []
        for i, s1 in enumerate(structures):
            for s2 in structures[i+1:]:
                try:
                    if hasattr(s1, 'prototype') and hasattr(s2, 'prototype'):
                        p1 = s1.prototype
                        p2 = s2.prototype
                    else:
                        p1 = s1.get_prototype()
                        p2 = s2.get_prototype()
                    
                    distance = np.linalg.norm(p1 - p2)
                    distances.append(distance)
                except:
                    pass
        
        if distances:
            return np.mean(distances) / np.max(distances) if np.max(distances) > 0 else 0.0
        else:
            return 0.0
    
    def visualize(self, filename: str = 'meta_learning_status.png'):
        """可视化元学习状态"""
        import matplotlib.pyplot as plt
        
        # 创建保存目录
        save_dir = 'visualizations'
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        
        # 1. 演化策略参数
        plt.subplot(2, 2, 1)
        strategy = self.meta_learner.evolution_strategy
        params = list(strategy.keys())
        values = list(strategy.values())
        plt.bar(params, values)
        plt.xticks(rotation=45, ha='right')
        plt.title('Evolution Strategy Parameters')
        plt.grid(True, alpha=0.3, axis='y')
        
        # 2. 记忆大小
        plt.subplot(2, 2, 2)
        memory_size = len(self.meta_learner.memory)
        buffer_size = len(self.meta_learner.experience_buffer)
        plt.bar(['Memory Size', 'Buffer Size'], [memory_size, buffer_size])
        plt.title('Meta Learning System State')
        plt.grid(True, alpha=0.3, axis='y')
        
        # 3. 结构多样性
        diversity = self._calculate_structure_diversity()
        plt.subplot(2, 2, 3)
        plt.bar(['Structure Diversity'], [diversity])
        plt.ylim(0, 1)
        plt.title('Current Structure Diversity')
        plt.grid(True, alpha=0.3, axis='y')
        
        # 4. 演化计划
        plt.subplot(2, 2, 4)
        context = self._extract_context({})
        plan = self.meta_learner.generate_evolution_plan(
            self.base_agent.structure_pool.structures,
            context
        )
        plan_items = list(plan.items())[:4]  # 只显示前4个项目
        plan_names = [item[0] for item in plan_items]
        plan_values = [item[1] for item in plan_items]
        plt.bar(plan_names, plan_values)
        plt.xticks(rotation=45, ha='right')
        plt.title('Current Evolution Plan')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved meta learning status visualization to {filepath}")
        plt.close()
    
    def save(self, path: str):
        """保存智能体状态"""
        self.base_agent.save(path)
        self.meta_learner.save(f"{path}_meta.pkl")
    
    def load(self, path: str):
        """加载智能体状态"""
        self.base_agent.load(path)
        try:
            self.meta_learner.load(f"{path}_meta.pkl")
        except:
            print("No meta learning data found, initializing new meta learner")


def run_episode_with_meta(env, agent, max_steps: int = 200) -> Dict:
    """运行一个元学习增强的回合"""
    obs = env._reset()
    agent.base_agent.reset()
    
    total_reward = 0
    structure_events = []
    meta_events = []
    
    for step in range(max_steps):
        # 智能体选择行动
        action, info = agent.step(obs)
        
        # 环境执行
        obs, reward, done = env.step(action)
        
        # 更新智能体
        agent.update(reward, done)
        
        total_reward += reward
        structure_events.append(info.get('structure_event', 'unknown'))
        meta_events.append(info.get('evolution_plan', {}))
        
        if done:
            break
    
    # 确保即使环境没有返回done=True，也会存储结构
    if not done:
        print("Episode reached max steps, forcing update with done=True")
        agent.update(total_reward, True)
    
    return {
        'steps': step + 1,
        'total_reward': total_reward,
        'structure_events': structure_events,
        'meta_events': meta_events,
        'final_state': agent.base_agent.get_state()
    }

def test_meta_enhanced_sdas():
    """测试元学习增强的SDAS智能体"""
    print("=" * 70)
    print("Testing Meta-Enhanced SDAS Agent")
    print("=" * 70)
    
    # 导入环境
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments'))
    from complex_petri_dish import ComplexPetriDish
    
    # 创建环境
    env = ComplexPetriDish(
        width=15,
        height=15,
        n_static_obstacles=20,
        n_dynamic_obstacles=3,
        n_energy_sources=5
    )
    
    # 创建元学习增强的智能体
    agent = MetaEnhancedSDASAgent()
    
    print("\nRunning 5 episodes...")
    
    # 运行几个回合
    rewards = []
    for ep in range(5):
        result = run_episode_with_meta(env, agent, max_steps=150)
        rewards.append(result['total_reward'])
        
        print(f"\nEpisode {ep+1}:")
        print(f"  Steps: {result['steps']}")
        print(f"  Reward: {result['total_reward']:.2f}")
        print(f"  Structures: {len(agent.base_agent.structure_pool.structures)}")
        
        # 每2个回合可视化一次
        if (ep + 1) % 2 == 0:
            agent.visualize(f'meta_learning_ep{ep+1}.png')
            agent.base_agent.visualize_network(f'structure_network_meta_ep{ep+1}.png')
    
    # 显示最终报告
    avg_reward = np.mean(rewards)
    print("\n" + "=" * 70)
    print("Final Report:")
    print("=" * 70)
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Meta Learning Memory Size: {len(agent.meta_learner.memory)}")
    print(f"Evolution Strategy: {agent.meta_learner.evolution_strategy}")
    
    print("\n" + "=" * 70)
    print("Meta-Enhanced SDAS Agent Test Complete!")
    print("=" * 70)


def compare_meta_vs_enhanced():
    """比较增强版和元学习增强版SDAS的性能"""
    print("=" * 70)
    print("Comparing Enhanced SDAS vs Meta-Enhanced SDAS")
    print("=" * 70)
    
    # 导入环境
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments'))
    from complex_petri_dish import ComplexPetriDish
    from enhanced_sdas import EnhancedSDASAgent
    
    # 创建环境
    env = ComplexPetriDish(
        width=15,
        height=15,
        n_static_obstacles=20,
        n_dynamic_obstacles=3,
        n_energy_sources=5
    )
    
    # 1. 测试增强版SDAS
    print("\n1. Testing Enhanced SDAS Agent:")
    enhanced_agent = EnhancedSDASAgent()
    enhanced_rewards = []
    
    for episode in range(10):
        from sdas import run_episode
        result = run_episode(env, enhanced_agent, max_steps=150)
        enhanced_rewards.append(result['total_reward'])
        if episode < 3:
            print(f"  Episode {episode+1}: Reward = {result['total_reward']:.2f}, Steps = {result['steps']}")
    
    avg_enhanced = np.mean(enhanced_rewards)
    print(f"  Average Reward: {avg_enhanced:.2f}")
    
    # 2. 测试元学习增强版SDAS
    print("\n2. Testing Meta-Enhanced SDAS Agent:")
    meta_agent = MetaEnhancedSDASAgent()
    meta_rewards = []
    
    for episode in range(10):
        result = run_episode_with_meta(env, meta_agent, max_steps=150)
        meta_rewards.append(result['total_reward'])
        if episode < 3:
            print(f"  Episode {episode+1}: Reward = {result['total_reward']:.2f}, Steps = {result['steps']}")
    
    avg_meta = np.mean(meta_rewards)
    print(f"  Average Reward: {avg_meta:.2f}")
    
    # 比较结果
    print("\n" + "=" * 70)
    print("Comparison Results:")
    print("=" * 70)
    print(f"Enhanced SDAS: {avg_enhanced:.2f}")
    print(f"Meta-Enhanced SDAS: {avg_meta:.2f}")
    print(f"Improvement: {avg_meta - avg_enhanced:.2f}")
    print(f"Relative Improvement: {((avg_meta - avg_enhanced) / abs(avg_enhanced) * 100):.1f}%" if avg_enhanced != 0 else "N/A")
    print("=" * 70)


if __name__ == "__main__":
    # 测试元学习增强的SDAS智能体
    test_meta_enhanced_sdas()
    
    # 比较性能
    print("\n")
    compare_meta_vs_enhanced()
