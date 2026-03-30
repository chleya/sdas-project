#!/usr/bin/env python3
"""
Enhanced SDAS Agent - 增强版SDAS智能体

集成结构自组织网络，提升信息处理能力和自适应能力
"""

import numpy as np
from typing import Dict, Tuple, List
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments'))

from sdas import SDASAgent, Config, run_episode
from structure_network import StructureNetwork
from complex_petri_dish import ComplexPetriDish


class EnhancedSDASAgent(SDASAgent):
    """增强版SDAS智能体，集成结构自组织网络"""
    
    def __init__(self, config: Config = None):
        super().__init__(config)
        
        # 初始化结构网络
        self.structure_network = StructureNetwork()
        
        # 同步结构到网络
        self._sync_structure_network()
        
        # 网络更新频率
        self.network_update_interval = 10
        self.steps_since_network_update = 0
    
    def _sync_structure_network(self):
        """同步结构池中的结构到网络"""
        # 添加新结构
        for structure in self.structure_pool.structures:
            if structure.id not in self.structure_network.structure_map:
                self.structure_network.add_structure(structure)
        
        # 移除已删除的结构
        current_ids = {s.id for s in self.structure_pool.structures}
        network_ids = set(self.structure_network.structure_map.keys())
        for structure_id in network_ids - current_ids:
            self.structure_network.remove_structure(structure_id)
    
    def step(self, obs: dict) -> Tuple[int, dict]:
        """智能体一步"""
        # 1. 编码观测
        latent = self.encoder.encode(obs)
        
        # 2. 结构池处理
        structure_signal = self.structure_pool.observe(
            latent, 
            label=self._generate_label(obs)
        )
        
        # 3. 定期更新结构网络
        self.steps_since_network_update += 1
        if self.steps_since_network_update >= self.network_update_interval:
            self._update_structure_network()
            self.steps_since_network_update = 0
        
        # 4. 通过网络激活结构
        activated_structures = self.structure_network.get_activated_structures(latent, top_k=3)
        
        # 5. 激活传播
        propagated_structures = []
        if activated_structures:
            initial_activations = [(s.id, a) for s, a in activated_structures]
            propagated = self.structure_network.propagate_activation(initial_activations, max_steps=2)
            propagated_structures = [(s, a) for s, a in propagated if s.id not in [x[0].id for x in activated_structures]]
        
        # 6. 行动选择（考虑网络激活信息）
        action = self._select_action_with_network(
            obs=obs,
            structure_signal=structure_signal,
            activated_structures=activated_structures,
            propagated_structures=propagated_structures,
            latent=latent
        )
        
        # 7. 记录
        self.total_steps += 1
        
        info = {
            'latent_norm': float(np.linalg.norm(latent)),
            'structure_event': structure_signal['event'],
            'active_structures': structure_signal.get('active_structures', []),
            'recommended_focus': structure_signal.get('recommended_focus', ''),
            'exploration_rate': max(0.05, 0.3 - self.total_steps * 0.001),
            'activated_structures': [(s.id, round(a, 3)) for s, a in activated_structures],
            'propagated_structures': [(s.id, round(a, 3)) for s, a in propagated_structures[:3]],
            'network_stats': self.structure_network.get_network_stats()
        }
        
        return action, info
    
    def _update_structure_network(self):
        """更新结构网络"""
        # 同步结构
        self._sync_structure_network()
        
        # 更新连接
        self.structure_network.update_connections()
    
    def _select_action_with_network(self, obs: dict, structure_signal: dict,
                                   activated_structures: List, 
                                   propagated_structures: List,
                                   latent: np.ndarray) -> int:
        """基于网络信息选择行动"""
        # 基础探索率
        epsilon = max(0.05, 0.3 - self.total_steps * 0.001)
        
        # 如果有强激活的结构，降低探索率
        if activated_structures and activated_structures[0][1] > 0.7:
            epsilon *= 0.5
        
        # Epsilon-greedy
        if np.random.random() < epsilon:
            return np.random.randint(0, self.config.n_actions)
        
        # 基于网络激活信息选择行动
        # 优先使用传播的结构信息
        all_relevant_structures = activated_structures + propagated_structures
        
        if all_relevant_structures:
            # 根据结构标签选择行动
            structure_labels = [s.label for s, _ in all_relevant_structures[:3]]
            
            # 简化的规则：根据标签选择行动
            if any('能量' in label for label in structure_labels):
                # 向能量移动
                return np.random.randint(0, 4)  # 随机移动
            elif any('避障' in label for label in structure_labels):
                # 避开障碍物
                return 4  # 等待
            elif any('探索' in label for label in structure_labels):
                # 探索
                return np.random.randint(0, 4)
        
        # 默认策略
        return self.policy.select_action(
            obs=obs,
            structure_signal=structure_signal,
            world_model=self.world_model,
            latent=latent,
            epsilon=epsilon
        )
    
    def visualize_network(self, filename: str = 'structure_network.png', 
                         observation: dict = None):
        """可视化结构网络"""
        self._update_structure_network()
        
        if observation is not None:
            latent = self.encoder.encode(observation)
            self.structure_network.visualize(filename, show_activation=True, 
                                           observation=latent)
        else:
            self.structure_network.visualize(filename)
    
    def visualize_network_evolution(self, filename: str = 'network_evolution.png'):
        """可视化网络演化"""
        self.structure_network.visualize_evolution(filename)
    
    def get_network_report(self) -> Dict:
        """获取网络报告"""
        stats = self.structure_network.get_network_stats()
        
        return {
            'network_stats': stats,
            'n_structures_in_pool': len(self.structure_pool.structures),
            'n_structures_in_network': len(self.structure_network.structure_map),
            'activation_history_length': len(self.structure_network.activation_history),
            'evolution_history_length': len(self.structure_network.evolution_history)
        }


def test_enhanced_sdas():
    """测试增强版SDAS智能体"""
    print("=" * 70)
    print("Testing Enhanced SDAS Agent with Structure Network")
    print("=" * 70)
    
    # 创建环境
    env = ComplexPetriDish(
        width=15,
        height=15,
        n_static_obstacles=20,
        n_dynamic_obstacles=3,
        n_energy_sources=5
    )
    
    # 创建增强版智能体
    agent = EnhancedSDASAgent()
    
    print("\nRunning 5 episodes...")
    
    # 运行几个回合
    for ep in range(5):
        result = run_episode(env, agent, max_steps=150)
        
        print(f"\nEpisode {ep+1}:")
        print(f"  Steps: {result['steps']}")
        print(f"  Reward: {result['total_reward']:.2f}")
        print(f"  Structures: {len(agent.structure_pool.structures)}")
        
        # 显示网络信息
        if 'network_stats' in result.get('final_state', {}):
            stats = result['final_state']['network_stats']
            print(f"  Network: {stats['n_structures']} nodes, {stats['n_connections']} edges")
        
        # 每2个回合可视化一次网络
        if (ep + 1) % 2 == 0:
            agent.visualize_network(f'structure_network_ep{ep+1}.png')
    
    # 可视化网络演化
    print("\nGenerating network evolution visualization...")
    agent.visualize_network_evolution()
    
    # 显示最终报告
    report = agent.get_network_report()
    print("\n" + "=" * 70)
    print("Final Network Report:")
    print("=" * 70)
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("Enhanced SDAS Agent Test Complete!")
    print("=" * 70)


def compare_with_original():
    """比较原始SDAS和增强版SDAS的性能"""
    print("=" * 70)
    print("Comparing Original SDAS vs Enhanced SDAS")
    print("=" * 70)
    
    # 创建环境
    env = ComplexPetriDish(
        width=15,
        height=15,
        n_static_obstacles=20,
        n_dynamic_obstacles=3,
        n_energy_sources=5
    )
    
    # 1. 测试原始SDAS
    print("\n1. Testing Original SDAS Agent:")
    original_agent = SDASAgent(Config())
    original_rewards = []
    
    for episode in range(10):
        result = run_episode(env, original_agent, max_steps=150)
        original_rewards.append(result['total_reward'])
        if episode < 3:  # 只显示前3个回合
            print(f"  Episode {episode+1}: Reward = {result['total_reward']:.2f}, Steps = {result['steps']}")
    
    avg_original = np.mean(original_rewards)
    print(f"  Average Reward: {avg_original:.2f}")
    
    # 2. 测试增强版SDAS
    print("\n2. Testing Enhanced SDAS Agent:")
    enhanced_agent = EnhancedSDASAgent(Config())
    enhanced_rewards = []
    
    for episode in range(10):
        result = run_episode(env, enhanced_agent, max_steps=150)
        enhanced_rewards.append(result['total_reward'])
        if episode < 3:  # 只显示前3个回合
            print(f"  Episode {episode+1}: Reward = {result['total_reward']:.2f}, Steps = {result['steps']}")
    
    avg_enhanced = np.mean(enhanced_rewards)
    print(f"  Average Reward: {avg_enhanced:.2f}")
    
    # 比较结果
    print("\n" + "=" * 70)
    print("Comparison Results:")
    print("=" * 70)
    print(f"Original SDAS: {avg_original:.2f}")
    print(f"Enhanced SDAS: {avg_enhanced:.2f}")
    print(f"Improvement: {avg_enhanced - avg_original:.2f}")
    print(f"Relative Improvement: {((avg_enhanced - avg_original) / abs(avg_original) * 100):.1f}%" if avg_original != 0 else "N/A")
    print("=" * 70)


if __name__ == "__main__":
    # 测试增强版SDAS
    test_enhanced_sdas()
    
    # 比较性能
    print("\n")
    compare_with_original()
