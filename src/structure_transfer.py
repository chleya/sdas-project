"""
Structure Transfer Learning - 结构迁移学习

核心功能：
1. 结构提取 - 从源环境中提取有用的结构
2. 结构评估 - 评估结构在目标环境中的适用性
3. 结构迁移 - 将适用的结构迁移到目标环境
4. 结构适应 - 让迁移的结构适应新环境
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pickle
import os

from structure_pool import Structure, StructurePool
from structure_pool_lora import LowRankStructure, LowRankStructurePool
from adaptive_structure_pool import AdaptiveStructurePool
from adaptive_low_rank_pool import AdaptiveLowRankStructurePool


@dataclass
class TransferConfig:
    """迁移配置"""
    # 迁移参数
    top_k_structures: int = 10  # 迁移前k个最有用的结构
    similarity_threshold: float = 0.7  # 结构相似度阈值
    adaptation_steps: int = 10  # 迁移后适应步数
    
    # 评估参数
    evaluation_episodes: int = 5  # 评估迁移效果的episode数
    max_steps_per_episode: int = 100  # 评估时的最大步数
    
    # 保存路径
    save_dir: str = "transferred_structures"


class StructureTransfer:
    """
    结构迁移学习管理器
    
    负责在不同环境之间迁移结构知识
    """
    
    def __init__(self, config: TransferConfig = None):
        self.config = config or TransferConfig()
        
        # 创建保存目录
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        print(f"Structure Transfer initialized")
        print(f"  Top K structures: {self.config.top_k_structures}")
        print(f"  Similarity threshold: {self.config.similarity_threshold}")
        print(f"  Adaptation steps: {self.config.adaptation_steps}")
        print(f"  Save directory: {self.config.save_dir}")
    
    def extract_structures(self, source_pool: any) -> List[any]:
        """
        从源结构池提取有用的结构
        
        Args:
            source_pool: 源结构池
        
        Returns:
            按效用排序的结构列表
        """
        if isinstance(source_pool, (AdaptiveStructurePool, AdaptiveLowRankStructurePool)):
            # 从自适应结构池提取
            if hasattr(source_pool, 'structure_stats'):
                # 按效用分数排序
                sorted_structures = sorted(
                    source_pool.structures,
                    key=lambda s: source_pool.structure_stats[s.id].utility_score,
                    reverse=True
                )
            else:
                # 按活力排序
                sorted_structures = sorted(
                    source_pool.structures,
                    key=lambda s: s.vigor(),
                    reverse=True
                )
        else:
            # 从普通结构池提取
            sorted_structures = sorted(
                source_pool.structures,
                key=lambda s: s.vigor(),
                reverse=True
            )
        
        # 取前k个
        return sorted_structures[:self.config.top_k_structures]
    
    def evaluate_structure(self, structure: any, target_env: any, 
                          target_pool: any) -> float:
        """
        评估结构在目标环境中的适用性
        
        Args:
            structure: 要评估的结构
            target_env: 目标环境
            target_pool: 目标结构池
        
        Returns:
            结构在目标环境中的适应度分数
        """
        # 重置环境
        target_env._reset()
        
        total_reward = 0
        steps = 0
        
        # 模拟使用该结构
        for _ in range(self.config.max_steps_per_episode):
            obs = target_env._get_obs()
            
            # 计算结构与当前观察的相似度
            if isinstance(structure, LowRankStructure):
                prototype = structure.get_prototype()
            else:
                prototype = structure.prototype
            
            # 简化的相似度计算
            if hasattr(target_pool, '_cosine_similarity'):
                similarity = target_pool._cosine_similarity(
                    np.array(list(obs.values())),
                    prototype
                )
            else:
                # 简单的欧氏距离
                similarity = 1.0 / (1.0 + np.linalg.norm(
                    np.array(list(obs.values())) - prototype
                ))
            
            # 基于相似度选择动作（简化）
            action = 0 if similarity > 0.5 else 4  # 0: 上, 4: 等待
            
            obs, reward, done = target_env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # 计算适应度分数
        fitness = total_reward / steps if steps > 0 else 0
        return fitness
    
    def transfer_structures(self, source_pool: any, target_pool: any, 
                           target_env: any) -> List[any]:
        """
        将结构从源池迁移到目标池
        
        Args:
            source_pool: 源结构池
            target_pool: 目标结构池
            target_env: 目标环境
        
        Returns:
            成功迁移的结构列表
        """
        print("\n" + "="*60)
        print("STRUCTURE TRANSFER")
        print("="*60)
        
        # 提取源结构
        source_structures = self.extract_structures(source_pool)
        print(f"Extracted {len(source_structures)} structures from source pool")
        
        # 评估并迁移结构
        transferred = []
        for i, structure in enumerate(source_structures):
            # 评估结构
            fitness = self.evaluate_structure(structure, target_env, target_pool)
            
            print(f"Structure {i+1}/{len(source_structures)}: fitness={fitness:.2f}")
            
            # 基于适应度决定是否迁移
            if fitness > 0:
                # 创建迁移的结构
                transferred_structure = self._create_transferred_structure(structure, target_pool)
                if transferred_structure:
                    # 添加到目标池
                    if hasattr(target_pool, 'structures'):
                        target_pool.structures.append(transferred_structure)
                        # 更新统计信息
                        if hasattr(target_pool, 'structure_stats'):
                            from adaptive_structure_pool import StructureStats
                            from adaptive_low_rank_pool import LowRankStructureStats
                            
                            if isinstance(target_pool, AdaptiveLowRankStructurePool):
                                stats_class = LowRankStructureStats
                            else:
                                stats_class = StructureStats
                            
                            target_pool.structure_stats[transferred_structure.id] = stats_class(
                                structure_id=transferred_structure.id,
                                creation_time=0
                            )
                    transferred.append(transferred_structure)
                    print(f"  ✓ Transferred structure {transferred_structure.id}")
            else:
                print(f"  ✗ Structure not transferred (low fitness)")
        
        # 适应迁移的结构
        if transferred:
            self._adapt_transferred_structures(transferred, target_pool, target_env)
        
        print(f"\nTransfer completed: {len(transferred)} structures transferred")
        return transferred
    
    def _create_transferred_structure(self, source_structure: any, 
                                    target_pool: any) -> any:
        """
        创建迁移的结构
        
        Args:
            source_structure: 源结构
            target_pool: 目标结构池
        
        Returns:
            迁移后的结构
        """
        if isinstance(source_structure, LowRankStructure):
            # 低秩结构迁移
            if isinstance(target_pool, LowRankStructurePool):
                # 复制低秩参数
                new_structure = LowRankStructure(
                    id=target_pool.next_id,
                    base_vector=target_pool.base_vector,
                    B=source_structure.B.copy(),
                    a=source_structure.a.copy(),
                    label=f"transferred_{source_structure.label}",
                    utility=source_structure.utility,
                    age=0
                )
                target_pool.next_id += 1
                return new_structure
        else:
            # 普通结构迁移
            if hasattr(target_pool, 'vector_dim'):
                new_structure = Structure(
                    id=getattr(target_pool, 'next_id', len(target_pool.structures)),
                    prototype=source_structure.prototype.copy(),
                    label=f"transferred_{source_structure.label}",
                    utility=source_structure.utility,
                    age=0
                )
                if hasattr(target_pool, 'next_id'):
                    target_pool.next_id += 1
                return new_structure
        
        return None
    
    def _adapt_transferred_structures(self, transferred_structures: List[any], 
                                    target_pool: any, target_env: any):
        """
        适应迁移的结构到目标环境
        
        Args:
            transferred_structures: 迁移的结构列表
            target_pool: 目标结构池
            target_env: 目标环境
        """
        print("\nAdapting transferred structures...")
        
        for step in range(self.config.adaptation_steps):
            # 重置环境
            target_env._reset()
            
            for _ in range(50):  # 每个适应步骤运行50步
                obs = target_env._get_obs()
                
                # 转换观察为向量
                obs_vector = np.array(list(obs.values()))
                
                # 观察并适应
                if hasattr(target_pool, 'observe'):
                    target_pool.observe(obs_vector)
                
                # 随机动作
                action = np.random.randint(5)
                obs, reward, done = target_env.step(action)
                
                if done:
                    break
        
        print(f"Adaptation completed in {self.config.adaptation_steps} steps")
    
    def save_structures(self, structures: List[any], filename: str):
        """
        保存结构到文件
        
        Args:
            structures: 结构列表
            filename: 文件名
        """
        filepath = os.path.join(self.config.save_dir, filename)
        
        # 保存结构数据
        structure_data = []
        for structure in structures:
            if isinstance(structure, LowRankStructure):
                data = {
                    'type': 'low_rank',
                    'id': structure.id,
                    'B': structure.B.tolist(),
                    'a': structure.a.tolist(),
                    'label': structure.label,
                    'utility': structure.utility
                }
            else:
                data = {
                    'type': 'normal',
                    'id': structure.id,
                    'prototype': structure.prototype.tolist(),
                    'label': structure.label,
                    'utility': structure.utility
                }
            structure_data.append(data)
        
        with open(filepath, 'wb') as f:
            pickle.dump(structure_data, f)
        
        print(f"Saved {len(structures)} structures to {filepath}")
    
    def load_structures(self, filename: str) -> List[any]:
        """
        从文件加载结构
        
        Args:
            filename: 文件名
        
        Returns:
            结构列表
        """
        filepath = os.path.join(self.config.save_dir, filename)
        
        with open(filepath, 'rb') as f:
            structure_data = pickle.load(f)
        
        structures = []
        for data in structure_data:
            if data['type'] == 'low_rank':
                structure = LowRankStructure(
                    id=data['id'],
                    base_vector=np.zeros(64),  # 占位，需要后续设置
                    B=np.array(data['B']),
                    a=np.array(data['a']),
                    label=data['label'],
                    utility=data['utility']
                )
            else:
                structure = Structure(
                    id=data['id'],
                    prototype=np.array(data['prototype']),
                    label=data['label'],
                    utility=data['utility']
                )
            structures.append(structure)
        
        print(f"Loaded {len(structures)} structures from {filepath}")
        return structures
    
    def evaluate_transfer_effectiveness(self, target_pool: any, target_env: any) -> Dict:
        """
        评估迁移效果
        
        Args:
            target_pool: 目标结构池
            target_env: 目标环境
        
        Returns:
            评估结果
        """
        print("\n" + "="*60)
        print("EVALUATING TRANSFER EFFECTIVENESS")
        print("="*60)
        
        rewards = []
        steps = []
        successes = []
        
        for episode in range(self.config.evaluation_episodes):
            # 重置环境
            target_env._reset()
            total_reward = 0
            step_count = 0
            
            for _ in range(self.config.max_steps_per_episode):
                obs = target_env._get_obs()
                
                # 转换观察为向量
                obs_vector = np.array(list(obs.values()))
                
                # 观察
                if hasattr(target_pool, 'observe'):
                    target_pool.observe(obs_vector)
                
                # 随机动作（简化）
                action = np.random.randint(5)
                obs, reward, done = target_env.step(action)
                
                total_reward += reward
                step_count += 1
                
                if done:
                    break
            
            rewards.append(total_reward)
            steps.append(step_count)
            successes.append(1 if done else 0)
            
            print(f"Episode {episode+1}: reward={total_reward:.2f}, steps={step_count}, done={done}")
        
        # 计算统计结果
        results = {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_steps': np.mean(steps),
            'success_rate': np.mean(successes),
            'n_episodes': self.config.evaluation_episodes
        }
        
        print(f"\nEvaluation Results:")
        print(f"  Avg reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Success rate: {results['success_rate']:.2f}")
        print(f"  Avg steps: {results['avg_steps']:.1f}")
        
        return results


if __name__ == "__main__":
    """测试结构迁移功能"""
    print("="*70)
    print("Structure Transfer Test")
    print("="*70)
    
    # 导入必要的模块
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
    from complex_petri_dish import ComplexPetriDish
    
    # 创建源环境和结构池
    source_env = ComplexPetriDish(
        width=10, height=10,
        n_static_obstacles=10,
        n_dynamic_obstacles=1,
        n_energy_sources=3
    )
    
    # 创建源结构池（自适应低秩）
    from adaptive_low_rank_pool import AdaptiveLowRankStructurePool, AdaptiveLowRankConfig
    source_config = AdaptiveLowRankConfig(
        min_structures=4,
        max_structures=16,
        initial_structures=8,
        vector_dim=64,
        rank=4
    )
    source_pool = AdaptiveLowRankStructurePool(source_config)
    
    # 模拟源环境中的学习
    print("Simulating learning in source environment...")
    for step in range(500):
        obs = source_env._get_obs()
        obs_vector = np.array(list(obs.values()))
        source_pool.observe(obs_vector)
        action = np.random.randint(5)
        obs, reward, done = source_env.step(action)
        if done:
            source_env._reset()
    
    # 创建目标环境
    target_env = ComplexPetriDish(
        width=15, height=15,
        n_static_obstacles=20,
        n_dynamic_obstacles=3,
        n_energy_sources=5
    )
    
    # 创建目标结构池
    target_config = AdaptiveLowRankConfig(
        min_structures=4,
        max_structures=16,
        initial_structures=4,
        vector_dim=64,
        rank=4
    )
    target_pool = AdaptiveLowRankStructurePool(target_config)
    
    # 创建迁移管理器
    transfer = StructureTransfer()
    
    # 执行迁移
    transferred = transfer.transfer_structures(source_pool, target_pool, target_env)
    
    # 评估迁移效果
    transfer.evaluate_transfer_effectiveness(target_pool, target_env)
    
    # 保存迁移的结构
    transfer.save_structures(transferred, "transferred_from_easy_to_medium.pkl")
    
    print("\n" + "="*70)
    print("Structure Transfer Test Complete!")
    print("="*70)
