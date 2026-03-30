#!/usr/bin/env python3
"""
演示自适应结构可视化功能
"""

import os
import sys
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))

from src.visualization import AdaptiveStructureVisualizer
from src.adaptive_low_rank_pool import AdaptiveLowRankStructurePool, AdaptiveLowRankConfig
from src.adaptive_structure_pool import AdaptiveStructurePool, AdaptiveConfig
from src.structure_transfer import StructureTransfer
from complex_petri_dish import ComplexPetriDish

def demo_adaptive_visualization():
    """
    演示自适应结构可视化功能
    """
    print("="*70)
    print("Adaptive Structure Visualization Demo")
    print("="*70)
    
    # 创建可视化器
    viz = AdaptiveStructureVisualizer(save_dir="visualizations")
    
    # 1. 测试自适应动态可视化
    print("\n1. Testing Adaptive Dynamics Visualization...")
    
    # 创建自适应低秩结构池
    config = AdaptiveLowRankConfig(
        min_structures=4,
        max_structures=16,
        initial_structures=8,
        vector_dim=28,
        rank=4
    )
    adaptive_pool = AdaptiveLowRankStructurePool(config)
    
    # 模拟一些结构
    for i in range(10):
        # 模拟观察
        obs = np.random.randn(28)
        adaptive_pool.observe(obs)
    
    # 可视化自适应动态
    viz.visualize_adaptive_dynamics(adaptive_pool, episode=0, save=True, show=False)
    
    # 2. 测试低秩分解可视化
    print("\n2. Testing Low-Rank Decomposition Visualization...")
    if adaptive_pool.structures:
        structure = adaptive_pool.structures[0]
        viz.visualize_low_rank_decomposition(structure, episode=0, save=True, show=False)
    
    # 3. 测试结构迁移可视化
    print("\n3. Testing Structure Transfer Visualization...")
    
    # 创建源环境和结构池
    source_env = ComplexPetriDish(
        width=10, height=10,
        n_static_obstacles=10,
        n_dynamic_obstacles=1,
        n_energy_sources=3
    )
    
    source_config = AdaptiveLowRankConfig(
        min_structures=4,
        max_structures=16,
        initial_structures=8,
        vector_dim=28,
        rank=4
    )
    source_pool = AdaptiveLowRankStructurePool(source_config)
    
    # 模拟源环境中的学习
    print("   Simulating learning in source environment...")
    for step in range(200):
        obs = source_env._get_obs()
        obs_vector = np.array(list(obs.values()))
        source_pool.observe(obs_vector)
        action = np.random.randint(5)
        obs, reward, done = source_env.step(action)
        if done:
            source_env._reset()
    
    # 创建目标环境和结构池
    target_env = ComplexPetriDish(
        width=15, height=15,
        n_static_obstacles=20,
        n_dynamic_obstacles=3,
        n_energy_sources=5
    )
    
    target_config = AdaptiveLowRankConfig(
        min_structures=4,
        max_structures=16,
        initial_structures=4,
        vector_dim=28,
        rank=4
    )
    target_pool = AdaptiveLowRankStructurePool(target_config)
    
    # 执行结构迁移
    transfer = StructureTransfer()
    transferred = transfer.transfer_structures(source_pool, target_pool, target_env)
    
    # 可视化结构迁移
    viz.visualize_structure_transfer(source_pool, target_pool, transferred, 
                                   episode=0, save=True, show=False)
    
    # 4. 测试多环境比较可视化
    print("\n4. Testing Multi-Environment Comparison Visualization...")
    
    # 创建多个环境
    environments = {
        "Easy": ComplexPetriDish(
            width=10, height=10,
            n_static_obstacles=5,
            n_dynamic_obstacles=0,
            n_energy_sources=3
        ),
        "Medium": ComplexPetriDish(
            width=15, height=15,
            n_static_obstacles=15,
            n_dynamic_obstacles=2,
            n_energy_sources=4
        ),
        "Hard": ComplexPetriDish(
            width=20, height=20,
            n_static_obstacles=30,
            n_dynamic_obstacles=5,
            n_energy_sources=5
        )
    }
    
    # 创建对应的结构池
    structure_pools = {}
    for env_name, env in environments.items():
        pool_config = AdaptiveLowRankConfig(
                min_structures=4,
                max_structures=16,
                initial_structures=8,
                vector_dim=28,
                rank=4
            )
        structure_pools[env_name] = AdaptiveLowRankStructurePool(pool_config)
        
        # 模拟学习
        for step in range(100):
            obs = env._get_obs()
            obs_vector = np.array(list(obs.values()))
            structure_pools[env_name].observe(obs_vector)
            action = np.random.randint(5)
            obs, reward, done = env.step(action)
            if done:
                env._reset()
    
    # 可视化多环境比较
    viz.visualize_multi_environment_comparison(environments, structure_pools, 
                                             episode=0, save=True, show=False)
    
    print("\n" + "="*70)
    print("All visualizations generated successfully!")
    print("="*70)
    print("\nGenerated visualization files:")
    for f in os.listdir("visualizations"):
        if f.startswith(('adaptive_', 'low_rank_', 'structure_transfer', 'multi_env_')):
            print(f"  - {f}")

if __name__ == "__main__":
    demo_adaptive_visualization()
