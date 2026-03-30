#!/usr/bin/env python3
"""
测试元学习增强的SDAS智能体

使用实际环境数据测试元学习增强的SDAS智能体性能，
并与其他智能体进行比较。
"""

import numpy as np
import sys
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))

# 导入智能体和环境
from sdas import SDASAgent, Config, run_episode
from enhanced_sdas import EnhancedSDASAgent
from meta_structure_learning import MetaEnhancedSDASAgent, run_episode_with_meta
from complex_petri_dish import ComplexPetriDish


@dataclass
class EnvironmentConfig:
    """环境配置"""
    name: str
    width: int
    height: int
    n_static_obstacles: int
    n_dynamic_obstacles: int
    n_energy_sources: int
    difficulty: float  # 0-1


@dataclass
class AgentConfig:
    """智能体配置"""
    name: str
    agent_type: str  # 'base', 'enhanced', 'meta_enhanced'
    params: Dict


class MetaEnhancedTester:
    """元学习增强测试器"""
    
    def __init__(self, env_config: EnvironmentConfig, agent_config: AgentConfig):
        self.env_config = env_config
        self.agent_config = agent_config
        self.results = []
    
    def create_agent(self):
        """创建智能体"""
        if self.agent_config.agent_type == 'base':
            config = Config(**self.agent_config.params)
            return SDASAgent(config)
        elif self.agent_config.agent_type == 'enhanced':
            config = Config(**self.agent_config.params)
            return EnhancedSDASAgent(config)
        elif self.agent_config.agent_type == 'meta_enhanced':
            config = Config(**self.agent_config.params)
            return MetaEnhancedSDASAgent(config)
        else:
            raise ValueError(f"Unknown agent type: {self.agent_config.agent_type}")
    
    def test_episode(self, max_steps: int = 150) -> Dict:
        """测试一个 episode"""
        env = ComplexPetriDish(
            width=self.env_config.width,
            height=self.env_config.height,
            n_static_obstacles=self.env_config.n_static_obstacles,
            n_dynamic_obstacles=self.env_config.n_dynamic_obstacles,
            n_energy_sources=self.env_config.n_energy_sources
        )
        
        agent = self.create_agent()
        total_reward = 0
        steps = 0
        structure_counts = []
        meta_events = []
        
        if self.agent_config.agent_type == 'meta_enhanced':
            # 使用元学习增强的运行函数
            result = run_episode_with_meta(env, agent, max_steps)
            total_reward = result['total_reward']
            steps = result['steps']
            meta_events = result['meta_events']
            
            # 收集结构数量
            for structure in agent.base_agent.structure_pool.structures:
                structure_counts.append(len(agent.base_agent.structure_pool.structures))
        else:
            # 使用普通运行函数
            result = run_episode(env, agent, max_steps)
            total_reward = result['total_reward']
            steps = result['steps']
            
            # 收集结构数量
            if hasattr(agent, 'structure_pool') and hasattr(agent.structure_pool, 'structures'):
                structure_counts.append(len(agent.structure_pool.structures))
            elif hasattr(agent, 'base_agent') and hasattr(agent.base_agent, 'structure_pool'):
                structure_counts.append(len(agent.base_agent.structure_pool.structures))
        
        # 计算统计信息
        avg_structures = np.mean(structure_counts) if structure_counts else 0
        
        # 获取智能体状态
        agent_state = {}
        if hasattr(agent, 'get_state'):
            agent_state = agent.get_state()
        elif hasattr(agent, 'base_agent') and hasattr(agent.base_agent, 'get_state'):
            agent_state = agent.base_agent.get_state()
        
        # 获取元学习信息
        meta_info = {}
        if self.agent_config.agent_type == 'meta_enhanced':
            meta_info = {
                'memory_size': len(agent.meta_learner.memory),
                'evolution_strategy': agent.meta_learner.evolution_strategy,
                'meta_events': meta_events
            }
        
        return {
            'total_reward': total_reward,
            'steps': steps,
            'avg_structures': avg_structures,
            'structure_counts': structure_counts,
            'agent_state': agent_state,
            'meta_info': meta_info
        }
    
    def run_tests(self, n_episodes: int = 10) -> Dict:
        """运行多个 episode 测试"""
        episode_results = []
        start_time = time.time()
        
        for episode in range(n_episodes):
            result = self.test_episode()
            episode_results.append(result)
            
            if (episode + 1) % 5 == 0:
                avg_reward = np.mean([r['total_reward'] for r in episode_results])
                print(f"  Episode {episode+1}/{n_episodes}: avg_reward={avg_reward:.2f}")
        
        total_time = time.time() - start_time
        
        # 计算统计结果
        rewards = [r['total_reward'] for r in episode_results]
        steps = [r['steps'] for r in episode_results]
        
        results = {
            'env_config': self.env_config.__dict__,
            'agent_config': self.agent_config.__dict__,
            'n_episodes': n_episodes,
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'avg_steps': np.mean(steps),
            'total_time': total_time,
            'time_per_episode': total_time / n_episodes,
            'episode_results': episode_results
        }
        
        # 添加元学习特定信息
        if self.agent_config.agent_type == 'meta_enhanced' and episode_results:
            meta_infos = [r['meta_info'] for r in episode_results if r['meta_info']]
            if meta_infos:
                avg_memory_size = np.mean([info['memory_size'] for info in meta_infos])
                results['avg_meta_memory_size'] = avg_memory_size
        
        self.results = results
        return results


def run_meta_enhanced_test():
    """运行元学习增强测试"""
    print("="*80)
    print("Meta-Enhanced SDAS Agent Test with Real Environment Data")
    print("="*80)
    
    # 环境配置
    environments = [
        EnvironmentConfig(
            name="Easy Small",
            width=10,
            height=10,
            n_static_obstacles=10,
            n_dynamic_obstacles=1,
            n_energy_sources=3,
            difficulty=0.3
        ),
        EnvironmentConfig(
            name="Medium Medium",
            width=15,
            height=15,
            n_static_obstacles=20,
            n_dynamic_obstacles=3,
            n_energy_sources=5,
            difficulty=0.5
        ),
        EnvironmentConfig(
            name="Hard Large",
            width=20,
            height=20,
            n_static_obstacles=30,
            n_dynamic_obstacles=5,
            n_energy_sources=8,
            difficulty=0.8
        )
    ]
    
    # 智能体配置
    agents = [
        AgentConfig(
            name="Base SDAS",
            agent_type="base",
            params={}
        ),
        AgentConfig(
            name="Enhanced SDAS",
            agent_type="enhanced",
            params={}
        ),
        AgentConfig(
            name="Meta-Enhanced SDAS",
            agent_type="meta_enhanced",
            params={}
        )
    ]
    
    # 运行所有组合测试
    all_results = []
    
    for env_config in environments:
        print(f"\n{'-'*80}")
        print(f"Testing Environment: {env_config.name}")
        print(f"  Size: {env_config.width}x{env_config.height}")
        print(f"  Obstacles: static={env_config.n_static_obstacles}, dynamic={env_config.n_dynamic_obstacles}")
        print(f"  Energy sources: {env_config.n_energy_sources}")
        print(f"  Difficulty: {env_config.difficulty:.1f}")
        print('-'*80)
        
        for agent_config in agents:
            print(f"\nTesting Agent: {agent_config.name}")
            tester = MetaEnhancedTester(env_config, agent_config)
            result = tester.run_tests(n_episodes=10)
            all_results.append(result)
            
            # 打印结果
            print(f"  Results:")
            print(f"    Avg reward: {result['avg_reward']:.2f} ± {result['std_reward']:.2f}")
            print(f"    Min/Max reward: {result['min_reward']:.2f} / {result['max_reward']:.2f}")
            print(f"    Avg steps: {result['avg_steps']:.1f}")
            print(f"    Time per episode: {result['time_per_episode']:.2f}s")
            
            # 打印结构信息
            if 'avg_structures' in result['episode_results'][0]:
                avg_structs = np.mean([r['avg_structures'] for r in result['episode_results']])
                print(f"    Avg structures: {avg_structs:.1f}")
            
            # 打印元学习信息
            if 'avg_meta_memory_size' in result:
                print(f"    Meta memory size: {result['avg_meta_memory_size']:.1f}")
    
    # 生成对比表格
    print("\n" + "="*100)
    print("COMPARISON RESULTS")
    print("="*100)
    
    print(f"\n{'Environment':<20} {'Agent':<25} {'Avg Reward':<12} {'Min/Max':<15} {'Avg Steps':<10} {'Time/Episode':<12} {'Meta Memory':<10}")
    print("-"*110)
    
    for result in all_results:
        env_name = result['env_config']['name']
        agent_name = result['agent_config']['name']
        avg_reward = result['avg_reward']
        min_reward = result['min_reward']
        max_reward = result['max_reward']
        avg_steps = result['avg_steps']
        time_per_ep = result['time_per_episode']
        meta_memory = result.get('avg_meta_memory_size', '-')
        
        min_max_str = f"{min_reward:.2f}/{max_reward:.2f}"
        print(f"{env_name:<20} {agent_name:<25} {avg_reward:<12.2f} {min_max_str:<15} {avg_steps:<10.1f} {time_per_ep:<12.2f} {meta_memory:<10}")
    
    # 生成详细对比
    print("\n" + "="*100)
    print("DETAILED COMPARISON BY ENVIRONMENT")
    print("="*100)
    
    for env_config in environments:
        print(f"\n{env_config.name}:")
        print("-"*80)
        
        # 筛选当前环境的结果
        env_results = [r for r in all_results if r['env_config']['name'] == env_config.name]
        
        # 排序结果
        env_results.sort(key=lambda x: x['avg_reward'], reverse=True)
        
        for i, result in enumerate(env_results, 1):
            agent_name = result['agent_config']['name']
            avg_reward = result['avg_reward']
            improvement = (avg_reward - env_results[-1]['avg_reward']) / abs(env_results[-1]['avg_reward']) * 100 if env_results[-1]['avg_reward'] != 0 else 0
            
            print(f"  {i}. {agent_name}: {avg_reward:.2f} (Improvement: {improvement:.1f}%)")
    
    return all_results


def run_ablation_study():
    """运行消融研究"""
    print("\n" + "="*80)
    print("ABLATION STUDY: Meta Learning Components")
    print("="*80)
    
    # 基础环境配置
    env_config = EnvironmentConfig(
        name="Medium Test",
        width=15,
        height=15,
        n_static_obstacles=20,
        n_dynamic_obstacles=3,
        n_energy_sources=5,
        difficulty=0.5
    )
    
    # 智能体配置（消融研究）
    ablation_agents = [
        AgentConfig(
            name="Base SDAS",
            agent_type="base",
            params={}
        ),
        AgentConfig(
            name="Enhanced SDAS",
            agent_type="enhanced",
            params={}
        ),
        AgentConfig(
            name="Meta-Enhanced SDAS (Full)",
            agent_type="meta_enhanced",
            params={}
        )
    ]
    
    print(f"Testing Environment: {env_config.name}")
    print(f"  Size: {env_config.width}x{env_config.height}")
    print(f"  Obstacles: static={env_config.n_static_obstacles}, dynamic={env_config.n_dynamic_obstacles}")
    print(f"  Energy sources: {env_config.n_energy_sources}")
    print('-'*80)
    
    ablation_results = []
    
    for agent_config in ablation_agents:
        print(f"\nTesting: {agent_config.name}")
        tester = MetaEnhancedTester(env_config, agent_config)
        result = tester.run_tests(n_episodes=15)  # 更多回合以获得更稳定的结果
        ablation_results.append(result)
        
        print(f"  Avg reward: {result['avg_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"  Time per episode: {result['time_per_episode']:.2f}s")
    
    # 打印消融研究结果
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    
    print(f"\n{'Agent':<30} {'Avg Reward':<12} {'Std Reward':<12} {'Time/Episode':<12}")
    print("-"*66)
    
    for result in ablation_results:
        agent_name = result['agent_config']['name']
        avg_reward = result['avg_reward']
        std_reward = result['std_reward']
        time_per_ep = result['time_per_episode']
        
        print(f"{agent_name:<30} {avg_reward:<12.2f} {std_reward:<12.2f} {time_per_ep:<12.2f}")
    
    # 计算性能提升
    base_reward = next(r['avg_reward'] for r in ablation_results if r['agent_config']['name'] == "Base SDAS")
    enhanced_reward = next(r['avg_reward'] for r in ablation_results if r['agent_config']['name'] == "Enhanced SDAS")
    meta_reward = next(r['avg_reward'] for r in ablation_results if r['agent_config']['name'] == "Meta-Enhanced SDAS (Full)")
    
    enhanced_improvement = (enhanced_reward - base_reward) / abs(base_reward) * 100 if base_reward != 0 else 0
    meta_improvement = (meta_reward - base_reward) / abs(base_reward) * 100 if base_reward != 0 else 0
    
    print("\nPerformance Improvements:")
    print(f"  Enhanced SDAS vs Base: {enhanced_improvement:.1f}%")
    print(f"  Meta-Enhanced SDAS vs Base: {meta_improvement:.1f}%")
    print(f"  Meta-Enhanced SDAS vs Enhanced: {(meta_reward - enhanced_reward) / abs(enhanced_reward) * 100:.1f}%")
    
    return ablation_results


if __name__ == "__main__":
    # 运行元学习增强测试
    all_results = run_meta_enhanced_test()
    
    # 运行消融研究
    ablation_results = run_ablation_study()
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
