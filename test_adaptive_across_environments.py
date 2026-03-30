#!/usr/bin/env python3
"""
跨环境测试自适应结构机制

测试不同环境配置下的自适应结构性能：
1. 不同难度级别
2. 不同环境大小
3. 不同动态性
4. 不同智能体配置
"""

import numpy as np
import sys
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))

from sdas import SDASAgent, Config
from adaptive_structure_pool import AdaptiveStructurePool, AdaptiveConfig
from adaptive_low_rank_pool import AdaptiveLowRankStructurePool, AdaptiveLowRankConfig
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
    pool_type: str  # 'fixed', 'adaptive', 'adaptive_low_rank'
    params: Dict


class EnvironmentTester:
    """环境测试器"""
    
    def __init__(self, env_config: EnvironmentConfig, agent_config: AgentConfig):
        self.env_config = env_config
        self.agent_config = agent_config
        self.results = []
    
    def create_agent(self) -> SDASAgent:
        """创建智能体"""
        config = Config()
        agent = SDASAgent(config)
        
        if self.agent_config.pool_type == 'adaptive':
            # 替换为自适应结构池
            adaptive_config = AdaptiveConfig(**self.agent_config.params)
            agent.structure_pool = AdaptiveStructurePool(
                adaptive_config,
                {'vector_dim': config.encoder_dim}
            )
        elif self.agent_config.pool_type == 'adaptive_low_rank':
            # 替换为自适应低秩结构池
            adaptive_config = AdaptiveLowRankConfig(**self.agent_config.params)
            agent.structure_pool = AdaptiveLowRankStructurePool(adaptive_config)
        # 'fixed' 保持默认结构池
        
        return agent
    
    def test_episode(self, max_steps: int = 200) -> Dict:
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
        
        for step in range(max_steps):
            obs = env._get_obs()
            action, _ = agent.step(obs)
            obs, reward, done = env.step(action)
            
            # 记录结构数量
            if hasattr(agent.structure_pool, 'structures'):
                structure_counts.append(len(agent.structure_pool.structures))
            
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # 计算统计信息
        avg_structures = np.mean(structure_counts) if structure_counts else 0
        
        # 获取结构池信息
        pool_info = {}
        if hasattr(agent.structure_pool, 'get_adaptation_report'):
            pool_info = agent.structure_pool.get_adaptation_report()
        elif hasattr(agent.structure_pool, 'get_state'):
            pool_info = agent.structure_pool.get_state()
        
        return {
            'total_reward': total_reward,
            'steps': steps,
            'done': done,
            'avg_structures': avg_structures,
            'structure_counts': structure_counts,
            'pool_info': pool_info
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
        done_counts = sum(1 for r in episode_results if r['done'])
        
        results = {
            'env_config': self.env_config.__dict__,
            'agent_config': self.agent_config.__dict__,
            'n_episodes': n_episodes,
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_steps': np.mean(steps),
            'success_rate': done_counts / n_episodes,
            'total_time': total_time,
            'time_per_episode': total_time / n_episodes,
            'episode_results': episode_results
        }
        
        self.results = results
        return results


def run_comprehensive_test():
    """运行综合测试"""
    print("="*80)
    print("Cross-Environment Adaptive Structure Test")
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
            name="Fixed Structure",
            pool_type="fixed",
            params={}
        ),
        AgentConfig(
            name="Adaptive Structure",
            pool_type="adaptive",
            params={
                'min_structures': 4,
                'max_structures': 32,
                'initial_structures': 8
            }
        ),
        AgentConfig(
            name="Adaptive Low-Rank",
            pool_type="adaptive_low_rank",
            params={
                'min_structures': 4,
                'max_structures': 32,
                'initial_structures': 8,
                'vector_dim': 64,
                'rank': 4
            }
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
            tester = EnvironmentTester(env_config, agent_config)
            result = tester.run_tests(n_episodes=10)
            all_results.append(result)
            
            # 打印结果
            print(f"  Results:")
            print(f"    Avg reward: {result['avg_reward']:.2f} ± {result['std_reward']:.2f}")
            print(f"    Success rate: {result['success_rate']:.2f}")
            print(f"    Avg steps: {result['avg_steps']:.1f}")
            print(f"    Time per episode: {result['time_per_episode']:.2f}s")
            
            # 打印结构信息
            if 'avg_structures' in result['episode_results'][0]:
                avg_structs = np.mean([r['avg_structures'] for r in result['episode_results']])
                print(f"    Avg structures: {avg_structs:.1f}")
            
            # 打印自适应信息
            if 'pool_info' in result['episode_results'][0]:
                pool_info = result['episode_results'][-1]['pool_info']
                if 'params_info' in pool_info:
                    print(f"    Params: {pool_info['params_info']['total_params']}")
                    print(f"    Compression: {pool_info['params_info']['compression_ratio']:.2f}x")
                elif 'total_params' in pool_info:
                    print(f"    Params: {pool_info['total_params']}")
    
    # 生成对比表格
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    print(f"\n{'Environment':<20} {'Agent':<25} {'Avg Reward':<12} {'Success Rate':<12} {'Avg Steps':<10} {'Time/Episode':<12}")
    print("-"*90)
    
    for result in all_results:
        env_name = result['env_config']['name']
        agent_name = result['agent_config']['name']
        avg_reward = result['avg_reward']
        success_rate = result['success_rate'] * 100
        avg_steps = result['avg_steps']
        time_per_ep = result['time_per_episode']
        
        print(f"{env_name:<20} {agent_name:<25} {avg_reward:<12.2f} {success_rate:<12.1f} {avg_steps:<10.1f} {time_per_ep:<12.2f}")
    
    return all_results


if __name__ == "__main__":
    run_comprehensive_test()
