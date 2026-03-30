# -*- coding: utf-8 -*-
import pytest
import numpy as np
from src.structure_pool import StructurePool
from experiments.digital_petri_dish import DigitalPetriDish
from src.sdas import SDASAgent


def test_structure_pool_creation():
    """测试结构池创建和基本功能"""
    pool = StructurePool()
    assert len(pool.structures) == 0


def test_structure_pool_observe():
    """测试结构池观测功能"""
    pool = StructurePool()
    
    # 第一次观测，应该创建新结构
    obs1 = np.random.randn(128)
    result1 = pool.observe(obs1, "测试主题A")
    assert result1['event'] == 'created'
    assert len(pool.structures) == 1
    
    # 相似观测，应该强化现有结构
    obs2 = obs1 + np.random.randn(128) * 0.1
    result2 = pool.observe(obs2, "测试主题A")
    assert result2['event'] == 'reinforced'
    assert len(pool.structures) == 1
    
    # 不同观测，应该创建新结构
    obs3 = np.random.randn(128) * 2
    result3 = pool.observe(obs3, "新主题B")
    assert result3['event'] == 'created'
    assert len(pool.structures) == 2


def test_digital_petri_dish():
    """测试数字培养皿环境"""
    env = DigitalPetriDish(width=10, height=10, n_obstacles=15, n_energy=3)
    obs = env._reset()
    assert 'agent_x' in obs
    assert 'agent_y' in obs
    assert 'nearest_energy_x' in obs
    assert 'nearest_energy_y' in obs


def test_sdas_agent():
    """测试SDAS智能体"""
    agent = SDASAgent()
    env = DigitalPetriDish(width=10, height=10, n_obstacles=15, n_energy=3)
    
    # 运行一个简单的episode
    obs = env._reset()
    agent.reset()
    
    total_reward = 0
    for _ in range(10):
        action, info = agent.step(obs)
        obs, reward, done = env.step(action)
        agent.update_structure(reward)
        total_reward += reward
        if done:
            break
    
    # 检查奖励是否合理
    assert isinstance(total_reward, float)
    # 检查结构池是否有结构
    assert len(agent.structure_pool.structures) > 0


if __name__ == "__main__":
    pytest.main([__file__])
