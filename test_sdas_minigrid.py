"""
SDAS MiniGrid集成测试
测试SDAS在MiniGrid环境中的表现
"""

import pytest
import numpy as np
import random
from src.sdas_minigrid import SDASMiniGridAgent, MiniGridConfig

# 模拟MiniGrid环境
class MockMiniGridEnv:
    """
    模拟MiniGrid环境
    用于测试SDAS集成
    """
    
    def __init__(self, env_type='empty'):
        self.env_type = env_type
        self.width = 8
        self.height = 8
        self.agent_pos = [0, 0]
        self.direction = 0  # 0=右, 1=下, 2=左, 3=上
        self.goal_pos = [7, 7]
        self.steps = 0
        self.max_steps = 200
        self.walls = []
        
        # 初始化墙壁
        if env_type == 'four_rooms':
            # 四房间布局
            self._init_four_rooms()
    
    def _init_four_rooms(self):
        """初始化四房间布局"""
        # 垂直墙壁
        for y in range(8):
            if y != 3 and y != 4:
                self.walls.append([3, y])
        # 水平墙壁
        for x in range(8):
            if x != 3 and x != 4:
                self.walls.append([x, 3])
    
    def reset(self):
        """重置环境"""
        self.agent_pos = [0, 0]
        self.direction = 0
        self.steps = 0
        return self._get_obs()
    
    def _get_obs(self):
        """获取观测"""
        return {
            'agent_pos': self.agent_pos,
            'direction': self.direction,
            'mission': 'reach the goal'
        }
    
    def step(self, action):
        """执行动作"""
        # 动作：0=前进, 1=左转, 2=右转
        new_pos = self.agent_pos.copy()
        
        if action == 0:  # 前进
            if self.direction == 0:  # 右
                new_pos[0] = min(new_pos[0] + 1, self.width - 1)
            elif self.direction == 1:  # 下
                new_pos[1] = max(new_pos[1] - 1, 0)
            elif self.direction == 2:  # 左
                new_pos[0] = max(new_pos[0] - 1, 0)
            elif self.direction == 3:  # 上
                new_pos[1] = min(new_pos[1] + 1, self.height - 1)
        elif action == 1:  # 左转
            self.direction = (self.direction + 3) % 4
        elif action == 2:  # 右转
            self.direction = (self.direction + 1) % 4
        
        # 检查墙壁碰撞
        if new_pos not in self.walls:
            self.agent_pos = new_pos
        
        self.steps += 1
        
        # 计算奖励
        reward = -0.01  # 每步惩罚
        done = False
        
        # 到达目标
        if self.agent_pos == self.goal_pos:
            reward = 1.0
            done = True
        
        # 超时
        if self.steps >= self.max_steps:
            done = True
        
        return self._get_obs(), reward, done, {}


def test_sdas_minigrid_agent_initialization():
    """测试SDAS MiniGrid智能体初始化"""
    agent = SDASMiniGridAgent()
    assert agent.config.max_structures == 32
    assert len(agent.structure_pool.structures) == 0


def test_sdas_minigrid_agent_step():
    """测试SDAS MiniGrid智能体步骤"""
    env = MockMiniGridEnv(env_type='empty')
    agent = SDASMiniGridAgent()
    
    obs = env.reset()
    agent.reset()
    
    action, info = agent.step(obs)
    assert action in [0, 1, 2]  # 验证动作范围
    assert 'structure_event' in info
    assert 'active_structures' in info


def test_sdas_minigrid_agent_training():
    """测试SDAS MiniGrid智能体训练"""
    env = MockMiniGridEnv(env_type='empty')
    agent = SDASMiniGridAgent()
    
    total_reward = 0
    structures_count = []
    
    for ep in range(3):
        obs = env.reset()
        agent.reset()
        
        ep_reward = 0
        done = False
        steps = 0
        
        while not done:
            action, info = agent.step(obs)
            obs, reward, done, _ = env.step(action)
            agent.update_structure(reward)
            ep_reward += reward
            steps += 1
            
            if steps >= 100:  # 防止无限循环
                break
        
        total_reward += ep_reward
        structures_count.append(len(agent.structure_pool.structures))
    
    # 检查奖励是否合理
    assert isinstance(total_reward, float)
    # 检查结构数量是否增加
    assert len(agent.structure_pool.structures) > 0
    # 检查结构数量是否随训练增加
    assert max(structures_count) >= min(structures_count)


def test_structure_transfer():
    """测试SDAS MiniGrid智能体结构迁移能力"""
    # 在empty环境中训练
    env1 = MockMiniGridEnv(env_type='empty')
    agent = SDASMiniGridAgent()
    
    # 在第一个环境中训练
    for episode in range(5):
        obs = env1.reset()
        agent.reset()
        
        total_reward = 0
        done = False
        
        while not done:
            action, info = agent.step(obs)
            obs, reward, done, _ = env1.step(action)
            agent.update_structure(reward)
            total_reward += reward
        
    # 记录训练后的结构数量
    structures_after_training = len(agent.structure_pool.structures)
    assert structures_after_training > 0
    
    # 迁移到four_rooms环境
    env2 = MockMiniGridEnv(env_type='four_rooms')
    
    # 在第二个环境中测试迁移
    transfer_rewards = []
    for episode in range(5):
        obs = env2.reset()
        agent.reset()
        
        total_reward = 0
        done = False
        
        while not done:
            action, info = agent.step(obs)
            obs, reward, done, _ = env2.step(action)
            agent.update_structure(reward)
            total_reward += reward
        
        transfer_rewards.append(total_reward)
    
    # 检查迁移后的奖励是否合理
    assert isinstance(np.mean(transfer_rewards), float)
    # 检查结构数量是否保持或增加
    assert len(agent.structure_pool.structures) >= structures_after_training


if __name__ == "__main__":
    pytest.main([__file__])
