"""
环境测试脚本
验证模拟MiniGrid环境是否正常工作
"""

import numpy as np
import random

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
        # 垂直墙壁（中间有门）
        for y in range(8):
            if y != 3:
                self.walls.append([3, y])
        # 水平墙壁（中间有门）
        for x in range(8):
            if x != 3:
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

# 测试环境
def test_environment():
    print("Testing Empty environment...")
    env = MockMiniGridEnv(env_type='empty')
    
    # 手动控制智能体到达目标
    obs = env.reset()
    print(f"Initial position: {obs['agent_pos']}")
    
    # 简单的导航策略：一直向右，然后向上
    steps = 0
    total_reward = 0
    done = False
    
    while not done:
        # 一直向右
        if obs['agent_pos'][0] < 7:
            action = 0  # 前进
        else:
            # 到达右边界后向上
            if obs['direction'] != 3:  # 不是向上
                action = 2  # 右转
            else:
                action = 0  # 前进
        
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        print(f"Step {steps}: Position = {obs['agent_pos']}, Direction = {obs['direction']}, Reward = {reward:.2f}")
        
        if done:
            print(f"Done! Total steps: {steps}, Total reward: {total_reward:.2f}")
            break

# 测试四房间环境
def test_four_rooms():
    print("\nTesting FourRooms environment...")
    env = MockMiniGridEnv(env_type='four_rooms')
    
    # 打印墙壁布局
    print("Walls:")
    for wall in env.walls:
        print(f"  {wall}")
    
    obs = env.reset()
    print(f"Initial position: {obs['agent_pos']}")
    print(f"Goal position: {env.goal_pos}")
    
    # 简单的导航策略
    steps = 0
    total_reward = 0
    done = False
    
    while not done and steps < 100:
        # 随机动作
        action = random.randint(0, 2)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        if steps % 10 == 0:
            print(f"Step {steps}: Position = {obs['agent_pos']}, Direction = {obs['direction']}")
    
    print(f"Done! Total steps: {steps}, Total reward: {total_reward:.2f}")

if __name__ == "__main__":
    test_environment()
    test_four_rooms()
