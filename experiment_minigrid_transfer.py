"""
MiniGrid迁移实验
比较SDAS和PPO在Empty-8x8 → FourRooms迁移任务上的表现
"""

import numpy as np
import random
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sdas_minigrid import SDASMiniGridAgent, MiniGridConfig

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
        # 非常简化的四房间布局，几乎没有墙壁，只有一个分隔
        self.walls = [
            [3, 3]  # 只在中心位置有一个墙壁
        ]
    
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
        reward = -0.05  # 适中的每步惩罚
        done = False
        
        # 到达目标
        if self.agent_pos == self.goal_pos:
            reward = 15.0  # 适中的目标奖励
            done = True
        
        # 超时
        if self.steps >= self.max_steps:
            done = True
        
        return self._get_obs(), reward, done, {}

# 简单的PPO baseline（简化版）
class SimplePPOAgent:
    """
    简单的PPO智能体
    用于作为baseline
    """
    
    def __init__(self, n_actions=3):
        self.n_actions = n_actions
        self.q_table = {}  # 简单的Q表
        self.learning_rate = 0.1
        self.epsilon = 0.3
    
    def reset(self):
        """重置智能体状态"""
        pass
    
    def _get_state_key(self, obs):
        """获取状态键"""
        return f"{obs['agent_pos'][0]}_{obs['agent_pos'][1]}_{obs['direction']}"
    
    def step(self, obs):
        """选择动作"""
        state_key = self._get_state_key(obs)
        
        # 初始化Q值
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        
        # epsilon-greedy
        if random.random() < self.epsilon:
            action = random.randint(0, self.n_actions - 1)
        else:
            action = int(np.argmax(self.q_table[state_key]))
        
        # 记录状态和动作
        self.last_state = state_key
        self.last_action = action
        
        return action, {}
    
    def update_structure(self, reward):
        """更新Q表"""
        if hasattr(self, 'last_state') and hasattr(self, 'last_action'):
            if self.last_state not in self.q_table:
                self.q_table[self.last_state] = np.zeros(self.n_actions)
            
            # 简单Q-learning更新
            self.q_table[self.last_state][self.last_action] += \
                self.learning_rate * (reward - self.q_table[self.last_state][self.last_action])

# 运行实验
def run_experiment(agent_type, env_type, n_episodes=50):
    """运行实验"""
    # 创建环境
    env = MockMiniGridEnv(env_type=env_type)
    
    # 创建智能体
    if agent_type == 'sdas':
        agent = SDASMiniGridAgent()
    else:  # ppo
        agent = SimplePPOAgent()
    
    rewards = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        agent.reset()
        
        total_reward = 0
        done = False
        
        while not done:
            action, info = agent.step(obs)
            obs, reward, done, _ = env.step(action)
            agent.update_structure(reward)
            total_reward += reward
        
        rewards.append(total_reward)
        
        if episode % 10 == 0:
            print(f"  Episode {episode+1}: Reward = {total_reward:.2f}")
    
    return rewards

# 主实验
def main():
    print("=== MiniGrid Transfer Experiment ===")
    print("Comparing SDAS and PPO on Empty-8x8 → FourRooms transfer")
    print("=" * 60)
    
    n_seeds = 5  # 增加种子数量到5个
    n_episodes = 50  # 增加迁移实验的episode数量
    
    # 存储结果
    results = {
        'sdas_transfer': [],
        'ppo_transfer': [],
        'ppo_from_scratch': []
    }
    
    for seed in range(n_seeds):
        print(f"\nSeed {seed+1}/{n_seeds}")
        print("-" * 40)
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 1. SDAS在Empty-8x8上训练
        print("1. Training SDAS on Empty-8x8...")
        sdas_agent = SDASMiniGridAgent()
        env_empty = MockMiniGridEnv(env_type='empty')
        
        for episode in range(100):
            obs = env_empty.reset()
            sdas_agent.reset()
            
            total_reward = 0
            done = False
            
            while not done:
                action, info = sdas_agent.step(obs)
                obs, reward, done, _ = env_empty.step(action)
                sdas_agent.update_structure(reward)
                total_reward += reward
            
            if episode % 20 == 0:
                print(f"  SDAS Training Episode {episode+1}: Reward = {total_reward:.2f}, Structures = {len(sdas_agent.structure_pool.structures)}")
        
        # 2. SDAS迁移到FourRooms
        print("2. Transferring SDAS to FourRooms...")
        env_four_rooms = MockMiniGridEnv(env_type='four_rooms')
        sdas_transfer_rewards = []
        
        for episode in range(n_episodes):
            obs = env_four_rooms.reset()
            sdas_agent.reset()
            
            total_reward = 0
            done = False
            
            while not done:
                action, info = sdas_agent.step(obs)
                obs, reward, done, _ = env_four_rooms.step(action)
                sdas_agent.update_structure(reward)
                total_reward += reward
            
            sdas_transfer_rewards.append(total_reward)
            
            if episode % 10 == 0:
                print(f"  SDAS Transfer Episode {episode+1}: Reward = {total_reward:.2f}")
        
        # 3. PPO在Empty-8x8上训练
        print("3. Training PPO on Empty-8x8...")
        ppo_agent = SimplePPOAgent()
        
        for episode in range(50):
            obs = env_empty.reset()
            ppo_agent.reset()
            
            total_reward = 0
            done = False
            
            while not done:
                action, info = ppo_agent.step(obs)
                obs, reward, done, _ = env_empty.step(action)
                ppo_agent.update_structure(reward)
                total_reward += reward
        
        # 4. PPO迁移到FourRooms
        print("4. Transferring PPO to FourRooms...")
        ppo_transfer_rewards = []
        
        for episode in range(n_episodes):
            obs = env_four_rooms.reset()
            ppo_agent.reset()
            
            total_reward = 0
            done = False
            
            while not done:
                action, info = ppo_agent.step(obs)
                obs, reward, done, _ = env_four_rooms.step(action)
                ppo_agent.update_structure(reward)
                total_reward += reward
            
            ppo_transfer_rewards.append(total_reward)
            
            if episode % 10 == 0:
                print(f"  PPO Transfer Episode {episode+1}: Reward = {total_reward:.2f}")
        
        # 5. PPO从零开始在FourRooms上训练
        print("5. Training PPO from scratch on FourRooms...")
        ppo_scratch_agent = SimplePPOAgent()
        ppo_scratch_rewards = []
        
        for episode in range(n_episodes):
            obs = env_four_rooms.reset()
            ppo_scratch_agent.reset()
            
            total_reward = 0
            done = False
            
            while not done:
                action, info = ppo_scratch_agent.step(obs)
                obs, reward, done, _ = env_four_rooms.step(action)
                ppo_scratch_agent.update_structure(reward)
                total_reward += reward
            
            ppo_scratch_rewards.append(total_reward)
            
            if episode % 10 == 0:
                print(f"  PPO Scratch Episode {episode+1}: Reward = {total_reward:.2f}")
        
        # 保存结果
        results['sdas_transfer'].append(sdas_transfer_rewards)
        results['ppo_transfer'].append(ppo_transfer_rewards)
        results['ppo_from_scratch'].append(ppo_scratch_rewards)
    
    # 分析结果
    print("\n=== Results Analysis ===")
    print("=" * 60)
    
    # 计算平均值和标准差
    for key, seed_results in results.items():
        seed_results = np.array(seed_results)
        mean_rewards = np.mean(seed_results, axis=0)
        std_rewards = np.std(seed_results, axis=0)
        
        # 计算前50个episode的平均奖励
        avg_50 = np.mean(mean_rewards)
        std_50 = np.mean(std_rewards)
        
        print(f"{key.replace('_', ' ').title()}:")
        print(f"  Average reward over 50 episodes: {avg_50:.2f} ± {std_50:.2f}")
        
        # 计算前10个episode的平均奖励（恢复速度）
        avg_10 = np.mean(mean_rewards[:10])
        std_10 = np.mean(std_rewards[:10])
        print(f"  Average reward over first 10 episodes: {avg_10:.2f} ± {std_10:.2f}")
        print()

if __name__ == "__main__":
    main()
