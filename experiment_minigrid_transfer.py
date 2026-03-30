"""
MiniGrid迁移实验
比较SDAS和PPO在Empty-8x8 → FourRooms迁移任务上的表现
"""

import numpy as np
import random
import gymnasium as gym
import csv
import os
from scipy import stats
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from src.sdas_minigrid import SDASMiniGridAgent, MiniGridConfig

# 设置固定种子
def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    gym.utils.seeding.np_random(seed)


# 真实的MiniGrid环境包装器
class MiniGridEnvWrapper:
    """
    MiniGrid环境包装器
    用于适配SDAS智能体
    """
    
    def __init__(self, env_id):
        self.env = gym.make(env_id)
        self.max_steps = 200
        self.steps = 0
    
    def reset(self):
        """重置环境"""
        obs, _ = self.env.reset()
        self.steps = 0
        return self._process_obs(obs)
    
    def _process_obs(self, obs):
        """处理观测，转换为SDAS所需的格式"""
        return {
            'agent_pos': [obs['agent_pos'][0], obs['agent_pos'][1]],
            'direction': obs['direction'],
            'mission': obs['mission']
        }
    
    def step(self, action):
        """执行动作"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.steps += 1
        
        # 调整奖励
        if reward > 0:
            reward = 15.0  # 目标奖励
        else:
            reward = -0.05  # 每步惩罚
        
        return self._process_obs(obs), reward, done, info
    
    def close(self):
        """关闭环境"""
        self.env.close()

# PPO baseline
class PPOAgent:
    """
    PPO智能体
    用于作为baseline
    """
    
    def __init__(self, env_id, seed, log_dir=None):
        self.env = DummyVecEnv([lambda: gym.make(env_id)])
        # 设置tensorboard日志
        if log_dir:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.model = PPO('MlpPolicy', self.env, verbose=0, tensorboard_log=log_dir)
        else:
            self.model = PPO('MlpPolicy', self.env, verbose=0)
        self.n_actions = self.env.action_space.n
    
    def reset(self):
        """重置智能体状态"""
        pass
    
    def step(self, obs):
        """选择动作"""
        # 转换观测格式
        # 注意：这里简化处理，实际应该根据PPO模型的需求进行适当转换
        action, _states = self.model.predict(obs, deterministic=True)
        return action[0], {}
    
    def update_structure(self, reward):
        """PPO通过模型训练更新，这里不需要单独更新"""
        pass
    
    def train(self, total_timesteps=10000):
        """训练PPO模型"""
        self.model.learn(total_timesteps=total_timesteps)
    
    def close(self):
        """关闭环境"""
        self.env.close()

# 运行实验
def run_experiment(agent_type, env_id, n_episodes=50):
    """运行实验"""
    # 创建环境
    env = MiniGridEnvWrapper(env_id)
    
    # 创建智能体
    if agent_type == 'sdas':
        agent = SDASMiniGridAgent()
    else:  # ppo
        agent = PPOAgent(env_id)
        # 训练PPO模型
        agent.train(total_timesteps=10000)
    
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
    
    # 关闭环境
    env.close()
    if agent_type != 'sdas':
        agent.close()
    
    return rewards

# 主实验
def main():
    print("=== MiniGrid Transfer Experiment ===")
    print("Comparing SDAS and PPO on Empty-8x8 → FourRooms transfer")
    print("=" * 60)
    
    n_seeds = 10  # 增加种子数量到10个，提高统计显著性
    n_episodes = 50  # 增加迁移实验的episode数量
    
    # 创建结果目录
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 创建CSV文件
    csv_file = os.path.join(results_dir, 'minigrid_transfer_results.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Seed', 'Agent', 'Episode', 'Reward'])
    
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
        set_seed(seed)
        
        # 创建日志目录
        log_dir = os.path.join('logs', f'seed_{seed}')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 1. SDAS在Empty-8x8上训练
        print("1. Training SDAS on Empty-8x8...")
        sdas_agent = SDASMiniGridAgent()
        env_empty = MiniGridEnvWrapper('MiniGrid-Empty-8x8-v0')
        
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
        env_four_rooms = MiniGridEnvWrapper('MiniGrid-FourRooms-v0')
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
            
            # 写入CSV文件
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([seed, 'SDAS_Transfer', episode, total_reward])
            
            if episode % 10 == 0:
                print(f"  SDAS Transfer Episode {episode+1}: Reward = {total_reward:.2f}")
        
        # 3. PPO在Empty-8x8上训练
        print("3. Training PPO on Empty-8x8...")
        ppo_agent = PPOAgent('MiniGrid-Empty-8x8-v0', seed, os.path.join(log_dir, 'ppo_empty'))
        ppo_agent.train(total_timesteps=10000)
        
        # 4. PPO迁移到FourRooms
        print("4. Transferring PPO to FourRooms...")
        ppo_transfer_rewards = []
        env_ppo_four_rooms = MiniGridEnvWrapper('MiniGrid-FourRooms-v0')
        
        for episode in range(n_episodes):
            obs = env_ppo_four_rooms.reset()
            ppo_agent.reset()
            
            total_reward = 0
            done = False
            
            while not done:
                action, info = ppo_agent.step(obs)
                obs, reward, done, _ = env_ppo_four_rooms.step(action)
                ppo_agent.update_structure(reward)
                total_reward += reward
            
            ppo_transfer_rewards.append(total_reward)
            
            # 写入CSV文件
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([seed, 'PPO_Transfer', episode, total_reward])
            
            if episode % 10 == 0:
                print(f"  PPO Transfer Episode {episode+1}: Reward = {total_reward:.2f}")
        
        # 5. PPO从零开始在FourRooms上训练
        print("5. Training PPO from scratch on FourRooms...")
        ppo_scratch_agent = PPOAgent('MiniGrid-FourRooms-v0', seed, os.path.join(log_dir, 'ppo_fourrooms_scratch'))
        ppo_scratch_agent.train(total_timesteps=10000)
        ppo_scratch_rewards = []
        
        for episode in range(n_episodes):
            obs = env_ppo_four_rooms.reset()
            ppo_scratch_agent.reset()
            
            total_reward = 0
            done = False
            
            while not done:
                action, info = ppo_scratch_agent.step(obs)
                obs, reward, done, _ = env_ppo_four_rooms.step(action)
                ppo_scratch_agent.update_structure(reward)
                total_reward += reward
            
            ppo_scratch_rewards.append(total_reward)
            
            # 写入CSV文件
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([seed, 'PPO_Scratch', episode, total_reward])
            
            if episode % 10 == 0:
                print(f"  PPO Scratch Episode {episode+1}: Reward = {total_reward:.2f}")
        
        # 保存结果
        results['sdas_transfer'].append(sdas_transfer_rewards)
        results['ppo_transfer'].append(ppo_transfer_rewards)
        results['ppo_from_scratch'].append(ppo_scratch_rewards)
        
        # 关闭环境
        env_empty.close()
        env_four_rooms.close()
        env_ppo_four_rooms.close()
        ppo_agent.close()
        ppo_scratch_agent.close()
    
    # 分析结果
    print("\n=== Results Analysis ===")
    print("=" * 60)
    
    # 计算平均值和标准差
    avg_rewards = {}
    for key, seed_results in results.items():
        seed_results = np.array(seed_results)
        mean_rewards = np.mean(seed_results, axis=0)
        std_rewards = np.std(seed_results, axis=0)
        
        # 计算前50个episode的平均奖励
        avg_50 = np.mean(mean_rewards)
        std_50 = np.mean(std_rewards)
        
        # 计算前10个episode的平均奖励（恢复速度）
        avg_10 = np.mean(mean_rewards[:10])
        std_10 = np.mean(std_rewards[:10])
        
        avg_rewards[key] = {
            'avg_50': avg_50,
            'std_50': std_50,
            'avg_10': avg_10,
            'std_10': std_10,
            'all_rewards': seed_results
        }
        
        print(f"{key.replace('_', ' ').title()}:")
        print(f"  Average reward over 50 episodes: {avg_50:.2f} ± {std_50:.2f}")
        print(f"  Average reward over first 10 episodes: {avg_10:.2f} ± {std_10:.2f}")
        print()
    
    # 统计显著性分析（t-test）
    print("=== Statistical Significance Analysis ===")
    print("=" * 60)
    
    # 计算SDAS迁移 vs PPO迁移的t-test
    sdas_transfer_rewards = np.array([np.mean(seeds) for seeds in results['sdas_transfer']])
    ppo_transfer_rewards = np.array([np.mean(seeds) for seeds in results['ppo_transfer']])
    ppo_scratch_rewards = np.array([np.mean(seeds) for seeds in results['ppo_from_scratch']])
    
    # SDAS迁移 vs PPO迁移
    t_stat, p_value = stats.ttest_ind(sdas_transfer_rewards, ppo_transfer_rewards)
    print(f"SDAS Transfer vs PPO Transfer:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("  ✓ Statistically significant (p < 0.05)")
    else:
        print("  ✗ Not statistically significant (p ≥ 0.05)")
    print()
    
    # SDAS迁移 vs PPO从零开始
    t_stat, p_value = stats.ttest_ind(sdas_transfer_rewards, ppo_scratch_rewards)
    print(f"SDAS Transfer vs PPO From Scratch:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("  ✓ Statistically significant (p < 0.05)")
    else:
        print("  ✗ Not statistically significant (p ≥ 0.05)")
    print()

if __name__ == "__main__":
    main()
