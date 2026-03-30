#!/usr/bin/env python3
"""
RL Integration - 强化学习与SDAS集成

将强化学习算法与SDAS结构驱动系统集成，提升智能体性能
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments'))

from sdas import SDASAgent, Config, run_episode
from adaptive_low_rank_pool import AdaptiveLowRankStructurePool, AdaptiveLowRankConfig
from digital_petri_dish import DigitalPetriDish
from complex_petri_dish import ComplexPetriDish


@dataclass
class PPOConfig:
    """PPO 配置"""
    # 网络参数
    hidden_dim: int = 64
    n_layers: int = 2
    activation: str = "relu"
    
    # PPO 超参数
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # 训练参数
    batch_size: int = 32
    n_epochs: int = 4
    learning_rate: float = 3e-4


class PPOActorCritic(nn.Module):
    """PPO 演员-评论家网络"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig):
        super().__init__()
        self.config = config
        
        # 特征提取网络
        layers = []
        input_dim = obs_dim
        
        for _ in range(config.n_layers):
            layers.append(nn.Linear(input_dim, config.hidden_dim))
            if config.activation == "relu":
                layers.append(nn.ReLU())
            elif config.activation == "tanh":
                layers.append(nn.Tanh())
            input_dim = config.hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 演员网络（策略）
        self.actor = nn.Linear(config.hidden_dim, action_dim)
        
        # 评论家网络（价值）
        self.critic = nn.Linear(config.hidden_dim, 1)
    
    def forward(self, obs: torch.Tensor):
        """前向传播"""
        features = self.feature_extractor(obs)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """获取行动和概率"""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, value, entropy


class PPOBuffer:
    """PPO 经验缓冲区"""
    
    def __init__(self, gamma: float, gae_lambda: float):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()
    
    def reset(self):
        """重置缓冲区"""
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def add(self, obs: np.ndarray, action: int, log_prob: float, 
            reward: float, done: bool, value: float):
        """添加经验"""
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
    
    def compute_advantages(self):
        """计算优势函数"""
        # 转换为numpy数组
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # 计算优势和回报
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        last_value = 0
        last_advantage = 0
        
        # 从后向前计算
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            last_advantage = advantages[t]
            returns[t] = advantages[t] + values[t]
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return (
            np.array(self.observations),
            np.array(self.actions),
            np.array(self.log_probs),
            returns,
            advantages
        )


class RLSDASAgent:
    """强化学习增强的SDAS智能体"""
    
    def __init__(self, config: Config = None, rl_config: PPOConfig = None):
        self.config = config or Config()
        self.rl_config = rl_config or PPOConfig()
        
        # 初始化SDAS智能体
        self.sdas_agent = SDASAgent(self.config)
        
        # 环境观测维度（ComplexPetriDish）
        self.obs_dim = 28
        # 行动维度
        self.action_dim = 5
        
        # 初始化PPO网络
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PPOActorCritic(self.obs_dim, self.action_dim, self.rl_config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.rl_config.learning_rate)
        
        # 经验缓冲区
        self.buffer = PPOBuffer(self.rl_config.gamma, self.rl_config.gae_lambda)
        
        # 训练状态
        self.step_count = 0
        self.episode_count = 0
    
    def reset(self):
        """重置智能体"""
        self.sdas_agent.reset()
        self.buffer.reset()
    
    def extract_obs_features(self, obs: dict) -> np.ndarray:
        """从观测字典提取特征向量"""
        # 提取ComplexPetriDish的所有特征
        features = np.array([
            # 智能体位置
            obs.get('agent_x', 0),
            obs.get('agent_y', 0),
            # 最近能量
            obs.get('nearest_energy_x', 0),
            obs.get('nearest_energy_y', 0),
            obs.get('energy_dist', 0),
            # 能量计数
            obs.get('n_energy_remaining', 0) / 10.0,
            # 熟悉度
            obs.get('familiarity', 0),
            # 障碍物附近
            float(obs.get('obstacle_nearby', False)),
            # 最近普通能量
            obs.get('nearest_normal_energy_x', 0),
            obs.get('nearest_normal_energy_y', 0),
            # 最近高能量
            obs.get('nearest_high_energy_x', 0),
            obs.get('nearest_high_energy_y', 0),
            # 最近稀有能量
            obs.get('nearest_rare_energy_x', 0),
            obs.get('nearest_rare_energy_y', 0),
            # 能量距离
            obs.get('normal_energy_dist', 0),
            obs.get('high_energy_dist', 0),
            obs.get('rare_energy_dist', 0),
            # 能量计数
            obs.get('n_normal_energy', 0) / 5.0,
            obs.get('n_high_energy', 0) / 3.0,
            obs.get('n_rare_energy', 0) / 2.0,
            # 动态障碍物
            obs.get('nearest_dynamic_obstacle_dist', 0),
            obs.get('n_dynamic_obstacles', 0) / 10.0,
            # 目标
            obs.get('goal_x', 0),
            obs.get('goal_y', 0),
            obs.get('goal_dist', 0),
            # 时间
            obs.get('time_step', 0),
            # 事件
            obs.get('n_events', 0)
        ])
        
        # 处理NaN值
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 确保值在合理范围内
        features = np.clip(features, -1.0, 1.0)
        
        # 确保维度正确
        if len(features) < self.obs_dim:
            features = np.pad(features, (0, self.obs_dim - len(features)))
        else:
            features = features[:self.obs_dim]
        
        return features
    
    def step(self, obs: dict) -> Tuple[int, dict]:
        """智能体一步"""
        # 1. 提取观测特征
        obs_features = self.extract_obs_features(obs)
        
        # 2. SDAS结构池处理
        latent = self.sdas_agent.encoder.encode(obs)
        structure_signal = self.sdas_agent.structure_pool.observe(latent, 
                                                               label=self.sdas_agent._generate_label(obs))
        
        # 3. RL策略选择行动
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_features).unsqueeze(0).to(self.device)
            action, log_prob, value, entropy = self.model.get_action(obs_tensor)
            action = action.item()
            log_prob = log_prob.item()
            value = value.item()
        
        # 4. 记录信息
        self.step_count += 1
        
        info = {
            'rl_action': action,
            'structure_event': structure_signal['event'],
            'active_structures': structure_signal.get('active_structures', []),
            'value': value,
            'log_prob': log_prob,
            'entropy': entropy.item()
        }
        
        # 保存到缓冲区（等待奖励）
        self.buffer.add(obs_features, action, log_prob, 0.0, False, value)
        
        return action, info
    
    def update(self, reward: float, done: bool):
        """更新智能体"""
        # 更新SDAS结构
        self.sdas_agent.update_structure(reward)
        
        # 更新缓冲区
        if self.buffer.values:
            last_value = self.buffer.values[-1]
        else:
            last_value = 0.0
        
        # 更新最后一个经验的奖励和完成状态
        if self.buffer.rewards:
            self.buffer.rewards[-1] = reward
            self.buffer.dones[-1] = done
        
        # 如果回合结束，添加最终价值
        if done:
            self.episode_count += 1
            # 添加最终状态的值（0）
            if self.buffer.observations:
                obs_features = self.buffer.observations[-1]
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs_features).unsqueeze(0).to(self.device)
                    _, value = self.model(obs_tensor)
                    value = value.item()
                self.buffer.values[-1] = value
    
    def train(self):
        """训练PPO网络"""
        if len(self.buffer.observations) < self.rl_config.batch_size:
            return
        
        # 计算优势和回报
        observations, actions, old_log_probs, returns, advantages = self.buffer.compute_advantages()
        
        # 转换为张量
        observations = torch.FloatTensor(observations).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # 训练多个epoch
        for epoch in range(self.rl_config.n_epochs):
            # 随机采样批次
            indices = torch.randperm(len(observations))
            for i in range(0, len(observations), self.rl_config.batch_size):
                batch_indices = indices[i:i+self.rl_config.batch_size]
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # 前向传播
                logits, values = self.model(batch_obs)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.rl_config.clip_epsilon, 
                                   1.0 + self.rl_config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = nn.functional.mse_loss(values.squeeze(), batch_returns)
                
                # 总损失
                total_loss = (policy_loss + 
                             self.rl_config.value_coef * value_loss - 
                             self.rl_config.entropy_coef * entropy)
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.rl_config.max_grad_norm)
                self.optimizer.step()
        
        # 重置缓冲区
        self.buffer.reset()
    
    def save(self, path: str):
        """保存模型"""
        torch.save(self.model.state_dict(), f"{path}_rl_model.pt")
        self.sdas_agent.save(path)
    
    def load(self, path: str):
        """加载模型"""
        self.model.load_state_dict(torch.load(f"{path}_rl_model.pt"))
        self.sdas_agent = SDASAgent.load(path)


def run_rl_episode(env: ComplexPetriDish, agent: RLSDASAgent, 
                  max_steps: int = 200) -> Dict:
    """运行一个强化学习回合"""
    obs = env._reset()
    agent.reset()
    
    total_reward = 0
    structure_events = []
    rl_actions = []
    
    for step in range(max_steps):
        # 智能体选择行动
        action, info = agent.step(obs)
        
        # 环境执行
        obs, reward, done = env.step(action)
        
        # 更新智能体
        agent.update(reward, done)
        
        total_reward += reward
        structure_events.append(info['structure_event'])
        rl_actions.append(info['rl_action'])
        
        if done:
            break
    
    # 训练
    agent.train()
    
    return {
        'steps': step + 1,
        'total_reward': total_reward,
        'structure_events': structure_events,
        'rl_actions': rl_actions,
        'episode_count': agent.episode_count
    }


def train_rl_sdas():
    """训练强化学习增强的SDAS智能体"""
    print("=" * 70)
    print("Training RL-Enhanced SDAS Agent")
    print("=" * 70)
    
    # 创建环境
    env = ComplexPetriDish(
        width=15,
        height=15,
        n_static_obstacles=20,
        n_dynamic_obstacles=3,
        n_energy_sources=5
    )
    
    # 创建智能体
    config = Config()
    rl_config = PPOConfig()
    agent = RLSDASAgent(config, rl_config)
    
    # 训练参数
    n_episodes = 1000
    max_steps_per_episode = 200
    
    # 记录
    rewards = []
    steps = []
    
    for episode in range(n_episodes):
        result = run_rl_episode(env, agent, max_steps_per_episode)
        
        rewards.append(result['total_reward'])
        steps.append(result['steps'])
        
        # 每10个回合打印一次
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            avg_steps = np.mean(steps[-10:])
            print(f"Episode {episode+1}/{n_episodes}: "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Avg Steps: {avg_steps:.1f}")
        
        # 每100个回合保存一次
        if (episode + 1) % 100 == 0:
            agent.save(f"rl_sdas_ep{episode+1}")
            print(f"Saved model at episode {episode+1}")
    
    # 保存最终模型
    agent.save("rl_sdas_final")
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    return agent


def test_rl_sdas():
    """测试强化学习增强的SDAS智能体"""
    print("=" * 70)
    print("Testing RL-Enhanced SDAS Agent")
    print("=" * 70)
    
    # 创建环境
    env = ComplexPetriDish(
        width=15,
        height=15,
        n_static_obstacles=20,
        n_dynamic_obstacles=3,
        n_energy_sources=5
    )
    
    # 创建智能体
    agent = RLSDASAgent()
    try:
        agent.load("rl_sdas_final")
        print("Loaded trained model")
    except:
        print("No trained model found, using random model")
    
    # 测试
    n_episodes = 10
    total_rewards = []
    
    for episode in range(n_episodes):
        result = run_rl_episode(env, agent, max_steps=200)
        total_rewards.append(result['total_reward'])
        print(f"Episode {episode+1}: Reward = {result['total_reward']:.2f}, Steps = {result['steps']}")
    
    avg_reward = np.mean(total_rewards)
    print(f"\nAverage Reward: {avg_reward:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    # 训练
    # train_rl_sdas()
    
    # 测试
    test_rl_sdas()
