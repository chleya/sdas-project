"""
SDAS - Structure-Driven Agent System
结构驱动型智能体系统

核心闭环：
  observation → encode → structure_pool → world_model → action_policy → environment → feedback → structure_update
  
  简化为：
  观测编码 → 结构池(竞争/衰减/分裂) → 世界模型(预测) → 行动选择 → 环境反馈 → 结构更新
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import random

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments'))

from structure_pool import StructurePool
from digital_petri_dish import DigitalPetriDish


@dataclass
class Config:
    """SDAS配置"""
    # 结构池
    max_structures: int = 32
    decay_rate: float = 0.08
    create_threshold: float = 0.65
    
    # 编码器
    encoder_dim: int = 64
    
    # 世界模型
    world_model_dim: int = 32
    horizon: int = 5  # 预测步数
    
    # 行动
    n_actions: int = 5
    
    # 混合目标权重
    lambda_pred: float = 1.0
    lambda_info: float = 0.1
    lambda_emp: float = 0.1
    lambda_sparse: float = 0.05
    lambda_homeo: float = 0.02


class Encoder:
    """
    观测编码器
    将环境观测编码为latent向量
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        # 简化：线性投影 + ReLU
        self.W = np.random.randn(input_dim, output_dim) * 0.1
        self.b = np.zeros(output_dim)
    
    def encode(self, obs: dict) -> np.ndarray:
        """编码观测为latent向量"""
        # 将dict观测展平并归一化
        features = np.array([
            obs.get('agent_x', 0),
            obs.get('agent_y', 0),
            obs.get('nearest_energy_x', 0),
            obs.get('nearest_energy_y', 0),
            obs.get('energy_dist', 0),
            obs.get('n_energy_remaining', 0) / 5.0,
            obs.get('familiarity', 0),
            obs.get('obstacle_nearby', 0),
        ])
        
        # 填充到input_dim
        if len(features) < self.input_dim:
            features = np.pad(features, (0, self.input_dim - len(features)))
        else:
            features = features[:self.input_dim]
        
        # 线性投影
        latent = features @ self.W + self.b
        latent = np.maximum(0, latent)  # ReLU
        
        # 归一化，使余弦相似度更稳定
        norm = np.linalg.norm(latent)
        if norm > 0:
            latent = latent / norm
        
        return latent


class WorldModel:
    """
    世界模型
    Dreamer风格的latent dynamics预测
    预测下一时刻的latent和reward
    """
    
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 64):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 简化的RSSM: latent + action -> next_latent
        self.fc1 = np.random.randn(latent_dim + action_dim, hidden_dim) * 0.1
        self.fc2 = np.random.randn(hidden_dim, latent_dim) * 0.1
        
        # reward预测
        self.fc_reward = np.random.randn(hidden_dim, 1) * 0.1
    
    def predict(self, latent: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
        """预测下一时刻latent和reward"""
        # One-hot编码行动
        a_onehot = np.zeros(self.action_dim)
        a_onehot[action] = 1.0
        
        # 拼接
        x = np.concatenate([latent.flatten(), a_onehot])
        
        # 前向
        h = np.maximum(0, x @ self.fc1)  # ReLU
        next_latent = h @ self.fc2
        next_latent = np.maximum(0, next_latent)  # ReLU
        
        # Reward预测（简化）
        reward_pred = (h @ self.fc_reward).item()
        
        return next_latent, reward_pred
    
    def compute_prediction_loss(self, latent: np.ndarray, action: int, 
                             next_latent_true: np.ndarray) -> float:
        """预测损失"""
        next_latent_pred, _ = self.predict(latent, action)
        return np.mean((next_latent_true - next_latent_pred) ** 2)


class ActionPolicy:
    """
    行动策略
    基于结构状态和世界模型选择行动
    """
    
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
    
    def select_action(
        self, 
        obs: dict, 
        structure_signal: dict,
        world_model: WorldModel,
        latent: np.ndarray,
        epsilon: float = 0.1
    ) -> int:
        """
        基于观测、结构信号、世界模型选择行动
        
        策略优先级：
        1. 如果有活跃结构 → 使用结构的Q值加权选择
        2. 否则 → epsilon-greedy探索
        """
        # epsilon-greedy
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        
        # 获取活跃结构对象列表
        active = structure_signal.get('active_structures_objects', [])
        if not active:
            # 没有活跃结构，随机探索
            return random.randint(0, self.n_actions - 1)
        
        # 用活跃度加权各结构的Q值
        q_values = np.zeros(self.n_actions)
        total_weight = 0
        for s, weight in active:  # (Structure对象, 相似度权重)
            q_values += weight * s.action_values
            total_weight += weight
        
        if total_weight > 0:
            q_values /= total_weight
            return int(np.argmax(q_values))
        
        # 如果没有权重，随机选择
        return random.randint(0, self.n_actions - 1)


class SDASAgent:
    """
    SDAS智能体主体
    
    整合所有模块的完整闭环
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # 初始化各模块
        self.structure_pool = StructurePool(
            max_structures=self.config.max_structures,
            decay_rate=self.config.decay_rate,
            create_threshold=self.config.create_threshold,
            vector_dim=self.config.encoder_dim
        )
        
        self.encoder = Encoder(
            input_dim=8,  # 观测特征数
            output_dim=self.config.encoder_dim
        )
        
        self.world_model = WorldModel(
            latent_dim=self.config.encoder_dim,
            action_dim=self.config.n_actions,
            hidden_dim=self.config.world_model_dim
        )
        
        self.policy = ActionPolicy(n_actions=self.config.n_actions)
        
        # 状态追踪
        self.total_steps = 0
        self.episode_count = 0
        self.total_reward = 0.0
        self.last_active_structure_id = None  # 记录上一步激活的结构ID
        self.last_action = None  # 记录上一步的动作
        
        # 历史记录
        self.history: List[Dict] = []
    
    def reset(self):
        """重置智能体状态"""
        # 不重置total_steps，这样epsilon可以基于全局步数持续衰减
        # self.total_steps = 0
        self.history = []
    
    def step(self, obs: dict) -> Tuple[int, dict]:
        """
        智能体一步
        
        Returns:
            (action, extra_info)
        """
        # 1. 编码观测
        latent = self.encoder.encode(obs)
        
        # 2. 结构池处理
        structure_signal = self.structure_pool.observe(
            latent, 
            label=self._generate_label(obs)
        )
        
        # 记录当前激活的结构ID
        self.last_active_structure_id = structure_signal.get('best_structure_id')
        
        # 3. 行动选择
        action = self.policy.select_action(
            obs=obs,
            structure_signal=structure_signal,
            world_model=self.world_model,
            latent=latent,
            epsilon=max(0.05, 0.3 - self.total_steps * 0.001)  # 衰减探索
        )
        
        # 记录当前动作
        self.last_action = action
        
        # 4. 记录
        self.total_steps += 1
        
        info = {
            'latent_norm': float(np.linalg.norm(latent)),
            'structure_event': structure_signal['event'],
            'active_structures': structure_signal.get('active_structures', []),
            'best_structure_id': self.last_active_structure_id,
            'recommended_focus': structure_signal.get('recommended_focus', ''),
            'exploration_rate': max(0.05, 0.3 - self.total_steps * 0.001)
        }
        
        return action, info
    
    def update_structure(self, reward: float):
        """根据奖励更新结构效用和动作Q值"""
        # 找到上一步激活的结构，给它奖励
        if self.last_active_structure_id is not None and self.last_action is not None:
            for s in self.structure_pool.structures:
                if s.id == self.last_active_structure_id:
                    # 更新效用
                    delta = reward * 0.1
                    s.utility = np.clip(s.utility + delta, 0, 1.0)
                    
                    # 简单Q-learning更新动作Q值
                    s.action_values[self.last_action] += 0.1 * (reward - s.action_values[self.last_action])
                    break
    
    def _generate_label(self, obs: dict) -> str:
        """根据观测生成标签"""
        if obs.get('obstacle_nearby', False):
            return "避障"
        elif obs.get('energy_dist', 1.0) < 0.2:
            return "获取能量"
        elif obs.get('familiarity', 0) > 0.5:
            return "探索熟悉区"
        else:
            return "探索新区域"
    
    def get_state(self) -> Dict:
        """获取智能体完整状态"""
        return {
            'total_steps': self.total_steps,
            'structure_pool': self.structure_pool.get_state(),
            'history_len': len(self.history)
        }
    
    def save(self, path: str):
        """保存状态"""
        self.structure_pool.save(path + '.structures.json')
        # TODO: 保存其他模块
    
    @classmethod
    def load(cls, path: str) -> 'SDASAgent':
        """加载状态"""
        agent = cls()
        agent.structure_pool = StructurePool.load(path + '.structures.json')
        return agent


def run_episode(env: DigitalPetriDish, agent: SDASAgent, max_steps: int = 200) -> Dict:
    """运行一个回合"""
    obs = env._reset()
    agent.reset()
    
    total_reward = 0
    structure_events = []
    
    for step in range(max_steps):
        # Agent选择行动
        action, info = agent.step(obs)
        
        # 环境执行
        obs, reward, done = env.step(action)
        
        # 更新结构
        agent.update_structure(reward)
        
        total_reward += reward
        structure_events.append(info['structure_event'])
        
        if done:
            break
    
    return {
        'steps': step + 1,
        'total_reward': total_reward,
        'structure_events': structure_events,
        'final_state': agent.get_state()
    }


if __name__ == "__main__":
    # 简单测试
    print("=== SDAS Agent Test ===\n")
    
    env = DigitalPetriDish(width=15, height=15, n_obstacles=25, n_energy=5)
    agent = SDASAgent()
    
    # 运行几个回合
    for ep in range(3):
        result = run_episode(env, agent, max_steps=100)
        print(f"Episode {ep+1}:")
        print(f"  Steps: {result['steps']}")
        print(f"  Reward: {result['total_reward']:.2f}")
        print(f"  Structures: {result['final_state']['structure_pool']['structure_count']}")
        print(f"  Events: {result['structure_events'][:5]}...")
        print()
    
    print("SDAS Framework OK!")
