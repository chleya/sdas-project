"""
SDAS与MiniGrid环境的集成
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import random
from src.structure_pool import StructurePool


@dataclass
class MiniGridConfig:
    """SDAS MiniGrid配置"""
    # 结构池
    max_structures: int = 8  # 进一步减少结构数量，提高质量
    decay_rate: float = 0.03  # 进一步减慢衰减速度
    create_threshold: float = 0.85  # 进一步提高创建阈值，减少结构创建
    
    # 编码器
    encoder_dim: int = 64
    
    # 行动
    n_actions: int = 3  # MiniGrid: 0=前进, 1=左转, 2=右转


class MiniGridEncoder:
    """
    MiniGrid观测编码器
    将MiniGrid的观测编码为latent向量
    """
    
    def __init__(self, output_dim: int):
        self.output_dim = output_dim
    
    def encode(self, obs: dict) -> np.ndarray:
        """编码MiniGrid观测为latent向量"""
        # 提取关键特征
        features = []
        
        # 位置信息（归一化）
        if 'agent_pos' in obs:
            x, y = obs['agent_pos']
            features.extend([x / 7.0, y / 7.0])  # 8x8网格归一化到[0,1]
        elif 'image' in obs:
            # 从图像中提取位置信息（简化）
            # 假设agent在图像中心
            features.extend([0.5, 0.5])
        else:
            features.extend([0.0, 0.0])
        
        # 方向信息
        if 'direction' in obs:
            dir_onehot = np.zeros(4)
            dir_onehot[obs['direction']] = 1.0
            features.extend(dir_onehot.tolist())
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # 目标信息
        if 'mission' in obs:
            # 简单的任务编码
            mission = obs['mission'].lower()
            if 'get' in mission or 'pick' in mission:
                features.append(1.0)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # 填充到固定长度
        while len(features) < 16:  # 基础特征数
            features.append(0.0)
        
        # 线性投影
        features = np.array(features)
        latent = np.random.randn(self.output_dim, len(features)) @ features
        latent = np.maximum(0, latent)  # ReLU
        
        # 归一化
        norm = np.linalg.norm(latent)
        if norm > 0:
            latent = latent / norm
        
        return latent


class MiniGridActionPolicy:
    """
    MiniGrid行动策略
    基于结构状态选择动作
    """
    
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
    
    def select_action(
        self, 
        obs: dict, 
        structure_signal: dict,
        latent: np.ndarray,
        epsilon: float = 0.1
    ) -> int:
        """
        基于观测、结构信号选择动作
        """
        # epsilon-greedy
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        
        # 获取活跃结构对象列表
        active = structure_signal.get('active_structures_objects', [])
        if not active:
            # 没有活跃结构，随机探索
            return random.randint(0, self.n_actions - 1)
        
        # 用相似度和效用的乘积作为权重
        q_values = np.zeros(self.n_actions)
        total_weight = 0
        for s, sim_weight in active:  # (Structure对象, 相似度权重)
            # 结合相似度和结构效用作为权重
            weight = sim_weight * (s.utility + 0.1)  # 加0.1避免权重为0
            q_values += weight * s.action_values[:self.n_actions]
            total_weight += weight
        
        if total_weight > 0:
            q_values /= total_weight
            return int(np.argmax(q_values))
        
        # 如果没有权重，随机选择
        return random.randint(0, self.n_actions - 1)


class SDASMiniGridAgent:
    """
    SDAS MiniGrid智能体
    """
    
    def __init__(self, config: MiniGridConfig = None):
        self.config = config or MiniGridConfig()
        
        # 初始化各模块
        self.structure_pool = StructurePool(
            max_structures=self.config.max_structures,
            decay_rate=self.config.decay_rate,
            create_threshold=self.config.create_threshold,
            vector_dim=self.config.encoder_dim
        )
        
        self.encoder = MiniGridEncoder(output_dim=self.config.encoder_dim)
        self.policy = MiniGridActionPolicy(n_actions=self.config.n_actions)
        
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
            # 只更新最后激活的结构，集中奖励信号
            for s in self.structure_pool.structures:
                if s.id == self.last_active_structure_id:
                    # 更新效用
                    delta = reward * 0.3  # 大幅增加学习率
                    s.utility = np.clip(s.utility + delta, 0, 1.0)
                    
                    # 简单Q-learning更新动作Q值
                    s.action_values[self.last_action] += 0.3 * (reward - s.action_values[self.last_action])  # 大幅增加学习率
                    break
    
    def _generate_label(self, obs: dict) -> str:
        """根据观测生成标签"""
        if 'mission' in obs:
            mission = obs['mission'].lower()
            if 'get' in mission or 'pick' in mission:
                return "获取目标"
            elif 'go' in mission or 'reach' in mission:
                return "导航"
        return "探索"
    
    def get_state(self) -> Dict:
        """获取智能体完整状态"""
        return {
            'total_steps': self.total_steps,
            'structure_pool': self.structure_pool.get_state(),
            'history_len': len(self.history)
        }
