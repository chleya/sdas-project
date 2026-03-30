"""
Adaptive Low-Rank Structure Pool
自适应低秩结构池 - 结合自适应调整和低秩分解

核心特点：
1. 结构利用率监控和自适应调整
2. 低秩分解的参数效率
3. 动态结构复杂度管理
4. 高效的参数优化
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from structure_pool_lora import LowRankStructure, LowRankStructurePool
from adaptive_strategy import AdaptiveStrategy, AdaptiveStrategyConfig


@dataclass
class LowRankStructureStats:
    """低秩结构统计信息"""
    structure_id: int
    activation_count: int = 0
    total_activation_strength: float = 0.0
    creation_time: int = 0
    last_activation_time: int = 0
    utility_score: float = 0.0  # 综合效用分数
    
    @property
    def avg_activation_strength(self) -> float:
        if self.activation_count == 0:
            return 0.0
        return self.total_activation_strength / self.activation_count
    
    @property
    def usage_frequency(self) -> float:
        """使用频率（基于生命周期）"""
        if self.creation_time == 0:
            return 0.0
        return self.activation_count / max(1, self.last_activation_time - self.creation_time)


@dataclass
class AdaptiveLowRankConfig:
    """自适应低秩配置"""
    # 结构数量范围
    min_structures: int = 4
    max_structures: int = 64
    initial_structures: int = 8
    
    # 低秩参数
    vector_dim: int = 64
    rank: int = 4  # 低秩维度
    
    # 调整阈值
    base_complexity_threshold: float = 0.7
    base_utilization_threshold: float = 0.3
    
    # 调整频率
    adaptation_interval: int = 100
    evaluation_window: int = 50
    
    # 增长/收缩策略
    growth_rate: float = 1.5
    shrink_rate: float = 0.8


class AdaptiveLowRankStructurePool(LowRankStructurePool):
    """
    自适应低秩结构池
    
    继承自低秩结构池，添加自适应调整功能
    """
    
    def __init__(self, config: AdaptiveLowRankConfig = None):
        self.adaptive_config = config or AdaptiveLowRankConfig()
        
        # 初始化低秩结构池
        super().__init__(
            vector_dim=self.adaptive_config.vector_dim,
            rank=self.adaptive_config.rank,
            max_structures=self.adaptive_config.initial_structures
        )
        
        # 结构统计信息
        self.structure_stats: Dict[int, LowRankStructureStats] = {}
        
        # 任务难度历史
        self.difficulty_history: deque = deque(
            maxlen=self.adaptive_config.evaluation_window
        )
        
        # 性能历史
        self.performance_history: deque = deque(
            maxlen=self.adaptive_config.evaluation_window
        )
        
        # 时间步
        self.time_step = 0
        self.last_adaptation_step = 0
        
        # 自适应历史记录
        self.adaptation_history: List[Dict] = []
        
        # 自适应策略
        self.strategy = AdaptiveStrategy(
            AdaptiveStrategyConfig(
                base_complexity_threshold=self.adaptive_config.base_complexity_threshold,
                base_utilization_threshold=self.adaptive_config.base_utilization_threshold
            )
        )
        
        # 初始化结构统计
        self._init_structure_stats()
        
        print(f"Adaptive Low-Rank Structure Pool initialized")
        print(f"  Initial structures: {self.adaptive_config.initial_structures}")
        print(f"  Min/Max: {self.adaptive_config.min_structures}/{self.adaptive_config.max_structures}")
        print(f"  Vector dim: {self.adaptive_config.vector_dim}, Rank: {self.adaptive_config.rank}")
        print(f"  Adaptation interval: {self.adaptive_config.adaptation_interval}")
    
    def _init_structure_stats(self):
        """初始化结构统计"""
        for struct in self.structures:
            self.structure_stats[struct.id] = LowRankStructureStats(
                structure_id=struct.id,
                creation_time=self.time_step
            )
    
    def observe(self, observation: np.ndarray, label: str = "",
                prediction_error: float = 0.0, info_gain: float = 0.0) -> Dict:
        """
        观察输入并更新统计信息
        
        Args:
            observation: 输入观察
            label: 标签
            prediction_error: 预测误差
            info_gain: 信息增益
        
        Returns:
            结构信号
        """
        # 调用父类的 observe 方法
        signal = super().observe(observation, label)
        
        # 从 signal 中提取活跃结构
        active_structures = []
        for info in signal.get('active_structures', []):
            struct_id = info['id']
            for s in self.structures:
                if s.id == struct_id:
                    active_structures.append(s)
                    break
        
        # 更新结构统计
        self._update_structure_stats(active_structures, observation)
        
        # 更新任务难度指标
        self._update_difficulty_metrics(prediction_error, info_gain)
        
        # 检查是否需要自适应调整
        self.time_step += 1
        if self._should_adapt():
            self._adapt_structure_complexity()
        
        # 在 signal 中添加自适应信息
        signal['adaptive_info'] = {
            'n_structures': len(self.structures),
            'utilization': self._calculate_utilization(),
            'time_step': self.time_step,
            'total_params': self.get_total_params(),
            'compression_ratio': self.get_compression_ratio()
        }
        
        return signal
    
    def _update_structure_stats(self, active_structures: List[LowRankStructure], 
                               observation: np.ndarray):
        """更新结构统计信息"""
        for struct in active_structures:
            if struct.id not in self.structure_stats:
                # 新创建的结构
                self.structure_stats[struct.id] = LowRankStructureStats(
                    structure_id=struct.id,
                    creation_time=self.time_step
                )
            
            stats = self.structure_stats[struct.id]
            stats.activation_count += 1
            
            # 计算激活强度（使用余弦相似度）
            prototype = struct.get_prototype()
            similarity = self._cosine_similarity(observation, prototype)
            stats.total_activation_strength += similarity
            stats.last_activation_time = self.time_step
            
            # 更新效用分数
            recency = 1.0 / (1.0 + 0.01 * (self.time_step - stats.last_activation_time))
            frequency = stats.usage_frequency
            strength = stats.avg_activation_strength
            
            stats.utility_score = (
                0.4 * recency +
                0.3 * frequency +
                0.3 * strength
            )
    
    def _update_difficulty_metrics(self, prediction_error: float, info_gain: float):
        """更新任务难度指标"""
        # 计算当前性能
        recent_rewards = list(self.performance_history)[-10:] if self.performance_history else [0]
        reward_variance = np.var(recent_rewards) if len(recent_rewards) > 1 else 0
        
        # 简单的难度计算
        difficulty = min(1.0, (prediction_error + (1 - info_gain)) / 2.0)
        self.difficulty_history.append(difficulty)
    
    def _should_adapt(self) -> bool:
        """检查是否应该进行自适应调整"""
        steps_since_last = self.time_step - self.last_adaptation_step
        return steps_since_last >= self.adaptive_config.adaptation_interval
    
    def _adapt_structure_complexity(self):
        """自适应调整结构复杂度"""
        self.last_adaptation_step = self.time_step
        
        # 评估当前状态
        avg_difficulty = np.mean(self.difficulty_history) if self.difficulty_history else 0.5
        utilization = self._calculate_utilization()
        n_structures = len(self.structures)
        
        # 计算近期性能
        recent_performance = np.mean(self.performance_history) if self.performance_history else 0.0
        
        # 使用自适应策略决定动作
        action = self.strategy.decide_action(
            difficulty=avg_difficulty,
            utilization=utilization,
            n_structures=n_structures,
            min_structures=self.adaptive_config.min_structures,
            max_structures=self.adaptive_config.max_structures
        )
        
        # 执行调整
        if action == 'grow':
            self._grow_structures()
        elif action == 'shrink':
            self._shrink_structures()
        elif action == 'prune':
            self._prune_low_utility_structures()
        
        # 更新策略历史
        self.strategy.update_history(
            difficulty=avg_difficulty,
            utilization=utilization,
            performance=recent_performance,
            action=action
        )
        
        # 记录历史
        self.adaptation_history.append({
            'time_step': self.time_step,
            'action': action,
            'n_structures': len(self.structures),
            'difficulty': avg_difficulty,
            'utilization': utilization,
            'current_thresholds': {
                'complexity': self.strategy.complexity_threshold,
                'utilization': self.strategy.utilization_threshold
            },
            'params_info': {
                'total_params': self.get_total_params(),
                'compression_ratio': self.get_compression_ratio()
            }
        })
        
        print(f"\n[Adaptive Low-Rank] Step {self.time_step}: {action}")
        print(f"  Structures: {len(self.structures)}, Difficulty: {avg_difficulty:.3f}, Utilization: {utilization:.3f}")
        print(f"  Current thresholds: complexity={self.strategy.complexity_threshold:.3f}, utilization={self.strategy.utilization_threshold:.3f}")
        print(f"  Params: {self.get_total_params()}, Compression: {self.get_compression_ratio():.2f}x")
    
    def _calculate_utilization(self) -> float:
        """计算整体结构利用率"""
        if not self.structures:
            return 0.0
        
        # 计算有效结构比例（近期被激活过的）
        recent_threshold = self.time_step - self.adaptive_config.evaluation_window
        active_recently = sum(
            1 for stats in self.structure_stats.values()
            if stats.last_activation_time >= recent_threshold
        )
        
        # 确保利用率在 0-1 之间
        utilization = active_recently / len(self.structures)
        return np.clip(utilization, 0.0, 1.0)
    
    def _grow_structures(self):
        """增加结构数量"""
        current_n = len(self.structures)
        target_n = min(
            int(current_n * self.adaptive_config.growth_rate),
            self.adaptive_config.max_structures
        )
        n_to_add = target_n - current_n
        
        # 基于现有结构生成新结构
        for _ in range(n_to_add):
            if self.structures:
                # 选择一个基础结构进行分支
                parent = self._select_parent_structure()
                # 生成随机观测作为新结构的目标
                new_observation = self.base_vector + np.random.randn(self.vector_dim) * 0.5
                new_struct = self._branch_structure(
                    parent, new_observation, f"adaptive_{self.next_id}"
                )
                self.structures.append(new_struct)
                
                # 初始化统计
                self.structure_stats[new_struct.id] = LowRankStructureStats(
                    structure_id=new_struct.id,
                    creation_time=self.time_step
                )
                
                self.next_id += 1
        
        # 更新 max_structures
        self.max_structures = min(self.adaptive_config.max_structures, target_n)
        print(f"  Added {n_to_add} structures (total: {len(self.structures)})")
    
    def _shrink_structures(self):
        """减少结构数量"""
        current_n = len(self.structures)
        target_n = max(
            int(current_n * self.adaptive_config.shrink_rate),
            self.adaptive_config.min_structures
        )
        n_to_remove = current_n - target_n
        
        # 按效用分数排序，移除最低效的
        sorted_stats = sorted(
            self.structure_stats.items(),
            key=lambda x: x[1].utility_score
        )
        
        removed = 0
        for struct_id, _ in sorted_stats:
            if removed >= n_to_remove:
                break
            
            # 找到并移除结构
            for i, struct in enumerate(self.structures):
                if struct.id == struct_id:
                    del self.structures[i]
                    del self.structure_stats[struct_id]
                    removed += 1
                    break
        
        # 更新 max_structures
        self.max_structures = max(self.adaptive_config.min_structures, target_n)
        print(f"  Removed {removed} structures (total: {len(self.structures)})")
    
    def _prune_low_utility_structures(self):
        """修剪低效结构"""
        threshold = 0.2
        to_prune = [
            struct_id for struct_id, stats in self.structure_stats.items()
            if stats.utility_score < threshold
        ]
        
        # 确保不低于最小结构数
        max_prune = len(self.structures) - self.adaptive_config.min_structures
        to_prune = to_prune[:max_prune]
        
        for struct_id in to_prune:
            for i, struct in enumerate(self.structures):
                if struct.id == struct_id:
                    del self.structures[i]
                    del self.structure_stats[struct_id]
                    break
        
        print(f"  Pruned {len(to_prune)} low-utility structures (total: {len(self.structures)})")
    
    def _select_parent_structure(self) -> LowRankStructure:
        """选择父结构（基于效用分数的轮盘赌选择）"""
        if not self.structures:
            return None
        
        utilities = np.array([
            self.structure_stats[s.id].utility_score 
            for s in self.structures
        ])
        
        # 添加小值避免除零
        utilities = utilities + 0.01
        probabilities = utilities / utilities.sum()
        
        idx = np.random.choice(len(self.structures), p=probabilities)
        return self.structures[idx]
    
    def get_adaptation_report(self) -> Dict:
        """获取自适应调整报告"""
        return {
            'current_structures': len(self.structures),
            'adaptation_count': len(self.adaptation_history),
            'recent_adaptations': self.adaptation_history[-5:],
            'avg_utilization': self._calculate_utilization(),
            'params_info': {
                'total_params': self.get_total_params(),
                'compression_ratio': self.get_compression_ratio()
            },
            'structure_utilities': {
                s.id: self.structure_stats[s.id].utility_score 
                for s in self.structures
            }
        }


if __name__ == "__main__":
    # 测试自适应低秩结构池
    print("="*70)
    print("Adaptive Low-Rank Structure Pool Test")
    print("="*70)
    
    # 创建配置
    config = AdaptiveLowRankConfig(
        min_structures=4,
        max_structures=32,
        initial_structures=8,
        vector_dim=64,
        rank=4,
        adaptation_interval=50
    )
    
    # 创建自适应低秩结构池
    pool = AdaptiveLowRankStructurePool(config)
    
    # 模拟观察
    print("\nSimulating observations...")
    for i in range(200):
        # 随机输入
        observation = np.random.randn(64)
        observation = observation / np.linalg.norm(observation)
        
        # 模拟难度变化
        if i < 50:
            difficulty = 0.3  # 简单
        elif i < 100:
            difficulty = 0.8  # 困难
        else:
            difficulty = 0.5  # 中等
        
        prediction_error = difficulty * 5.0
        info_gain = 1.0 - difficulty * 0.5
        
        # 观察
        signal = pool.observe(
            observation, 
            prediction_error=prediction_error, 
            info_gain=info_gain
        )
        
        # 模拟性能
        performance = 1.0 - difficulty + np.random.randn() * 0.1
        pool.performance_history.append(performance)
        
        if i % 50 == 0:
            print(f"\nStep {i}:")
            print(f"  Active structures: {len(signal['active_structures'])}")
            print(f"  Total structures: {signal['adaptive_info']['n_structures']}")
            print(f"  Params: {signal['adaptive_info']['total_params']}")
            print(f"  Compression: {signal['adaptive_info']['compression_ratio']:.2f}x")
            print(f"  Difficulty: {difficulty:.3f}")
    
    # 打印报告
    print("\n" + "="*70)
    print("Adaptation Report")
    print("="*70)
    report = pool.get_adaptation_report()
    print(f"Final structure count: {report['current_structures']}")
    print(f"Total adaptations: {report['adaptation_count']}")
    print(f"Average utilization: {report['avg_utilization']:.3f}")
    print(f"Total params: {report['params_info']['total_params']}")
    print(f"Compression ratio: {report['params_info']['compression_ratio']:.2f}x")
    
    if report['recent_adaptations']:
        print("\nRecent adaptations:")
        for adapt in report['recent_adaptations']:
            print(f"  Step {adapt['time_step']}: {adapt['action']} "
                  f"(n={adapt['n_structures']}, diff={adapt['difficulty']:.3f})")
