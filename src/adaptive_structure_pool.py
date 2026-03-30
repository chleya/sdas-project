"""
Adaptive Structure Pool - 自适应结构池

核心功能：
1. 结构利用率监控
2. 任务难度评估
3. 自适应结构增减
4. 复杂度与性能平衡
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from structure_pool import Structure, StructurePool
from adaptive_strategy import AdaptiveStrategy, AdaptiveStrategyConfig


@dataclass
class StructureStats:
    """结构统计信息"""
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
class TaskDifficultyMetrics:
    """任务难度指标"""
    reward_variance: float = 0.0      # 奖励方差
    success_rate: float = 0.0         # 成功率
    episode_length: float = 0.0       # 平均 episode 长度
    prediction_error: float = 0.0     # 世界模型预测误差
    info_gain: float = 0.0            # 信息增益
    
    @property
    def composite_difficulty(self) -> float:
        """综合难度分数（0-1，越高越难）"""
        # 奖励方差高、成功率低、预测误差大 = 难度高
        difficulty = (
            0.3 * min(1.0, self.reward_variance / 10.0) +
            0.3 * (1.0 - self.success_rate) +
            0.2 * min(1.0, self.prediction_error / 5.0) +
            0.2 * (1.0 - min(1.0, self.info_gain))
        )
        return np.clip(difficulty, 0.0, 1.0)


@dataclass
class AdaptiveConfig:
    """自适应配置"""
    # 结构数量范围
    min_structures: int = 4
    max_structures: int = 64
    initial_structures: int = 16
    
    # 调整阈值
    utilization_threshold: float = 0.3  # 利用率低于此值考虑删除
    complexity_threshold: float = 0.7   # 难度高于此值考虑增加结构
    
    # 调整频率
    adaptation_interval: int = 100      # 每多少步调整一次
    evaluation_window: int = 50         # 评估窗口大小
    
    # 增长/收缩策略
    growth_rate: float = 1.5            # 增长时的乘数
    shrink_rate: float = 0.8            # 收缩时的乘数
    
    # 平滑参数
    smoothing_factor: float = 0.9       # 指数平滑因子


class AdaptiveStructurePool(StructurePool):
    """
    自适应结构池
    
    继承自基础 StructurePool，添加自适应调整功能
    """
    
    def __init__(self, config: AdaptiveConfig = None, base_config: dict = None):
        self.adaptive_config = config or AdaptiveConfig()
        
        # 初始化基础结构池
        base_config = base_config or {}
        base_config['max_structures'] = self.adaptive_config.initial_structures
        super().__init__(**base_config)
        
        # 结构统计信息
        self.structure_stats: Dict[int, StructureStats] = {}
        
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
        self.strategy = AdaptiveStrategy()
        
        # 初始化结构统计
        self._init_structure_stats()
        
        print(f"Adaptive Structure Pool initialized")
        print(f"  Initial structures: {self.adaptive_config.initial_structures}")
        print(f"  Min/Max: {self.adaptive_config.min_structures}/{self.adaptive_config.max_structures}")
        print(f"  Adaptation interval: {self.adaptive_config.adaptation_interval}")
    
    def _init_structure_stats(self):
        """初始化结构统计"""
        for i, struct in enumerate(self.structures):
            self.structure_stats[struct.id] = StructureStats(
                structure_id=struct.id,
                creation_time=self.time_step
            )
    
    def observe(self, encoder_output: np.ndarray, 
                prediction_error: float = 0.0,
                info_gain: float = 0.0,
                label: str = "") -> Dict:
        """
        观察输入并更新统计信息
        
        Args:
            encoder_output: 编码器输出
            prediction_error: 世界模型预测误差
            info_gain: 信息增益
            label: 标签
        
        Returns:
            signal: 包含活跃结构等信息的字典
        """
        # 调用父类的 observe 方法
        signal = super().observe(encoder_output, label)
        
        # 从 signal 中提取活跃结构
        active_structures = []
        for info in signal.get('active_structures', []):
            struct_id = info['id']
            for s in self.structures:
                if s.id == struct_id:
                    active_structures.append(s)
                    break
        
        # 更新结构统计
        self._update_structure_stats(active_structures, encoder_output)
        
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
            'time_step': self.time_step
        }
        
        return signal
    
    def _update_structure_stats(self, active_structures: List[Structure], 
                                 encoder_output: np.ndarray):
        """更新结构统计信息"""
        for struct in active_structures:
            if struct.id not in self.structure_stats:
                # 新创建的结构
                self.structure_stats[struct.id] = StructureStats(
                    structure_id=struct.id,
                    creation_time=self.time_step
                )
            
            stats = self.structure_stats[struct.id]
            stats.activation_count += 1
            
            # 计算激活强度（使用余弦相似度）
            similarity = self._cosine_similarity(
                struct.prototype, encoder_output
            )
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
        
        metrics = TaskDifficultyMetrics(
            reward_variance=reward_variance,
            prediction_error=prediction_error,
            info_gain=info_gain
        )
        
        self.difficulty_history.append(metrics.composite_difficulty)
    
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
            }
        })
        
        print(f"\n[Adaptive] Step {self.time_step}: {action}")
        print(f"  Structures: {len(self.structures)}, Difficulty: {avg_difficulty:.3f}, Utilization: {utilization:.3f}")
        print(f"  Current thresholds: complexity={self.strategy.complexity_threshold:.3f}, utilization={self.strategy.utilization_threshold:.3f}")
    
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
        
        return active_recently / len(self.structures)
    
    def _decide_adaptation(self, difficulty: float, utilization: float) -> str:
        """
        决定自适应动作
        
        Returns:
            'grow': 增加结构
            'shrink': 减少结构
            'prune': 修剪低效结构
            'maintain': 保持不变
        """
        n_structures = len(self.structures)
        
        # 难度高且利用率也高 -> 增加结构
        if (difficulty > self.adaptive_config.complexity_threshold and 
            utilization > 0.7 and 
            n_structures < self.adaptive_config.max_structures):
            return 'grow'
        
        # 利用率低 -> 减少结构或修剪
        if utilization < self.adaptive_config.utilization_threshold:
            if n_structures > self.adaptive_config.min_structures:
                # 如果有低效结构，优先修剪
                low_utility_count = sum(
                    1 for stats in self.structure_stats.values()
                    if stats.utility_score < 0.2
                )
                if low_utility_count > n_structures * 0.2:  # 超过20%低效
                    return 'prune'
                else:
                    return 'shrink'
        
        return 'maintain'
    
    def _grow_structures(self):
        """增加结构数量"""
        current_n = len(self.structures)
        target_n = min(
            int(current_n * self.adaptive_config.growth_rate),
            self.adaptive_config.max_structures
        )
        n_to_add = target_n - current_n
        
        for _ in range(n_to_add):
            # 基于现有高效结构生成新结构
            parent = self._select_parent_structure()
            new_structure = self._create_child_structure(parent)
            self.structures.append(new_structure)
            
            # 初始化统计
            self.structure_stats[new_structure.id] = StructureStats(
                structure_id=new_structure.id,
                creation_time=self.time_step
            )
        
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
    
    def _select_parent_structure(self) -> Structure:
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
    
    def _create_child_structure(self, parent: Structure) -> Structure:
        """基于父结构创建子结构"""
        # 添加小的随机扰动
        noise = np.random.randn(self.prototype_dim) * 0.1
        child_prototype = parent.prototype + noise
        child_prototype = child_prototype / (np.linalg.norm(child_prototype) + 1e-8)
        
        child = Structure(
            id=self.next_structure_id,
            prototype=child_prototype,
            label=f"struct_{self.next_structure_id}",
            creation_time=self.time_step
        )
        
        self.next_structure_id += 1
        return child
    
    def get_adaptation_report(self) -> Dict:
        """获取自适应调整报告"""
        return {
            'current_structures': len(self.structures),
            'adaptation_count': len(self.adaptation_history),
            'recent_adaptations': self.adaptation_history[-5:],
            'avg_utilization': self._calculate_utilization(),
            'structure_utilities': {
                s.id: self.structure_stats[s.id].utility_score 
                for s in self.structures
            }
        }


if __name__ == "__main__":
    # 测试自适应结构池
    print("="*70)
    print("Adaptive Structure Pool Test")
    print("="*70)
    
    # 创建配置
    config = AdaptiveConfig(
        min_structures=4,
        max_structures=32,
        initial_structures=8,
        adaptation_interval=50
    )
    
    # 创建自适应结构池
    pool = AdaptiveStructurePool(config, {'vector_dim': 32})
    
    # 模拟观察
    print("\nSimulating observations...")
    for i in range(200):
        # 随机输入
        encoder_output = np.random.randn(32)
        encoder_output = encoder_output / np.linalg.norm(encoder_output)
        
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
        signal = pool.observe(encoder_output, prediction_error, info_gain)
        
        # 模拟性能
        performance = 1.0 - difficulty + np.random.randn() * 0.1
        pool.performance_history.append(performance)
        
        if i % 50 == 0:
            print(f"\nStep {i}:")
            print(f"  Active structures: {len(signal['active_structures'])}")
            print(f"  Total structures: {signal['adaptive_info']['n_structures']}")
            print(f"  Difficulty: {difficulty:.3f}")
    
    # 打印报告
    print("\n" + "="*70)
    print("Adaptation Report")
    print("="*70)
    report = pool.get_adaptation_report()
    print(f"Final structure count: {report['current_structures']}")
    print(f"Total adaptations: {report['adaptation_count']}")
    print(f"Average utilization: {report['avg_utilization']:.3f}")
    
    if report['recent_adaptations']:
        print("\nRecent adaptations:")
        for adapt in report['recent_adaptations']:
            print(f"  Step {adapt['time_step']}: {adapt['action']} "
                  f"(n={adapt['n_structures']}, diff={adapt['difficulty']:.3f})")
