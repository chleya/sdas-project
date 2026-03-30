"""
Adaptive Strategy - 高级自适应策略

核心功能：
1. 动态阈值调整
2. 基于趋势的决策
3. 多目标优化
4. 策略评估
"""

import numpy as np
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Tuple, Optional


@dataclass
class AdaptiveStrategyConfig:
    """自适应策略配置"""
    # 基础阈值
    base_complexity_threshold: float = 0.7
    base_utilization_threshold: float = 0.3
    
    # 动态调整参数
    threshold_adaptation_rate: float = 0.05  # 阈值调整速率
    trend_window: int = 50  # 趋势分析窗口
    
    # 调整频率
    min_adaptation_interval: int = 50
    max_adaptation_interval: int = 200
    
    # 多目标权重
    performance_weight: float = 0.4
    efficiency_weight: float = 0.3
    stability_weight: float = 0.3
    
    # 探索参数
    exploration_rate: float = 0.1  # 探索新策略的概率


class AdaptiveStrategy:
    """
    自适应策略管理器
    
    负责：
    1. 动态调整阈值
    2. 分析历史趋势
    3. 做出最优决策
    4. 评估策略效果
    """
    
    def __init__(self, config: AdaptiveStrategyConfig = None):
        self.config = config or AdaptiveStrategyConfig()
        
        # 当前阈值
        self.complexity_threshold = self.config.base_complexity_threshold
        self.utilization_threshold = self.config.base_utilization_threshold
        
        # 历史数据
        self.difficulty_history: deque = deque(
            maxlen=self.config.trend_window
        )
        self.utilization_history: deque = deque(
            maxlen=self.config.trend_window
        )
        self.performance_history: deque = deque(
            maxlen=self.config.trend_window
        )
        self.action_history: deque = deque(
            maxlen=self.config.trend_window
        )
        
        # 策略评估
        self.reward_history: List[float] = []
        self.action_count: Dict[str, int] = {
            'grow': 0,
            'shrink': 0,
            'prune': 0,
            'maintain': 0
        }
        
        # 时间步
        self.time_step = 0
        
        print(f"Adaptive Strategy initialized")
        print(f"  Base complexity threshold: {self.config.base_complexity_threshold}")
        print(f"  Base utilization threshold: {self.config.base_utilization_threshold}")
        print(f"  Trend window: {self.config.trend_window}")
    
    def update_history(self, difficulty: float, utilization: float, 
                      performance: float, action: str):
        """更新历史数据"""
        self.difficulty_history.append(difficulty)
        self.utilization_history.append(utilization)
        self.performance_history.append(performance)
        self.action_history.append(action)
        
        self.action_count[action] += 1
        self.time_step += 1
    
    def analyze_trends(self) -> Dict[str, float]:
        """分析历史趋势"""
        trends = {}
        
        # 难度趋势
        if len(self.difficulty_history) >= 5:
            difficulty_array = np.array(self.difficulty_history)
            trends['difficulty_trend'] = np.polyfit(
                range(len(difficulty_array)),
                difficulty_array,
                1
            )[0]
        else:
            trends['difficulty_trend'] = 0.0
        
        # 利用率趋势
        if len(self.utilization_history) >= 5:
            utilization_array = np.array(self.utilization_history)
            trends['utilization_trend'] = np.polyfit(
                range(len(utilization_array)),
                utilization_array,
                1
            )[0]
        else:
            trends['utilization_trend'] = 0.0
        
        # 性能趋势
        if len(self.performance_history) >= 5:
            performance_array = np.array(self.performance_history)
            trends['performance_trend'] = np.polyfit(
                range(len(performance_array)),
                performance_array,
                1
            )[0]
        else:
            trends['performance_trend'] = 0.0
        
        # 近期动作分布
        recent_actions = list(self.action_history)[-10:]
        action_counts = {}
        for action in ['grow', 'shrink', 'prune', 'maintain']:
            action_counts[action] = recent_actions.count(action) / max(1, len(recent_actions))
        trends['action_distribution'] = action_counts
        
        return trends
    
    def adjust_thresholds(self, trends: Dict[str, float]):
        """动态调整阈值"""
        # 基于难度趋势调整复杂度阈值
        difficulty_trend = trends['difficulty_trend']
        if difficulty_trend > 0.01:  # 难度上升
            self.complexity_threshold = min(
                0.9, 
                self.complexity_threshold + self.config.threshold_adaptation_rate
            )
        elif difficulty_trend < -0.01:  # 难度下降
            self.complexity_threshold = max(
                0.5, 
                self.complexity_threshold - self.config.threshold_adaptation_rate
            )
        
        # 基于利用率趋势调整利用率阈值
        utilization_trend = trends['utilization_trend']
        if utilization_trend < -0.01:  # 利用率下降
            self.utilization_threshold = max(
                0.1, 
                self.utilization_threshold - self.config.threshold_adaptation_rate
            )
        elif utilization_trend > 0.01:  # 利用率上升
            self.utilization_threshold = min(
                0.5, 
                self.utilization_threshold + self.config.threshold_adaptation_rate
            )
    
    def calculate_adaptation_interval(self, trends: Dict[str, float]) -> int:
        """计算调整间隔"""
        # 基于任务复杂性和稳定性计算间隔
        difficulty_std = np.std(self.difficulty_history) if self.difficulty_history else 0
        performance_std = np.std(self.performance_history) if self.performance_history else 0
        
        # 标准差越大，调整间隔越小
        instability = difficulty_std + performance_std
        
        interval = int(
            self.config.min_adaptation_interval + 
            (self.config.max_adaptation_interval - self.config.min_adaptation_interval) * 
            max(0, 1 - instability * 2)
        )
        
        return max(self.config.min_adaptation_interval, interval)
    
    def decide_action(self, difficulty: float, utilization: float, 
                    n_structures: int, min_structures: int, 
                    max_structures: int) -> str:
        """
        决定最优动作
        
        Returns:
            'grow', 'shrink', 'prune', 'maintain'
        """
        # 分析趋势
        trends = self.analyze_trends()
        
        # 调整阈值
        self.adjust_thresholds(trends)
        
        # 多目标评分
        scores = self._calculate_action_scores(
            difficulty, utilization, n_structures, 
            min_structures, max_structures, trends
        )
        
        # 探索机制
        if np.random.random() < self.config.exploration_rate:
            # 探索次优动作
            sorted_actions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_actions) > 1:
                return sorted_actions[1][0]
        
        # 选择最优动作
        best_action = max(scores.items(), key=lambda x: x[1])[0]
        
        return best_action
    
    def _calculate_action_scores(self, difficulty: float, utilization: float, 
                               n_structures: int, min_structures: int, 
                               max_structures: int, trends: Dict[str, float]) -> Dict[str, float]:
        """计算每个动作的评分"""
        scores = {}
        
        # 计算基础评分
        # grow 评分
        if n_structures < max_structures:
            grow_score = 0.0
            # 难度高时增长更有利
            grow_score += difficulty * 0.5
            # 利用率高时增长更有利
            grow_score += utilization * 0.3
            # 难度上升趋势时增长更有利
            grow_score += max(0, trends['difficulty_trend']) * 0.2
            scores['grow'] = grow_score
        else:
            scores['grow'] = -float('inf')
        
        # shrink 评分
        if n_structures > min_structures:
            shrink_score = 0.0
            # 利用率低时收缩更有利
            shrink_score += (1 - utilization) * 0.5
            # 性能稳定时收缩更有利
            shrink_score += (1 - abs(trends['performance_trend'])) * 0.3
            # 难度下降趋势时收缩更有利
            shrink_score += max(0, -trends['difficulty_trend']) * 0.2
            scores['shrink'] = shrink_score
        else:
            scores['shrink'] = -float('inf')
        
        # prune 评分
        if n_structures > min_structures:
            prune_score = 0.0
            # 利用率低时修剪更有利
            prune_score += (1 - utilization) * 0.4
            # 基于近期动作分布，避免过度修剪
            recent_prune = trends['action_distribution'].get('prune', 0)
            prune_score += (1 - recent_prune) * 0.6
            scores['prune'] = prune_score
        else:
            scores['prune'] = -float('inf')
        
        # maintain 评分
        maintain_score = 0.0
        # 性能稳定时保持更有利
        maintain_score += (1 - abs(trends['performance_trend'])) * 0.4
        # 难度稳定时保持更有利
        maintain_score += (1 - abs(trends['difficulty_trend'])) * 0.3
        # 利用率适中时保持更有利
        maintain_score += (1 - abs(utilization - 0.5)) * 0.3
        scores['maintain'] = maintain_score
        
        return scores
    
    def evaluate_strategy(self) -> Dict[str, float]:
        """评估策略效果"""
        if not self.reward_history:
            return {
                'avg_reward': 0.0,
                'reward_trend': 0.0,
                'action_diversity': 0.0,
                'stability': 0.0
            }
        
        # 平均奖励
        avg_reward = np.mean(self.reward_history)
        
        # 奖励趋势
        if len(self.reward_history) >= 5:
            reward_array = np.array(self.reward_history)
            reward_trend = np.polyfit(
                range(len(reward_array)),
                reward_array,
                1
            )[0]
        else:
            reward_trend = 0.0
        
        # 动作多样性
        total_actions = sum(self.action_count.values())
        if total_actions > 0:
            action_probs = [count / total_actions for count in self.action_count.values()]
            action_diversity = -sum(p * np.log(p) for p in action_probs if p > 0)
        else:
            action_diversity = 0.0
        
        # 稳定性（奖励方差的倒数）
        if len(self.reward_history) > 1:
            stability = 1.0 / np.std(self.reward_history)
        else:
            stability = 0.0
        
        return {
            'avg_reward': avg_reward,
            'reward_trend': reward_trend,
            'action_diversity': action_diversity,
            'stability': stability
        }
    
    def get_state(self) -> Dict:
        """获取策略状态"""
        return {
            'current_thresholds': {
                'complexity': self.complexity_threshold,
                'utilization': self.utilization_threshold
            },
            'trends': self.analyze_trends(),
            'action_stats': self.action_count,
            'evaluation': self.evaluate_strategy()
        }


if __name__ == "__main__":
    # 测试自适应策略
    print("="*70)
    print("Adaptive Strategy Test")
    print("="*70)
    
    strategy = AdaptiveStrategy()
    
    # 模拟不同场景
    scenarios = [
        # (difficulty, utilization, n_structures, performance)
        (0.8, 0.8, 10, 1.5),  # 高难度高利用率
        (0.3, 0.2, 20, -0.5),  # 低难度低利用率
        (0.5, 0.5, 15, 0.5),  # 中等难度中等利用率
        (0.7, 0.6, 8, 1.0),   # 中高难度中高利用率
        (0.2, 0.3, 25, -1.0),  # 低难度低利用率
    ]
    
    for i, (difficulty, utilization, n_structures, performance) in enumerate(scenarios):
        action = strategy.decide_action(
            difficulty, utilization, n_structures, 4, 32
        )
        
        strategy.update_history(difficulty, utilization, performance, action)
        
        print(f"\nScenario {i+1}:")
        print(f"  Difficulty: {difficulty:.2f}, Utilization: {utilization:.2f}")
        print(f"  Structures: {n_structures}, Performance: {performance:.2f}")
        print(f"  Selected action: {action}")
        print(f"  Current thresholds: complexity={strategy.complexity_threshold:.3f}, "
              f"utilization={strategy.utilization_threshold:.3f}")
    
    # 打印策略评估
    print("\n" + "="*70)
    print("Strategy Evaluation")
    print("="*70)
    evaluation = strategy.evaluate_strategy()
    for key, value in evaluation.items():
        print(f"  {key}: {value:.3f}")
    
    print("\nAction distribution:")
    for action, count in strategy.action_count.items():
        print(f"  {action}: {count}")
