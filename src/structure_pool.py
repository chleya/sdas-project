"""
Structure Pool - 竞争性结构池

核心概念：
- 结构单元：可在长期交互中存活并参与行动决策的中间对象
- 32-64个槽位，每个维护 prototype、age、utility、surprise history
- 竞争机制：top-k稀疏激活
- 生灭机制：低utility衰减、高冗余merge、高surprise分裂
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import json


@dataclass
class Structure:
    """结构单元"""
    id: int
    prototype: np.ndarray       # 原型向量 (词频向量)
    label: str                 # 关键词组合
    age: int = 0               # 存活时间步
    utility: float = 0.0       # 效用值
    surprise_history: List[float] = field(default_factory=list)  # 惊讶历史
    activation_trace: List[int] = field(default_factory=list)  # 激活轨迹
    mean_similarity: float = 0.0  # 平均相似度
    action_values: np.ndarray = field(default_factory=lambda: np.zeros(5))  # 每个动作的Q值
    
    def vigor(self) -> float:
        """活力值 = utility（去掉age衰减，让有经验的老结构更有话语权）"""
        return self.utility


class StructurePool:
    """结构池管理器"""
    
    def __init__(
        self,
        max_structures: int = 32,
        decay_rate: float = 0.08,
        create_threshold: float = 0.42,
        vector_dim: int = 128
    ):
        self.max_structures = max_structures
        self.decay_rate = decay_rate
        self.create_threshold = create_threshold
        self.vector_dim = vector_dim
        
        self.structures: List[Structure] = []
        self.next_id = 0
        
    def observe(self, observation: np.ndarray, label: str = "") -> Dict:
        """
        观察输入，返回结构信号
        
        Returns:
            dict with keys: novelty, best_similarity, event, active_structures, 
                           recommended_focus, state_summary
        """
        if len(self.structures) == 0:
            # 首次观测，创建第一个结构
            new_struct = Structure(
                id=self.next_id,
                prototype=observation.copy(),
                label=label or "initial",
                utility=1.0,  # 最高初始utility，让第一个结构完全主导
                age=0
            )
            self.structures.append(new_struct)
            self.next_id += 1
            # 传递(Structure, similarity)元组，首次创建时相似度为1.0
            return self._build_signal("created", [(new_struct, 1.0)])
        
        # 计算与现有结构的相似度
        similarities = []
        for s in self.structures:
            sim = self._cosine_similarity(observation, s.prototype)
            similarities.append((s, sim))
        
        # 找到最佳匹配
        best_struct, best_similarity = max(similarities, key=lambda x: x[1])
        novelty = 1.0 - best_similarity
        
        # 根据novelty决定事件类型
        if novelty > self.create_threshold:
            # 新主题：创建新结构
            new_struct = Structure(
                id=self.next_id,
                prototype=observation.copy(),
                label=label or f"structure_{self.next_id}",
                utility=0.8,  # 非常高的初始utility，让新结构立即产生影响
                age=0
            )
            self.structures.append(new_struct)
            self.next_id += 1
            event = "created"
            # 重新计算相似度，包括新结构
            similarities = []
            for s in self.structures:
                sim = self._cosine_similarity(observation, s.prototype)
                similarities.append((s, sim))
        elif best_similarity > 0.6:
            # 强化现有结构
            best_struct.utility = min(1.0, best_struct.utility + 0.2)  # 大幅增加强化幅度
            best_struct.surprise_history.append(novelty)
            best_struct.mean_similarity = (
                0.8 * best_struct.mean_similarity + 0.2 * best_similarity
            )
            # prototype 快速向当前观测漂移，让结构能够快速适应环境变化
            best_struct.prototype = 0.8 * best_struct.prototype + 0.2 * observation  # 大幅增加学习率
            event = "reinforced"
        else:
            # 分支：从最佳匹配分裂出新结构
            new_struct = Structure(
                id=self.next_id,
                prototype=0.5 * best_struct.prototype + 0.5 * observation,  # 平衡保留和创新
                label=label or f"branch_{self.next_id}",
                utility=0.7,  # 很高的初始utility
                age=0
            )
            self.structures.append(new_struct)
            self.next_id += 1
            event = "branched"
            # 重新计算相似度，包括新结构
            similarities = []
            for s in self.structures:
                sim = self._cosine_similarity(observation, s.prototype)
                similarities.append((s, sim))
        
        # 衰减所有结构
        self._decay_all()
        
        # 剪枝：移除极低utility的结构
        self._prune()
        
        # 按相似度和utility的乘积排序，取前5个，确保只有最有用的结构被激活
        similarities_with_utility = [(s, sim * s.utility) for s, sim in similarities]
        similarities_with_utility.sort(key=lambda x: x[1], reverse=True)
        top_similar = [(s, sim) for s, sim in similarities_with_utility[:5]]
        
        return self._build_signal(event, top_similar)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """余弦相似度"""
        a = a.flatten()
        b = b.flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)
    
    def _decay_all(self):
        """衰减所有结构"""
        for s in self.structures:
            s.age += 1
            # 每步缓慢衰减utility，让不活跃的结构逐渐失去影响力
            s.utility = max(0.0, s.utility - 0.002)
    
    def _prune(self):
        """剪枝：移除低utility结构并合并高冗余结构"""
        if len(self.structures) <= self.max_structures:
            return
        
        # 首先合并高冗余结构
        self._merge_redundant_structures()
        
        # 如果仍然超过最大结构数，按utility排序，保留最高的
        if len(self.structures) > self.max_structures:
            self.structures.sort(key=lambda s: s.utility, reverse=True)
            self.structures = self.structures[:self.max_structures]
    
    def _merge_redundant_structures(self):
        """合并高冗余结构"""
        if len(self.structures) < 2:
            return
        
        # 计算结构之间的相似度
        redundant_pairs = []
        for i in range(len(self.structures)):
            for j in range(i + 1, len(self.structures)):
                sim = self._cosine_similarity(
                    self.structures[i].prototype,
                    self.structures[j].prototype
                )
                if sim > 0.8:  # 相似度阈值
                    redundant_pairs.append((i, j, sim))
        
        # 按相似度排序
        redundant_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # 合并冗余结构
        merged = set()
        for i, j, _ in redundant_pairs:
            if i not in merged and j not in merged:
                # 保留utility高的结构，合并utility低的结构
                if self.structures[i].utility >= self.structures[j].utility:
                    # 合并到i结构
                    self.structures[i].utility = (
                        self.structures[i].utility + self.structures[j].utility
                    ) / 2
                    # 原型向量取平均
                    self.structures[i].prototype = (
                        self.structures[i].prototype + self.structures[j].prototype
                    ) / 2
                    # 动作值取平均
                    self.structures[i].action_values = (
                        self.structures[i].action_values + self.structures[j].action_values
                    ) / 2
                    merged.add(j)
                else:
                    # 合并到j结构
                    self.structures[j].utility = (
                        self.structures[i].utility + self.structures[j].utility
                    ) / 2
                    # 原型向量取平均
                    self.structures[j].prototype = (
                        self.structures[i].prototype + self.structures[j].prototype
                    ) / 2
                    # 动作值取平均
                    self.structures[j].action_values = (
                        self.structures[i].action_values + self.structures[j].action_values
                    ) / 2
                    merged.add(i)
        
        # 移除被合并的结构
        self.structures = [s for i, s in enumerate(self.structures) if i not in merged]
    
    def _build_signal(self, event: str, active: List[Tuple[Structure, float]]) -> Dict:
        """构建返回信号"""
        active_info = []
        active_objects = []  # 存储(Structure对象, 权重)用于策略选择
        for s, sim in active[:3]:
            active_info.append({
                "id": s.id,
                "label": s.label,
                "vigor": round(s.vigor(), 3),
                "similarity": round(sim, 3),
                "age": s.age
            })
            # 使用权重=相似度，而不是vigor
            weight = sim
            active_objects.append((s, weight))
        
        # 推荐聚焦
        if len(active) > 0 and active[0][0].vigor() > 0.5:
            focus = f"继续围绕「{active[0][0].label}」推进"
        elif event == "created" and active:
            focus = f"新主题「{active[0][0].label}」，确认用户意图"
        else:
            focus = "保持当前方向"
        
        # 获取最佳匹配结构的ID
        best_structure_id = active[0][0].id if active else None
        best_similarity = active[0][1] if active else 0.0
        
        return {
            "novelty": round(1 - best_similarity if active else 1.0, 3),
            "best_similarity": round(best_similarity, 3),
            "event": event,
            "active_structures": active_info,
            "active_structures_objects": active_objects,  # 用于策略选择
            "best_structure_id": best_structure_id,  # 最佳匹配结构ID
            "recommended_focus": focus,
            "state_summary": {
                "structure_count": len(self.structures),
                "labels": [s.label for s in self.structures]
            }
        }
    
    def get_state(self) -> Dict:
        """获取当前状态"""
        return {
            "structure_count": len(self.structures),
            "structures": [
                {
                    "id": s.id,
                    "label": s.label,
                    "age": s.age,
                    "utility": round(s.utility, 3),
                    "vigor": round(s.vigor(), 3)
                }
                for s in self.structures
            ]
        }
    
    def save(self, path: str):
        """保存状态"""
        data = {
            "max_structures": self.max_structures,
            "decay_rate": self.decay_rate,
            "next_id": self.next_id,
            "structures": [
                {
                    "id": s.id,
                    "prototype": s.prototype.tolist(),
                    "label": s.label,
                    "age": s.age,
                    "utility": s.utility,
                    "surprise_history": s.surprise_history,
                    "mean_similarity": s.mean_similarity
                }
                for s in self.structures
            ]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'StructurePool':
        """加载状态"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        pool = cls(
            max_structures=data["max_structures"],
            decay_rate=data["decay_rate"]
        )
        pool.next_id = data["next_id"]
        
        for s_data in data["structures"]:
            s = Structure(
                id=s_data["id"],
                prototype=np.array(s_data["prototype"]),
                label=s_data["label"],
                age=s_data["age"],
                utility=s_data["utility"],
                surprise_history=s_data["surprise_history"],
                mean_similarity=s_data["mean_similarity"]
            )
            pool.structures.append(s)
        
        return pool


if __name__ == "__main__":
    # 简单测试
    pool = StructurePool(max_structures=16)
    
    # 模拟观测
    obs1 = np.random.randn(128)
    result1 = pool.observe(obs1, "测试主题A")
    print(f"Observation 1: {result1['event']}")
    print(f"Active: {[s['label'] for s in result1['active_structures']]}")
    print(f"Focus: {result1['recommended_focus']}")
    
    # 继续同一主题
    obs2 = obs1 + np.random.randn(128) * 0.1
    result2 = pool.observe(obs2, "测试主题A")
    print(f"\nObservation 2: {result2['event']}")
    print(f"Active: {[s['label'] for s in result2['active_structures']]}")
    print(f"Focus: {result2['recommended_focus']}")
    
    # 新主题
    obs3 = np.random.randn(128) * 2
    result3 = pool.observe(obs3, "新主题B")
    print(f"\nObservation 3: {result3['event']}")
    print(f"Active: {[s['label'] for s in result3['active_structures']]}")
    print(f"Focus: {result3['recommended_focus']}")
