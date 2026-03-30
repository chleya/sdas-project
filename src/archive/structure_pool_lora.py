"""
Structure Pool with Low-Rank Prototypes
低秩原型结构池 - EGGROLL 风格的高效参数优化

核心思想：
- 每个结构的原型不再直接存储完整向量
- 而是存储为：prototype = base_vector + B @ a
- 其中 B 是 (vector_dim x rank) 矩阵，a 是 (rank,) 向量
- 这样每个结构只需存储 rank 个参数而非 vector_dim 个
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import json
import copy


@dataclass
class LowRankStructure:
    """
    低秩结构单元
    使用简化的低秩分解：prototype = base + B @ a
    """
    id: int
    base_vector: np.ndarray      # 共享的基础向量
    B: np.ndarray                # 低秩矩阵 (vector_dim x rank)
    a: np.ndarray                # 低秩系数向量 (rank,)
    label: str
    
    age: int = 0
    utility: float = 0.0
    surprise_history: List[float] = field(default_factory=list)
    mean_similarity: float = 0.0
    
    def get_prototype(self) -> np.ndarray:
        """计算完整原型向量：base + B @ a"""
        return self.base_vector + self.B @ self.a
    
    def get_lora_params(self) -> np.ndarray:
        """获取低秩参数的展平向量"""
        return np.concatenate([self.B.flatten(), self.a.flatten()])
    
    def set_lora_params(self, params: np.ndarray, vector_dim: int, rank: int):
        """从展平向量设置低秩参数"""
        B_size = vector_dim * rank
        self.B = params[:B_size].reshape(vector_dim, rank)
        self.a = params[B_size:B_size + rank]
    
    def vigor(self) -> float:
        """活力值 = utility * (1 - decay_rate)^age"""
        return self.utility * (0.92 ** self.age)


class LowRankStructurePool:
    """
    低秩结构池管理器
    所有结构共享一个基础向量，通过低秩扰动来区分
    """
    
    def __init__(
        self,
        vector_dim: int = 64,
        rank: int = 4,                    # 低秩维度
        max_structures: int = 32,
        decay_rate: float = 0.08,
        create_threshold: float = 0.42,
        base_vector: np.ndarray = None
    ):
        self.vector_dim = vector_dim
        self.rank = rank
        self.max_structures = max_structures
        self.decay_rate = decay_rate
        self.create_threshold = create_threshold
        
        # 共享基础向量
        if base_vector is not None:
            self.base_vector = base_vector.copy()
        else:
            self.base_vector = np.random.randn(vector_dim) * 0.1
        
        self.structures: List[LowRankStructure] = []
        self.next_id = 0
        
        # 统计信息
        self.total_observations = 0
        self.creation_events = 0
        self.reinforcement_events = 0
    
    def observe(self, observation: np.ndarray, label: str = "") -> Dict:
        """观察输入，返回结构信号"""
        self.total_observations += 1
        
        if len(self.structures) == 0:
            # 首次观测
            new_struct = self._create_structure(observation, label or "initial")
            self.structures.append(new_struct)
            self.next_id += 1
            self.creation_events += 1
            return self._build_signal("created", [new_struct])
        
        # 计算相似度
        similarities = []
        for s in self.structures:
            prototype = s.get_prototype()
            sim = self._cosine_similarity(observation, prototype)
            similarities.append((s, sim))
        
        best_struct, best_similarity = max(similarities, key=lambda x: x[1])
        novelty = 1.0 - best_similarity
        
        if novelty > self.create_threshold:
            new_struct = self._create_structure(observation, label or f"structure_{self.next_id}")
            self.structures.append(new_struct)
            self.next_id += 1
            self.creation_events += 1
            event = "created"
        elif best_similarity > 0.8:
            self._reinforce_structure(best_struct, observation, best_similarity)
            event = "reinforced"
        else:
            new_struct = self._branch_structure(best_struct, observation, 
                                               label or f"branch_{self.next_id}")
            self.structures.append(new_struct)
            self.next_id += 1
            self.creation_events += 1
            event = "branched"
        
        self._decay_all()
        self._prune()
        
        active = sorted(self.structures, key=lambda s: s.vigor(), reverse=True)[:5]
        return self._build_signal(event, active)
    
    def _create_structure(self, observation: np.ndarray, label: str) -> LowRankStructure:
        """创建新结构"""
        # 计算目标扰动
        target_delta = observation - self.base_vector
        
        # 使用随机低秩矩阵，使得 B @ a ≈ target_delta
        B = np.random.randn(self.vector_dim, self.rank) * 0.1
        
        # 通过最小二乘求解最优 a
        a = np.linalg.lstsq(B, target_delta, rcond=None)[0]
        
        return LowRankStructure(
            id=self.next_id,
            base_vector=self.base_vector,
            B=B,
            a=a,
            label=label,
            utility=1.0,
            age=0
        )
    
    def _reinforce_structure(self, struct: LowRankStructure, 
                            observation: np.ndarray, similarity: float):
        """强化结构"""
        lr = 0.05
        
        prototype = struct.get_prototype()
        error = observation - prototype
        
        # 梯度更新：让 B @ a 更接近目标
        # d(B @ a)/dB = a, d(B @ a)/da = B
        grad_B = np.outer(error, struct.a)
        grad_a = struct.B.T @ error
        
        struct.B += lr * grad_B
        struct.a += lr * grad_a
        
        struct.utility = min(1.0, struct.utility + 0.05)
        struct.surprise_history.append(1 - similarity)
        self.reinforcement_events += 1
    
    def _branch_structure(self, parent: LowRankStructure, 
                         observation: np.ndarray, label: str) -> LowRankStructure:
        """从父结构分支"""
        # 继承并添加扰动
        B_new = parent.B + np.random.randn(*parent.B.shape) * 0.02
        a_new = parent.a + np.random.randn(*parent.a.shape) * 0.02
        
        # 微调以匹配观测
        target_delta = observation - self.base_vector
        for _ in range(3):
            current = B_new @ a_new
            error = target_delta - current
            B_new += 0.01 * np.outer(error, a_new)
            a_new += 0.01 * B_new.T @ error
        
        return LowRankStructure(
            id=self.next_id,
            base_vector=self.base_vector,
            B=B_new,
            a=a_new,
            label=label,
            utility=0.5,
            age=0
        )
    
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
    
    def _prune(self):
        """剪枝"""
        if len(self.structures) <= self.max_structures:
            return
        
        self.structures.sort(key=lambda s: s.vigor(), reverse=True)
        self.structures = self.structures[:self.max_structures]
    
    def _build_signal(self, event: str, active: List[LowRankStructure]) -> Dict:
        """构建返回信号"""
        active_info = []
        for s in active[:3]:
            active_info.append({
                "id": s.id,
                "label": s.label,
                "vigor": round(s.vigor(), 3),
                "similarity": round(s.mean_similarity, 3),
                "age": s.age
            })
        
        if len(active) > 0 and active[0].vigor() > 0.5:
            focus = f"继续围绕「{active[0].label}」推进"
        elif event == "created":
            focus = f"新主题「{active[0].label}」，确认用户意图"
        else:
            focus = "保持当前方向"
        
        return {
            "novelty": round(1 - active[0].mean_similarity if active else 1.0, 3),
            "best_similarity": round(active[0].mean_similarity if active else 0.0, 3),
            "event": event,
            "active_structures": active_info,
            "recommended_focus": focus,
            "state_summary": {
                "structure_count": len(self.structures),
                "labels": [s.label for s in self.structures],
                "total_params": self.get_total_params(),
                "compression_ratio": self.get_compression_ratio()
            }
        }
    
    def get_total_params(self) -> int:
        """计算总参数量"""
        # 基础向量 + 每个结构的低秩参数 (B: dim*rank + a: rank)
        base_params = self.vector_dim
        lora_params_per_struct = self.vector_dim * self.rank + self.rank
        return base_params + len(self.structures) * lora_params_per_struct
    
    def get_compression_ratio(self) -> float:
        """计算压缩比（全秩参数量 / 低秩参数量）"""
        full_rank_params = self.vector_dim * len(self.structures)
        low_rank_params = self.get_total_params()
        return full_rank_params / low_rank_params if low_rank_params > 0 else 1.0
    
    def get_all_lora_params(self) -> np.ndarray:
        """获取所有低秩参数用于 ES 优化"""
        params = [self.base_vector.copy()]
        for s in self.structures:
            params.append(s.get_lora_params())
        return np.concatenate(params)
    
    def set_all_lora_params(self, params_vector: np.ndarray):
        """从 ES 优化的参数向量恢复"""
        idx = 0
        
        # 更新基础向量
        self.base_vector = params_vector[idx:idx + self.vector_dim].copy()
        idx += self.vector_dim
        
        # 更新每个结构的低秩参数
        for s in self.structures:
            lora_size = self.vector_dim * self.rank + self.rank
            s.set_lora_params(params_vector[idx:idx + lora_size], self.vector_dim, self.rank)
            s.base_vector = self.base_vector.copy()
            idx += lora_size
    
    def get_state(self) -> Dict:
        """获取状态"""
        return {
            "structure_count": len(self.structures),
            "vector_dim": self.vector_dim,
            "rank": self.rank,
            "total_params": self.get_total_params(),
            "compression_ratio": self.get_compression_ratio(),
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


def test_low_rank_pool():
    """测试低秩结构池"""
    print("=" * 60)
    print("Testing Low-Rank Structure Pool")
    print("=" * 60)
    
    vector_dim = 64
    rank = 4
    max_struct = 16
    
    pool = LowRankStructurePool(
        vector_dim=vector_dim,
        rank=rank,
        max_structures=max_struct
    )
    
    print(f"\nConfiguration:")
    print(f"  Vector dim: {vector_dim}")
    print(f"  Rank: {rank}")
    print(f"  Max structures: {max_struct}")
    
    # 对比参数量
    full_params_per_struct = vector_dim
    lowrank_params_per_struct = vector_dim * rank + rank
    
    print(f"\nParameter Efficiency (per structure):")
    print(f"  Full-rank: {full_params_per_struct} params")
    print(f"  Low-rank:  {lowrank_params_per_struct} params")
    print(f"  Compression: {full_params_per_struct / lowrank_params_per_struct:.2f}x")
    
    # 模拟观测
    print("\nSimulating observations...")
    
    obs1 = np.random.randn(vector_dim)
    result1 = pool.observe(obs1, "主题A")
    print(f"\nObs 1: {result1['event']} - {result1['state_summary']['structure_count']} structures")
    print(f"  Compression: {result1['state_summary']['compression_ratio']:.2f}x")
    
    obs2 = obs1 + np.random.randn(vector_dim) * 0.1
    result2 = pool.observe(obs2, "主题A")
    print(f"Obs 2: {result2['event']} - {result2['state_summary']['structure_count']} structures")
    
    obs3 = np.random.randn(vector_dim) * 2
    result3 = pool.observe(obs3, "主题B")
    print(f"Obs 3: {result3['event']} - {result3['state_summary']['structure_count']} structures")
    
    # 最终状态
    state = pool.get_state()
    print(f"\nFinal State:")
    print(f"  Total structures: {state['structure_count']}")
    print(f"  Total params: {state['total_params']}")
    print(f"  Compression: {state['compression_ratio']:.2f}x")
    
    # 测试 ES 参数提取和恢复
    print("\nTesting ES parameter interface...")
    params = pool.get_all_lora_params()
    print(f"  Total ES parameters: {len(params)}")
    
    # 修改参数并恢复
    params += np.random.randn(len(params)) * 0.01
    pool.set_all_lora_params(params)
    print("  Parameter update: OK")
    
    print("\n" + "=" * 60)
    print("Low-Rank Structure Pool Test PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_low_rank_pool()
