"""
Structure Interpretability Analysis
结构可解释性分析模块

功能：
1. 分析每个结构代表什么模式
2. 结构聚类分析
3. 结构激活模式分析
4. 生成结构解释报告
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import json


class StructureAnalyzer:
    """结构分析器"""
    
    def __init__(self, structure_pool):
        self.pool = structure_pool
        self.analysis_results = {}
    
    def analyze_all_structures(self) -> Dict:
        """
        分析所有结构，生成完整报告
        """
        structures = self.pool.structures
        
        if len(structures) == 0:
            return {"error": "No structures to analyze"}
        
        report = {
            "total_structures": len(structures),
            "summary": self._generate_summary(),
            "structure_details": self._analyze_each_structure(),
            "clustering": self._cluster_structures(),
            "activation_patterns": self._analyze_activation_patterns(),
            "lifecycle_analysis": self._analyze_lifecycle(),
            "interpretability_score": self._calculate_interpretability()
        }
        
        self.analysis_results = report
        return report
    
    def _generate_summary(self) -> Dict:
        """生成结构池摘要统计"""
        structures = self.pool.structures
        
        utilities = [s.utility for s in structures]
        ages = [s.age for s in structures]
        vigors = [s.vigor() for s in structures]
        
        return {
            "avg_utility": float(np.mean(utilities)),
            "std_utility": float(np.std(utilities)),
            "avg_age": float(np.mean(ages)),
            "max_age": int(max(ages)),
            "avg_vigor": float(np.mean(vigors)),
            "high_utility_count": sum(1 for u in utilities if u > 0.7),
            "low_utility_count": sum(1 for u in utilities if u < 0.3),
            "labels": list(set(s.label for s in structures))
        }
    
    def _analyze_each_structure(self) -> List[Dict]:
        """分析每个结构的详细信息"""
        structures = self.pool.structures
        details = []
        
        for s in structures:
            detail = {
                "id": s.id,
                "label": s.label,
                "age": s.age,
                "utility": s.utility,
                "vigor": s.vigor(),
                "status": self._classify_structure_status(s),
                "interpretation": self._interpret_structure(s)
            }
            
            # 如果有原型向量，分析其特性
            if hasattr(s, 'prototype'):
                prototype = s.prototype
            elif hasattr(s, 'get_prototype'):
                prototype = s.get_prototype()
            else:
                prototype = None
            
            if prototype is not None:
                detail["prototype_stats"] = {
                    "mean": float(np.mean(prototype)),
                    "std": float(np.std(prototype)),
                    "min": float(np.min(prototype)),
                    "max": float(np.max(prototype)),
                    "sparsity": float(np.mean(np.abs(prototype) < 0.01))
                }
            
            details.append(detail)
        
        return details
    
    def _classify_structure_status(self, structure) -> str:
        """分类结构状态"""
        vigor = structure.vigor()
        age = structure.age
        
        if vigor > 0.7:
            return "active"  # 活跃
        elif vigor > 0.3:
            return "stable"  # 稳定
        elif age < 10:
            return "young"   # 年轻
        else:
            return "decaying"  # 衰减中
    
    def _interpret_structure(self, structure) -> str:
        """解释结构代表什么"""
        label = structure.label
        
        # 基于标签的解释
        interpretations = {
            "避障": "Avoids obstacles, focuses on safety",
            "获取能量": "Seeks energy sources, goal-directed",
            "探索新区域": "Explores unknown areas, curious",
            "探索熟悉区": "Stays in familiar areas, conservative",
            "initial": "Initial exploration pattern"
        }
        
        for key, value in interpretations.items():
            if key in label:
                return value
        
        # 基于效用的解释
        if structure.utility > 0.8:
            return "Highly useful pattern"
        elif structure.utility > 0.5:
            return "Moderately useful pattern"
        else:
            return "Developing pattern"
    
    def _cluster_structures(self) -> Dict:
        """
        对结构进行聚类分析
        """
        structures = self.pool.structures
        n = len(structures)
        
        if n < 2:
            return {"n_clusters": 0, "clusters": []}
        
        # 计算相似度矩阵
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if hasattr(structures[i], 'get_prototype'):
                    p1 = structures[i].get_prototype()
                    p2 = structures[j].get_prototype()
                else:
                    p1 = structures[i].prototype
                    p2 = structures[j].prototype
                
                similarity_matrix[i, j] = self._cosine_similarity(p1, p2)
        
        # 简单的层次聚类
        clusters = self._hierarchical_clustering(similarity_matrix, threshold=0.7)
        
        return {
            "n_clusters": len(clusters),
            "clusters": [
                {
                    "cluster_id": i,
                    "structure_ids": [structures[idx].id for idx in cluster],
                    "labels": [structures[idx].label for idx in cluster],
                    "avg_similarity": float(np.mean([
                        similarity_matrix[i, j] 
                        for i in cluster for j in cluster if i != j
                    ])) if len(cluster) > 1 else 1.0
                }
                for i, cluster in enumerate(clusters)
            ]
        }
    
    def _hierarchical_clustering(self, similarity_matrix: np.ndarray, 
                                 threshold: float = 0.7) -> List[List[int]]:
        """简单的层次聚类"""
        n = len(similarity_matrix)
        clusters = [[i] for i in range(n)]
        
        merged = True
        while merged and len(clusters) > 1:
            merged = False
            new_clusters = []
            skip = set()
            
            for i, c1 in enumerate(clusters):
                if i in skip:
                    continue
                
                for j, c2 in enumerate(clusters[i+1:], i+1):
                    if j in skip:
                        continue
                    
                    # 计算两个簇之间的平均相似度
                    avg_sim = np.mean([
                        similarity_matrix[a, b] 
                        for a in c1 for b in c2
                    ])
                    
                    if avg_sim > threshold:
                        new_clusters.append(c1 + c2)
                        skip.add(i)
                        skip.add(j)
                        merged = True
                        break
                
                if i not in skip:
                    new_clusters.append(c1)
            
            clusters = new_clusters
        
        return clusters
    
    def _analyze_activation_patterns(self) -> Dict:
        """分析结构激活模式"""
        structures = self.pool.structures
        
        # 统计标签分布
        label_counts = defaultdict(int)
        for s in structures:
            label_counts[s.label] += 1
        
        # 统计效用分布
        utility_bins = {
            "high (0.7-1.0)": sum(1 for s in structures if s.utility >= 0.7),
            "medium (0.3-0.7)": sum(1 for s in structures if 0.3 <= s.utility < 0.7),
            "low (0.0-0.3)": sum(1 for s in structures if s.utility < 0.3)
        }
        
        # 年龄分布
        age_bins = {
            "young (0-10)": sum(1 for s in structures if s.age < 10),
            "mature (10-50)": sum(1 for s in structures if 10 <= s.age < 50),
            "old (50+)": sum(1 for s in structures if s.age >= 50)
        }
        
        return {
            "label_distribution": dict(label_counts),
            "utility_distribution": utility_bins,
            "age_distribution": age_bins
        }
    
    def _analyze_lifecycle(self) -> Dict:
        """分析结构生命周期"""
        structures = self.pool.structures
        
        if not structures:
            return {}
        
        # 按创建顺序排序
        sorted_structs = sorted(structures, key=lambda s: s.id)
        
        lifecycle = {
            "creation_order": [
                {
                    "id": s.id,
                    "label": s.label,
                    "current_age": s.age,
                    "current_utility": s.utility,
                    "survived": s.vigor() > 0.1
                }
                for s in sorted_structs
            ],
            "survival_rate": sum(1 for s in structures if s.vigor() > 0.1) / len(structures),
            "avg_lifespan": np.mean([s.age for s in structures])
        }
        
        return lifecycle
    
    def _calculate_interpretability(self) -> float:
        """计算可解释性分数"""
        structures = self.pool.structures
        
        if not structures:
            return 0.0
        
        # 基于多个因素计算
        scores = []
        
        # 1. 标签多样性（有标签比无标签好）
        unique_labels = len(set(s.label for s in structures))
        scores.append(min(unique_labels / len(structures), 1.0))
        
        # 2. 效用分布（不要太集中）
        utilities = [s.utility for s in structures]
        utility_std = np.std(utilities)
        scores.append(min(utility_std * 2, 1.0))  # 标准化
        
        # 3. 年龄分布（有老有新比较好）
        ages = [s.age for s in structures]
        age_range = max(ages) - min(ages)
        scores.append(min(age_range / 100, 1.0))
        
        return float(np.mean(scores))
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        a = a.flatten()
        b = b.flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)
    
    def print_report(self, report: Dict = None):
        """打印分析报告"""
        if report is None:
            report = self.analysis_results
        
        print("\n" + "="*70)
        print("STRUCTURE POOL ANALYSIS REPORT")
        print("="*70)
        
        # 摘要
        print("\n📊 SUMMARY")
        print("-"*70)
        summary = report.get("summary", {})
        print(f"  Total Structures: {report.get('total_structures', 0)}")
        print(f"  Average Utility: {summary.get('avg_utility', 0):.3f}")
        print(f"  Average Age: {summary.get('avg_age', 0):.1f}")
        print(f"  High Utility Structures: {summary.get('high_utility_count', 0)}")
        print(f"  Interpretability Score: {report.get('interpretability_score', 0):.3f}")
        
        # 聚类
        print("\n🔗 CLUSTERING")
        print("-"*70)
        clustering = report.get("clustering", {})
        print(f"  Number of Clusters: {clustering.get('n_clusters', 0)}")
        for cluster in clustering.get("clusters", []):
            print(f"    Cluster {cluster['cluster_id']}: "
                  f"{cluster['structure_ids']} "
                  f"(avg sim: {cluster['avg_similarity']:.3f})")
        
        # 激活模式
        print("\n📈 ACTIVATION PATTERNS")
        print("-"*70)
        patterns = report.get("activation_patterns", {})
        print("  Utility Distribution:")
        for bin_name, count in patterns.get("utility_distribution", {}).items():
            print(f"    {bin_name}: {count}")
        
        print("\n  Age Distribution:")
        for bin_name, count in patterns.get("age_distribution", {}).items():
            print(f"    {bin_name}: {count}")
        
        # 结构详情
        print("\n🔍 STRUCTURE DETAILS")
        print("-"*70)
        for detail in report.get("structure_details", [])[:5]:  # 只显示前5个
            print(f"\n  Structure {detail['id']}: {detail['label']}")
            print(f"    Status: {detail['status']}")
            print(f"    Utility: {detail['utility']:.3f}, "
                  f"Age: {detail['age']}, "
                  f"Vigor: {detail['vigor']:.3f}")
            print(f"    Interpretation: {detail['interpretation']}")
        
        print("\n" + "="*70)


def demo_analysis():
    """演示结构分析"""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
    
    from sdas import SDASAgent, Config
    from digital_petri_dish import DigitalPetriDish
    
    print("="*70)
    print("Structure Analysis Demo")
    print("="*70)
    
    # 创建环境和智能体
    print("\n1. Running agent to generate structures...")
    env = DigitalPetriDish(width=15, height=15, n_obstacles=20, n_energy=5)
    agent = SDASAgent(Config())
    
    # 运行多个回合
    for ep in range(3):
        obs = env._reset()
        agent.reset()
        
        for step in range(100):
            action, info = agent.step(obs)
            obs, reward, done = env.step(action)
            if done:
                break
    
    print(f"   Generated {len(agent.structure_pool.structures)} structures")
    
    # 分析结构池
    print("\n2. Analyzing structures...")
    analyzer = StructureAnalyzer(agent.structure_pool)
    report = analyzer.analyze_all_structures()
    
    # 打印报告
    analyzer.print_report(report)
    
    # 保存报告
    report_file = "structure_analysis_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Full report saved to: {report_file}")


if __name__ == "__main__":
    import os
    demo_analysis()
