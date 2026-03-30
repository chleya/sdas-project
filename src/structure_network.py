#!/usr/bin/env python3
"""
Structure Network - 结构自组织网络

实现结构之间的智能连接和信息传递，形成自组织网络
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import os


class StructureNetwork:
    """结构自组织网络"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.graph = nx.Graph()
        self.activation_history = []
        self.structure_map = {}
        self.evolution_history = []
        
    def add_structure(self, structure):
        """添加结构到网络"""
        self.graph.add_node(structure.id, structure=structure)
        self.structure_map[structure.id] = structure
        self._record_evolution()
    
    def remove_structure(self, structure_id):
        """从网络中移除结构"""
        if structure_id in self.graph.nodes:
            self.graph.remove_node(structure_id)
            if structure_id in self.structure_map:
                del self.structure_map[structure_id]
            self._record_evolution()
    
    def update_connections(self):
        """更新结构之间的连接"""
        # 计算结构之间的相似度
        nodes = list(self.graph.nodes(data=True))
        
        for i, (id1, data1) in enumerate(nodes):
            structure1 = data1['structure']
            for j, (id2, data2) in enumerate(nodes[i+1:], i+1):
                structure2 = data2['structure']
                
                # 计算相似度
                similarity = self._calculate_similarity(structure1, structure2)
                
                if similarity > 0.6:  # 相似度阈值
                    # 添加或更新连接
                    if self.graph.has_edge(id1, id2):
                        self.graph[id1][id2]['weight'] = similarity
                    else:
                        self.graph.add_edge(id1, id2, weight=similarity)
                else:
                    # 移除低相似度连接
                    if self.graph.has_edge(id1, id2):
                        self.graph.remove_edge(id1, id2)
        
        self._record_evolution()
    
    def _calculate_similarity(self, structure1, structure2):
        """计算两个结构的相似度"""
        try:
            if hasattr(structure1, 'get_prototype') and hasattr(structure2, 'get_prototype'):
                p1 = structure1.get_prototype()
                p2 = structure2.get_prototype()
            else:
                p1 = structure1.prototype
                p2 = structure2.prototype
                
            # 确保向量维度相同
            if len(p1) != len(p2):
                min_len = min(len(p1), len(p2))
                p1 = p1[:min_len]
                p2 = p2[:min_len]
                
            # 计算余弦相似度
            dot_product = np.dot(p1, p2)
            norm_p1 = np.linalg.norm(p1)
            norm_p2 = np.linalg.norm(p2)
            
            if norm_p1 == 0 or norm_p2 == 0:
                return 0.0
                
            return dot_product / (norm_p1 * norm_p2)
        except Exception:
            return 0.0
    
    def activate_structures(self, observation):
        """激活与观测相关的结构"""
        activations = []
        
        for node_id, data in self.graph.nodes(data=True):
            structure = data['structure']
            
            # 计算激活程度
            try:
                if hasattr(structure, 'get_prototype'):
                    prototype = structure.get_prototype()
                else:
                    prototype = structure.prototype
                    
                # 确保向量维度相同
                if len(observation) != len(prototype):
                    min_len = min(len(observation), len(prototype))
                    obs = observation[:min_len]
                    proto = prototype[:min_len]
                else:
                    obs = observation
                    proto = prototype
                    
                activation = np.dot(obs, proto) / (np.linalg.norm(obs) * np.linalg.norm(proto) + 1e-8)
                activations.append((node_id, activation))
            except Exception:
                activations.append((node_id, 0.0))
        
        # 按激活程度排序
        activations.sort(key=lambda x: x[1], reverse=True)
        
        # 记录激活历史
        self.activation_history.append(activations[:3])  # 记录前3个激活的结构
        
        return activations
    
    def get_activated_structures(self, observation, top_k=3):
        """获取激活的结构"""
        activations = self.activate_structures(observation)
        activated_structures = []
        
        for node_id, activation in activations[:top_k]:
            if activation > 0.3:  # 激活阈值
                if node_id in self.structure_map:
                    activated_structures.append((self.structure_map[node_id], activation))
        
        return activated_structures
    
    def propagate_activation(self, initial_activations, max_steps=3):
        """激活传播"""
        activation_values = {node_id: activation for node_id, activation in initial_activations}
        
        for step in range(max_steps):
            new_activations = activation_values.copy()
            
            for node_id, current_activation in activation_values.items():
                if current_activation > 0.1:  # 只有激活值大于阈值的节点才传播
                    neighbors = list(self.graph.neighbors(node_id))
                    for neighbor_id in neighbors:
                        edge_weight = self.graph[node_id][neighbor_id]['weight']
                        propagated_activation = current_activation * edge_weight * 0.5
                        
                        if neighbor_id not in new_activations or propagated_activation > new_activations.get(neighbor_id, -1):
                            new_activations[neighbor_id] = propagated_activation
            
            activation_values = new_activations
        
        # 转换为结构列表
        propagated_structures = []
        for node_id, activation in activation_values.items():
            if activation > 0.1 and node_id in self.structure_map:
                propagated_structures.append((self.structure_map[node_id], activation))
        
        propagated_structures.sort(key=lambda x: x[1], reverse=True)
        return propagated_structures
    
    def _record_evolution(self):
        """记录演化历史"""
        snapshot = {
            'n_structures': len(self.graph.nodes),
            'n_connections': len(self.graph.edges),
            'average_degree': np.mean([d for n, d in self.graph.degree()]) if self.graph.nodes else 0,
            'structures': list(self.structure_map.values())
        }
        self.evolution_history.append(snapshot)
    
    def get_network_stats(self):
        """获取网络统计信息"""
        if not self.graph.nodes:
            return {
                'n_structures': 0,
                'n_connections': 0,
                'average_degree': 0,
                'clustering_coefficient': 0,
                'average_path_length': 0,
                'n_connected_components': 0
            }
        
        # 计算连通组件数
        n_connected_components = nx.number_connected_components(self.graph)
        
        # 计算平均路径长度（只对连通图计算）
        if n_connected_components == 1 and len(self.graph) > 1:
            try:
                average_path_length = nx.average_shortest_path_length(self.graph)
            except:
                average_path_length = 0
        else:
            average_path_length = 0
        
        return {
            'n_structures': len(self.graph.nodes),
            'n_connections': len(self.graph.edges),
            'average_degree': np.mean([d for n, d in self.graph.degree()]),
            'clustering_coefficient': nx.average_clustering(self.graph),
            'average_path_length': average_path_length,
            'n_connected_components': n_connected_components
        }
    
    def get_structure_centrality(self, structure_id):
        """获取结构的网络中心度"""
        if not self.graph.nodes or structure_id not in self.graph.nodes:
            return 0.0
        
        try:
            # 计算度中心性
            degree_centrality = nx.degree_centrality(self.graph)[structure_id]
            
            # 计算介数中心性
            betweenness_centrality = nx.betweenness_centrality(self.graph, weight='weight').get(structure_id, 0.0)
            
            # 计算接近中心性
            try:
                closeness_centrality = nx.closeness_centrality(self.graph, distance='weight').get(structure_id, 0.0)
            except:
                closeness_centrality = 0.0
            
            # 综合中心度
            centrality = (degree_centrality * 0.4 + betweenness_centrality * 0.4 + closeness_centrality * 0.2)
            
            return centrality
        except Exception as e:
            print(f"Error calculating centrality: {e}")
            return 0.0
    
    def visualize(self, filename='structure_network.png', show_activation=False, observation=None):
        """可视化结构网络"""
        if not self.graph.nodes:
            print("No structures to visualize")
            return
        
        # 创建保存目录
        save_dir = 'visualizations'
        os.makedirs(save_dir, exist_ok=True)
        
        # 计算布局
        pos = nx.spring_layout(self.graph, weight='weight', iterations=100)
        
        # 创建图形和轴
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 计算节点大小和颜色
        node_sizes = []
        node_colors = []
        node_labels = {}
        
        if show_activation and observation is not None:
            activations = dict(self.activate_structures(observation))
            for node_id in self.graph.nodes:
                structure = self.structure_map[node_id]
                activation = activations.get(node_id, 0.0)
                node_sizes.append(100 + activation * 1000)
                node_colors.append(activation)
                node_labels[node_id] = f"S{node_id}\n{activation:.2f}"
        else:
            for node_id in self.graph.nodes:
                structure = self.structure_map[node_id]
                node_sizes.append(100 + structure.utility * 1000)
                node_colors.append(structure.vigor())
                node_labels[node_id] = f"S{node_id}\n{structure.utility:.2f}"
        
        # 绘制节点
        nodes = nx.draw_networkx_nodes(
            self.graph, pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=plt.cm.RdYlGn,
            alpha=0.7,
            edgecolors='black',
            linewidths=1,
            ax=ax
        )
        
        # 绘制边
        edges = self.graph.edges(data=True)
        edge_weights = [d['weight'] for _, _, d in edges]
        nx.draw_networkx_edges(
            self.graph, pos,
            edgelist=edges,
            width=[w * 3 for w in edge_weights],
            alpha=0.5,
            edge_color='#888888',
            ax=ax
        )
        
        # 绘制标签
        nx.draw_networkx_labels(
            self.graph, pos, node_labels,
            font_size=8, font_weight='bold',
            ax=ax
        )
        
        # 添加颜色条
        cbar = plt.colorbar(nodes, ax=ax)
        cbar.set_label('Vigor' if not show_activation else 'Activation')
        
        # 添加标题
        stats = self.get_network_stats()
        title = f'Structure Self-Organizing Network\n'
        title += f'Nodes: {stats["n_structures"]}, Edges: {stats["n_connections"]}, '
        title += f'Avg Degree: {stats["average_degree"]:.2f}'
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # 保存和显示
        filepath = os.path.join(save_dir, filename)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved structure network visualization to {filepath}")
        plt.close()
    
    def visualize_evolution(self, filename='network_evolution.png'):
        """可视化网络演化过程"""
        if len(self.evolution_history) < 2:
            print("Not enough evolution history to visualize")
            return
        
        # 创建保存目录
        save_dir = 'visualizations'
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        
        # 1. 结构数量变化
        plt.subplot(2, 2, 1)
        structure_counts = [s['n_structures'] for s in self.evolution_history]
        plt.plot(structure_counts)
        plt.xlabel('Step')
        plt.ylabel('Structure Count')
        plt.title('Structure Count Over Time')
        plt.grid(True, alpha=0.3)
        
        # 2. 连接数量变化
        plt.subplot(2, 2, 2)
        connection_counts = [s['n_connections'] for s in self.evolution_history]
        plt.plot(connection_counts)
        plt.xlabel('Step')
        plt.ylabel('Connection Count')
        plt.title('Connection Count Over Time')
        plt.grid(True, alpha=0.3)
        
        # 3. 平均度变化
        plt.subplot(2, 2, 3)
        avg_degrees = [s['average_degree'] for s in self.evolution_history]
        plt.plot(avg_degrees)
        plt.xlabel('Step')
        plt.ylabel('Average Degree')
        plt.title('Average Degree Over Time')
        plt.grid(True, alpha=0.3)
        
        # 4. 网络密度变化
        plt.subplot(2, 2, 4)
        densities = []
        for s in self.evolution_history:
            n = s['n_structures']
            if n > 1:
                max_edges = n * (n - 1) / 2
                density = s['n_connections'] / max_edges
            else:
                density = 0
            densities.append(density)
        plt.plot(densities)
        plt.xlabel('Step')
        plt.ylabel('Network Density')
        plt.title('Network Density Over Time')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved network evolution visualization to {filepath}")
        plt.close()


if __name__ == "__main__":
    """测试结构网络"""
    print("=" * 70)
    print("Structure Network Test")
    print("=" * 70)
    
    # 模拟结构
    class MockStructure:
        def __init__(self, id, prototype, utility=0.5):
            self.id = id
            self.prototype = prototype
            self.utility = utility
            self.age = 0
            
        def vigor(self):
            return self.utility * (1 - 0.1 * self.age)
    
    # 创建结构网络
    network = StructureNetwork()
    
    # 添加一些结构
    for i in range(10):
        prototype = np.random.randn(64)
        prototype = prototype / np.linalg.norm(prototype)
        structure = MockStructure(i, prototype, utility=np.random.rand())
        network.add_structure(structure)
    
    # 更新连接
    network.update_connections()
    
    # 可视化网络
    network.visualize('test_structure_network.png')
    
    # 模拟观测
    observation = np.random.randn(64)
    observation = observation / np.linalg.norm(observation)
    
    # 激活结构
    activations = network.activate_structures(observation)
    print(f"Top 3 activated structures: {activations[:3]}")
    
    # 可视化激活状态
    network.visualize('test_structure_network_activation.png', 
                    show_activation=True, observation=observation)
    
    # 测试激活传播
    propagated = network.propagate_activation(activations[:2])
    print(f"Propagated structures: {[(s.id, a) for s, a in propagated[:3]]}")
    
    # 显示网络统计
    stats = network.get_network_stats()
    print(f"Network stats: {stats}")
    
    print("\n" + "=" * 70)
    print("Structure Network Test Complete!")
    print("=" * 70)
