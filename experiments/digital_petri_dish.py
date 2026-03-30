"""
Digital Petri Dish - MVP实验环境

简单但真实的环境：
- 有限网格世界
- Agent需要寻找能量
- 避开障碍
- 回到熟悉区域
- 过程中更新内部结构
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
import random


@dataclass
class Position:
    """位置"""
    x: int
    y: int
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def distance_to(self, other: 'Position') -> float:
        return abs(self.x - other.x) + abs(self.y - other.y)


class DigitalPetriDish:
    """
    数字培养皿环境
    
    任务：寻找能量 - 避障 - 回到熟悉区域
    
    状态空间：
    - agent位置
    - 能量位置
    - 障碍布局
    - 访问历史
    
    行动空间：
    - 上/下/左/右移动
    - 等待
    """
    
    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        n_obstacles: int = 30,
        n_energy: int = 5,
        energy_reward: float = 1.0,
        step_cost: float = -0.01,
        obstacle_penalty: float = -0.5
    ):
        self.width = width
        self.height = height
        self.n_obstacles = n_obstacles
        self.n_energy = n_energy
        self.energy_reward = energy_reward
        self.step_cost = step_cost
        self.obstacle_penalty = obstacle_penalty
        
        self.agent_pos = Position(0, 0)
        self.energy_positions: List[Position] = []
        self.obstacles: List[Position] = []
        self.visited: set = set()
        
        self._reset()
    
    def _reset(self):
        """重置环境"""
        # 放置障碍物（随机但不在左上角）
        self.obstacles = []
        while len(self.obstacles) < self.n_obstacles:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            pos = Position(x, y)
            if (x, y) != (0, 0) and pos not in self.obstacles:
                self.obstacles.append(pos)
        
        # 放置能量
        self.energy_positions = []
        while len(self.energy_positions) < self.n_energy:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            pos = Position(x, y)
            if pos not in self.obstacles and pos not in self.energy_positions:
                self.energy_positions.append(pos)
        
        # Agent从左上角开始
        self.agent_pos = Position(0, 0)
        self.visited = {self.agent_pos}
        
        return self._get_obs()
    
    def _get_obs(self) -> dict:
        """获取观测"""
        # 找到最近能量
        min_energy_dist = float('inf')
        nearest_energy = None
        for e in self.energy_positions:
            d = self.agent_pos.distance_to(e)
            if d < min_energy_dist:
                min_energy_dist = d
                nearest_energy = e
        
        # 计算熟悉度（访问过的格子比例）
        familiarity = len(self.visited) / (self.width * self.height)
        
        return {
            "agent_x": self.agent_pos.x / self.width,
            "agent_y": self.agent_pos.y / self.height,
            "nearest_energy_x": nearest_energy.x / self.width if nearest_energy else 0,
            "nearest_energy_y": nearest_energy.y / self.height if nearest_energy else 0,
            "energy_dist": min_energy_dist / (self.width + self.height),
            "n_energy_remaining": len(self.energy_positions),
            "familiarity": familiarity,
            "visited_count": len(self.visited),
            "obstacle_nearby": any(
                self.agent_pos.distance_to(o) <= 2 
                for o in self.obstacles
            )
        }
    
    def step(self, action: int) -> Tuple[dict, float, bool]:
        """
        执行行动
        
        Actions:
        0: 上, 1: 下, 2: 左, 3: 右, 4: 等待
        """
        dx, dy = [(0, 1), (0, -1), (-1, 0), (1, 0), (0, 0)][action]
        
        new_x = self.agent_pos.x + dx
        new_y = self.agent_pos.y + dy
        
        # 边界检查
        if new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height:
            return self._get_obs(), self.step_cost, False
        
        new_pos = Position(new_x, new_y)
        
        # 障碍物检查
        if new_pos in self.obstacles:
            return self._get_obs(), self.obstacle_penalty, False
        
        # 移动
        self.agent_pos = new_pos
        self.visited.add(new_pos)
        
        reward = self.step_cost
        
        # 能量检查
        if new_pos in self.energy_positions:
            self.energy_positions.remove(new_pos)
            reward += self.energy_reward
            
            # 如果所有能量都被收集，重置能量
            if len(self.energy_positions) == 0:
                self._spawn_new_energy()
        
        done = len(self.energy_positions) == 0 and len(self.visited) > self.width * self.height * 0.8
        
        return self._get_obs(), reward, done
    
    def _spawn_new_energy(self):
        """生成新能量"""
        while len(self.energy_positions) < self.n_energy:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            pos = Position(x, y)
            if pos not in self.obstacles and pos != self.agent_pos:
                self.energy_positions.append(pos)
    
    def get_familiar_region_center(self) -> Optional[Position]:
        """获取熟悉区域中心"""
        if not self.visited:
            return None
        
        xs = [p.x for p in self.visited]
        ys = [p.y for p in self.visited]
        return Position(round(np.mean(xs)), round(np.mean(ys)))
    
    def render(self) -> str:
        """文本渲染"""
        lines = []
        for y in range(self.height - 1, -1, -1):
            line = ""
            for x in range(self.width):
                pos = Position(x, y)
                if pos == self.agent_pos:
                    line += "A"
                elif pos in self.energy_positions:
                    line += "E"
                elif pos in self.obstacles:
                    line += "#"
                elif pos in self.visited:
                    line += "."
                else:
                    line += " "
            lines.append(line)
        return "\n".join(lines)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from src.sdas import SDASAgent
    
    # 测试环境基本功能
    env = DigitalPetriDish(width=10, height=10, n_obstacles=15, n_energy=3)
    
    print("=== Digital Petri Dish ===")
    print(env.render())
    print()
    
    # 随机行动
    for i in range(20):
        action = random.randint(0, 4)
        obs, reward, done = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward:.2f}")
        print(f"  能量剩余: {obs['n_energy_remaining']}, 熟悉度: {obs['familiarity']:.2f}")
        
        if done:
            print("任务完成!")
            break
    
    print("\n" + env.render())
    
    # 测试结构数量随环境复杂度变化
    print("\n=== 测试结构数量随环境复杂度变化 ===")
    
    # 不同复杂度的环境
    complexities = [
        (10, 10, 5, 2),   # 简单
        (15, 15, 15, 3),  # 中等
        (20, 20, 30, 5),  # 复杂
        (25, 25, 50, 8)   # 非常复杂
    ]
    
    structure_counts = []
    complexity_labels = []
    
    for width, height, n_obstacles, n_energy in complexities:
        print(f"\n测试环境: {width}x{height}, 障碍物: {n_obstacles}, 能量: {n_energy}")
        
        # 创建环境和智能体
        env = DigitalPetriDish(width=width, height=height, n_obstacles=n_obstacles, n_energy=n_energy)
        agent = SDASAgent()
        
        # 运行一定步数
        max_steps = 200
        for step in range(max_steps):
            obs = env._get_obs()
            action, info = agent.step(obs)
            obs, reward, done = env.step(action)
            agent.update_structure(reward)
            
            if done:
                break
        
        # 记录结构数量
        structure_count = len(agent.structure_pool.structures)
        structure_counts.append(structure_count)
        complexity_labels.append(f"{width}x{height}, {n_obstacles}障, {n_energy}能")
        
        print(f"  结构数量: {structure_count}")
    
    # 绘制结构数量随复杂度变化的曲线
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(complexities)), structure_counts, tick_label=complexity_labels)
    plt.xlabel('环境复杂度')
    plt.ylabel('结构数量')
    plt.title('结构数量随环境复杂度的变化')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('structure_count_vs_complexity.png')
    print("\n结构数量变化曲线已保存为 structure_count_vs_complexity.png")
    plt.show()
