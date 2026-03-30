"""
Complex Digital Petri Dish - 更复杂的实验环境

增强特性：
1. 动态障碍物（会移动）
2. 多类型能量（不同奖励值）
3. 目标区域（需要到达）
4. 环境变化（随机事件）
5. 更丰富的状态表示
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import random

from digital_petri_dish import Position, DigitalPetriDish


@dataclass
class DynamicObstacle:
    """动态障碍物"""
    pos: Position
    direction: int  # 0: 上, 1: 下, 2: 左, 3: 右
    speed: float    # 移动概率


@dataclass
class EnergySource:
    """能量源"""
    pos: Position
    value: float    # 能量值
    type: str       # 能量类型: 'normal', 'high', 'rare'
    respawn_time: int  # 重生时间


class ComplexPetriDish(DigitalPetriDish):
    """
    复杂数字培养皿环境
    
    任务：
    - 收集不同类型的能量
    - 避开动态障碍物
    - 到达目标区域
    - 适应环境变化
    """
    
    def __init__(
        self,
        width: int = 25,
        height: int = 25,
        n_static_obstacles: int = 30,
        n_dynamic_obstacles: int = 5,
        n_energy_sources: int = 8,
        energy_rewards: Dict[str, float] = None,
        step_cost: float = -0.01,
        obstacle_penalty: float = -0.5,
        goal_reward: float = 5.0,
        random_event_prob: float = 0.1
    ):
        # 复杂环境参数
        self.n_dynamic_obstacles = n_dynamic_obstacles
        self.energy_rewards = energy_rewards or {
            'normal': 1.0,
            'high': 2.5,
            'rare': 5.0
        }
        self.goal_reward = goal_reward
        self.random_event_prob = random_event_prob
        
        # 动态障碍物
        self.dynamic_obstacles: List[DynamicObstacle] = []
        
        # 能量源（替换父类的能量位置）
        self.energy_sources: List[EnergySource] = []
        
        # 目标区域
        self.goal_pos: Position = None
        
        # 环境状态
        self.time_step = 0
        self.event_history = []
        
        # 调用父类初始化
        super().__init__(
            width=width,
            height=height,
            n_obstacles=n_static_obstacles,
            n_energy=n_energy_sources,
            energy_reward=1.0,  # 父类的奖励会被覆盖
            step_cost=step_cost,
            obstacle_penalty=obstacle_penalty
        )
    
    def _reset(self):
        """重置环境"""
        # 重置父类
        super()._reset()
        
        # 清空父类的能量位置（我们使用自己的能量源）
        self.energy_positions = []
        
        # 放置动态障碍物
        self.dynamic_obstacles = []
        while len(self.dynamic_obstacles) < self.n_dynamic_obstacles:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            pos = Position(x, y)
            if (x, y) != (0, 0) and pos not in self.obstacles:
                direction = random.randint(0, 3)
                speed = random.uniform(0.1, 0.3)
                self.dynamic_obstacles.append(DynamicObstacle(pos, direction, speed))
        
        # 放置能量源
        self.energy_sources = []
        energy_types = ['normal', 'high', 'rare']
        type_probs = [0.7, 0.2, 0.1]  # 概率分布
        
        while len(self.energy_sources) < self.n_energy:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            pos = Position(x, y)
            
            # 检查位置是否合法
            is_valid = True
            if pos in self.obstacles:
                is_valid = False
            for dyn_obs in self.dynamic_obstacles:
                if dyn_obs.pos == pos:
                    is_valid = False
                    break
            for es in self.energy_sources:
                if es.pos == pos:
                    is_valid = False
                    break
            
            if is_valid:
                energy_type = random.choices(energy_types, type_probs)[0]
                value = self.energy_rewards[energy_type]
                respawn_time = random.randint(10, 30)
                self.energy_sources.append(EnergySource(pos, value, energy_type, respawn_time))
        
        # 放置目标区域（右下角）
        self.goal_pos = Position(self.width - 1, self.height - 1)
        
        # 重置时间和事件历史
        self.time_step = 0
        self.event_history = []
        
        return self._get_obs()
    
    def _get_obs(self) -> dict:
        """获取观测"""
        # 基础观测
        base_obs = super()._get_obs()
        
        # 找到最近的不同类型能量
        min_dist_by_type = {
            'normal': float('inf'),
            'high': float('inf'),
            'rare': float('inf')
        }
        nearest_by_type = {
            'normal': None,
            'high': None,
            'rare': None
        }
        
        for es in self.energy_sources:
            d = self.agent_pos.distance_to(es.pos)
            if d < min_dist_by_type[es.type]:
                min_dist_by_type[es.type] = d
                nearest_by_type[es.type] = es.pos
        
        # 动态障碍物信息
        nearest_dynamic_obstacle_dist = float('inf')
        for obs in self.dynamic_obstacles:
            d = self.agent_pos.distance_to(obs.pos)
            if d < nearest_dynamic_obstacle_dist:
                nearest_dynamic_obstacle_dist = d
        
        # 目标距离
        if self.goal_pos:
            goal_dist = self.agent_pos.distance_to(self.goal_pos) / (self.width + self.height)
        else:
            goal_dist = 1.0  # 初始状态时的默认值
        
        # 能量统计
        energy_counts = {
            'normal': 0,
            'high': 0,
            'rare': 0
        }
        for es in self.energy_sources:
            energy_counts[es.type] += 1
        
        # 扩展观测
        obs = {
            **base_obs,
            # 能量信息
            'nearest_normal_energy_x': nearest_by_type['normal'].x / self.width if nearest_by_type['normal'] else 0,
            'nearest_normal_energy_y': nearest_by_type['normal'].y / self.height if nearest_by_type['normal'] else 0,
            'nearest_high_energy_x': nearest_by_type['high'].x / self.width if nearest_by_type['high'] else 0,
            'nearest_high_energy_y': nearest_by_type['high'].y / self.height if nearest_by_type['high'] else 0,
            'nearest_rare_energy_x': nearest_by_type['rare'].x / self.width if nearest_by_type['rare'] else 0,
            'nearest_rare_energy_y': nearest_by_type['rare'].y / self.height if nearest_by_type['rare'] else 0,
            'normal_energy_dist': min_dist_by_type['normal'] / (self.width + self.height),
            'high_energy_dist': min_dist_by_type['high'] / (self.width + self.height),
            'rare_energy_dist': min_dist_by_type['rare'] / (self.width + self.height),
            'n_normal_energy': energy_counts['normal'],
            'n_high_energy': energy_counts['high'],
            'n_rare_energy': energy_counts['rare'],
            # 动态障碍物信息
            'nearest_dynamic_obstacle_dist': nearest_dynamic_obstacle_dist / (self.width + self.height),
            'n_dynamic_obstacles': len(self.dynamic_obstacles),
            # 目标信息
            'goal_x': self.goal_pos.x / self.width if self.goal_pos else 1.0,
            'goal_y': self.goal_pos.y / self.height if self.goal_pos else 1.0,
            'goal_dist': goal_dist,
            # 时间信息
            'time_step': self.time_step / 1000,  # 归一化
            # 环境状态
            'n_events': len(self.event_history) / 10,  # 归一化
        }
        
        return obs
    
    def _move_dynamic_obstacles(self):
        """移动动态障碍物"""
        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        
        for obs in self.dynamic_obstacles:
            if random.random() < obs.speed:
                dx, dy = directions[obs.direction]
                new_x = obs.pos.x + dx
                new_y = obs.pos.y + dy
                
                # 边界检查
                if 0 <= new_x < self.width and 0 <= new_y < self.height:
                    new_pos = Position(new_x, new_y)
                    
                    # 检查是否与静态障碍物或其他动态障碍物碰撞
                    collision = False
                    if new_pos in self.obstacles:
                        collision = True
                    for other_obs in self.dynamic_obstacles:
                        if other_obs != obs and other_obs.pos == new_pos:
                            collision = True
                            break
                    
                    if not collision:
                        obs.pos = new_pos
                    else:
                        # 碰撞时随机改变方向
                        obs.direction = random.randint(0, 3)
                else:
                    # 碰到边界时改变方向
                    obs.direction = random.randint(0, 3)
    
    def _check_random_events(self):
        """检查随机事件"""
        if random.random() < self.random_event_prob:
            event_type = random.choice([
                'energy_boost',    # 能量值提升
                'obstacle_shift',  # 障碍物移动
                'goal_relocate',   # 目标位置改变
                'energy_surge'     # 大量能量生成
            ])
            
            if event_type == 'energy_boost':
                # 提升所有能量值
                for es in self.energy_sources:
                    es.value *= 1.5
                self.event_history.append('energy_boost')
                
            elif event_type == 'obstacle_shift':
                # 随机移动一个静态障碍物
                if self.obstacles:
                    obs_idx = random.randint(0, len(self.obstacles) - 1)
                    old_pos = self.obstacles[obs_idx]
                    
                    # 找新位置
                    for _ in range(100):
                        x = random.randint(0, self.width - 1)
                        y = random.randint(0, self.height - 1)
                        new_pos = Position(x, y)
                        if (x, y) != (0, 0) and new_pos not in self.obstacles:
                            self.obstacles[obs_idx] = new_pos
                            self.event_history.append('obstacle_shift')
                            break
                
            elif event_type == 'goal_relocate':
                # 随机改变目标位置
                for _ in range(100):
                    x = random.randint(0, self.width - 1)
                    y = random.randint(0, self.height - 1)
                    new_pos = Position(x, y)
                    if new_pos not in self.obstacles:
                        self.goal_pos = new_pos
                        self.event_history.append('goal_relocate')
                        break
                
            elif event_type == 'energy_surge':
                # 生成额外的能量
                for _ in range(3):
                    x = random.randint(0, self.width - 1)
                    y = random.randint(0, self.height - 1)
                    new_pos = Position(x, y)
                    
                    # 检查位置是否合法
                    is_valid = True
                    if new_pos in self.obstacles:
                        is_valid = False
                    for dyn_obs in self.dynamic_obstacles:
                        if dyn_obs.pos == new_pos:
                            is_valid = False
                            break
                    for es in self.energy_sources:
                        if es.pos == new_pos:
                            is_valid = False
                            break
                    
                    if is_valid:
                        energy_type = random.choices(['normal', 'high', 'rare'], [0.7, 0.2, 0.1])[0]
                        value = self.energy_rewards[energy_type]
                        respawn_time = random.randint(10, 30)
                        self.energy_sources.append(EnergySource(new_pos, value, energy_type, respawn_time))
                self.event_history.append('energy_surge')
    
    def step(self, action: int) -> Tuple[dict, float, bool]:
        """
        执行行动
        
        Actions:
        0: 上, 1: 下, 2: 左, 3: 右, 4: 等待
        """
        # 移动动态障碍物
        self._move_dynamic_obstacles()
        
        # 检查随机事件
        self._check_random_events()
        
        # 执行行动
        dx, dy = [(0, 1), (0, -1), (-1, 0), (1, 0), (0, 0)][action]
        
        new_x = self.agent_pos.x + dx
        new_y = self.agent_pos.y + dy
        
        # 边界检查
        if new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height:
            self.time_step += 1
            return self._get_obs(), self.step_cost, False
        
        new_pos = Position(new_x, new_y)
        
        # 静态障碍物检查
        if new_pos in self.obstacles:
            self.time_step += 1
            return self._get_obs(), self.obstacle_penalty, False
        
        # 动态障碍物检查
        for obs in self.dynamic_obstacles:
            if obs.pos == new_pos:
                self.time_step += 1
                return self._get_obs(), self.obstacle_penalty, False
        
        # 移动
        self.agent_pos = new_pos
        self.visited.add(new_pos)
        
        reward = self.step_cost
        
        # 能量检查
        collected_energy = None
        for i, es in enumerate(self.energy_sources):
            if es.pos == new_pos:
                collected_energy = es
                del self.energy_sources[i]
                reward += es.value
                break
        
        # 目标检查
        if new_pos == self.goal_pos:
            reward += self.goal_reward
            done = True
        else:
            done = False
        
        # 能量重生
        self._respawn_energy()
        
        # 更新时间
        self.time_step += 1
        
        return self._get_obs(), reward, done
    
    def _respawn_energy(self):
        """重生能量"""
        # 简单的能量重生逻辑
        if len(self.energy_sources) < self.n_energy:
            # 有几率生成新能量
            if random.random() < 0.1:
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                new_pos = Position(x, y)
                
                # 检查位置是否合法
                is_valid = True
                if new_pos in self.obstacles:
                    is_valid = False
                for dyn_obs in self.dynamic_obstacles:
                    if dyn_obs.pos == new_pos:
                        is_valid = False
                        break
                for es in self.energy_sources:
                    if es.pos == new_pos:
                        is_valid = False
                        break
                
                if is_valid:
                    energy_type = random.choices(['normal', 'high', 'rare'], [0.7, 0.2, 0.1])[0]
                    value = self.energy_rewards[energy_type]
                    respawn_time = random.randint(10, 30)
                    self.energy_sources.append(EnergySource(new_pos, value, energy_type, respawn_time))
    
    def render(self) -> str:
        """文本渲染"""
        lines = []
        for y in range(self.height - 1, -1, -1):
            line = ""
            for x in range(self.width):
                pos = Position(x, y)
                if pos == self.agent_pos:
                    line += "A"
                elif pos == self.goal_pos:
                    line += "G"
                elif any(es.pos == pos for es in self.energy_sources):
                    # 不同类型能量显示不同字符
                    for es in self.energy_sources:
                        if es.pos == pos:
                            if es.type == 'normal':
                                line += "e"
                            elif es.type == 'high':
                                line += "E"
                            else:  # rare
                                line += "★"
                            break
                elif pos in self.obstacles:
                    line += "#"
                elif any(obs.pos == pos for obs in self.dynamic_obstacles):
                    line += "O"
                elif pos in self.visited:
                    line += "."
                else:
                    line += " "
            lines.append(line)
        return "\n".join(lines)


if __name__ == "__main__":
    env = ComplexPetriDish(
        width=15,
        height=15,
        n_static_obstacles=20,
        n_dynamic_obstacles=3,
        n_energy_sources=5
    )
    
    print("=== Complex Digital Petri Dish ===")
    print(env.render())
    print()
    
    # 随机行动测试
    for i in range(50):
        action = random.randint(0, 4)
        obs, reward, done = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward:.2f}")
        print(f"  能量: N={obs['n_normal_energy']}, H={obs['n_high_energy']}, R={obs['n_rare_energy']}")
        print(f"  目标距离: {obs['goal_dist']:.2f}, 动态障碍物距离: {obs['nearest_dynamic_obstacle_dist']:.2f}")
        
        if done:
            print("任务完成!")
            break
        
        # 每10步渲染一次
        if (i + 1) % 10 == 0:
            print("\n" + env.render())
            print()
    
    print("\n最终状态:")
    print(env.render())
