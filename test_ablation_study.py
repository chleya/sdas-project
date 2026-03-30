"""
消融研究 - 验证SDAS各组件的贡献

测试4个变体：
1. 无结构池（Flat Q-learning）
2. 无分支机制（只有创建/衰减）
3. 无Q-learning（纯相似度选择）
4. 无相似度激活（纯Q-learning）

与完整SDAS进行对比
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass

from experiments.digital_petri_dish import DigitalPetriDish
from src.sdas import SDASAgent, Config
from src.structure_pool import StructurePool, Structure


class FlatQLearningAgent:
    """无结构池的Flat Q-learning智能体"""
    
    def __init__(self, n_actions=5, learning_rate=0.1, discount_factor=0.99):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = {}
        self.total_steps = 0
    
    def reset(self):
        pass
    
    def _get_state_key(self, obs):
        """获取状态键"""
        return (int(obs['agent_x'] * 5), int(obs['agent_y'] * 5), 
                int(obs['nearest_energy_x'] * 5), int(obs['nearest_energy_y'] * 5),
                int(obs['energy_dist'] * 10))
    
    def step(self, obs):
        """选择动作"""
        state_key = self._get_state_key(obs)
        
        # epsilon-greedy
        epsilon = max(0.05, 0.3 - self.total_steps * 0.001)
        if random.random() < epsilon:
            action = random.randint(0, self.n_actions - 1)
        else:
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.n_actions)
            action = int(np.argmax(self.q_table[state_key]))
        
        self.total_steps += 1
        return action, {}
    
    def update_structure(self, reward):
        """更新Q表"""
        if hasattr(self, 'last_state') and hasattr(self, 'last_action'):
            if self.last_state not in self.q_table:
                self.q_table[self.last_state] = np.zeros(self.n_actions)
            
            # 简单Q-learning更新
            self.q_table[self.last_state][self.last_action] += \
                self.learning_rate * (reward - self.q_table[self.last_state][self.last_action])
    
    def get_state(self):
        return {}


class NoBranchingStructurePool(StructurePool):
    """无分支机制的结构池"""
    
    def observe(self, observation, label=""):
        """观察输入，返回结构信号（无分支）"""
        if len(self.structures) == 0:
            # 首次观测，创建第一个结构
            new_struct = Structure(
                id=self.next_id,
                prototype=observation.copy(),
                label=label or "initial",
                utility=0.5,
                age=0
            )
            self.structures.append(new_struct)
            self.next_id += 1
            return self._build_signal("created", [(new_struct, 1.0)])
        
        # 计算与现有结构的相似度
        similarities = []
        for s in self.structures:
            sim = self._cosine_similarity(observation, s.prototype)
            similarities.append((s, sim))
        
        # 找到最佳匹配
        best_struct, best_similarity = max(similarities, key=lambda x: x[1])
        novelty = 1.0 - best_similarity
        
        # 根据novelty决定事件类型（无分支）
        if novelty > self.create_threshold:
            # 新主题：创建新结构
            new_struct = Structure(
                id=self.next_id,
                prototype=observation.copy(),
                label=label or f"structure_{self.next_id}",
                utility=0.5,
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
        else:
            # 强化现有结构（无分支）
            best_struct.utility = min(1.0, best_struct.utility + 0.1)
            best_struct.surprise_history.append(novelty)
            best_struct.mean_similarity = (
                0.9 * best_struct.mean_similarity + 0.1 * best_similarity
            )
            best_struct.prototype = 0.95 * best_struct.prototype + 0.05 * observation
            event = "reinforced"
        
        # 衰减所有结构
        self._decay_all()
        
        # 剪枝：移除极低utility的结构
        self._prune()
        
        # 按相似度排序，取前5个
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar = similarities[:5]
        
        return self._build_signal(event, top_similar)


class NoQLearningActionPolicy:
    """无Q-learning的动作策略（基于相似度和环境观测选择）"""
    
    def __init__(self, n_actions=5):
        self.n_actions = n_actions
    
    def select_action(self, obs, structure_signal, world_model, latent, epsilon=0.1):
        """基于相似度和环境观测选择动作"""
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        
        # 基于结构相似度选择动作
        active = structure_signal.get('active_structures_objects', [])
        if not active:
            # 没有活跃结构，基于环境观测选择动作
            return self._select_based_on_obs(obs)
        
        # 选择相似度最高的结构
        active.sort(key=lambda x: x[1], reverse=True)
        best_structure = active[0][0]
        best_similarity = active[0][1]
        
        # 基于结构相似度和环境观测选择动作
        if best_similarity > 0.7:
            # 高相似度：基于结构的标签和环境观测选择动作
            return self._select_based_on_structure_and_obs(best_structure, obs)
        else:
            # 低相似度：基于环境观测选择动作
            return self._select_based_on_obs(obs)
    
    def _select_based_on_obs(self, obs):
        """基于环境观测选择动作"""
        # 基于最近能量的位置选择动作
        agent_x = obs.get('agent_x', 0)
        agent_y = obs.get('agent_y', 0)
        energy_x = obs.get('nearest_energy_x', 0)
        energy_y = obs.get('nearest_energy_y', 0)
        energy_dist = obs.get('energy_dist', 1.0)
        obstacle_nearby = obs.get('obstacle_nearby', False)
        
        # 如果有障碍物附近，随机选择一个方向
        if obstacle_nearby:
            return random.choice([0, 1, 2, 3])
        
        # 如果能量很近，向能量移动
        if energy_dist < 0.3:
            if abs(energy_x - agent_x) > abs(energy_y - agent_y):
                return 3 if energy_x > agent_x else 2  # 右或左
            else:
                return 0 if energy_y > agent_y else 1  # 上或下
        
        # 否则随机探索
        return random.randint(0, 3)
    
    def _select_based_on_structure_and_obs(self, structure, obs):
        """基于结构和环境观测选择动作"""
        # 基于结构的标签和环境观测选择动作
        label = structure.label
        
        if "避障" in label:
            # 避障模式：避开障碍物
            if obs.get('obstacle_nearby', False):
                return random.choice([0, 1, 2, 3])
        elif "获取能量" in label:
            # 获取能量模式：向能量移动
            agent_x = obs.get('agent_x', 0)
            agent_y = obs.get('agent_y', 0)
            energy_x = obs.get('nearest_energy_x', 0)
            energy_y = obs.get('nearest_energy_y', 0)
            
            if abs(energy_x - agent_x) > abs(energy_y - agent_y):
                return 3 if energy_x > agent_x else 2  # 右或左
            else:
                return 0 if energy_y > agent_y else 1  # 上或下
        elif "探索熟悉区" in label:
            # 探索熟悉区模式：随机移动
            return random.randint(0, 3)
        elif "探索新区域" in label:
            # 探索新区域模式：随机移动
            return random.randint(0, 3)
        
        # 默认基于环境观测选择
        return self._select_based_on_obs(obs)


class NoSimilarityActionPolicy:
    """无相似度激活的动作策略（纯Q-learning）"""
    
    def __init__(self, n_actions=5):
        self.n_actions = n_actions
        self.q_table = {}
    
    def _get_state_key(self, obs):
        """获取状态键"""
        return (int(obs['agent_x'] * 5), int(obs['agent_y'] * 5), 
                int(obs['nearest_energy_x'] * 5), int(obs['nearest_energy_y'] * 5),
                int(obs['energy_dist'] * 10))
    
    def select_action(self, obs, structure_signal, world_model, latent, epsilon=0.1):
        """基于Q-learning选择动作"""
        state_key = self._get_state_key(obs)
        
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.n_actions)
            return int(np.argmax(self.q_table[state_key]))


class SDASNoBranchingAgent(SDASAgent):
    """无分支机制的SDAS智能体"""
    
    def __init__(self, config=None):
        super().__init__(config)
        # 替换为无分支结构池
        self.structure_pool = NoBranchingStructurePool(
            max_structures=self.config.max_structures,
            decay_rate=self.config.decay_rate,
            create_threshold=self.config.create_threshold,
            vector_dim=self.config.encoder_dim
        )


class SDASNoQLearningAgent(SDASAgent):
    """无Q-learning的SDAS智能体"""
    
    def __init__(self, config=None):
        super().__init__(config)
        # 替换为无Q-learning策略
        self.policy = NoQLearningActionPolicy(n_actions=self.config.n_actions)
    
    def update_structure(self, reward):
        """只更新结构效用，不更新Q值"""
        if self.last_active_structure_id is not None:
            for s in self.structure_pool.structures:
                if s.id == self.last_active_structure_id:
                    delta = reward * 0.1
                    s.utility = np.clip(s.utility + delta, 0, 1.0)
                    break


class SDASNoSimilarityAgent(SDASAgent):
    """无相似度激活的SDAS智能体"""
    
    def __init__(self, config=None):
        super().__init__(config)
        # 替换为无相似度策略
        self.policy = NoSimilarityActionPolicy(n_actions=self.config.n_actions)


def run_episode(env, agent, max_steps=200):
    """运行一个回合"""
    obs = env._reset()
    agent.reset()
    
    total_reward = 0
    
    for step in range(max_steps):
        action, info = agent.step(obs)
        obs, reward, done = env.step(action)
        agent.update_structure(reward)
        total_reward += reward
        
        if done:
            break
    
    return total_reward

def run_ablation_study(n_episodes=100, n_seeds=5):
    """运行消融研究"""
    agents = {
        "Full SDAS": SDASAgent,
        "Flat Q-learning": FlatQLearningAgent,
        "No Branching": SDASNoBranchingAgent,
        "No Q-learning": SDASNoQLearningAgent,
        "No Similarity": SDASNoSimilarityAgent
    }
    
    results = {name: [] for name in agents.keys()}
    
    for seed in range(n_seeds):
        print(f"Seed {seed+1}/{n_seeds}")
        
        # 固定随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 创建环境
        env = DigitalPetriDish(width=15, height=15, n_obstacles=25, n_energy=5)
        
        # 初始化智能体
        agent_instances = {name: cls() for name, cls in agents.items()}
        
        # 记录每个智能体的奖励
        seed_results = {name: [] for name in agents.keys()}
        
        for episode in range(n_episodes):
            if episode % 10 == 0:
                print(f"  Episode {episode+1}/{n_episodes}")
            
            for name, agent in agent_instances.items():
                reward = run_episode(env, agent)
                seed_results[name].append(reward)
        
        # 保存种子结果
        for name in agents.keys():
            results[name].append(seed_results[name])
    
    return results

def plot_results(results):
    """绘制结果"""
    plt.figure(figsize=(12, 8))
    
    # 计算均值和标准差
    for name, seed_results in results.items():
        seed_results = np.array(seed_results)
        mean_rewards = np.mean(seed_results, axis=0)
        std_rewards = np.std(seed_results, axis=0)
        
        # 平滑处理
        window = 5
        if len(mean_rewards) >= window:
            mean_rewards = np.convolve(mean_rewards, np.ones(window)/window, mode='same')
        
        episodes = range(1, len(mean_rewards) + 1)
        plt.plot(episodes, mean_rewards, label=name)
        plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
    
    plt.title('Ablation Study: SDAS Components Contribution')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('ablation_study_results.png')
    plt.show()

def print_summary(results):
    """打印总结"""
    print("\n=== Ablation Study Summary ===")
    
    for name, seed_results in results.items():
        seed_results = np.array(seed_results)
        final_rewards = seed_results[:, -20:].mean(axis=1)  # 最后20个回合的平均
        mean_final = np.mean(final_rewards)
        std_final = np.std(final_rewards)
        
        print(f"{name}: {mean_final:.2f} ± {std_final:.2f}")


if __name__ == "__main__":
    print("=== SDAS Ablation Study ===")
    print("Testing 5 variants over 100 episodes with 5 seeds...")
    
    results = run_ablation_study(n_episodes=100, n_seeds=5)
    
    print_summary(results)
    plot_results(results)
    
    print("\nAblation study completed!")
    print("Results saved to 'ablation_study_results.png'")
