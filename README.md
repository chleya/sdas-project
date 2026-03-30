# Structure-Driven Agent System (SDAS)

## 项目简介

SDAS是一个基于结构驱动的智能体系统，旨在通过竞争性结构池实现持续学习和迁移学习能力。该系统能够在不同环境间转移知识，并在环境变化时快速适应。

## 核心特性

- **结构池机制**：通过竞争性结构池实现知识的存储和迁移
- **持续学习**：能够在环境变化时保持和更新知识
- **迁移学习**：在不同环境间转移学习到的结构和策略
- **MiniGrid集成**：支持在MiniGrid环境中进行实验

## 安装指南

### 依赖项

- Python 3.8+
- NumPy 1.26+
- Matplotlib
- Gymnasium with MiniGrid
- Stable-Baselines3
- CleanRL

### 安装步骤

1. 克隆仓库
   ```bash
   git clone https://github.com/chleya/sdas-project.git
   cd sdas-project
   ```

2. 安装依赖
   ```bash
   pip install -e .
   ```

## 项目结构

```
sdas-project/
├── src/                  # 核心代码
│   ├── sdas.py           # 主SDAS智能体实现
│   ├── sdas_minigrid.py  # MiniGrid环境集成
│   ├── structure_pool.py # 结构池实现
│   └── ...
├── experiments/          # 实验相关代码
├── visualizations/       # 可视化结果
├── experiment_minigrid_transfer.py  # MiniGrid迁移实验
├── test_cross_episode_transfer_fixed.py  # 跨episode迁移实验
├── test_ablation_study.py  # 消融研究
└── README.md            # 项目说明
```

## 架构图

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  环境观测   │────>│   编码器    │────>│  结构池     │────>│  行动策略   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       ^                   ^                   ^                   │
       │                   │                   │                   │
       │                   │                   │                   v
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  环境反馈   │<────│  结构更新   │<────│  奖励处理   │<────│  环境执行   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

## 实验复现

### MiniGrid迁移实验

运行从Empty-8x8到FourRooms的迁移实验：

```bash
python experiment_minigrid_transfer.py
```

### 跨Episode迁移实验

运行跨Episode的结构迁移实验：

```bash
python test_cross_episode_transfer_fixed.py
```

### 消融研究

运行消融实验，验证SDAS各组件的贡献：

```bash
python test_ablation_study.py
```

## 实验结果

### MiniGrid迁移实验

| 智能体类型 | 50个episode平均奖励 | 前10个episode平均奖励 |
|-----------|-------------------|---------------------|
| SDAS迁移   | -4.25 ± 7.25      | -3.16 ± 7.92        |
| PPO迁移    | -5.05 ± 6.16      | -6.99 ± 4.58        |
| PPO从零开始 | -5.04 ± 5.95      | -5.13 ± 4.54        |

### 跨Episode迁移实验

SDAS在环境变化时表现出更快的恢复速度，结构池能够有效保留和迁移知识。

### 消融研究

| 变体 | 平均奖励 |
|------|----------|
| 完整SDAS | -2.09 ± 0.08 |
| 无分支 | -2.24 ± 0.17 |
| 无Q学习 | -27.98 ± 12.58 |
| 无相似度 | -60.17 ± 10.09 |
| 扁平Q学习 | -47.42 ± 12.44 |

## 报告链接

- [SDAS最终报告](SDAS_Final_Report.md)

## 未来发展

1. **真实MiniGrid环境集成**：进一步优化与官方MiniGrid环境的集成
2. **更复杂任务变体**：测试SDAS在更复杂的MiniGrid任务上的表现
3. **参数优化**：进一步优化SDAS的参数，提高稳定性和性能
4. **与标准算法比较**：与更多标准算法进行全面比较
5. **结构可视化**：开发工具可视化结构池的演化过程

## 贡献

欢迎通过Issue和Pull Request贡献代码和建议。

## 许可证

MIT License
