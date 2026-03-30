# SDAS - Structure-Driven Agent System

Structure-Driven Agent System (SDAS) - 验证"竞争性持久结构池"是否能提升长期适应

## 项目结构

```
SDAS/
├── src/
│   ├── structure_pool.py    # 结构池核心
│   ├── world_model.py       # 世界模型
│   ├── agent.py             # Agent主体
│   └── __init__.py
├── experiments/
│   └── digital_petri_dish.py  # MVP环境
├── scripts/
│   └── run_demo.py
└── docs/
    └── proposal.md          # 项目企划书
```

## 核心概念

- **结构池**: 32-64个槽位，每个维护prototype、age、utility、surprise history
- **竞争机制**: top-k稀疏激活
- **生灭机制**: 低utility衰减、高冗余merge、高surprise分裂
- **世界模型**: Dreamer风格latent dynamics
- **混合目标**: L = L_pred + λ1·L_info + λ2·L_emp + λ3·L_sparse + λ4·L_homeo

## 状态

项目启动中...
