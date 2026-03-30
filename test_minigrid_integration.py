"""
MiniGrid集成测试
测试SDAS在MiniGrid环境中的表现
"""

import numpy as np
import random
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 测试MiniGrid环境
print("Testing MiniGrid environment...")
try:
    from minigrid.envs import EmptyEnv, FourRoomsEnv
    from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
    print("MiniGrid imported successfully!")
except ImportError as e:
    print(f"Error importing MiniGrid: {e}")
    sys.exit(1)

# 测试Empty-8x8环境
def test_empty_env():
    print("\nTesting Empty-8x8 environment...")
    env = EmptyEnv(size=8)
    obs = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Observation keys: {list(obs.keys())}")
    
    # 测试动作空间
    print(f"Action space: {env.action_space}")
    print(f"Action meanings: {env.actions}")
    
    # 测试几步
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward}, done={done}")
        if done:
            break

# 测试FourRooms环境
def test_four_rooms_env():
    print("\nTesting FourRooms environment...")
    env = FourRoomsEnv()
    obs = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Observation keys: {list(obs.keys())}")
    
    # 测试动作空间
    print(f"Action space: {env.action_space}")
    print(f"Action meanings: {env.actions}")
    
    # 测试几步
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward}, done={done}")
        if done:
            break

# 测试图像观测
def test_rgb_obs():
    print("\nTesting RGB image observation...")
    env = EmptyEnv(size=8)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    obs = env.reset()
    print(f"RGB observation shape: {obs.shape}")
    print(f"RGB observation dtype: {obs.dtype}")

if __name__ == "__main__":
    print("=== MiniGrid Integration Test ===")
    test_empty_env()
    test_four_rooms_env()
    test_rgb_obs()
    print("\nMiniGrid integration test completed!")
