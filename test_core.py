# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.structure_pool import StructurePool
import numpy as np

# Test Structure Pool
print("Testing Structure Pool...")
pool = StructurePool()

# First observation
obs1 = np.random.randn(128)
result1 = pool.observe(obs1, "测试主题A")
print(f"1st observe: {result1['event']}, focus: {result1['recommended_focus']}")

# Similar observation
obs2 = obs1 + np.random.randn(128) * 0.1
result2 = pool.observe(obs2, "测试主题A")
print(f"2nd observe: {result2['event']}, focus: {result2['recommended_focus']}")

# Different observation
obs3 = np.random.randn(128) * 2
result3 = pool.observe(obs3, "新主题B")
print(f"3rd observe: {result3['event']}, focus: {result3['recommended_focus']}")

print(f"\nTotal structures: {result3['state_summary']['structure_count']}")

# Test Digital Petri Dish
print("\n" + "="*40)
print("Testing Digital Petri Dish...")
from experiments.digital_petri_dish import DigitalPetriDish

env = DigitalPetriDish(width=10, height=10, n_obstacles=15, n_energy=3)
print(env.render())

print("\nSDAS Core Modules OK!")
