# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Add src to path
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))

print("Testing SDAS Framework...")
print("="*50)

from sdas import SDASAgent, run_episode, Config
from digital_petri_dish import DigitalPetriDish

# Create environment and agent
env = DigitalPetriDish(width=12, height=12, n_obstacles=20, n_energy=4)
agent = SDASAgent(Config(
    max_structures=16,
    encoder_dim=32,
    world_model_dim=24
))

# Run a few episodes
for ep in range(3):
    result = run_episode(env, agent, max_steps=80)
    print(f"\nEpisode {ep+1}:")
    print(f"  Steps: {result['steps']}")
    print(f"  Reward: {result['total_reward']:.3f}")
    print(f"  Final structures: {result['final_state']['structure_pool']['structure_count']}")
    event_counts = {}
    for e in result['structure_events']:
        event_counts[e] = event_counts.get(e, 0) + 1
    print(f"  Events: {event_counts}")

# Show final structure state
print("\n" + "="*50)
print("Final Structure Pool State:")
state = agent.get_state()
for s in state['structure_pool']['structures']:
    print(f"  {s['label']}: vigor={s['vigor']:.3f}, age={s['age']}")

print("\nSDAS Framework Test PASSED!")
