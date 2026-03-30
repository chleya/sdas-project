"""
完整训练运行脚本
依次运行所有三个方案的 ES 训练
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))

import numpy as np
import json


def print_header(title):
    """打印标题"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def run_scheme_a():
    """运行方案 A：ES 优化超参数"""
    print_header("SCHEME A: ES Hyperparameter Optimization")
    
    from es_trainer import ESTrainer, ESConfig
    
    es_config = ESConfig(
        population_size=30,
        n_generations=20,
        sigma=0.1,
        learning_rate=0.05,
        n_eval_episodes=3,
        max_steps_per_episode=100
    )
    
    trainer = ESTrainer(es_config)
    best_params = trainer.train(n_generations=20)
    
    # 保存结果
    results_a = {
        'scheme': 'A - ES Hyperparameters',
        'best_fitness': trainer.best_fitness,
        'params': {
            'decay_rate': best_params.decay_rate,
            'create_threshold': best_params.create_threshold,
            'max_structures': best_params.max_structures,
            'utility_lr': best_params.utility_lr
        }
    }
    
    with open('results_scheme_a.json', 'w') as f:
        json.dump(results_a, f, indent=2)
    
    print("\nResults saved to results_scheme_a.json")
    return results_a


def run_scheme_c():
    """运行方案 C：端到端全参数优化（先运行因为更耗时）"""
    print_header("SCHEME C: End-to-End Full Parameter Optimization")
    
    from es_trainer_end2end import End2EndESTrainer, End2EndESConfig
    
    es_config = End2EndESConfig(
        population_size=15,
        n_generations=15,
        sigma=0.02,
        learning_rate=0.02,
        n_eval_episodes=2,
        max_steps_per_episode=100,
        optimize_encoder=True,
        optimize_world_model=True,
        optimize_structure_pool_hparams=True
    )
    
    trainer = End2EndESTrainer(es_config)
    best_params = trainer.train(n_generations=15)
    
    # 保存结果
    trainer.save_best_agent('results_scheme_c_agent.npy')
    
    results_c = {
        'scheme': 'C - End-to-End Optimization',
        'best_fitness': trainer.best_fitness,
        'total_params': len(best_params)
    }
    
    with open('results_scheme_c.json', 'w') as f:
        json.dump(results_c, f, indent=2)
    
    print("\nResults saved to results_scheme_c.json and results_scheme_c_agent.npy")
    return results_c


def run_scheme_b():
    """运行方案 B：低秩 ES 优化原型"""
    print_header("SCHEME B: Low-Rank (LoRA) Prototype Optimization")
    
    from es_trainer_lora import LoRAESTrainer, LoRAESConfig
    
    es_config = LoRAESConfig(
        population_size=20,
        n_generations=15,
        sigma=0.05,
        learning_rate=0.03,
        n_eval_episodes=2,
        max_steps_per_episode=100,
        rank=2
    )
    
    trainer = LoRAESTrainer(es_config)
    best_params = trainer.train(n_generations=15)
    
    # 保存结果
    trainer.save_best_params('results_scheme_b_params.npy')
    
    results_b = {
        'scheme': 'B - Low-Rank (LoRA) Optimization',
        'best_fitness': trainer.best_fitness,
        'total_params': len(best_params),
        'rank': es_config.rank
    }
    
    with open('results_scheme_b.json', 'w') as f:
        json.dump(results_b, f, indent=2)
    
    print("\nResults saved to results_scheme_b.json and results_scheme_b_params.npy")
    return results_b


def run_comparison_all_methods():
    """运行所有方法对比（不训练，直接测试）"""
    print_header("COMPARISON: All Methods Test (No Training)")
    
    exec(open('compare_all_methods.py').read())


def main():
    print("="*80)
    print("  SDAS COMPLETE TRAINING RUNNER")
    print("="*80)
    print("\nSelect what to run:")
    print("  1. Run all training (A, B, C) - will take time!")
    print("  2. Run only Scheme A (ES Hyperparameters)")
    print("  3. Run only Scheme B (Low-Rank)")
    print("  4. Run only Scheme C (End-to-End)")
    print("  5. Run comparison test (no training)")
    print("  6. Run quick test of all trainers")
    print("="*80)
    
    choice = input("\nSelect option (1-6): ").strip()
    
    results = {}
    
    if choice == "1":
        # 运行所有训练 - 注意：会很长时间
        print("\nWARNING: This will take a long time!")
        confirm = input("Are you sure? (y/n): ").strip().lower()
        if confirm == 'y':
            results['A'] = run_scheme_a()
            results['B'] = run_scheme_b()
            results['C'] = run_scheme_c()
    
    elif choice == "2":
        results['A'] = run_scheme_a()
    
    elif choice == "3":
        results['B'] = run_scheme_b()
    
    elif choice == "4":
        results['C'] = run_scheme_c()
    
    elif choice == "5":
        run_comparison_all_methods()
    
    elif choice == "6":
        print_header("QUICK TEST: All Trainers")
        
        # 快速测试每个训练器
        print("\n--- Testing Scheme A Trainer ---")
        from es_trainer import quick_test as test_a
        test_a()
        
        print("\n--- Testing Scheme B Trainer ---")
        from es_trainer_lora import quick_test as test_b
        test_b()
        
        print("\n--- Testing Scheme C Trainer ---")
        from es_trainer_end2end import quick_test as test_c
        test_c()
        
        print("\n--- All Quick Tests Complete ---")
    
    else:
        print("Invalid choice. Exiting.")
        return
    
    # 总结
    if results:
        print("\n" + "="*80)
        print("  TRAINING SUMMARY")
        print("="*80)
        
        for key, r in results.items():
            print(f"\n{key}: {r['scheme']}")
            print(f"  Best Fitness: {r.get('best_fitness', 'N/A'):.3f}")
            if 'total_params' in r:
                print(f"  Params: {r['total_params']}")
        
        print("\nAll results saved to results_scheme_*.json files")


if __name__ == "__main__":
    main()
