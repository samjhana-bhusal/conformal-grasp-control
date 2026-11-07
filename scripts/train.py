#!/usr/bin/env python3
"""
Train PPO agent for UR5 bin sorting.

Saves checkpoints, training logs, and plots.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml

from src.env.bin_sort_env import BinSortEnv
from src.rl.ppo import PPOTrainer, PPOConfig


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(os.path.join(root, "configs", "default.yaml")) as f:
        config = yaml.safe_load(f)

    sim_config = config["simulation"]
    train_config = config["training"]

    env = BinSortEnv(sim_config)

    ppo_config = PPOConfig(
        lr=train_config["learning_rate"],
        gamma=train_config["gamma"],
        gae_lambda=train_config["gae_lambda"],
        clip_eps=train_config["clip_epsilon"],
        n_epochs=train_config["n_epochs"],
        batch_size=train_config["batch_size"],
        entropy_coef=train_config["entropy_coef"],
        value_coef=train_config["value_coef"],
        max_grad_norm=train_config["max_grad_norm"],
        n_steps=train_config["n_steps_per_update"],
    )

    trainer = PPOTrainer(
        obs_dim=BinSortEnv.OBS_DIM,
        act_dim=BinSortEnv.ACT_DIM,
        config=ppo_config,
    )

    total_steps = train_config["total_timesteps"]
    steps_per_update = train_config["n_steps_per_update"]
    n_updates = total_steps // steps_per_update
    save_interval = train_config["save_interval"]

    ckpt_dir = os.path.join(root, "data", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"Training PPO for {total_steps} timesteps ({n_updates} updates)")
    print(f"  Steps per update: {steps_per_update}")
    print(f"  Learning rate: {ppo_config.lr}")
    print(f"  Batch size: {ppo_config.batch_size}")
    print()

    start_time = time.time()

    for update in range(1, n_updates + 1):
        rollout_info = trainer.collect_rollout(env, steps_per_update)
        stats = trainer.update()

        elapsed = time.time() - start_time
        total_timesteps_so_far = update * steps_per_update

        # Print progress
        recent_rewards = trainer.logs["episode_rewards"][-10:]
        recent_picks = trainer.logs["episode_picks"][-10:]
        recent_sorts = trainer.logs["episode_correct_sorts"][-10:]

        if recent_rewards:
            avg_r = np.mean(recent_rewards)
            avg_picks = np.mean(recent_picks)
            avg_sorts = np.mean(recent_sorts)
        else:
            avg_r = avg_picks = avg_sorts = 0.0

        print(f"Update {update}/{n_updates} | "
              f"Steps: {total_timesteps_so_far} | "
              f"Reward: {avg_r:.1f} | "
              f"Picks: {avg_picks:.1f} | "
              f"Sorts: {avg_sorts:.1f} | "
              f"P.Loss: {stats['policy_loss']:.4f} | "
              f"V.Loss: {stats['value_loss']:.4f} | "
              f"Entropy: {stats['entropy']:.3f} | "
              f"Time: {elapsed:.0f}s")

        if update % save_interval == 0:
            path = os.path.join(ckpt_dir, f"ppo_step{total_timesteps_so_far}.pt")
            trainer.save(path)
            print(f"  Checkpoint saved: {path}")

    # Save final model
    final_path = os.path.join(ckpt_dir, "ppo_final.pt")
    trainer.save(final_path)
    print(f"\nFinal model saved: {final_path}")

    # Save training logs
    log_path = os.path.join(root, "data", "results", "training_logs.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(trainer.logs, f, indent=2, default=lambda x: float(x))
    print(f"Training logs saved: {log_path}")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Total episodes: {len(trainer.logs['episode_rewards'])}")

    if trainer.logs["episode_rewards"]:
        print(f"Final avg reward (last 20): {np.mean(trainer.logs['episode_rewards'][-20:]):.2f}")
        print(f"Final avg picks (last 20): {np.mean(trainer.logs['episode_picks'][-20:]):.2f}")
        print(f"Final avg sorts (last 20): {np.mean(trainer.logs['episode_correct_sorts'][-20:]):.2f}")

    env.close()


if __name__ == "__main__":
    main()
