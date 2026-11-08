#!/usr/bin/env python3
"""
Run the full experiment pipeline for recycling waste sorting.

Design decision log:
  - Scripted expert as primary baseline: RL needs millions of steps for
    manipulation. The scripted expert demonstrates the task pipeline clearly
    while the RL curves show learning progress. Standard practice in
    manipulation research (CLIPort, SORNet).
  - Contamination modeled via object ambiguity: high-ambiguity items (soiled
    cardboard, wax cups) are misclassified at rates proportional to their
    ambiguity score. This is the core mechanism that makes the safety gate
    necessary — without it, contaminants enter recycling streams.
  - VLM scores reflect visual confidence, not ground truth: ambiguous items
    that are misclassified can still get moderate VLM scores, making them
    hard for a fixed threshold to catch.
  - 200 trials per condition: tight confidence intervals (±5-7%).
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
from src.env.bin_sort_env import BinSortEnv
from src.safety.conformal_gate import ConformalGate


def scripted_pick_and_place(env, target_obj, max_steps=200):
    """Scripted expert: kinematic IK with proximity-based grasping."""
    import mujoco

    bid = env._object_body_ids.get(target_obj["name"], 0)
    if bid <= 0:
        return {"pick_success": False, "correct_class": False, "steps": 0}

    site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
    jac_p = np.zeros((3, env.model.nv))
    jac_r = np.zeros((3, env.model.nv))

    def ik_solve(q_start, target_pos, tol=0.005, max_iters=200):
        q = q_start.copy()
        for _ in range(max_iters):
            env.data.qpos[:6] = q
            mujoco.mj_forward(env.model, env.data)
            err = target_pos - env.data.site_xpos[site_id]
            if np.linalg.norm(err) < tol:
                return q, True
            mujoco.mj_jacSite(env.model, env.data, jac_p, jac_r, site_id)
            j = jac_p[:, :6]
            dq = j.T @ np.linalg.solve(j @ j.T + 1e-6 * np.eye(3), err)
            q = np.clip(q + np.clip(dq, -0.1, 0.1), -3.14, 3.14)
        return q, False

    def set_arm(q_target, n_interp=40):
        q0 = env.data.qpos[:6].copy()
        for i in range(n_interp):
            t = (i + 1) / n_interp
            q_i = q0 + t * (q_target - q0)
            env.data.qpos[:6] = q_i
            env.data.qvel[:6] = 0.0
            env.data.ctrl[:6] = q_i
            mujoco.mj_forward(env.model, env.data)
            env._update_attached()

    obj_pos = env.data.xpos[bid].copy()
    q_current = env.data.qpos[:6].copy()

    above = obj_pos.copy()
    above[2] += 0.12
    q_above, ok = ik_solve(q_current, above)
    if not ok:
        return {"pick_success": False, "correct_class": False, "steps": 0}
    set_arm(q_above, n_interp=30)

    grasp_target = obj_pos.copy()
    grasp_target[2] += 0.04
    q_grasp, ok = ik_solve(q_above, grasp_target)
    if not ok:
        return {"pick_success": False, "correct_class": False, "steps": 0}
    set_arm(q_grasp, n_interp=20)

    env.data.qpos[:6] = q_grasp
    mujoco.mj_forward(env.model, env.data)
    grip_pos = env.data.site_xpos[site_id].copy()
    obj_now = env.data.xpos[bid].copy()
    dist = np.linalg.norm(grip_pos - obj_now)

    if dist > 0.06:
        return {"pick_success": False, "correct_class": False, "steps": 0}

    env._attach(target_obj["name"])

    lift = np.array([obj_pos[0], obj_pos[1], 0.65])
    q_lift, _ = ik_solve(q_grasp, lift)
    set_arm(q_lift, n_interp=40)

    if env._attached_obj is None:
        return {"pick_success": False, "correct_class": False, "steps": 0}

    waypoint = np.array([0.0, 0.0, 0.70])
    q_way, _ = ik_solve(q_lift, waypoint, tol=0.02)
    set_arm(q_way, n_interp=50)

    from src.env.ur5_scene import get_bin_position
    bin_pos = get_bin_position(target_obj["target_bin"])
    above_bin = bin_pos.copy()
    above_bin[2] = 0.55
    q_bin, _ = ik_solve(q_way, above_bin, tol=0.02)
    set_arm(q_bin, n_interp=50)

    if env._attached_obj is None:
        return {"pick_success": False, "correct_class": False, "steps": 0}

    env._detach()
    for _ in range(80):
        mujoco.mj_step(env.model, env.data)

    final_pos = env.data.xpos[bid].copy()
    correct_bin_pos = get_bin_position(target_obj["target_bin"])
    dist_to_bin = np.linalg.norm(final_pos[:2] - correct_bin_pos[:2])
    correct = dist_to_bin < 0.15

    return {"pick_success": True, "correct_class": correct, "steps": 1}


def simulate_vlm_score(obj, is_correct, rng):
    """Simulate VLM confidence with ambiguity-aware confusion.

    Key insight: ambiguous items that are MISCLASSIFIED can still produce
    moderate VLM scores (the VLM is fooled by visual similarity). This is
    what makes a fixed threshold insufficient — the conformal gate learns
    from calibration data that moderate scores for certain item types are
    unreliable.
    """
    ambiguity = obj.get("ambiguity", 0.1)

    if is_correct:
        alpha = max(3.0, 12.0 * (1.0 - ambiguity * 0.5))
        score = rng.beta(alpha, 2.0)
    else:
        if ambiguity > 0.3:
            score = rng.beta(5.0, 4.0)
        else:
            score = rng.beta(2.0, 6.0)

    return float(np.clip(score + rng.normal(0, 0.02), 0.01, 0.99))


def run_trials(env, config, n_trials, base_seed, fault_level=0.0):
    """Run n_trials of scripted pick-and-sort attempts.

    Args:
        fault_level: extra misclassification probability (0-1) simulating
            degraded conditions (dirty sensor, unusual waste stream).
    """
    rng = np.random.RandomState(base_seed)
    results = []

    for i in range(n_trials):
        seed = base_seed + i
        env.reset(seed=seed)
        objects = env.objects

        graspable = [o for o in objects if o["size"] <= 0.030]
        if graspable:
            target = graspable[rng.randint(0, len(graspable))]
        else:
            target = objects[rng.randint(0, len(objects))]

        trial = scripted_pick_and_place(env, target)

        if trial["pick_success"]:
            ambiguity = target.get("ambiguity", 0.1)
            misclass_prob = min(0.85, ambiguity * 0.8 + fault_level)
            correct_class = rng.random() >= misclass_prob
        else:
            correct_class = False

        overall_success = trial["pick_success"] and correct_class
        is_contamination = trial["pick_success"] and not correct_class

        score = simulate_vlm_score(target, overall_success, rng)

        results.append({
            "seed": seed,
            "object": target["name"],
            "category": target["category"],
            "ambiguity": target.get("ambiguity", 0.1),
            "size": target["size"],
            "pick_success": trial["pick_success"],
            "correct_class": correct_class,
            "success": overall_success,
            "contamination": is_contamination,
            "vlm_score": score,
        })

        if (i + 1) % 50 == 0:
            sr = np.mean([r["success"] for r in results])
            cr = np.mean([r["contamination"] for r in results])
            print(f"    Trial {i+1}/{n_trials}: SR={sr:.2%}, Contam={cr:.2%}")

    return results


def clopper_pearson(k, n, alpha=0.05):
    from scipy import stats as sp_stats
    if k == 0:
        return (0.0, 1.0 - (alpha/2)**(1.0/n))
    if k == n:
        return ((alpha/2)**(1.0/n), 1.0)
    lo = sp_stats.beta.ppf(alpha/2, k, n - k + 1)
    hi = sp_stats.beta.ppf(1 - alpha/2, k + 1, n - k)
    return (lo, hi)


def eval_with_gate(results, threshold):
    """Evaluate gated performance with contamination tracking."""
    scores = np.array([r["vlm_score"] for r in results])
    successes = np.array([r["success"] for r in results])
    contaminations = np.array([r["contamination"] for r in results])
    pick_successes = np.array([r["pick_success"] for r in results])

    mask = scores >= threshold
    n_attempted = mask.sum()
    if n_attempted == 0:
        return {
            "sr": 0, "sr_ci": (0, 0), "contamination_rate": 0,
            "coverage": 0, "abstention": 1.0,
            "n_attempted": 0, "n_success": 0, "n_contamination": 0,
            "n_total": len(results), "diversion_rate": 0,
        }

    n_success = int(successes[mask].sum())
    n_contamination = int(contaminations[mask].sum())
    n_picked = int(pick_successes[mask].sum())
    sr = n_success / n_attempted
    contamination_rate = n_contamination / n_attempted if n_attempted > 0 else 0
    ci = clopper_pearson(n_success, n_attempted)

    total_recyclable = sum(1 for r in results if r["category"] != "landfill")
    diverted = sum(1 for j, r in enumerate(results)
                   if mask[j] and r["success"] and r["category"] != "landfill")
    diversion_rate = diverted / total_recyclable if total_recyclable > 0 else 0

    return {
        "sr": float(sr),
        "sr_ci": (float(ci[0]), float(ci[1])),
        "contamination_rate": float(contamination_rate),
        "coverage": float(mask.mean()),
        "abstention": float(1 - mask.mean()),
        "n_attempted": int(n_attempted),
        "n_success": n_success,
        "n_contamination": n_contamination,
        "n_total": len(results),
        "diversion_rate": float(diversion_rate),
    }


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "configs", "default.yaml")) as f:
        config = yaml.safe_load(f)

    sim_config = config["simulation"]
    env = BinSortEnv(sim_config)

    all_results = {}
    n_trials = 200

    # ── Phase 1: Calibration ──────────────────────────────────────
    print("Phase 1: Calibration (200 trials, mixed waste stream)")
    cal_data = run_trials(env, sim_config, n_trials, base_seed=1000)
    cal_scores = np.array([r["vlm_score"] for r in cal_data])
    cal_labels = np.array([r["success"] for r in cal_data])

    gate = ConformalGate(alpha=0.10, delta=0.05)
    cal_result = gate.calibrate(cal_scores, cal_labels)

    cal_contam = np.mean([r["contamination"] for r in cal_data])
    print(f"  Calibration: threshold={cal_result.threshold:.3f}, "
          f"SR={cal_result.empirical_success_rate:.2%}, "
          f"coverage={cal_result.coverage:.2%}, "
          f"raw contamination={cal_contam:.2%}")

    all_results["calibration"] = {
        "threshold": cal_result.threshold,
        "n_trials": n_trials,
        "raw_success_rate": float(cal_labels.mean()),
        "raw_contamination_rate": float(cal_contam),
    }

    # ── Phase 2: In-distribution test ─────────────────────────────
    print("\nPhase 2: In-distribution test (3 seeds x 200 trials)")
    id_results = {}
    for seed in [42, 123, 456]:
        print(f"  Seed {seed}:")
        test_data = run_trials(env, sim_config, n_trials, base_seed=seed * 100)

        no_gate = eval_with_gate(test_data, 0.0)
        raw_gate = eval_with_gate(test_data, 0.5)
        conf_gate = eval_with_gate(test_data, cal_result.threshold)

        id_results[str(seed)] = {
            "no_gate": no_gate,
            "raw_0.5": raw_gate,
            "conformal": conf_gate,
        }
        print(f"    No gate: SR={no_gate['sr']:.2%}, Contam={no_gate['contamination_rate']:.2%}")
        print(f"    Conformal: SR={conf_gate['sr']:.2%}, Contam={conf_gate['contamination_rate']:.2%}, "
              f"Cov={conf_gate['coverage']:.2%}")

    seed_keys = ["42", "123", "456"]
    for method in ["no_gate", "raw_0.5", "conformal"]:
        srs = [id_results[s][method]["sr"] for s in seed_keys]
        covs = [id_results[s][method]["coverage"] for s in seed_keys]
        contams = [id_results[s][method]["contamination_rate"] for s in seed_keys]
        divs = [id_results[s][method]["diversion_rate"] for s in seed_keys]
        id_results[f"agg_{method}"] = {
            "mean_sr": float(np.mean(srs)),
            "std_sr": float(np.std(srs)),
            "mean_cov": float(np.mean(covs)),
            "mean_contamination": float(np.mean(contams)),
            "mean_diversion": float(np.mean(divs)),
        }

    all_results["in_distribution"] = id_results

    # ── Phase 3: Fault injection (contamination wave) ─────────────
    print("\nPhase 3: Fault injection — degraded waste stream (+20% misclass)")
    fault_data = run_trials(env, sim_config, n_trials, base_seed=7000, fault_level=0.20)
    fault_no_gate = eval_with_gate(fault_data, 0.0)
    fault_conf = eval_with_gate(fault_data, cal_result.threshold)

    all_results["fault_injection"] = {
        "no_gate": fault_no_gate,
        "conformal": fault_conf,
    }
    print(f"  No gate:    SR={fault_no_gate['sr']:.2%}, Contam={fault_no_gate['contamination_rate']:.2%}")
    print(f"  Conformal:  SR={fault_conf['sr']:.2%}, Contam={fault_conf['contamination_rate']:.2%}, "
          f"Cov={fault_conf['coverage']:.2%}")

    # ── Phase 4: Recalibration ────────────────────────────────────
    print("\nPhase 4: Recalibration with degraded-stream data")
    clean_cal = run_trials(env, sim_config, 100, base_seed=13000)
    blend_scores = np.concatenate([
        np.array([r["vlm_score"] for r in clean_cal]),
        np.array([r["vlm_score"] for r in fault_data]),
    ])
    blend_labels = np.concatenate([
        np.array([r["success"] for r in clean_cal]),
        np.array([r["success"] for r in fault_data]),
    ])

    gate_recal = ConformalGate(alpha=0.10, delta=0.05)
    recal_result = gate_recal.calibrate(blend_scores, blend_labels)

    test_recal = run_trials(env, sim_config, n_trials, base_seed=15000, fault_level=0.20)
    before = eval_with_gate(test_recal, cal_result.threshold)
    after = eval_with_gate(test_recal, recal_result.threshold)

    all_results["recalibration"] = {
        "original_threshold": cal_result.threshold,
        "new_threshold": recal_result.threshold,
        "before": before,
        "after": after,
    }
    print(f"  Before recal: SR={before['sr']:.2%}, Contam={before['contamination_rate']:.2%}, "
          f"Cov={before['coverage']:.2%}")
    print(f"  After recal:  SR={after['sr']:.2%}, Contam={after['contamination_rate']:.2%}, "
          f"Cov={after['coverage']:.2%}")

    # ── Save ──────────────────────────────────────────────────────
    out_path = os.path.join(root, "data", "results", "experiment_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print(f"\nResults saved: {out_path}")

    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
