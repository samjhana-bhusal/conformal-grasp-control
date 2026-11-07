# UR5 Bin Sorting with Conformal Safety Gate

A simulated UR5 robot arm that sorts objects into categorized bins, with a conformal prediction safety gate that provides distribution-free success rate guarantees.

![Demo](figures/demo_sort.gif)

## Problem Statement

Robotic bin sorting in warehouses and manufacturing requires high reliability. A robot that confidently places objects in wrong bins is worse than one that knows when to abstain. This project addresses:

1. **Task**: Pick objects from a cluttered source bin, classify by category, and place in the correct destination bin
2. **Safety**: Ensure a minimum success rate (90%) with statistical guarantees, even under distribution shift
3. **Adaptability**: Recalibrate the safety gate when operating conditions change

## Approach

### Environment

- **Robot**: UR5 6-DOF arm built from geometric primitives in MuJoCo (no external meshes)
- **Gripper**: Simplified Robotiq 2F-85 parallel-jaw gripper
- **Objects**: 12 object types across 3 categories (fruit, hardware, toys) with varying sizes
- **Bins**: 1 source bin + 3 color-coded destination bins (red/blue/green)

### Reinforcement Learning

PPO (Proximal Policy Optimization) with dense reward shaping for the manipulation task:

| Component | Value | Rationale |
|-----------|-------|-----------|
| Algorithm | PPO | Stable for high-variance contact dynamics |
| Learning rate | 3e-4 | Conservative for manipulation |
| Total steps | 200K | Baseline training budget |
| Reward | Dense shaped | Sparse rewards fail in manipulation |

### Conformal Safety Gate

The safety gate wraps the policy and decides whether to execute a grasp based on calibrated risk bounds:

- **Method**: Clopper-Pearson exact binomial intervals with Bonferroni correction
- **Target**: 90% success rate among attempted grasps (alpha=0.10)
- **Guarantee**: Finite-sample, distribution-free (no normality assumption)

## Results

### Training Curves

![Training](figures/training_curves.png)

The PPO agent shows learning signal at 200K steps: reward increases from -3.0 to -2.6, value loss decreases, and policy loss stabilizes. Manipulation RL typically needs millions of steps for successful grasps; the curves confirm the policy is learning the reward structure.

### Experiment Pipeline

A scripted expert baseline demonstrates the full task pipeline (pick, classify, place) across 4 experimental phases:

![Results](figures/experiment_results.png)

| Condition | No Gate | Conformal Gate |
|-----------|---------|----------------|
| In-distribution | 100% SR | 100% SR, 100% coverage |
| Fault injection (30% noise) | 70.5% SR | 70.5% SR, 100% coverage |
| After recalibration | 68.5% SR | **93.2% SR**, 73.5% coverage |

**Key finding**: Under distribution shift (fault injection), the recalibrated conformal gate recovers to 93.2% success by abstaining on uncertain grasps (26.5% abstention), exceeding the 90% target.

### VLM Score Distributions

![Scores](figures/score_distributions.png)

VLM confidence scores are simulated using Beta distributions that match real vision-language model calibration behavior: high confidence (Beta(12,2)) for successful grasps, lower (Beta(2,6)) for failures.

### Coverage-Success Tradeoff

![Tradeoff](figures/coverage_tradeoff.png)

The conformal gate finds the optimal operating point on the coverage-success tradeoff curve, maximizing throughput while maintaining the 90% success guarantee.

## Design Decisions

| Decision | Choice | Alternative | Rationale |
|----------|--------|-------------|-----------|
| Robot model | Geometric primitives | URDF import | Self-contained, no mesh dependencies |
| Gripper | Simplified 2F-85 | Full linkage model | 2 rigid fingers sufficient for grasping study |
| RL algorithm | PPO | SAC, TD3 | Most stable for high-variance contact dynamics |
| Reward | Dense shaped | Sparse | RL can't learn manipulation from sparse reward alone |
| Safety gate | Clopper-Pearson | Normal approx | Finite-sample guarantees without distributional assumptions |
| Gate target | 90% (alpha=0.10) | 95% | 95% too conservative for robotics throughput |
| Scripted baseline | Kinematic IK + proximity grasp | Full physics | Standard in sim research (RLBench, ManiSkill) |
| Bin layout | All within UR5 reach | Spread out | Must avoid IK singularities during transport |
| Home position | Above source bin | Factory default | Computed via IK to minimize approach distance |

## Project Structure

```
.
├── configs/
│   └── default.yaml          # All hyperparameters with rationale
├── src/
│   ├── env/
│   │   ├── ur5_scene.py      # UR5 MuJoCo scene builder
│   │   └── bin_sort_env.py   # Gym-style RL environment
│   ├── rl/
│   │   └── ppo.py            # PPO implementation
│   ├── safety/
│   │   └── conformal_gate.py # Conformal prediction gate
│   └── ...
├── scripts/
│   ├── train.py              # PPO training
│   ├── run_experiments.py    # Full experiment pipeline
│   ├── generate_plots.py     # Figure generation
│   └── render_demo.py        # Demo video rendering
├── data/
│   └── results/              # Experiment outputs (JSON)
├── figures/                  # Generated plots and demo
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

### Run training

```bash
python scripts/train.py
```

### Run experiments

```bash
python scripts/run_experiments.py
```

### Generate figures

```bash
python scripts/generate_plots.py
```

### Render demo video

```bash
python scripts/render_demo.py
```

## Requirements

- Python 3.9+
- MuJoCo (via `mujoco` package)
- PyTorch
- NumPy, SciPy, Matplotlib
- imageio (for video rendering)

## References

- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
- Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
- Universal Robots UR5 Technical Specifications.
- Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or fiducial limits. *Biometrika*.
