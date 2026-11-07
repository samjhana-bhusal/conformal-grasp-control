#!/usr/bin/env python3
"""Render a demo video of the scripted expert performing bin sorting."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco
import yaml
import imageio

from src.env.bin_sort_env import BinSortEnv
from src.env.ur5_scene import get_home_qpos, get_bin_position


def render_frame(env, camera="side_cam"):
    env.renderer.update_scene(env.data, camera)
    return env.renderer.render()


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "configs", "default.yaml")) as f:
        config = yaml.safe_load(f)

    env = BinSortEnv(config["simulation"])
    env.reset(seed=42)

    site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
    jac_p = np.zeros((3, env.model.nv))
    jac_r = np.zeros((3, env.model.nv))

    def ik_solve(q_start, target_pos, tol=0.005):
        q = q_start.copy()
        for _ in range(200):
            env.data.qpos[:6] = q
            mujoco.mj_forward(env.model, env.data)
            err = target_pos - env.data.site_xpos[site_id]
            if np.linalg.norm(err) < tol:
                return q
            mujoco.mj_jacSite(env.model, env.data, jac_p, jac_r, site_id)
            j = jac_p[:, :6]
            dq = j.T @ np.linalg.solve(j @ j.T + 1e-6 * np.eye(3), err)
            q = np.clip(q + np.clip(dq, -0.1, 0.1), -3.14, 3.14)
        return q

    def set_arm_with_frames(q_target, n_interp=40, every=2):
        q0 = env.data.qpos[:6].copy()
        for i in range(n_interp):
            t = (i + 1) / n_interp
            q_i = q0 + t * (q_target - q0)
            env.data.qpos[:6] = q_i
            env.data.qvel[:6] = 0.0
            env.data.ctrl[:6] = q_i
            mujoco.mj_forward(env.model, env.data)
            env._update_attached()
            if i % every == 0:
                frames.append(render_frame(env))

    frames = []

    # Render initial scene from multiple cameras
    for _ in range(15):
        frames.append(render_frame(env, "side_cam"))

    target = [o for o in env.objects if o["size"] <= 0.030][0]
    bid = env._object_body_ids[target["name"]]
    obj_pos = env.data.xpos[bid].copy()
    q_current = env.data.qpos[:6].copy()
    print(f"Picking {target['name']} -> {target['target_bin']}")

    # Move above object
    above = obj_pos.copy()
    above[2] += 0.12
    q_above = ik_solve(q_current, above)
    set_arm_with_frames(q_above, n_interp=40, every=2)

    # Descend
    grasp = obj_pos.copy()
    grasp[2] += 0.04
    q_grasp = ik_solve(q_above, grasp)
    set_arm_with_frames(q_grasp, n_interp=30, every=2)

    # Pause at grasp
    for _ in range(10):
        frames.append(render_frame(env))

    # Attach
    env._attach(target["name"])

    # Lift
    lift = np.array([obj_pos[0], obj_pos[1], 0.65])
    q_lift = ik_solve(q_grasp, lift)
    set_arm_with_frames(q_lift, n_interp=40, every=2)

    # Waypoint
    waypoint = np.array([0.0, 0.0, 0.70])
    q_way = ik_solve(q_lift, waypoint, tol=0.02)
    set_arm_with_frames(q_way, n_interp=50, every=2)

    # Above bin
    bin_pos = get_bin_position(target["target_bin"])
    above_bin = bin_pos.copy()
    above_bin[2] = 0.55
    q_bin = ik_solve(q_way, above_bin, tol=0.02)
    set_arm_with_frames(q_bin, n_interp=50, every=2)

    # Pause
    for _ in range(10):
        frames.append(render_frame(env))

    # Release
    env._detach()
    for i in range(40):
        mujoco.mj_step(env.model, env.data)
        if i % 2 == 0:
            frames.append(render_frame(env))

    # Hold on final frame
    for _ in range(20):
        frames.append(render_frame(env))

    # Save GIF
    gif_path = os.path.join(root, "figures", "demo_sort.gif")
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=15, loop=0)
    print(f"Saved GIF: {gif_path} ({len(frames)} frames)")

    # Also save MP4
    mp4_path = os.path.join(root, "figures", "demo_sort.mp4")
    imageio.mimsave(mp4_path, frames, fps=15)
    print(f"Saved MP4: {mp4_path}")

    env.close()


if __name__ == "__main__":
    main()
