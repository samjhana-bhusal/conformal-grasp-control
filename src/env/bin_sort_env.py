"""
Recycling sorting environment for RL training.

The agent controls a UR5 robot arm to:
  1. Pick waste items from a mixed-waste source bin
  2. Classify them (via VLM confidence score)
  3. Place them in the correct stream: recyclable, compost, or landfill

Observation space:
  - 6 joint angles (UR5)
  - 2 finger positions
  - 3D grip site position
  - 3D target object position
  - object size
  - VLM confidence score (simulated, ambiguity-aware)
  - category encoding (one-hot, 3 classes: recyclable/compost/landfill)
  Total: 6 + 2 + 3 + 3 + 1 + 1 + 3 = 19

Action space:
  - 6 delta joint angles (continuous, clipped)
  - 1 gripper command (open/close, continuous)
  Total: 7

Reward structure:
  Design decision: asymmetric dense reward reflecting real-world contamination costs.
  Rationale: in recycling, contamination (wrong item in recycling/compost stream)
  is far more costly than a missed recyclable routed to landfill. One contaminated
  item can render an entire batch non-recyclable.

  +10.0  correct sort to any bin
  +3.0   successful pick (object lifted)
  -20.0  contamination (wrong item in recyclable or compost stream)
  -2.0   missed value (recyclable/compostable sent to landfill)
  -5.0   dropped object
  -0.01  per-step cost (encourages efficiency)
  +0.5   distance reduction toward target (shaping)
"""

from __future__ import annotations

import numpy as np
import mujoco
from typing import Optional

from .ur5_scene import (
    build_scene_xml, create_random_objects, get_home_qpos, get_bin_position
)


class BinSortEnv:
    """Gym-style environment for UR5 bin sorting."""

    N_JOINTS = 6
    N_FINGERS = 2
    OBS_DIM = 19
    ACT_DIM = 7
    MAX_STEPS = 300

    FINGER_OPEN = 0.0
    FINGER_CLOSED = -0.04

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.data = None
        self.renderer = None
        self.objects = []
        self._object_body_ids = {}
        self._step_count = 0
        self._target_idx = 0
        self._phase = "approach"  # approach | grasp | transport | place
        self._attached_obj = None
        self._attached_offset = None
        self._attached_body_id = None
        self._prev_dist = None
        self._episode_reward = 0.0
        self._episode_picks = 0
        self._episode_correct = 0

    def reset(self, seed: int = 0) -> np.ndarray:
        """Reset environment with new random objects. Returns observation."""
        rng = np.random.RandomState(seed)
        self.objects = create_random_objects(
            rng,
            n_objects=self.config.get("n_objects", 6),
            table_h=self.config.get("table_height", 0.4),
        )

        xml = build_scene_xml(self.objects, self.config)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        home = get_home_qpos()
        self.data.qpos[:self.N_JOINTS] = home
        self.data.ctrl[:self.N_JOINTS] = home
        self.data.ctrl[self.N_JOINTS:self.N_JOINTS + self.N_FINGERS] = self.FINGER_OPEN

        self._object_body_ids = {}
        for obj in self.objects:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj["name"])
            if bid >= 0:
                self._object_body_ids[obj["name"]] = bid

        # Settle
        for _ in range(500):
            mujoco.mj_step(self.model, self.data)

        if self.renderer is not None:
            self.renderer.close()
        self.renderer = mujoco.Renderer(
            self.model,
            height=self.config.get("render_height", 480),
            width=self.config.get("render_width", 640),
        )

        self._step_count = 0
        self._target_idx = 0
        self._phase = "approach"
        self._attached_obj = None
        self._attached_offset = None
        self._attached_body_id = None
        self._prev_dist = None
        self._episode_reward = 0.0
        self._episode_picks = 0
        self._episode_correct = 0

        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Take one step. Returns (obs, reward, done, info)."""
        action = np.clip(action, -1.0, 1.0)

        # Scale actions
        delta_joints = action[:self.N_JOINTS] * 0.05  # max 0.05 rad per step
        gripper_cmd = action[self.N_JOINTS]  # -1 = close, +1 = open

        # Apply joint deltas
        current_q = self.data.qpos[:self.N_JOINTS].copy()
        new_q = np.clip(current_q + delta_joints, -3.14, 3.14)
        self.data.ctrl[:self.N_JOINTS] = new_q

        # Gripper
        finger_target = self.FINGER_OPEN if gripper_cmd > 0 else self.FINGER_CLOSED
        self.data.ctrl[self.N_JOINTS] = finger_target
        self.data.ctrl[self.N_JOINTS + 1] = finger_target

        # Physics steps
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
            self._update_attached()

        self._step_count += 1

        # Compute reward
        reward = self._compute_reward()
        self._episode_reward += reward

        done = self._step_count >= self.MAX_STEPS or self._target_idx >= len(self.objects)

        info = {
            "step": self._step_count,
            "phase": self._phase,
            "picks": self._episode_picks,
            "correct_sorts": self._episode_correct,
            "episode_reward": self._episode_reward,
        }

        return self._get_obs(), reward, done, info

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        joint_angles = self.data.qpos[:self.N_JOINTS].copy()
        finger_pos = self.data.qpos[self.N_JOINTS:self.N_JOINTS + self.N_FINGERS].copy()

        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
        grip_pos = self.data.site_xpos[site_id].copy()

        if self._target_idx < len(self.objects):
            target = self.objects[self._target_idx]
            bid = self._object_body_ids.get(target["name"], 0)
            obj_pos = self.data.xpos[bid].copy() if bid > 0 else np.zeros(3)
            obj_size = target["size"]
            vlm_score = self._simulate_vlm_score(target)
            category = self._category_onehot(target["category"])
        else:
            obj_pos = np.zeros(3)
            obj_size = 0.0
            vlm_score = 0.0
            category = np.zeros(3)

        obs = np.concatenate([
            joint_angles,       # 6
            finger_pos,         # 2
            grip_pos,           # 3
            obj_pos,            # 3
            [obj_size],         # 1
            [vlm_score],        # 1
            category,           # 3
        ]).astype(np.float32)

        return obs

    def _simulate_vlm_score(self, obj: dict) -> float:
        """Simulate VLM classification confidence with ambiguity modeling.

        Design decision: confidence depends on item ambiguity, not just size.
        A wax-coated paper cup (ambiguity=0.6) gets a lower, noisier score than
        a clean plastic bottle (ambiguity=0.05).  This drives the need for the
        conformal gate: high-ambiguity landfill items produce moderate VLM scores
        that a fixed threshold cannot reliably separate.
        """
        rng = np.random.RandomState(hash(obj["name"]) % 2**31)
        ambiguity = obj.get("ambiguity", 0.1)
        alpha_param = max(2.0, 15.0 * (1.0 - ambiguity))
        beta_param = max(1.5, 2.0 + 10.0 * ambiguity)
        return float(np.clip(rng.beta(alpha_param, beta_param), 0.01, 0.99))

    def _category_onehot(self, category: str) -> np.ndarray:
        cats = {"recyclable": 0, "compost": 1, "landfill": 2}
        vec = np.zeros(3)
        vec[cats.get(category, 0)] = 1.0
        return vec

    def _compute_reward(self) -> float:
        """Dense reward with asymmetric contamination penalties."""
        if self._target_idx >= len(self.objects):
            return 0.0

        reward = -0.01  # step cost

        target = self.objects[self._target_idx]
        bid = self._object_body_ids.get(target["name"], 0)
        if bid <= 0:
            return reward

        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
        grip_pos = self.data.site_xpos[site_id].copy()
        obj_pos = self.data.xpos[bid].copy()

        if self._phase == "approach":
            dist = np.linalg.norm(grip_pos - obj_pos)
            if self._prev_dist is not None:
                reward += 0.5 * (self._prev_dist - dist)
            self._prev_dist = dist

            if self._check_grasp(target["name"]):
                self._phase = "transport"
                self._attach(target["name"])
                self._episode_picks += 1
                reward += 3.0
                self._prev_dist = None

        elif self._phase == "transport":
            target_bin_pos = get_bin_position(target["target_bin"])
            dist = np.linalg.norm(grip_pos - target_bin_pos)
            if self._prev_dist is not None:
                reward += 0.5 * (self._prev_dist - dist)
            self._prev_dist = dist

            # Check proximity to ANY bin (correct or wrong)
            all_bins = {"bin_a": "recyclable", "bin_b": "compost", "bin_c": "landfill"}
            for bin_name, bin_stream in all_bins.items():
                bp = get_bin_position(bin_name)
                d = np.linalg.norm(grip_pos - bp)
                if d < 0.06:
                    self._detach()
                    if bin_name == target["target_bin"]:
                        self._episode_correct += 1
                        reward += 10.0
                    elif bin_stream in ("recyclable", "compost"):
                        reward -= 20.0  # contamination
                    else:
                        reward -= 2.0  # missed value (sent to landfill)
                    self._phase = "approach"
                    self._target_idx += 1
                    self._prev_dist = None
                    break

            if self._attached_obj is None and self._phase == "transport":
                reward -= 5.0
                self._phase = "approach"
                self._prev_dist = None

        return reward

    def _check_grasp(self, obj_name: str) -> bool:
        """Check if fingers are contacting the target object."""
        left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "finger_left_pad")
        right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "finger_right_pad")
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{obj_name}_geom")

        finger_ids = {i for i in (left_id, right_id) if i >= 0}
        if obj_id < 0:
            return False

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 in finger_ids and c.geom2 == obj_id) or \
               (c.geom2 in finger_ids and c.geom1 == obj_id):
                return True
        return False

    def _attach(self, obj_name: str):
        """Lock object to gripper."""
        bid = self._object_body_ids.get(obj_name)
        if bid is None:
            return
        hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        if hand_id < 0:
            return
        self._attached_obj = obj_name
        self._attached_offset = self.data.xpos[bid].copy() - self.data.xpos[hand_id].copy()
        self._attached_body_id = bid

    def _detach(self):
        """Release object from gripper."""
        self._attached_obj = None
        self._attached_offset = None
        self._attached_body_id = None

    def _update_attached(self):
        """Move attached object with gripper each physics step."""
        if self._attached_obj is None:
            return
        hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        if hand_id < 0:
            return
        new_pos = self.data.xpos[hand_id].copy() + self._attached_offset
        bid = self._attached_body_id
        jnt_adr = self.model.body_jntadr[bid]
        if jnt_adr >= 0:
            qadr = self.model.jnt_qposadr[jnt_adr]
            self.data.qpos[qadr:qadr+3] = new_pos

    def render(self, camera: str = "side_cam") -> np.ndarray:
        """Render RGB frame from specified camera."""
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def get_grip_pos(self) -> np.ndarray:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
        return self.data.site_xpos[site_id].copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
