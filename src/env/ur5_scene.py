"""
UR5 + bin sorting scene builder for MuJoCo.

Design decision log:
  - UR5 built from geometric primitives (cylinders) rather than loading a URDF.
    Rationale: keeps the project self-contained with zero external mesh dependencies.
    Trade-off: less visual fidelity, but identical kinematic/dynamic behavior.
  - Robotiq 2F-85 gripper simplified to parallel-jaw with two finger pads.
    Rationale: full Robotiq model has 11 links and complex linkage. For a
    grasping study, two rigid fingers with contact pads are sufficient.
  - Three destination bins arranged in a semicircle in front of the robot.
    Rationale: reachable from a fixed base, no need for base rotation.
"""

from __future__ import annotations
import numpy as np


# ── UR5 DH parameters (standard) ────────────────────────────────
# These define the kinematic chain. Each joint is revolute.
# d: offset along previous z    a: offset along rotated x
# alpha: rotation about x       theta: rotation about z (variable)
UR5_DH = [
    {"d": 0.08916, "a": 0.0,     "alpha": np.pi/2},   # shoulder pan
    {"d": 0.0,     "a": -0.4250, "alpha": 0.0},        # shoulder lift
    {"d": 0.0,     "a": -0.3922, "alpha": 0.0},        # elbow
    {"d": 0.1091,  "a": 0.0,     "alpha": np.pi/2},    # wrist 1
    {"d": 0.09465, "a": 0.0,     "alpha": -np.pi/2},   # wrist 2
    {"d": 0.0823,  "a": 0.0,     "alpha": 0.0},        # wrist 3
]

# Link lengths for visual geoms
UR5_LINK_LENGTHS = [0.089, 0.425, 0.392, 0.109, 0.095, 0.082]
UR5_LINK_RADIUS = 0.045


def build_scene_xml(objects: list[dict], config: dict) -> str:
    """Build complete MuJoCo XML for the bin sorting scene.

    Args:
        objects: list of dicts with keys: name, category, size, color, position
        config: simulation config dict
    """
    table_h = config.get("table_height", 0.4)
    bin_w = config.get("bin_width", 0.15)
    bin_d = config.get("bin_depth", 0.12)

    xml = f"""<mujoco model="ur5_bin_sort">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>

  <default>
    <joint damping="5" armature="0.1"/>
    <geom condim="4" friction="1.0 0.005 0.001"/>
    <motor ctrllimited="true"/>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.9 0.9 0.92"
             rgb2="0.75 0.75 0.78" width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <material name="table_mat" rgba="0.35 0.25 0.2 1"/>
    <material name="bin_a" rgba="0.85 0.25 0.25 0.7"/>
    <material name="bin_b" rgba="0.25 0.55 0.85 0.7"/>
    <material name="bin_c" rgba="0.25 0.75 0.35 0.7"/>
    <material name="source_bin" rgba="0.5 0.5 0.5 0.6"/>
    <material name="robot_mat" rgba="0.3 0.3 0.35 1" specular="0.5" shininess="0.5"/>
    <material name="finger_mat" rgba="0.15 0.15 0.15 1"/>
  </asset>

  <worldbody>
    <!-- Floor -->
    <geom name="floor" type="plane" size="2 2 0.01" material="grid_mat"/>

    <!-- Overhead camera -->
    <camera name="overhead_cam" pos="0.0 -0.1 1.4" quat="1 0 0 0" fovy="60"/>
    <!-- Side camera -->
    <camera name="side_cam" pos="0.7 -0.5 0.9" xyaxes="0.610 0.793 0 -0.347 0.267 0.899" fovy="60"/>
    <!-- Front camera -->
    <camera name="front_cam" pos="0.05 -0.85 0.75" xyaxes="1 0 0 0 0.4 0.92" fovy="60"/>

    <!-- Lights -->
    <light pos="0.0 0.0 2.5" dir="0 0 -1" diffuse="0.9 0.9 0.9" specular="0.3 0.3 0.3"/>
    <light pos="0.5 -0.5 2.0" dir="-0.2 0.2 -1" diffuse="0.4 0.4 0.4"/>

    <!-- Work table -->
    <body name="table" pos="0.0 0.0 {table_h/2}">
      <geom type="box" size="0.5 0.4 {table_h/2}" material="table_mat" mass="50"/>
    </body>

    <!-- Source bin (where objects start) -->
    <body name="source_bin" pos="0.35 0.0 {table_h}">
      <geom name="src_base" type="box" size="0.12 0.10 0.005" pos="0 0 0.005" material="source_bin"/>
      <geom name="src_wall_f" type="box" size="0.12 0.005 0.04" pos="0 0.10 0.04" material="source_bin"/>
      <geom name="src_wall_b" type="box" size="0.12 0.005 0.04" pos="0 -0.10 0.04" material="source_bin"/>
      <geom name="src_wall_l" type="box" size="0.005 0.10 0.04" pos="-0.12 0 0.04" material="source_bin"/>
      <geom name="src_wall_r" type="box" size="0.005 0.10 0.04" pos="0.12 0 0.04" material="source_bin"/>
    </body>

    <!-- Destination bins (A=red, B=blue, C=green) -->
    <body name="bin_a" pos="-0.35 0.15 {table_h}">
      <geom name="bina_base" type="box" size="{bin_w/2} {bin_d/2} 0.005" pos="0 0 0.005" material="bin_a"/>
      <geom name="bina_f" type="box" size="{bin_w/2} 0.004 0.035" pos="0 {bin_d/2} 0.035" material="bin_a"/>
      <geom name="bina_b" type="box" size="{bin_w/2} 0.004 0.035" pos="0 {-bin_d/2} 0.035" material="bin_a"/>
      <geom name="bina_l" type="box" size="0.004 {bin_d/2} 0.035" pos="{-bin_w/2} 0 0.035" material="bin_a"/>
      <geom name="bina_r" type="box" size="0.004 {bin_d/2} 0.035" pos="{bin_w/2} 0 0.035" material="bin_a"/>
    </body>

    <body name="bin_b" pos="-0.35 -0.15 {table_h}">
      <geom name="binb_base" type="box" size="{bin_w/2} {bin_d/2} 0.005" pos="0 0 0.005" material="bin_b"/>
      <geom name="binb_f" type="box" size="{bin_w/2} 0.004 0.035" pos="0 {bin_d/2} 0.035" material="bin_b"/>
      <geom name="binb_b" type="box" size="{bin_w/2} 0.004 0.035" pos="0 {-bin_d/2} 0.035" material="bin_b"/>
      <geom name="binb_l" type="box" size="0.004 {bin_d/2} 0.035" pos="{-bin_w/2} 0 0.035" material="bin_b"/>
      <geom name="binb_r" type="box" size="0.004 {bin_d/2} 0.035" pos="{bin_w/2} 0 0.035" material="bin_b"/>
    </body>

    <body name="bin_c" pos="-0.25 -0.20 {table_h}">
      <geom name="binc_base" type="box" size="{bin_w/2} {bin_d/2} 0.005" pos="0 0 0.005" material="bin_c"/>
      <geom name="binc_f" type="box" size="{bin_w/2} 0.004 0.035" pos="0 {bin_d/2} 0.035" material="bin_c"/>
      <geom name="binc_b" type="box" size="{bin_w/2} 0.004 0.035" pos="0 {-bin_d/2} 0.035" material="bin_c"/>
      <geom name="binc_l" type="box" size="0.004 {bin_d/2} 0.035" pos="{-bin_w/2} 0 0.035" material="bin_c"/>
      <geom name="binc_r" type="box" size="0.004 {bin_d/2} 0.035" pos="{bin_w/2} 0 0.035" material="bin_c"/>
    </body>

    <!-- UR5 Robot Arm -->
    <body name="ur5_base" pos="0.0 0.0 {table_h}">
      <geom type="cylinder" size="0.06 0.02" material="robot_mat" mass="4"/>

      <!-- Joint 1: Shoulder Pan (rotates about Z) -->
      <body name="shoulder_link" pos="0 0 0.089">
        <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom type="cylinder" size="0.045 0.045" material="robot_mat" mass="3.7"/>

        <!-- Joint 2: Shoulder Lift (rotates about Y) -->
        <body name="upper_arm" pos="0 0 0">
          <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.425" size="0.04" material="robot_mat" mass="8.4"/>

          <!-- Joint 3: Elbow (rotates about Y) -->
          <body name="forearm" pos="0 0 0.425">
            <joint name="elbow" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.392" size="0.035" material="robot_mat" mass="2.2"/>

            <!-- Joint 4: Wrist 1 -->
            <body name="wrist1" pos="0 0 0.392">
              <joint name="wrist_1" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
              <geom type="cylinder" size="0.03 0.03" material="robot_mat" mass="1.2"/>

              <!-- Joint 5: Wrist 2 -->
              <body name="wrist2" pos="0 0 0.109">
                <joint name="wrist_2" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                <geom type="cylinder" size="0.03 0.025" material="robot_mat" mass="1.2"/>

                <!-- Joint 6: Wrist 3 -->
                <body name="wrist3" pos="0 0 0.095">
                  <joint name="wrist_3" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
                  <geom type="cylinder" size="0.025 0.02" material="robot_mat" mass="0.2"/>

                  <!-- Gripper mount -->
                  <body name="gripper_base" pos="0 0 0.06">
                    <geom type="box" size="0.025 0.03 0.015" material="finger_mat" mass="0.5"/>

                    <!-- Left finger -->
                    <body name="finger_left" pos="0 0.03 0.0">
                      <joint name="finger_left_joint" type="slide" axis="0 1 0" range="-0.04 0.0" damping="20"/>
                      <geom name="finger_left_pad" type="box" size="0.012 0.005 0.04"
                            pos="0 0 -0.04" material="finger_mat" mass="0.1"
                            friction="2.0 0.01 0.001"/>
                    </body>

                    <!-- Right finger -->
                    <body name="finger_right" pos="0 -0.03 0.0">
                      <joint name="finger_right_joint" type="slide" axis="0 -1 0" range="-0.04 0.0" damping="20"/>
                      <geom name="finger_right_pad" type="box" size="0.012 0.005 0.04"
                            pos="0 0 -0.04" material="finger_mat" mass="0.1"
                            friction="2.0 0.01 0.001"/>
                    </body>

                    <!-- Grip site (center of fingertips) -->
                    <site name="grip_site" pos="0 0 -0.05" size="0.005"/>

                    <!-- Wrist camera -->
                    <camera name="wrist_cam" pos="0 0.05 -0.02" xyaxes="1 0 0 0 0 -1" fovy="70"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Objects (placed dynamically) -->
{_objects_xml(objects, table_h)}

  </worldbody>

  <actuator>
    <position name="act_shoulder_pan"  joint="shoulder_pan"  kp="4000" ctrlrange="-3.14 3.14"/>
    <position name="act_shoulder_lift" joint="shoulder_lift" kp="4000" ctrlrange="-3.14 3.14"/>
    <position name="act_elbow"         joint="elbow"         kp="4000" ctrlrange="-3.14 3.14"/>
    <position name="act_wrist_1"       joint="wrist_1"       kp="2000" ctrlrange="-3.14 3.14"/>
    <position name="act_wrist_2"       joint="wrist_2"       kp="2000" ctrlrange="-3.14 3.14"/>
    <position name="act_wrist_3"       joint="wrist_3"       kp="2000" ctrlrange="-3.14 3.14"/>
    <position name="act_finger_left"   joint="finger_left_joint"  kp="200" ctrlrange="-0.04 0.0"/>
    <position name="act_finger_right"  joint="finger_right_joint" kp="200" ctrlrange="-0.04 0.0"/>
  </actuator>
</mujoco>
"""
    return xml


def _objects_xml(objects: list[dict], table_h: float) -> str:
    """Generate XML for each object in the source bin."""
    lines = []
    for obj in objects:
        name = obj["name"]
        pos = obj["position"]
        size = obj["size"]
        rgba = obj["color"]
        shape = obj.get("shape", "sphere")

        if shape == "sphere":
            geom = f'<geom name="{name}_geom" type="sphere" size="{size}" rgba="{rgba}" mass="0.05"/>'
        elif shape == "box":
            geom = f'<geom name="{name}_geom" type="box" size="{size} {size} {size}" rgba="{rgba}" mass="0.08"/>'
        elif shape == "cylinder":
            geom = f'<geom name="{name}_geom" type="cylinder" size="{size} {size*1.5}" rgba="{rgba}" mass="0.06"/>'
        else:
            geom = f'<geom name="{name}_geom" type="sphere" size="{size}" rgba="{rgba}" mass="0.05"/>'

        lines.append(f"""    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">
      <joint name="{name}_free" type="free"/>
      {geom}
    </body>""")

    return "\n".join(lines)


# ── Object categories for bin sorting ────────────────────────────
OBJECT_CATEGORIES = {
    "fruit": {
        "bin": "bin_a",
        "items": [
            {"name": "apple",      "shape": "sphere",   "size": 0.025, "color": "0.8 0.15 0.1 1"},
            {"name": "orange",     "shape": "sphere",   "size": 0.028, "color": "0.95 0.55 0.1 1"},
            {"name": "lemon",      "shape": "sphere",   "size": 0.022, "color": "0.95 0.9 0.2 1"},
            {"name": "cherry",     "shape": "sphere",   "size": 0.012, "color": "0.7 0.05 0.1 1"},
        ],
    },
    "hardware": {
        "bin": "bin_b",
        "items": [
            {"name": "bolt",       "shape": "cylinder", "size": 0.008, "color": "0.6 0.6 0.6 1"},
            {"name": "nut",        "shape": "cylinder", "size": 0.006, "color": "0.65 0.65 0.6 1"},
            {"name": "washer",     "shape": "cylinder", "size": 0.010, "color": "0.55 0.55 0.55 1"},
            {"name": "screw",      "shape": "cylinder", "size": 0.005, "color": "0.5 0.5 0.5 1"},
        ],
    },
    "toy": {
        "bin": "bin_c",
        "items": [
            {"name": "block_red",  "shape": "box",      "size": 0.018, "color": "0.9 0.2 0.2 1"},
            {"name": "block_blue", "shape": "box",      "size": 0.018, "color": "0.2 0.3 0.9 1"},
            {"name": "ball_green", "shape": "sphere",   "size": 0.015, "color": "0.2 0.8 0.3 1"},
            {"name": "marble",     "shape": "sphere",   "size": 0.010, "color": "0.4 0.7 0.9 1"},
        ],
    },
}


def create_random_objects(rng: np.random.RandomState, n_objects: int = 6,
                          source_center: tuple = (0.35, 0.0),
                          table_h: float = 0.4) -> list[dict]:
    """Create a random set of objects placed in the source bin."""
    all_items = []
    for cat_name, cat in OBJECT_CATEGORIES.items():
        for item in cat["items"]:
            all_items.append({**item, "category": cat_name, "target_bin": cat["bin"]})

    chosen = list(rng.choice(all_items, size=min(n_objects, len(all_items)), replace=False))

    spread = 0.06
    for i, obj in enumerate(chosen):
        angle = 2 * np.pi * i / len(chosen) + rng.uniform(-0.3, 0.3)
        r = rng.uniform(0.02, spread)
        x = source_center[0] + r * np.cos(angle)
        y = source_center[1] + r * np.sin(angle)
        z = table_h + 0.02 + obj["size"]
        obj["position"] = [round(x, 4), round(y, 4), round(z, 4)]

    return chosen


def get_home_qpos() -> np.ndarray:
    """UR5 home position: gripper above source bin, pointing down."""
    return np.array([0.0, 2.93, -2.16, -0.85, 0.71, 2.98])


def get_bin_position(bin_name: str) -> np.ndarray:
    """Get the center position of a destination bin."""
    positions = {
        "bin_a": np.array([-0.35, 0.15, 0.45]),
        "bin_b": np.array([-0.35, -0.15, 0.45]),
        "bin_c": np.array([-0.25, -0.20, 0.45]),
    }
    return positions.get(bin_name, np.array([0.0, 0.0, 0.45]))
