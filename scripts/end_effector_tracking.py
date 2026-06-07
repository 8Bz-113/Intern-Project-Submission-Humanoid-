
from __future__ import annotations

import os
import sys
import csv
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
from isaacsim import SimulationApp

#create isaacsim
simulation_app = SimulationApp({
    "headless": False,
    "disable_viewport_updates": False,
})

import omni.timeline  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim, XformPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from isaacsim.sensors.physics import ContactSensor  # noqa: E402
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path  # noqa: E402
from omni.isaac.core.utils.stage import get_current_stage  # noqa: E402
from pxr import UsdPhysics  # noqa: E402

sys.path.append(os.path.dirname(__file__))
from controller import RobotType, TrossenAIController  # noqa: E402


# Scene / robot configuration

ROBOT_USD_PATH = "./assets/robots/wxai/wxai_base.usd"
ROBOT_SCENE_PATH = "/World/wxai_robot"
GROUND_SCENE_PATH = "/World/ground"
GRIPPER_LINK_PATH = "/World/wxai_robot/gripper_right"

WXAI_ARM_DOF_INDICES = [0, 1, 2, 3, 4, 5]
WXAI_GRIPPER_DOF_INDEX = 6
WXAI_DEFAULT_DOF_POSITIONS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.044, 0.044]

DOWNWARD_ORIENTATION = np.array([-0.7071068, 0.0, -0.7071068, 0.0])

PARKING_START = np.array([-1.0, -2.0, 0.05])
PARKING_SPACING = np.array([0.0, 0.3, 0.0])

# Five objects from the original data-collection file.
YCB_PROP_PATHS = {
    #"006_mustard_bottle": "./assets/ycb/_06_mustard_bottle.usd",
    "007_tuna_fish_can": "./assets/ycb/_07_tuna_fish_can.usd",
    #"009_gelatin_box": "./assets/ycb/_09_gelatin_box.usd",
    #"010_potted_meat_can": "./assets/ycb/_10_potted_meat_can.usd",
    "025_mug": "./assets/ycb/_25_mug.usd",
}

#masses
YCB_MASSES = {
    "006_mustard_bottle": 0.60,
    "007_tuna_fish_can": 0.17,
    "009_gelatin_box": 0.15,
    "010_potted_meat_can": 0.37,
    "025_mug": 0.35,
}

#friction coeffs 
YCB_FRICTION = {
    "006_mustard_bottle": 0.60,
    "007_tuna_fish_can": 0.40,
    "009_gelatin_box": 0.50,
    "010_potted_meat_can": 0.45,
    "025_mug": 0.65,
}

YCB_ORIENTATION_OFFSETS = {
    "006_mustard_bottle": [0.707, 0.707, 0.0, 0.0],
}

# Task / RL configuration

DT = 1.0 / 60.0
MIN_SAFE_Z = 0.28
CLEARANCE_HEIGHT = 0.035
APPROACH_OFFSET = np.array([0.0, 0.0, 0.012])
PLACE_OFFSET = np.array([0.0, 0.0, 0.045])
HOME_POSITION = np.array([0.2, 0.0, 0.3])

# Timed non-RL phases. (Keyframe Times)
MOVE_ABOVE_STEPS = 80
DESCEND_STEPS = 50
GRASP_STEPS = 60
LIFT_STEPS = 80
MOVE_TO_ORBIT_STEPS = 80
PLACE_STEPS = 100
OPEN_STEPS = 30
RETREAT_STEPS = 100

# Slip detection confirmation. A force drop must persist this long before
# the episode is terminated as a real slip.
SLIP_CONFIRMATION_TIME = 1.0
SLIP_CONFIRMATION_STEPS = int(SLIP_CONFIRMATION_TIME / DT)

ORBIT_STEPS = 420
ORBIT_Z = 0.40
ORBIT_RADIUS = 0.15
ORBIT_CENTER = np.array([0.30, 0.0, ORBIT_Z])

# Cartesian residual during orbit: [dx, dy, dz] in metres.
ACTION_LOW = np.array([-0.12, -0.12, -0.08])
ACTION_HIGH = np.array([0.12, 0.12, 0.08])

# Uncertainty source: state noise, small action noise, control delay, and occasional unreachable desired point.
STATE_NOISE_STD = 0.000002
ACTION_NOISE_STD = 0.000002
CONTROL_DELAY_STEPS = 2

# Low-pass filtering and per-frame joint rate limits for smooth 360 motion.
ACTION_FILTER_ALPHA = 0.12
MAX_LOWER_JOINT_STEP = 0.010
MAX_WRIST_JOINT_STEP = 0.012
UNREACHABLE_PROBABILITY = 0.00
UNREACHABLE_SHIFT = np.array([0.16, 0.12, 0.0])

# Each episode starts with the current
# target grip force. The target begins at 10 N and is increased only when
# the episode fails due to missed pick or slip
ADAPTIVE_GRIP_FORCE_START = 10.0
ADAPTIVE_GRIP_FORCE_MIN = 10.0
ADAPTIVE_GRIP_FORCE_MAX = 60.0
FORCE_DROP_THRESHOLD = 2.0
MAX_GRIP_FORCE = 60.0

# Placement-only trajectory constants.
# These only affect the scripted place phase after orbit scoring has ended.
PLACE_HOVER_Z = 0.40
PLACE_RELEASE_Z = 0.115
PLACE_LIFT_Z = 0.28

# Timings for Placement step
PLACE_APPROACH_STEPS = 80
PLACE_DESCEND_STEPS = 80
PLACE_SETTLE_STEPS = 30
PLACE_OPEN_STEPS = 45
PLACE_LIFT_STEPS = 70
PLACE_HOME_STEPS = 80

# A trajectory is considered successful when the average orbit tracking
# accuracy is at least 80/100, i.e. within 10% of the requested 360 path.
TARGET_MEAN_ERROR = 0.10
TRAJECTORY_SUCCESS_SCORE = 80.0
REQUIRED_CONSECUTIVE_SUCCESSES = 3

# Overall Workspace Safe limits to spawn pick object
WORKSPACE_LIMITS = {
    "x": [(-0.40, -0.12), (0.12, 0.40)],
    "y": [(-0.30, -0.10), (0.10, 0.30)],
    "z": (0.10, 0.60),
}

# Always aplce in the same location
PLACE_LOCATIONS = [
    np.array([0.30, 0.0, 0.03]),
]


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict

#Rl Policy

class ResidualRLPolicy:
    """Cartesian residual learner using the completed episode mean error."""

    def __init__(self):
        self.residual_table = np.zeros((ORBIT_STEPS, 3), dtype=np.float64)

        self.large_error_learning_rate = 0.50
        self.moderate_error_learning_rate = 0.2
        self.small_error_learning_rate = 0.1

        self.max_residual = np.array([0.12, 0.12, 0.08], dtype=np.float64)
        self.decay = 0.999

        self.last_learning_rate = 0.0
        self.last_error_magnitude = 0.0
        self.last_average_error = 0.0

        # Store the complete orbit errors, then update the policy once the
        # episode mean error is known.
        self.episode_error_vectors = []

    def act(
        self,
        observation: np.ndarray,
        orbit_step: int | None = None,
    ) -> np.ndarray:
        if orbit_step is None:
            return np.zeros(3, dtype=np.float64)

        orbit_step = int(np.clip(orbit_step, 0, ORBIT_STEPS - 1))
        return self.residual_table[orbit_step].copy()

    def record_error(self, orbit_step: int, error_xyz: np.ndarray) -> None:
        orbit_step = int(np.clip(orbit_step, 0, ORBIT_STEPS - 1))
        error_xyz = np.asarray(error_xyz, dtype=np.float64).copy()
        self.episode_error_vectors.append((orbit_step, error_xyz))

    def apply_episode_update(self, mean_error: float) -> None:
        """Update all orbit residuals using the final logged mean error."""
        if not self.episode_error_vectors:
            self.last_learning_rate = 0.0
            self.last_average_error = float("nan")
            return
        # Bigger Error = Higher Learning Rate
        if mean_error >= 0.16:
            learning_rate = self.large_error_learning_rate
        elif mean_error >= 0.13:
            learning_rate = self.moderate_error_learning_rate
        else:
            learning_rate = self.small_error_learning_rate
        #Lower Error = Lower Learning Rate
        self.last_learning_rate = float(learning_rate)
        self.last_average_error = float(mean_error)

        for orbit_step, error_xyz in self.episode_error_vectors:
            self.last_error_magnitude = float(np.linalg.norm(error_xyz))

            self.residual_table[orbit_step] = (
                self.decay * self.residual_table[orbit_step]
                + learning_rate * error_xyz
            )

            self.residual_table[orbit_step] = np.clip(
                self.residual_table[orbit_step],
                -self.max_residual,
                self.max_residual,
            )

        self.episode_error_vectors = []
class WXAIOrbit360RLEnv:
    """
    Isaac Sim task:
    pick each of the five YCB objects, track a 360 orbit while holding it,
    then place it. During the 360 orbit section, the commanded motion is
    generated through the lower arm joints, while the wrist joints are held
    near their entry posture. RL learns lower-joint/wrist residual corrections and smoothness. Grip force is adapted between episodes, then locked during the orbit.
    """
    # Declaring Variables 
    def __init__(self):
        self.robot = None
        self.tactile = None
        self.object_pool = {}
        self.active_object = None
        self.active_object_name = None

        self.object_order = list(YCB_PROP_PATHS.keys())
        self.object_index = 0
        self.episode_id = 0

        self.pick_position = np.array([0.22, -0.18, 0.03])
        self.place_position = PLACE_LOCATIONS[0].copy()
        self.place_open_start = 0
        self.place_open_end = 0

        self.last_policy_learning_rate = 0.0
        self.last_policy_average_error = 0.0
        self.mean_residual_magnitude = 0.0
        self.max_residual_magnitude = 0.0
        self.prev_commanded_position = None

        self.phase = "idle"
        self.phase_step = 0
        self.orbit_step = 0
        self.done = False
        self.last_debug_phase = None
        self.preplanned_trajectory = []
        self.preplanned_index = 0



        self.action_delay = deque(
            [np.zeros(3, dtype=np.float64) for _ in range(CONTROL_DELAY_STEPS)],
            maxlen=CONTROL_DELAY_STEPS,
        )

        self.filtered_orbit_action = np.zeros(3, dtype=np.float64)
        self.locked_gripper_position = None
        self.prev_commanded_full_joints = None

        self.prev_ee_pos = None
        self.prev_ee_vel_vec = np.zeros(3)
        self.prev_force = 0.0
        self.initial_transport_force = None
        self.slip_detected = False
        self.missed_pick = False
        self.force_drop_counter = 0
        self.slip_confirmation_steps = SLIP_CONFIRMATION_STEPS

        self.reward_log = []
        self.log_dir = "rl_orbit_logs"
        os.makedirs(self.log_dir, exist_ok=True)

        self.joint_lower_limits = None
        self.joint_upper_limits = None

        # Joint-space 360 orbit state. These are initialised at the moment
        # the robot reaches the 360 orbit centre.
        self.orbit_joint_center = None
        self.orbit_wrist_start = None
        self.prev_commanded_lower_joints = None

        # Adaptive grip / completion tracking.
        self.target_grip_force = ADAPTIVE_GRIP_FORCE_START
        self.consecutive_tracking_successes = 0
        self.training_complete = False
        self.episode_summary_log = []
        self.orbit_tracking_errors = []
        self.orbit_rewards = []
        self.episode_finalised = False
        self.prev_commanded_position = None
        self.last_policy_learning_rate = 0.0
        self.mean_residual_magnitude = 0.0
        self.max_residual_magnitude = 0.0


    # Scene setup
    # Add arm and objects to scene
    def setup_scene(self) -> None:
        stage_utils.create_new_stage(template="sunlight")

        stage_utils.add_reference_to_stage(
            usd_path=ROBOT_USD_PATH,
            path=ROBOT_SCENE_PATH,
        )

        self.robot = TrossenAIController(
            robot_path=ROBOT_SCENE_PATH,
            robot_type=RobotType.WXAI,
            arm_dof_indices=WXAI_ARM_DOF_INDICES,
            gripper_dof_index=WXAI_GRIPPER_DOF_INDEX,
            default_dof_positions=WXAI_DEFAULT_DOF_POSITIONS,
        )

        stage_utils.add_reference_to_stage(
            usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
            path=GROUND_SCENE_PATH,
        )

        lower, upper = self.robot.get_dof_limits()
        self.joint_lower_limits = lower.numpy()[0][:6]
        self.joint_upper_limits = upper.numpy()[0][:6]

        self.spawn_ycb_objects()
        self.create_contact_sensor()

    # Spawns YCB objects at the stage of the sim in the parking area
    def spawn_ycb_objects(self) -> None:
        stage = get_current_stage()
        define_prim("/World/YCB", "Xform")

        for i, (name, usd_path) in enumerate(YCB_PROP_PATHS.items()):
            prim_path = f"/World/YCB/ycb_{name}"

            stage_utils.add_reference_to_stage(usd_path=usd_path, path=prim_path)

            root = stage.GetPrimAtPath(prim_path)
            children = [p for p in root.GetChildren()]
            real_path = str(children[0].GetPath()) if children else prim_path

            xform = XformPrim(real_path)
            xform.reset_xform_op_properties()

            GeomPrim(paths=[real_path + "/.*"], apply_collision_apis=True)
            obj = RigidPrim(paths=[real_path])

            orientation = np.array([YCB_ORIENTATION_OFFSETS.get(name, [0.0, 0.0, 0.0, 1.0])])
            parking_pos = PARKING_START + i * PARKING_SPACING

            obj.set_local_poses(
                translations=np.array([[parking_pos[0], parking_pos[1], parking_pos[2]]]),
                orientations=orientation,
            )

            self.object_pool[name] = obj
    # Create the contact sensor to know when to stop closing arm. 
    def create_contact_sensor(self) -> None:
        mesh_path = GRIPPER_LINK_PATH + "/collisions/gripper_right/node_STL_BINARY_/mesh"
        prim = get_prim_at_path(mesh_path)
        if prim and prim.IsValid() and not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)

        # Keep the same contact-sensor approach as the original file.
        omni.kit.commands.execute(
            "IsaacSensorCreateContactSensor",
            path="/World/wxai_robot/gripper_right/contact_sensor",
            parent="/World/wxai_robot/gripper_right",
            min_threshold=0.0001,
            sensor_period=1,
        )

        self.tactile = ContactSensor(
            prim_path="/World/wxai_robot/gripper_right/gripper_right/contact_sensor"
        )


    # Episode must be reset after P&P movement 

    def reset_env(self) -> np.ndarray:
        self.episode_id += 1
        self.done = False
        self.slip_detected = False
        self.missed_pick = False
        self.initial_transport_force = None
        self.force_drop_counter = 0
        self.prev_ee_pos = None
        self.prev_ee_vel_vec = np.zeros(3)
        self.prev_force = 0.0
        self.reward_log = []
        self.orbit_tracking_errors = []
        self.orbit_rewards = []
        self.episode_finalised = False
        self.orbit_step = 0
        self.orbit_joint_center = None
        self.orbit_wrist_start = None
        self.prev_commanded_lower_joints = None
        self.prev_commanded_full_joints = None
        self.filtered_orbit_action = np.zeros(3, dtype=np.float64)
        self.locked_gripper_position = None
        self.phase = "preplanned"
        self.phase_step = 0
        self.preplanned_index = 0
        self.action_delay.clear()
        for _ in range(CONTROL_DELAY_STEPS):
            self.action_delay.append(np.zeros(3, dtype=np.float64))

        self.robot.reset_to_default_pose()
        self.park_all_objects()

        self.active_object_name = self.object_order[self.object_index]
        self.active_object = self.object_pool[self.active_object_name]
        self.object_index = (self.object_index + 1) % len(self.object_order)

        self.pick_position = self.sample_pick_position()
        self.place_position = PLACE_LOCATIONS[0].copy()

        orientation = np.array([YCB_ORIENTATION_OFFSETS.get(self.active_object_name, [0.0, 0.0, 0.0, 1.0])])
        self.active_object.set_world_poses(
            positions=np.array([[self.pick_position[0], self.pick_position[1], 0.03]]),
            orientations=orientation,
        )

        for _ in range(5):
            simulation_app.update()

        self.build_preplanned_pick_trajectory()

        return self.get_observation(desired_pos=ORBIT_CENTER)

    # When YCB objects are spawned and reset they are returned to the parking area 
    def park_all_objects(self) -> None:
        for i, name in enumerate(self.object_order):
            orientation = np.array([YCB_ORIENTATION_OFFSETS.get(name, [0.0, 0.0, 0.0, 1.0])])
            parking_pos = PARKING_START + i * PARKING_SPACING
            self.object_pool[name].set_world_poses(
                positions=np.array([[parking_pos[0], parking_pos[1], parking_pos[2]]]),
                orientations=orientation,
            )

    # The system picks a random section from the workplace limits to spawn the objects
    # Z is always 0.03 however
    def sample_pick_position(self) -> np.ndarray:
        x_ranges = WORKSPACE_LIMITS["x"]
        y_ranges = WORKSPACE_LIMITS["y"]
        x = np.random.uniform(*random.choice(x_ranges))
        y = np.random.uniform(*random.choice(y_ranges))
        return np.array([x, y, 0.03])

    # Trajectory generation
    def build_preplanned_pick_trajectory(self) -> None:
        _, current_ee_pos, _ = self.robot.get_current_state()
        current_ee_pos = np.array(current_ee_pos[0])

        object_pos = self.active_object.get_world_poses()[0].numpy().flatten()

        # Key frames, pick, move to home, circle movement, place, then return to home 
        key_frames = [
            current_ee_pos,
            np.array([object_pos[0], object_pos[1], current_ee_pos[2] + 0.06]),
            object_pos + np.array([0.0, 0.0, CLEARANCE_HEIGHT]),
            object_pos + APPROACH_OFFSET,
            object_pos + np.array([0.0, 0.0, CLEARANCE_HEIGHT]),
            self.orbit_start_position(),
        ]

        #Time steps for each keyframe
        segment_steps = [
            MOVE_ABOVE_STEPS,
            DESCEND_STEPS,
            GRASP_STEPS,
            LIFT_STEPS,
            MOVE_TO_ORBIT_STEPS,
        ]

        self.preplanned_trajectory = self.interpolate_waypoints(key_frames, segment_steps)

    # Place has a seperate trajectory to keep the palce movement safe
    def build_place_trajectory(self) -> None:
        _, current_ee_pos, _ = self.robot.get_current_state()
        current_ee_pos = np.array(current_ee_pos[0], dtype=np.float64)

        place_xy = self.place_position[:2]

        # Keep the first move at the current/orbit height to avoid dipping early.
        transport_z = max(float(current_ee_pos[2]), PLACE_HOVER_Z)

        place_above_high = np.array(
            [place_xy[0], place_xy[1], transport_z],
            dtype=np.float64,
        )

        place_hover = np.array(
            [place_xy[0], place_xy[1], PLACE_HOVER_Z],
            dtype=np.float64,
        )

        place_release = np.array(
            [place_xy[0], place_xy[1], PLACE_RELEASE_Z],
            dtype=np.float64,
        )

        place_lift = np.array(
            [place_xy[0], place_xy[1], PLACE_LIFT_Z],
            dtype=np.float64,
        )

        self.place_open_start = (
            PLACE_APPROACH_STEPS
            + PLACE_DESCEND_STEPS
            + PLACE_SETTLE_STEPS
        )
        self.place_open_end = self.place_open_start + PLACE_OPEN_STEPS

        key_frames = [
            current_ee_pos,
            place_above_high,       # move horizontally at safe height
            place_hover,            # small vertical adjustment
            place_release,          # descend only once above place target
            place_release.copy(),   # settle
            place_release.copy(),   # release while stationary
            place_lift,             # lift vertically
            HOME_POSITION.copy(),   # then return home
        ]

        segment_steps = [
            PLACE_APPROACH_STEPS,
            40,
            PLACE_DESCEND_STEPS,
            PLACE_SETTLE_STEPS,
            PLACE_OPEN_STEPS,
            PLACE_LIFT_STEPS,
            PLACE_HOME_STEPS,
        ]

        self.preplanned_trajectory = self.interpolate_waypoints_no_z_clamp(
            key_frames,
            segment_steps,
        )
        self.preplanned_index = 0

    # build way points to know how to get to end position
    def interpolate_waypoints(self, key_frames: list[np.ndarray], segment_steps: list[int]) -> list[np.ndarray]:
        if len(key_frames) != len(segment_steps) + 1:
            raise ValueError("Expected len(key_frames) = len(segment_steps) + 1")

        trajectory = []
        for i, n_steps in enumerate(segment_steps):
            start = np.array(key_frames[i], dtype=np.float64)
            end = np.array(key_frames[i + 1], dtype=np.float64)

            for step in range(n_steps):
                alpha = step / max(n_steps, 1)
                pos = start + alpha * (end - start)
                pos[2] = max(pos[2], MIN_SAFE_Z) if i >= 3 else pos[2]
                trajectory.append(pos)

        trajectory.append(np.array(key_frames[-1], dtype=np.float64))
        return trajectory

    def interpolate_waypoints_no_z_clamp(
        self,
        key_frames: list[np.ndarray],
        segment_steps: list[int],
    ) -> list[np.ndarray]:
        """Interpolate placement waypoints without forcing z above MIN_SAFE_Z."""
        if len(key_frames) != len(segment_steps) + 1:
            raise ValueError("Expected len(key_frames) = len(segment_steps) + 1")

        trajectory = []
        for i, n_steps in enumerate(segment_steps):
            start = np.array(key_frames[i], dtype=np.float64)
            end = np.array(key_frames[i + 1], dtype=np.float64)

            for step in range(n_steps):
                alpha = step / max(n_steps, 1)
                pos = start + alpha * (end - start)
                trajectory.append(pos)

        trajectory.append(np.array(key_frames[-1], dtype=np.float64))
        return trajectory

    #Desired end-effector position for one complete reachable 360-degree circle
    def desired_orbit_position(self, step: int) -> np.ndarray:
        theta = 2.0 * np.pi * step / max(ORBIT_STEPS - 1, 1)
        radius = getattr(self, "orbit_radius", ORBIT_RADIUS)

        desired = ORBIT_CENTER + np.array([
            radius * np.cos(theta),
            radius * np.sin(theta),
            0.0,
        ])

        # Unreachable-position uncertainty: the desired target is sometimes shifted
        # outside the comfortable workspace. The reward teaches the policy to reduce
        # error without becoming unstable.
        if random.random() < UNREACHABLE_PROBABILITY:
            desired = desired + UNREACHABLE_SHIFT

        desired[2] = max(desired[2], MIN_SAFE_Z)
        return desired
    
    def desired_wrist_roll(self, step: int) -> float:
        if self.orbit_wrist_start is None:
            return 0.0
        return self.orbit_wrist_start

    # RL step
    def step(self, action: np.ndarray) -> StepResult:
        self.debug_phase()

        if self.done:
            return StepResult(
                self.get_observation(ORBIT_CENTER),
                0.0,
                True,
                {"phase": self.phase},
            )

        force = self.get_contact_force()

        if self.phase == "preplanned":
            self.execute_preplanned_step(force)
            observation = self.get_observation(ORBIT_CENTER)
            return StepResult(
                observation,
                0.0,
                self.done,
                {"phase": self.phase},
            )

        if self.phase == "orbit_rl":
            executed_orbit_step = self.orbit_step
            desired_pos = self.desired_orbit_position(executed_orbit_step)

            clean_action = np.clip(
                np.asarray(action, dtype=np.float64)[:3],
                ACTION_LOW,
                ACTION_HIGH,
            )

            noisy_action = clean_action + np.random.normal(
                0.0,
                ACTION_NOISE_STD,
                size=3,
            )
            noisy_action = np.clip(noisy_action, ACTION_LOW, ACTION_HIGH)

            self.action_delay.append(noisy_action)
            delayed_action = self.action_delay[0]

            self.filtered_orbit_action = (
                (1.0 - ACTION_FILTER_ALPHA) * self.filtered_orbit_action
                + ACTION_FILTER_ALPHA * delayed_action
            )

            cartesian_residual = self.filtered_orbit_action.copy()
            commanded_position = self.clamp_workspace(
                desired_pos + cartesian_residual
            )

            self.robot.set_end_effector_pose(
                position=commanded_position.reshape(1, -1),
                orientation=DOWNWARD_ORIENTATION.reshape(1, -1),
            )

            observation = self.get_observation(desired_pos)
            reward, info = self.compute_reward(
                desired_pos=desired_pos,
                commanded_position=commanded_position,
                action=clean_action,
                force=force,
            )

            terminal_reason = None
            if self.slip_detected:
                reward -= 100.0
                terminal_reason = "confirmed_slip"
                self.done = True

            _, actual_ee_pos, _ = self.robot.get_current_state()
            actual_ee_pos = np.asarray(actual_ee_pos[0], dtype=np.float64)

            self.orbit_tracking_errors.append(info["tracking_error"])
            self.orbit_rewards.append(reward)

            # Add rewards
            self.reward_log.append([
                self.episode_id,
                self.active_object_name,
                executed_orbit_step,
                desired_pos[0],
                desired_pos[1],
                desired_pos[2],
                actual_ee_pos[0],
                actual_ee_pos[1],
                actual_ee_pos[2],
                commanded_position[0],
                commanded_position[1],
                commanded_position[2],
                cartesian_residual[0],
                cartesian_residual[1],
                cartesian_residual[2],
                self.target_grip_force,
                force,
                reward,
                info["tracking_error"],
                info["smoothness_penalty"],
                info["command_smoothness"],
                int(info["slip"]),
            ])

            if self.done:
                if not self.episode_finalised:
                    self.episode_finalised = True
                    self.finalise_episode()

                return StepResult(
                    observation,
                    reward,
                    True,
                    {
                        "phase": self.phase,
                        "terminal_reason": terminal_reason,
                        "orbit_step": executed_orbit_step,
                        "orbit_complete": False,
                        **info,
                    },
                )

            self.orbit_step += 1
            orbit_complete = self.orbit_step >= ORBIT_STEPS

            if orbit_complete:
                # Main finalises after updating the policy metrics for the final
                # orbit sample. Placement remains outside orbit scoring.
                self.phase = "place"
                self.build_place_trajectory()

            return StepResult(
                observation,
                reward,
                self.done,
                {
                    "phase": self.phase,
                    "orbit_step": executed_orbit_step,
                    "orbit_complete": orbit_complete,
                    **info,
                },
            )

        if self.phase == "place":
            self.execute_place_step()
            observation = self.get_observation(
                self.place_position + PLACE_OFFSET
            )
            return StepResult(
                observation,
                0.0,
                self.done,
                {"phase": self.phase},
            )

        self.done = True
        return StepResult(
            self.get_observation(ORBIT_CENTER),
            0.0,
            True,
            {"phase": self.phase},
        )
    def orbit_start_position(self) -> np.ndarray:
        """Start at +X on one fixed, reachable 360-degree circle."""
        self.orbit_radius = ORBIT_RADIUS
        return np.array([ORBIT_RADIUS, 0.0, ORBIT_Z], dtype=np.float64)
    def execute_preplanned_step(self, force: float) -> None:
        if self.preplanned_index >= len(self.preplanned_trajectory):
            self.phase = "orbit_rl"
            self.orbit_step = 0
            self.initialise_orbit_joint_posture()
            return

        goal = self.preplanned_trajectory[self.preplanned_index]
        self.robot.set_end_effector_pose(
            position=goal.reshape(1, -1),
            orientation=DOWNWARD_ORIENTATION.reshape(1, -1),
        )

        # Close gripper during descend / grasp / early lift.
        if MOVE_ABOVE_STEPS + DESCEND_STEPS <= self.preplanned_index < (
            MOVE_ABOVE_STEPS + DESCEND_STEPS + GRASP_STEPS + LIFT_STEPS
        ):
            if force < self.target_grip_force:
                self.close_gripper(step=0.004)

        # Simple missed-pick check after lift section.
        object_z = self.get_object_position()[2]
        if self.preplanned_index > MOVE_ABOVE_STEPS + DESCEND_STEPS + GRASP_STEPS + LIFT_STEPS:
            if object_z < 0.055:
                self.missed_pick = True
                self.done = True
                if not self.episode_finalised:
                    self.episode_finalised = True
                    self.finalise_episode()
                print("[EPISODE] Missed pick; ending early.")

        self.preplanned_index += 1

    def execute_place_step(self) -> None:
        """Run the scripted place phase without changing the orbit result."""
        if self.preplanned_index >= len(self.preplanned_trajectory):
            self.done = True
            return

        goal = self.preplanned_trajectory[self.preplanned_index]
        self.robot.set_end_effector_pose(
            position=goal.reshape(1, -1),
            orientation=DOWNWARD_ORIENTATION.reshape(1, -1),
        )

        if self.preplanned_index < self.place_open_start:
            # Approach, descend and settle: keep holding the object.
            pass
        elif self.place_open_start <= self.preplanned_index < self.place_open_end:
            # Release slowly while stationary at the release height.
            self.apply_gripper_delta(delta=0.0015)
        else:
            # After release, the path lifts vertically first, then returns home.
            pass
        #Debug
        #if self.preplanned_index % 30 == 0:
        #    print(
        #        f"[PLACE DEBUG] index={self.preplanned_index}/"
        #        f"{len(self.preplanned_trajectory)} "
        #        f"open_start={self.place_open_start} "
        #        f"open_end={self.place_open_end} "
        #        f"done={self.done}"
        #    )

        self.preplanned_index += 1

    # Lower-joint + wrist 360 orbit control

    def initialise_orbit_joint_posture(self) -> None:
        q = self.robot.get_dof_positions().numpy()[0].copy()
        self.orbit_joint_center = q[:6].copy()
        self.orbit_wrist_start = float(q[5])
        self.prev_commanded_lower_joints = q[:3].copy()
        self.prev_commanded_full_joints = q[:6].copy()

        self.orbit_radius = ORBIT_RADIUS

    def lower_joint_orbit_target(self, step: int) -> np.ndarray:
        """Generate the 360 orbit using the lower arm joints."""
        if self.orbit_joint_center is None:
            self.initialise_orbit_joint_posture()

        theta = 2.0 * np.pi * step / max(ORBIT_STEPS - 1, 1)

        # move the lower joints to get a smoother trajectory 
        lower_target = self.orbit_joint_center[:3] + np.array([
            0.65 * np.sin(theta),
            0.12 * np.sin(theta),
            -0.10 * np.sin(theta),
        ])

        for j in range(3):
            lower_target[j] = np.clip(
                lower_target[j],
                self.joint_lower_limits[j],
                self.joint_upper_limits[j],
            )

        return lower_target

    def command_lower_joint_orbit(self, joint_residual: np.ndarray) -> tuple[np.ndarray, float]:
        """Command lower-joint orbit plus wrist 360 rotation."""
        base_target = self.lower_joint_orbit_target(self.orbit_step)
        lower_command = base_target + np.array(joint_residual[:3], dtype=np.float64)

        for j in range(3):
            lower_command[j] = np.clip(
                lower_command[j],
                self.joint_lower_limits[j],
                self.joint_upper_limits[j],
            )

        wrist_target = self.desired_wrist_roll(self.orbit_step) + float(joint_residual[3])
        wrist_target = float(np.clip(wrist_target, self.joint_lower_limits[5], self.joint_upper_limits[5]))

        q = self.robot.get_dof_positions().numpy()

        # Build the desired full arm command, then rate-limit it. Directly jumping
        # to each new target every frame is the main cause of jerky motion.
        desired_arm = q[0, :6].copy()
        desired_arm[0:3] = lower_command
        desired_arm[3] = self.orbit_joint_center[3]
        desired_arm[4] = self.orbit_joint_center[4]
        desired_arm[5] = wrist_target

        if self.prev_commanded_full_joints is None:
            smoothed_arm = desired_arm
        else:
            delta = desired_arm - self.prev_commanded_full_joints
            delta[0:3] = np.clip(delta[0:3], -MAX_LOWER_JOINT_STEP, MAX_LOWER_JOINT_STEP)
            delta[3:5] = 0.0
            delta[5] = np.clip(delta[5], -MAX_WRIST_JOINT_STEP, MAX_WRIST_JOINT_STEP)
            smoothed_arm = self.prev_commanded_full_joints + delta

        q[0, :6] = smoothed_arm
        self.robot.set_dof_positions(q)
        self.prev_commanded_full_joints = smoothed_arm.copy()

        return smoothed_arm[:3], float(smoothed_arm[5])

    # Observation / reward

    def get_observation(self, desired_pos: np.ndarray) -> np.ndarray:
        _, ee_pos, ee_rot = self.robot.get_current_state()
        ee_pos = np.array(ee_pos[0], dtype=np.float64)
        obj_pos = self.get_object_position()
        force = self.get_contact_force()
        q = self.robot.get_dof_positions().numpy()[0][:6]
        qd = self.robot.get_dof_velocities().numpy()[0][:6]
        gripper_open = self.get_gripper_opening()

        noisy_ee = ee_pos + np.random.normal(0.0, STATE_NOISE_STD, size=3)
        noisy_obj = obj_pos + np.random.normal(0.0, STATE_NOISE_STD, size=3)

        return np.concatenate([
            noisy_ee,
            desired_pos,
            desired_pos - noisy_ee,
            noisy_obj,
            noisy_obj - noisy_ee,
            np.array([force, gripper_open, YCB_MASSES[self.active_object_name], YCB_FRICTION[self.active_object_name]]),
            q,
            qd,
        ]).astype(np.float64)

    # add up all varaibles and generate reward variable
    def compute_reward(
        self,
        desired_pos: np.ndarray,
        commanded_position: np.ndarray,
        action: np.ndarray,
        force: float,
    ) -> tuple[float, dict]:
        _, ee_pos, _ = self.robot.get_current_state()
        ee_pos = np.asarray(ee_pos[0], dtype=np.float64)

        tracking_error = float(np.linalg.norm(desired_pos - ee_pos))

        ee_vel_vec = self.compute_ee_velocity_vector(ee_pos)
        smoothness_penalty = float(
            np.linalg.norm(ee_vel_vec - self.prev_ee_vel_vec)
        )
        self.prev_ee_vel_vec = ee_vel_vec

        action_penalty = float(np.linalg.norm(action[:3]))

        if self.prev_commanded_position is None:
            command_smoothness = 0.0
        else:
            command_smoothness = float(
                np.linalg.norm(
                    commanded_position - self.prev_commanded_position
                )
            )
        self.prev_commanded_position = commanded_position.copy()

        grip_penalty = 0.0
        slip = False

        if (
            self.initial_transport_force is None
            and force > max(0.5, 0.75 * self.target_grip_force)
        ):
            self.initial_transport_force = force
            self.force_drop_counter = 0

        if self.initial_transport_force is not None:
            force_drop = self.initial_transport_force - force

            if force_drop > FORCE_DROP_THRESHOLD:
                self.force_drop_counter += 1
            else:
                self.force_drop_counter = 0

            if self.force_drop_counter >= self.slip_confirmation_steps:
                object_z = self.get_object_position()[2]

                if object_z < 0.12:
                    slip = True
                    self.slip_detected = True
                    print(
                        "[SLIP DETECTED] Force dropped and object height "
                        f"is low: object_z={object_z:.3f}"
                    )
                else:
                    self.force_drop_counter = 0
                    print(
                        "[SLIP IGNORED] Force dropped but object is still "
                        f"held: object_z={object_z:.3f}"
                    )

                print(
                    f"[SLIP CHECK] Force drop={force_drop:.2f} N, "
                    f"counter={self.force_drop_counter}, "
                    f"time={self.force_drop_counter * DT:.2f}s"
                )

        if force < self.target_grip_force:
            grip_penalty += (
                self.target_grip_force - force
            ) / max(self.target_grip_force, 1e-6)

        if force > MAX_GRIP_FORCE:
            grip_penalty += (
                force - MAX_GRIP_FORCE
            ) / MAX_GRIP_FORCE
        # reward weights
        reward = (
            2.0
            - 20.0 * tracking_error
            - 0.10 * smoothness_penalty
            - 1.0 * action_penalty
            - 1.0 * command_smoothness
            - 2.0 * grip_penalty
            - (8.0 if slip else 0.0)
        )
        # send reward as a variable to be used elsewhere
        return reward, {
            "tracking_error": tracking_error,
            "smoothness_penalty": smoothness_penalty,
            "command_smoothness": command_smoothness,
            "slip": slip,
            "force": force,
            "target_grip_force": self.target_grip_force,
            "grip_penalty": grip_penalty,
        }
        # prevent robot from hitting the ground 
    def clamp_workspace(self, pos: np.ndarray) -> np.ndarray:
        pos = np.array(pos, dtype=np.float64).copy()
        pos[0] = np.clip(pos[0], -0.45, 0.45)
        pos[1] = np.clip(pos[1], -0.35, 0.35)
        pos[2] = np.clip(pos[2], MIN_SAFE_Z, 0.60)
        return pos

    def get_contact_force(self) -> float:
        contact_data = self.tactile.get_current_frame()
        if contact_data is None or "force" not in contact_data:
            return 0.0
        return float(np.linalg.norm(contact_data["force"]))
        
    # DEBUG to check what keyframe the robot is in 
    def debug_phase(self) -> None:
        if self.phase != self.last_debug_phase:
            print(
                f"[PHASE] episode={self.episode_id} "
                f"phase={self.phase} "
                f"preplanned_index={self.preplanned_index} "
                f"orbit_step={self.orbit_step} "
                f"done={self.done}"
            )
            self.last_debug_phase = self.phase

    def get_object_position(self) -> np.ndarray:
        return self.active_object.get_world_poses()[0].numpy().flatten()
    
    def get_gripper_opening(self) -> float:
        dof_pos = self.robot.get_dof_positions().numpy()[0]
        return float(dof_pos[self.robot.gripper_dof_index])

    def lock_gripper_for_orbit(self) -> None:
        """Freeze the gripper opening before the 360 orbit starts.

        The adaptive grip force is only used during grasping. During the orbit,
        the gripper should not open/close every frame because that creates visible
        vibration and can destabilise the object.
        """
        positions = self.robot.get_dof_positions().numpy()
        self.locked_gripper_position = float(positions[0, self.robot.gripper_dof_index])

    def hold_locked_gripper(self) -> None:
        if self.locked_gripper_position is None:
            self.lock_gripper_for_orbit()

        positions = self.robot.get_dof_positions().numpy()
        positions[0, self.robot.gripper_dof_index] = self.locked_gripper_position
        self.robot.set_dof_positions(positions)

    def close_gripper(self, step: float = 0.002) -> None:
        positions = self.robot.get_dof_positions().numpy()
        gripper_idx = self.robot.gripper_dof_index

        lower, _ = self.robot.get_dof_limits()
        lower_limit = lower.numpy()[0, gripper_idx]

        positions[0, gripper_idx] = max(positions[0, gripper_idx] - step, lower_limit)
        self.robot.set_dof_positions(positions)

    def apply_gripper_delta(self, delta: float) -> None:
        positions = self.robot.get_dof_positions().numpy()
        gripper_idx = self.robot.gripper_dof_index

        lower, upper = self.robot.get_dof_limits()
        lower_limit = lower.numpy()[0, gripper_idx]
        upper_limit = upper.numpy()[0, gripper_idx]

        # Negative delta closes, positive opens.
        positions[0, gripper_idx] = np.clip(
            positions[0, gripper_idx] + delta,
            lower_limit,
            upper_limit,
        )

        self.robot.set_dof_positions(positions)

    def compute_ee_velocity_vector(self, ee_pos: np.ndarray) -> np.ndarray:
        if self.prev_ee_pos is None:
            self.prev_ee_pos = ee_pos.copy()
            return np.zeros(3, dtype=np.float64)

        vel = (ee_pos - self.prev_ee_pos) / DT
        self.prev_ee_pos = ee_pos.copy()
        return vel

    # Compute score / 100, negative scores are allowed to better see improvement
    def compute_trajectory_score(self) -> tuple[float, float, float]:
        """Return trajectory quality relative to the 0.10 m mean-error goal.

        Score interpretation:
        - 100 means the mean tracking error is 0.10 m or lower.
        - 50 means the mean tracking error is 0.20 m.
        - 25 means the mean tracking error is 0.40 m.

        Scores are not forced to zero, so improvement remains visible even
        when the trajectory is still far from the target.
        """
        if len(self.orbit_tracking_errors) == 0:
            return float("nan"), float("nan"), float("nan")

        mean_error = float(np.mean(self.orbit_tracking_errors))
        max_error = float(np.max(self.orbit_tracking_errors))

        if mean_error <= 1e-9:
            score = 100.0
        else:
            score = 100.0 * TARGET_MEAN_ERROR / mean_error
            score = float(min(score, 100.0))

        return score, mean_error, max_error
        
    def compute_failure_severity(
        self,
        trajectory_score: float,
        average_reward: float,
    ) -> tuple[bool, int, str]:

        if self.missed_pick:
            return True, 5, "missed pick"

        if self.slip_detected:
            return True, 5, "slip detected"

        if trajectory_score >= TRAJECTORY_SUCCESS_SCORE:
            return False, 0, "trajectory success"
        
        if trajectory_score < 60.0:
            return True, 0, "large tracking failure"

        if trajectory_score < 70.0:
            return True, 0, "moderate tracking failure"

        return True, 0, "trajectory near miss"

    # once the episode is over compute the rewards and new variables based on RL program
    def finalise_episode(self) -> None:
        """Save orbit metrics and adapt grip only for grasp failures."""
        trajectory_score, mean_error, max_error = self.compute_trajectory_score()
        average_reward = (
            float(np.mean(self.orbit_rewards))
            if self.orbit_rewards
            else -999.0
        )
        total_reward = (
            float(np.sum(self.orbit_rewards))
            if self.orbit_rewards
            else 0.0
        )

        failed, grip_increase, failure_reason = self.compute_failure_severity(
            trajectory_score=trajectory_score,
            average_reward=average_reward,
        )

        old_grip = self.target_grip_force

        if failed:
            self.consecutive_tracking_successes = 0
            if grip_increase > 0:
                self.target_grip_force = float(np.clip(
                    self.target_grip_force + grip_increase,
                    ADAPTIVE_GRIP_FORCE_MIN,
                    ADAPTIVE_GRIP_FORCE_MAX,
                ))
        else:
            self.consecutive_tracking_successes += 1

        if (
            self.consecutive_tracking_successes
            >= REQUIRED_CONSECUTIVE_SUCCESSES
        ):
            self.training_complete = True
            
        # print all variables so user can see improvement 
        self.episode_summary_log.append({
            "episode": self.episode_id,
            "object": self.active_object_name,
            "trajectory_score": trajectory_score,
            "mean_tracking_error": mean_error,
            "max_tracking_error": max_error,
            "average_reward": average_reward,
            "total_reward": total_reward,
            "failed": failed,
            "failure_reason": failure_reason,
            "old_grip_force": old_grip,
            "new_grip_force": self.target_grip_force,
            "grip_increase": grip_increase,
            "consecutive_successes": self.consecutive_tracking_successes,
            "learning_rate": self.last_policy_learning_rate,
            "policy_average_error": self.last_policy_average_error,
            "mean_residual": self.mean_residual_magnitude,
            "max_residual": self.max_residual_magnitude,
        })

        self.save_reward_log()
        self.save_episode_summary_log()

        score_text = (
            f"{trajectory_score:.1f}/100"
            if np.isfinite(trajectory_score)
            else "N/A"
        )
        mean_error_text = (
            f"{mean_error:.4f}m"
            if np.isfinite(mean_error)
            else "N/A"
        )
        # print to terminal 
        print(
            f"[SUMMARY]\n"
            f"episode={self.episode_id}\n"
            f"object={self.active_object_name}\n"
            f"score={score_text}\n"
            f"mean_error={mean_error_text}\n"
            f"avg_reward={average_reward:.2f}\n"
            f"failed={failed}\n"
            f"reason='{failure_reason}'\n"
            f"grip={old_grip:.1f}N->{self.target_grip_force:.1f}N\n"
            f"lr={self.last_policy_learning_rate:.3f}\n"
            f"policy_avg_error={self.last_policy_average_error:.4f}m\n"
            f"mean_residual={self.mean_residual_magnitude:.4f}m\n"
            f"max_residual={self.max_residual_magnitude:.4f}m\n"
            f"success_streak={self.consecutive_tracking_successes}/"
            f"{REQUIRED_CONSECUTIVE_SUCCESSES}"
        )

        if self.training_complete:
            print(
                "[COMPLETE] Trajectory tracking was within 5% for "
                "3 consecutive episodes."
            )

    # save to csv file 
    def save_episode_summary_log(self) -> None:
        if len(self.episode_summary_log) == 0:
            return

        path = os.path.join(self.log_dir, "episode_summary.csv")
        fieldnames = list(self.episode_summary_log[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.episode_summary_log)

    # save to csv file
    def save_reward_log(self) -> None:
        if len(self.reward_log) == 0:
            return

        path = os.path.join(
            self.log_dir,
            f"episode_{self.episode_id:04d}.csv",
        )
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Episode",
                "Object",
                "Orbit_Step",
                "Desired_X",
                "Desired_Y",
                "Desired_Z",
                "Actual_EE_X",
                "Actual_EE_Y",
                "Actual_EE_Z",
                "Commanded_X",
                "Commanded_Y",
                "Commanded_Z",
                "Residual_X",
                "Residual_Y",
                "Residual_Z",
                "Target_Grip_Force",
                "Grip_Force",
                "Reward",
                "Tracking_Error",
                "Smoothness_Penalty",
                "Command_Smoothness",
                "Slip",
            ])
            writer.writerows(self.reward_log)

        print(f"[LOG] Saved RL 360 orbit episode: {path}")

# main loop 
def main() -> None:
    print("WidowX 360-Orbit Cartesian Residual Learning Task")

    env = WXAIOrbit360RLEnv()
    env.setup_scene()

    policy = ResidualRLPolicy()

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    observation = env.reset_env()
    policy.episode_error_vectors = []

    while simulation_app.is_running():
        if SimulationManager.is_simulating():
            if env.phase == "orbit_rl":
                action = policy.act(observation, env.orbit_step)
            else:
                action = np.zeros(3, dtype=np.float64)

            result = env.step(action)

            if result is None:
                print(
                    f"[ERROR] env.step() returned None. "
                    f"phase={env.phase}, done={env.done}"
                )
                observation = env.get_observation(ORBIT_CENTER)
                continue

            if (
                "tracking_error" in result.info
                and "orbit_step" in result.info
                and result.info.get("terminal_reason")
                != "confirmed_slip"
            ):
                desired = result.observation[3:6]
                actual = result.observation[0:3]
                error_xyz = desired - actual

                policy.record_error(
                    int(result.info["orbit_step"]),
                    error_xyz,
                )

            # Finalise a normal full orbit only after the final policy update,
            # so the printed learning metrics include the final orbit sample.
            if (
                result.info.get("orbit_complete", False)
                and not env.episode_finalised
            ):
                mean_error = float(np.mean(env.orbit_tracking_errors))
                policy.apply_episode_update(mean_error)

                env.last_policy_learning_rate = policy.last_learning_rate
                env.last_policy_average_error = policy.last_average_error

                residual_magnitudes = np.linalg.norm(
                    policy.residual_table,
                    axis=1,
                )
                env.mean_residual_magnitude = float(
                    np.mean(residual_magnitudes)
                )
                env.max_residual_magnitude = float(
                    np.max(residual_magnitudes)
                )

                env.episode_finalised = True
                env.finalise_episode()

            observation = result.observation

            if result.done:
                print(
                    f"[EPISODE COMPLETE] id={env.episode_id}, "
                    f"object={env.active_object_name}, "
                    f"slip={env.slip_detected}, "
                    f"missed_pick={env.missed_pick}"
                )

                if env.training_complete:
                    print(
                        "[TASK COMPLETE] Stopping simulation after "
                        "3 consecutive 360 orbits within the 0.10 m mean-error target."
                    )
                    break

                for _ in range(10):
                    simulation_app.update()

                observation = env.reset_env()
                policy.episode_error_vectors = []

        simulation_app.update()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping 360 orbit RL task...")
    except Exception as exc:
        print(f"Error: {exc}")
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
