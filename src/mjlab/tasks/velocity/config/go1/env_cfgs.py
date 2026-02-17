"""Unitree Go1 velocity environment configurations."""

import math
from dataclasses import replace
from typing import Literal

from mjlab.asset_zoo.robots import (
  GO1_ACTION_SCALE,
  get_go1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  HeightSensorCfg,
  ObjRef,
  RayCastSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

TerrainType = Literal["rough", "obstacles"]


def unitree_go1_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_go1_robot_cfg()}

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  # Set per-robot sensor config.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "trunk"
    elif sensor.name in ("foot_clearance_scan", "foot_height_scan"):
      assert isinstance(sensor, HeightSensorCfg)
      sensor.sites = tuple(
        ObjRef(type="site", name=name, entity="robot") for name in site_names
      )
    elif sensor.name == "base_normal_scan":
      assert isinstance(sensor, HeightSensorCfg)
      sensor.sites = (ObjRef(type="site", name="imu", entity="robot"),)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  thigh_geom_names = tuple(
    f"{leg}_thigh_collision{i}" for leg in foot_names for i in (1, 2, 3)
  )
  unwanted_contact_ground_cfg = ContactSensorCfg(
    name="unwanted_contact_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=("trunk_collision", "head_collision"),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  thigh_ground_cfg = ContactSensorCfg(
    name="thigh_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=thigh_geom_names,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  calf_geom_names = tuple(
    f"{leg}_calf_collision{i}" for leg in foot_names for i in (1, 2)
  )
  shank_ground_cfg = ContactSensorCfg(
    name="shank_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=calf_geom_names,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
    unwanted_contact_ground_cfg,
    shank_ground_cfg,
    thigh_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = GO1_ACTION_SCALE

  cfg.viewer.body_name = "trunk"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction_slide"].params["asset_cfg"].geom_names = geom_names
  cfg.events["foot_friction_spin"].params["asset_cfg"].geom_names = geom_names
  cfg.events["foot_friction_roll"].params["asset_cfg"].geom_names = geom_names

  cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
  }
  cfg.rewards["upright"].weight = 0.1
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk",)
  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names
  cfg.rewards["body_ang_vel"].weight = -1.0e-4
  cfg.rewards["angular_momentum"].weight = -1.0e-4
  cfg.rewards["air_time"].weight = 0.0
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name},
  )
  cfg.rewards["shank_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": shank_ground_cfg.name},
  )
  cfg.rewards["thigh_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": thigh_ground_cfg.name},
  )
  cfg.rewards["joint_vel_l2"].weight = 0.0
  cfg.rewards["joint_acc_l2"].weight = 0.0

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": unwanted_contact_ground_cfg.name},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 10
        cfg.scene.terrain.terrain_generator.num_rows = 10
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go1 flat terrain velocity configuration."""
  cfg = unitree_go1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s
    for s in (cfg.scene.sensors or ())
    if s.name
    not in (
      "terrain_scan",
      "foot_clearance_scan",
      "foot_height_scan",
      "base_normal_scan",
    )
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Flat terrain only needs fell_over, not trunk-ground contact.
  del cfg.terminations["illegal_contact"]
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(50.0)},
  )

  # Disable terrain curriculum.
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


def unitree_go1_stairs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  from mjlab.terrains.config import STAIRS_TERRAINS_CFG

  cfg = unitree_go1_rough_env_cfg(play=play)

  # Use stairs-specific terrain.
  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_generator is not None
  cfg.scene.terrain.terrain_generator = replace(STAIRS_TERRAINS_CFG)

  return cfg
