from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor, HeightSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  return torch.exp(-ang_vel_error / std**2)


def flat_orientation(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  normal_sensor_name: str | None = None,
  flatness_threshold: float = 0.087,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.

  Args:
    normal_sensor_name: Optional HeightSensor for terrain normals below the
      base. When provided, the upright reward is scaled by terrain flatness:
      on sloped terrain (normal far from vertical) the reward is reduced so
      the robot is not penalized for leaning with the slope.
    flatness_threshold: Maximum XY component of terrain normal for full
      reward. Above this the reward linearly decays to 0. Default 0.087
      (~5° slope).
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

  reward = torch.exp(-xy_squared / std**2)

  # Gate by terrain flatness: disable upright penalty on slopes.
  if normal_sensor_name is not None:
    height_sensor: HeightSensor = env.scene[normal_sensor_name]
    terrain_normal = height_sensor.data.normals_w  # [B, S, 3]
    # Use first (and only) site — the base. XY magnitude = slope indicator.
    normal_xy = terrain_normal[:, 0, :2]  # [B, 2]
    slope = torch.norm(normal_xy, dim=-1)  # [B], 0 = flat, 1 = vertical wall
    # Hard gate: full reward on flat terrain, zero on slopes.
    is_flat = (slope < flatness_threshold).float()
    reward = reward * is_flat

  return reward


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  # Sum over all match dims (patterns, slots) to get a scalar per env.
  found = data.found
  while found.dim() > 1:
    found = found.sum(dim=-1)
  return found


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_mid: float = 0.2,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward feet air time with trapezoidal shape.

  The per-foot reward is:
    - 0 when air_time <= threshold_min or air_time >= threshold_max
    - linearly increasing from 0 to 1 between threshold_min and threshold_mid
    - 1 (flat) between threshold_mid and threshold_max
  """
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  # Linear ramp from 0 to 1 between threshold_min and threshold_mid.
  ramp = (current_air_time - threshold_min) / (threshold_mid - threshold_min)
  ramp = ramp.clamp(0.0, 1.0)
  # Zero out above threshold_max.
  active = (current_air_time < threshold_max).float()
  per_foot_reward = ramp * active
  reward = torch.sum(per_foot_reward, dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_stance_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_mid: float = 0.2,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward feet ground contact (stance) time with trapezoidal shape.

  Encourages the robot to keep each foot on the ground for a target duration,
  producing a slower, more deliberate gait cadence.

  The per-foot reward is:
    - 0 when contact_time <= threshold_min or contact_time >= threshold_max
    - linearly increasing from 0 to 1 between threshold_min and threshold_mid
    - 1 (flat) between threshold_mid and threshold_max
  """
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_contact_time = sensor_data.current_contact_time
  assert current_contact_time is not None
  # Linear ramp from 0 to 1 between threshold_min and threshold_mid.
  ramp = (current_contact_time - threshold_min) / (threshold_mid - threshold_min)
  ramp = ramp.clamp(0.0, 1.0)
  # Zero out above threshold_max.
  active = (current_contact_time < threshold_max).float()
  per_foot_reward = ramp * active
  reward = torch.sum(per_foot_reward, dim=1)
  in_contact = current_contact_time > 0
  num_in_contact = torch.sum(in_contact.float())
  mean_contact_time = torch.sum(
    current_contact_time * in_contact.float()
  ) / torch.clamp(num_in_contact, min=1)
  env.extras["log"]["Metrics/stance_time_mean"] = mean_contact_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  height_sensor_name: str | None = None,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity.

  Args:
    height_sensor_name: Optional HeightSensor name for terrain-relative heights.
      When None, uses raw site Z (correct on flat ground only).
  """
  asset: Entity = env.scene[asset_cfg.name]
  if height_sensor_name is not None:
    height_sensor: HeightSensor = env.scene[height_sensor_name]
    foot_z = height_sensor.data.heights  # [B, N] terrain-relative
  else:
    foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  # Guard against NaN/Inf from diverged physics.
  foot_z = torch.nan_to_num(foot_z, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  foot_vel_xy = torch.nan_to_num(foot_vel_xy, nan=0.0, posinf=0.0, neginf=0.0)
  vel_norm = torch.norm(foot_vel_xy, dim=-1).clamp_(max=10.0)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def terrain_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  sensor_name: str,
  height_sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Positive reward for lifting feet above terrain during swing phase.

  Unlike ``feet_clearance`` (a penalty gated on foot velocity that rewards
  stopping), this function gives a *positive* reward proportional to how
  close each swing foot is to the target clearance height.

  When ``command_name`` is provided, the reward is gated on command velocity
  so that standing still earns zero reward.

  Reward per foot (during swing only)::

      clamp(foot_height_above_terrain / target_height, 0, 1)

  Averaged across all feet. Feet on the ground contribute 0.

  Args:
    target_height: Desired clearance above terrain (m).
    sensor_name: ContactSensor name for swing/stance detection.
    height_sensor_name: HeightSensor name for terrain-relative foot heights.
    command_name: Optional velocity command name.  When set, the reward is
      zero for envs whose total command magnitude is below
      ``command_threshold``.
    command_threshold: Minimum command magnitude to activate the reward.
    asset_cfg: Entity config (unused except for scene lookup).
  """
  # Swing detection via contact sensor.
  contact_sensor: ContactSensor = env.scene[sensor_name]
  in_swing = contact_sensor.data.found == 0  # [B, N]

  # Terrain-relative foot heights from a min-reduction height sensor.
  height_sensor: HeightSensor = env.scene[height_sensor_name]
  foot_z = height_sensor.data.heights  # [B, N]
  foot_z = torch.nan_to_num(foot_z, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 2.0)

  # Proportional achievement, capped at 1.0 (no penalty for going higher).
  achievement = (foot_z / target_height).clamp_(0.0, 1.0)  # [B, N]

  # Average over swing feet only.  Feet on the ground contribute 0.
  reward = (achievement * in_swing.float()).mean(dim=1)  # [B]

  # Gate on command velocity — standing still earns nothing.
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      reward = reward * active

  return reward


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing.

  Supports optional ``height_sensor_name`` param for terrain-relative foot
  heights. When omitted, uses raw site Z (correct on flat ground only).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.height_sensor_name = cfg.params.get("height_sensor_name", None)
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
    height_sensor_name: str | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    if self.height_sensor_name is not None:
      height_sensor: HeightSensor = env.scene[self.height_sensor_name]
      foot_heights = height_sensor.data.heights  # [B, N] terrain-relative
    else:
      foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    # Guard against NaN/Inf from diverged physics.
    foot_heights = torch.nan_to_num(
      foot_heights, nan=0.0, posinf=1.0, neginf=0.0
    ).clamp_(0.0, 1.0)
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))
