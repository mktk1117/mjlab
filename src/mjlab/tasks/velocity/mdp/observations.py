from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, HeightSensor
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_normal_from_sensors
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  sensor_name: str | None = None,
) -> torch.Tensor:
  """Return foot heights, either terrain-relative or absolute.

  When ``sensor_name`` is provided (a configured :class:`HeightSensor`),
  returns terrain-relative height via raycasting (works on rough terrain).
  When ``sensor_name`` is ``None``, falls back to the raw site Z-coordinate
  (fast, correct on flat ground).

  Args:
    env: The environment.
    asset_cfg: Entity with ``site_ids`` specifying the foot sites.
    sensor_name: Optional HeightSensor name for terrain-relative heights.

  Returns:
    Tensor of shape ``(num_envs, num_sites)``.
  """
  if sensor_name is not None:
    sensor: HeightSensor = env.scene[sensor_name]
    return sensor.data.heights  # [B, S] terrain-relative
  # Flat terrain fallback: raw site Z coordinate.
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_terrain_heights(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Return per-ray terrain heights from a concentric ring scan.

  Designed for use with a :class:`HeightSensor` configured with
  ``reduction="none"`` and multi-ring sampling. Returns raw per-ray
  heights as a flat observation vector.

  Heights are returned **relative to site height** (i.e. subtracted from
  the center site height) so the observation is zero-centered: 0 means
  the terrain is at the same height as the site, negative means the
  terrain is lower.

  Args:
    env: The environment.
    sensor_name: Name of the :class:`HeightSensor` in the scene.

  Returns:
    Tensor of shape ``(num_envs, num_sites * num_samples_per_site)``.
  """
  sensor: HeightSensor = env.scene[sensor_name]
  data = sensor.data
  # heights shape: [B, S, K] with reduction="none", or [B, S] otherwise.
  return data.heights.flatten(start_dim=1)  # [B, S*K]


def foot_ground_normal(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Return terrain normals below each foot site.

  Args:
    env: The environment.
    sensor_name: Name of the :class:`HeightSensor` in the scene.

  Returns:
    Tensor of shape ``(num_envs, num_sites * 3)`` — normals flattened.
  """
  sensor: HeightSensor = env.scene[sensor_name]
  return sensor.data.normals_w.flatten(start_dim=1)  # [B, S*3]


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def terrain_projected_gravity(
  env: ManagerBasedRlEnv,
  sensor_names: tuple[str, ...],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terrain normal projected into body frame."""
  asset: Entity = env.scene[asset_cfg.name]
  terrain_normal_w = terrain_normal_from_sensors(env, sensor_names)
  return quat_apply_inverse(asset.data.root_link_quat_w, terrain_normal_w)


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))
