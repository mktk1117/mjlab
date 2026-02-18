from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_sampled_ray_ids: dict[tuple[str, int, int], torch.Tensor] = {}
_cached_grid_fit: dict[tuple[str, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_sampled_ray_ids(
  *, sensor_name: str, total_rays: int, max_points: int, device: torch.device
) -> torch.Tensor:
  key = (sensor_name, total_rays, max_points)
  ray_ids = _sampled_ray_ids.get(key)
  if ray_ids is None or ray_ids.device != device:
    if total_rays > max_points:
      ray_ids = torch.linspace(
        0, total_rays - 1, max_points, dtype=torch.long, device=device
      )
    else:
      ray_ids = torch.arange(total_rays, device=device)
    _sampled_ray_ids[key] = ray_ids
  return ray_ids


def _fit_normal_masked_covariance(
  points: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
  """Fit normal from points using masked covariance eigen-decomposition."""
  valid = valid_mask.unsqueeze(-1).float()
  valid_count = valid.sum(dim=1).clamp(min=1.0)
  centroid = (points * valid).sum(dim=1, keepdim=True) / valid_count.unsqueeze(-1)
  centered = (points - centroid) * valid
  cov = torch.einsum("bni,bnj->bij", centered, centered)
  _, eigvecs = torch.linalg.eigh(cov)
  normal = eigvecs[..., 0]
  normal = torch.where(normal[:, 2:3] < 0, -normal, normal)
  normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-8)
  enough_points = valid_count.squeeze(-1) >= 3
  fallback = torch.zeros_like(normal)
  fallback[:, 2] = 1.0
  return torch.where(enough_points.unsqueeze(-1), normal, fallback)


def terrain_normal_from_sensors(
  env: ManagerBasedRlEnv,
  sensor_names: tuple[str, ...],
  max_points: int = 8,
) -> torch.Tensor:
  """Fit a terrain normal from ray hits with masked handling for misses.

  Fast path:
    - Single sensor with fixed local grid offsets.
    - Uses precomputed least-squares projection for z = ax + by + c.
    - Falls back to masked normal-equation solve when rays miss.

  General fallback:
    - Multi-sensor or no local offsets available.
    - Uses masked covariance + eigen decomposition.
  """
  if len(sensor_names) == 1:
    sensor_name = sensor_names[0]
    sensor = env.scene[sensor_name]
    local_offsets = getattr(sensor, "_local_offsets", None)
    if local_offsets is not None:
      total_rays = local_offsets.shape[0]
      ray_ids = _get_sampled_ray_ids(
        sensor_name=sensor_name,
        total_rays=total_rays,
        max_points=max_points,
        device=local_offsets.device,
      )
      xy = local_offsets[ray_ids, :2]
      h = sensor.data.pos_w[:, 2].unsqueeze(-1) - sensor.data.hit_pos_w[:, ray_ids, 2]
      valid_mask = sensor.data.distances[:, ray_ids] >= 0

      # Fast path when all sampled rays hit.
      if bool(valid_mask.all()):
        cache_key = (sensor_name, total_rays, max_points)
        cached = _cached_grid_fit.get(cache_key)
        if cached is None or cached[0].device != xy.device:
          ones = torch.ones((xy.shape[0], 1), device=xy.device, dtype=xy.dtype)
          design = torch.cat([xy, ones], dim=1)  # [K, 3]
          pinv = torch.linalg.pinv(design)  # [3, K]
          _cached_grid_fit[cache_key] = (pinv, ray_ids)
        else:
          pinv, _ = cached
        pinv = _cached_grid_fit[cache_key][0]
        theta = h @ pinv.T  # [B, 3], columns are [a, b, c]
      else:
        # Masked weighted least squares with tiny Tikhonov regularization.
        x = xy[:, 0].unsqueeze(0)
        y = xy[:, 1].unsqueeze(0)
        m = valid_mask.float()
        one = m
        s_xx = torch.sum(m * x * x, dim=1)
        s_xy = torch.sum(m * x * y, dim=1)
        s_yy = torch.sum(m * y * y, dim=1)
        s_x = torch.sum(m * x, dim=1)
        s_y = torch.sum(m * y, dim=1)
        s_1 = torch.sum(one, dim=1)
        b_x = torch.sum(m * x * h, dim=1)
        b_y = torch.sum(m * y * h, dim=1)
        b_1 = torch.sum(m * h, dim=1)
        mat = torch.stack(
          [
            torch.stack([s_xx, s_xy, s_x], dim=1),
            torch.stack([s_xy, s_yy, s_y], dim=1),
            torch.stack([s_x, s_y, s_1], dim=1),
          ],
          dim=1,
        )
        rhs = torch.stack([b_x, b_y, b_1], dim=1)
        reg = 1e-6 * torch.eye(3, device=mat.device, dtype=mat.dtype).unsqueeze(0)
        theta = torch.linalg.solve(mat + reg, rhs)

      normal_sensor = torch.stack(
        [-theta[:, 0], -theta[:, 1], torch.ones_like(theta[:, 0])], dim=1
      )
      normal_sensor = normal_sensor / (
        torch.norm(normal_sensor, dim=-1, keepdim=True) + 1e-8
      )

      # Rotate from sensor frame to world frame. For yaw-aligned scans this is
      # equivalent to rotating around +Z only.
      normal_world = quat_apply(sensor.data.quat_w, normal_sensor)
      normal_world = torch.where(normal_world[:, 2:3] < 0, -normal_world, normal_world)
      return normal_world / (torch.norm(normal_world, dim=-1, keepdim=True) + 1e-8)

  all_points = []
  all_valid = []
  for name in sensor_names:
    sensor = env.scene[name]
    total_rays = sensor.data.hit_pos_w.shape[1]
    ray_ids = _get_sampled_ray_ids(
      sensor_name=name,
      total_rays=total_rays,
      max_points=max_points,
      device=sensor.data.hit_pos_w.device,
    )
    all_points.append(sensor.data.hit_pos_w[:, ray_ids])
    all_valid.append(sensor.data.distances[:, ray_ids] >= 0)
  points = torch.cat(all_points, dim=1)
  valid_mask = torch.cat(all_valid, dim=1)
  return _fit_normal_masked_covariance(points, valid_mask)
