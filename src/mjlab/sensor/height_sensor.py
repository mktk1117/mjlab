"""Height sensor for per-site ground height and surface normal measurement.

Casts downward rays per site to measure the height above ground and
the surface normal at the ground contact point. This is useful for:

- Foot height above terrain (for clearance rewards, swing height tracking)
- Base height above terrain (for posture control, terminations)
- Terrain normal at each site (for foot/base alignment on rough terrain)
- Local terrain height map as observation (concentric ring scan)

Usage::

    from mjlab.sensor import HeightSensorCfg, ObjRef

    # Single ray per site (simplest):
    cfg = HeightSensorCfg(
        name="foot_height",
        sites=(
            ObjRef(type="site", name="foot_fl", entity="robot"),
            ObjRef(type="site", name="foot_fr", entity="robot"),
        ),
    )

    # Multi-ray sampling with reduction (e.g., min height in 50cm radius):
    cfg = HeightSensorCfg(
        name="foot_height",
        sites=(...),
        sampling=HeightSensorCfg.SamplingCfg(
            radius=0.5,
            num_samples=8,
        ),
        reduction="min",  # or "mean", "max", "median"
    )

    # Concentric ring terrain scan (raw per-ray heights as observation):
    cfg = HeightSensorCfg(
        name="foot_terrain_scan",
        sites=(...),
        sampling=HeightSensorCfg.SamplingCfg(
            rings=(
                HeightSensorCfg.RingCfg(radius=0.05, num_samples=4),
                HeightSensorCfg.RingCfg(radius=0.10, num_samples=6),
                HeightSensorCfg.RingCfg(radius=0.20, num_samples=8),
                HeightSensorCfg.RingCfg(radius=0.35, num_samples=10),
                HeightSensorCfg.RingCfg(radius=0.50, num_samples=12),
            ),
        ),
        reduction="none",  # raw per-ray data for observation
    )

Output Data
-----------

Access sensor data via the ``data`` property, which returns ``HeightSensorData``:

- ``heights``: [B, S] Height of each site above ground (positive = above). -1 if
  no ground hit or beyond max_distance. After reduction when using multi-ray sampling.
  Shape [B, S, K] when reduction="none".
- ``normals_w``: [B, S, 3] Surface normal at the ground point below each site
  (world frame). Zero if no hit. Shape [B, S, K, 3] when reduction="none".
- ``hit_pos_w``: [B, S, 3] Ground hit position in world frame (center ray only).
- ``site_pos_w``: [B, S, 3] World-space site positions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp
from mujoco_warp import rays

from mjlab.entity import Entity
from mjlab.sensor.builtin_sensor import ObjRef
from mjlab.sensor.sensor import Sensor, SensorCfg

if TYPE_CHECKING:
  from mjlab.sensor.sensor_context import SensorContext
  from mjlab.viewer.debug_visualizer import DebugVisualizer

# NOTE: Need to define this here because it's not publicly exposed by mujoco_warp.
vec6 = wp.types.vector(length=6, dtype=float)

logger = logging.getLogger(__name__)

# Type aliases for configuration choices.
RayAlignment = Literal["base", "yaw", "world"]
Reduction = Literal["mean", "min", "max", "median", "none"]


@dataclass
class HeightSensorData:
  """Height sensor output data.

  Note:
    Fields are views into GPU buffers and are valid until the next
    ``sense()`` call.
  """

  heights: torch.Tensor
  """[B, S] or [B, S, K] Height of each site above ground.

  Shape is [B, S] when reduction is applied (default), or [B, S, K] when
  reduction="none" (K = num_samples per site). -1 for misses.
  """

  normals_w: torch.Tensor
  """[B, S, 3] or [B, S, K, 3] Surface normal at ground point (world frame).

  Shape is [B, S, 3] when reduction is applied, or [B, S, K, 3] when
  reduction="none". Zero for misses.
  """

  hit_pos_w: torch.Tensor
  """[B, S, 3] Ground hit position for center ray (world frame).

  Always shape [B, S, 3] regardless of reduction (uses center/first ray).
  """

  site_pos_w: torch.Tensor
  """[B, S, 3] Site position in world frame."""


@dataclass
class HeightSensorCfg(SensorCfg):
  """Height sensor configuration.

  Casts downward rays per site to measure ground-relative height and
  surface normal. Supports multi-ray sampling with aggregation.
  """

  @dataclass
  class VizCfg:
    """Visualization settings for debug rendering."""

    hit_color: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.8)
    """RGBA color for rays that hit a surface."""

    miss_color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.4)
    """RGBA color for rays that miss."""

    hit_sphere_color: tuple[float, float, float, float] = (0.0, 1.0, 1.0, 1.0)
    """RGBA color for spheres drawn at hit points."""

    hit_sphere_radius: float = 0.5
    """Radius of spheres drawn at hit points (multiplier of meansize)."""

    show_rays: bool = True
    """Whether to draw ray arrows."""

    show_normals: bool = False
    """Whether to draw surface normals at hit points."""

    normal_color: tuple[float, float, float, float] = (1.0, 1.0, 0.0, 1.0)
    """RGBA color for surface normal arrows."""

    normal_length: float = 5.0
    """Length of surface normal arrows (multiplier of meansize)."""

  @dataclass
  class RingCfg:
    """Configuration for a single ring in a concentric pattern."""

    radius: float
    """Radius of this ring in meters."""

    num_samples: int
    """Number of evenly spaced sample points on this ring."""

  @dataclass
  class SamplingCfg:
    """Multi-ray sampling pattern around each site.

    Supports two modes:

    1. **Single ring**: Set ``radius`` and ``num_samples`` for one ring.
    2. **Concentric rings**: Set ``rings`` to a tuple of :class:`RingCfg`
       for multiple concentric rings at different radii.

    When ``rings`` is provided, ``radius`` and ``num_samples`` are ignored.

    With ``include_center=True`` (default), a center ray is always included.

    Example (concentric rings around each foot)::

        SamplingCfg(
            rings=(
                HeightSensorCfg.RingCfg(radius=0.05, num_samples=4),
                HeightSensorCfg.RingCfg(radius=0.10, num_samples=6),
                HeightSensorCfg.RingCfg(radius=0.20, num_samples=8),
                HeightSensorCfg.RingCfg(radius=0.35, num_samples=10),
                HeightSensorCfg.RingCfg(radius=0.50, num_samples=12),
            ),
        )
    """

    radius: float = 0.1
    """Radius for single-ring mode. Ignored when ``rings`` is set."""

    num_samples: int = 8
    """Samples for single-ring mode. Ignored when ``rings`` is set."""

    rings: tuple[HeightSensorCfg.RingCfg, ...] | None = None
    """Concentric ring definitions. Overrides ``radius``/``num_samples``."""

    include_center: bool = True
    """Whether to include a ray at the site center (in addition to rings)."""

  sites: tuple[ObjRef, ...] = ()
  """Sites to measure height for."""

  direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
  """Ray direction in frame-local coordinates (default: straight down)."""

  ray_alignment: RayAlignment = "world"
  """How the ray direction aligns with the frame.

  - "world": Fixed in world frame (always shoots -Z). Default — best for
    ground height measurement regardless of body orientation.
  - "yaw": Position + yaw only, ignores pitch/roll.
  - "base": Full position + rotation. Ray direction rotates with body.
  """

  sampling: SamplingCfg | None = None
  """Optional multi-ray sampling pattern.

  When None (default), one ray is cast per site. When specified, multiple
  rays are cast around each site in a ring pattern and aggregated using
  the ``reduction`` method.
  """

  reduction: Reduction = "mean"
  """How to aggregate multi-ray samples per site.

  Only used when ``sampling`` is specified. Options:
  - "mean": Average of all sample heights/normals.
  - "min": Minimum height per site (useful for foot clearance).
  - "max": Maximum height per site.
  - "median": Median height per site.
  - "none": No reduction — output raw per-ray data [B, S, K].
  """

  max_distance: float = 10.0
  """Maximum ray distance. Rays beyond this report -1."""

  exclude_parent_body: bool = True
  """Exclude parent body from ray intersection tests."""

  include_geom_groups: tuple[int, ...] | None = (0, 1, 2)
  """Geom groups (0-5) to include in raycasting.

  Defaults to (0, 1, 2). Set to None to include all groups.
  """

  debug_vis: bool = False
  """Enable debug visualization."""

  viz: VizCfg = field(default_factory=VizCfg)
  """Visualization settings."""

  def build(self) -> HeightSensor:
    return HeightSensor(self)


class HeightSensor(Sensor[HeightSensorData]):
  """Height sensor that measures per-site ground height via raycasting."""

  requires_sensor_context = True

  def __init__(self, cfg: HeightSensorCfg) -> None:
    super().__init__()
    self.cfg = cfg
    self._data: mjwarp.Data | None = None
    self._model: mjwarp.Model | None = None
    self._mj_model: mujoco.MjModel | None = None
    self._device: str | None = None
    self._wp_device: wp.context.Device | None = None

    self._site_ids: list[int] = []
    self._site_body_ids: list[int] = []
    self._num_sites: int = 0
    self._num_samples_per_site: int = 1  # 1 = single ray (no sampling)
    self._num_total_rays: int = 0

    # Ray direction (normalized).
    self._ray_direction: torch.Tensor | None = None

    # Horizontal offsets for multi-ray sampling [K, 3].
    self._sample_offsets: torch.Tensor | None = None

    # Warp arrays for raycasting.
    self._ray_pnt: wp.array | None = None
    self._ray_vec: wp.array | None = None
    self._ray_dist: wp.array | None = None
    self._ray_geomid: wp.array | None = None
    self._ray_normal: wp.array | None = None
    self._ray_bodyexclude: wp.array | None = None
    self._geomgroup = vec6(-1, -1, -1, -1, -1, -1)

    # Output tensors.
    self._heights: torch.Tensor | None = None
    self._normals_w: torch.Tensor | None = None
    self._hit_pos_w: torch.Tensor | None = None
    self._site_pos_w: torch.Tensor | None = None

    # Cache for post-processing.
    self._cached_world_origins: torch.Tensor | None = None
    self._cached_world_rays: torch.Tensor | None = None

    self._ctx: SensorContext | None = None
    self._warned_normal_reduction: bool = False

  def edit_spec(
    self,
    scene_spec: mujoco.MjSpec,
    entities: dict[str, Entity],
  ) -> None:
    del scene_spec, entities

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    self._data = data
    self._model = model
    self._mj_model = mj_model
    self._device = device
    self._wp_device = wp.get_device(device)
    num_envs = data.nworld

    # Resolve site IDs.
    self._site_ids = []
    self._site_body_ids = []
    for site_ref in self.cfg.sites:
      site_name = site_ref.prefixed_name()
      site_id = mj_model.site(site_name).id
      self._site_ids.append(site_id)
      self._site_body_ids.append(int(mj_model.site_bodyid[site_id]))
    self._num_sites = len(self._site_ids)

    if self._num_sites == 0:
      return

    # Compute sampling pattern.
    self._sample_offsets = self._build_sample_offsets(device)
    self._num_samples_per_site = self._sample_offsets.shape[0]
    self._num_total_rays = self._num_sites * self._num_samples_per_site

    # Normalize ray direction.
    direction = torch.tensor(
      self.cfg.direction, device=device, dtype=torch.float32
    )
    self._ray_direction = direction / direction.norm()

    # Allocate warp arrays: [B, S*K] where K = samples per site.
    self._ray_pnt = wp.zeros(
      (num_envs, self._num_total_rays), dtype=wp.vec3, device=device
    )
    self._ray_vec = wp.zeros(
      (num_envs, self._num_total_rays), dtype=wp.vec3, device=device
    )
    self._ray_dist = wp.zeros(
      (num_envs, self._num_total_rays), dtype=float, device=device
    )
    self._ray_geomid = wp.zeros(
      (num_envs, self._num_total_rays), dtype=int, device=device
    )
    self._ray_normal = wp.zeros(
      (num_envs, self._num_total_rays), dtype=wp.vec3, device=device
    )

    # Body exclusion: each ray excludes its site's parent body.
    # Repeat each body ID K times for multi-ray.
    if self.cfg.exclude_parent_body:
      body_excludes = []
      for body_id in self._site_body_ids:
        body_excludes.extend([body_id] * self._num_samples_per_site)
    else:
      body_excludes = [-1] * self._num_total_rays
    self._ray_bodyexclude = wp.array(
      body_excludes,
      dtype=int,
      device=device,
    )

    # Convert include_geom_groups to vec6 format (-1 = include, 0 = exclude).
    if self.cfg.include_geom_groups is not None:
      groups = [0, 0, 0, 0, 0, 0]
      for g in self.cfg.include_geom_groups:
        if 0 <= g <= 5:
          groups[g] = -1
      self._geomgroup = vec6(*groups)
    else:
      self._geomgroup = vec6(-1, -1, -1, -1, -1, -1)

    # Pre-allocate output tensors.
    self._heights = torch.zeros(num_envs, self._num_sites, device=device)
    self._normals_w = torch.zeros(
      num_envs, self._num_sites, 3, device=device
    )
    self._hit_pos_w = torch.zeros(
      num_envs, self._num_sites, 3, device=device
    )
    self._site_pos_w = torch.zeros(
      num_envs, self._num_sites, 3, device=device
    )

    assert self._wp_device is not None

  def _build_sample_offsets(self, device: str) -> torch.Tensor:
    """Build horizontal offsets for multi-ray sampling.

    Returns:
      Tensor of shape [K, 3] — offsets in XY plane (Z=0).
      When no sampling is configured, returns a single zero offset [1, 3].
    """
    if self.cfg.sampling is None:
      # Single ray at center.
      return torch.zeros((1, 3), device=device, dtype=torch.float32)

    sampling = self.cfg.sampling
    offsets: list[torch.Tensor] = []

    if sampling.include_center:
      offsets.append(torch.zeros(3, device=device, dtype=torch.float32))

    # Normalize single-ring to a list so the loop is the same.
    rings = sampling.rings or (
      HeightSensorCfg.RingCfg(sampling.radius, sampling.num_samples),
    )
    for ring in rings:
      for i in range(ring.num_samples):
        angle = 2.0 * math.pi * i / ring.num_samples
        offsets.append(torch.tensor(
          [ring.radius * math.cos(angle), ring.radius * math.sin(angle), 0.0],
          device=device,
          dtype=torch.float32,
        ))

    return torch.stack(offsets)  # [K, 3]

  def set_context(self, ctx: SensorContext) -> None:
    """Wire this sensor to a SensorContext for BVH-accelerated raycasting."""
    self._ctx = ctx

  @property
  def num_sites(self) -> int:
    return self._num_sites

  @property
  def num_samples_per_site(self) -> int:
    return self._num_samples_per_site

  def _compute_data(self) -> HeightSensorData:
    if self._ctx is None:
      raise RuntimeError(
        "HeightSensor requires a SensorContext. "
        "Ensure the sensor is part of a scene with "
        "sim.sense() calls."
      )
    assert self._heights is not None and self._normals_w is not None
    assert self._hit_pos_w is not None and self._site_pos_w is not None
    return HeightSensorData(
      heights=self._heights,
      normals_w=self._normals_w,
      hit_pos_w=self._hit_pos_w,
      site_pos_w=self._site_pos_w,
    )

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self.cfg.debug_vis:
      return
    assert self._data is not None

    data = self.data
    env_indices = list(visualizer.get_env_indices(data.heights.shape[0]))
    if not env_indices:
      return

    meansize = visualizer.meansize
    ray_width = 0.1 * meansize
    sphere_radius = self.cfg.viz.hit_sphere_radius * meansize
    normal_length = self.cfg.viz.normal_length * meansize
    normal_width = 0.1 * meansize
    miss_extent = min(0.5, self.cfg.max_distance * 0.05)
    name = self.cfg.name

    site_pos = data.site_pos_w[env_indices].cpu().numpy()
    hit_pos = data.hit_pos_w[env_indices].cpu().numpy()
    heights = data.heights[env_indices].cpu().numpy()
    normals = data.normals_w[env_indices].cpu().numpy()
    direction = self._ray_direction
    assert direction is not None
    dir_np = direction.cpu().numpy()

    for k in range(len(env_indices)):
      for i in range(self._num_sites):
        origin = site_pos[k, i]
        hit = heights[k, i] >= 0

        if hit:
          end = hit_pos[k, i]
          color = self.cfg.viz.hit_color
        else:
          end = origin + dir_np * miss_extent
          color = self.cfg.viz.miss_color

        if self.cfg.viz.show_rays:
          visualizer.add_arrow(
            start=origin,
            end=end,
            color=color,
            width=ray_width,
            label=f"{name}_ray_{i}",
          )

        if hit:
          visualizer.add_sphere(
            center=end,
            radius=sphere_radius,
            color=self.cfg.viz.hit_sphere_color,
            label=f"{name}_hit_{i}",
          )
          if self.cfg.viz.show_normals:
            normal_end = end + normals[k, i] * normal_length
            visualizer.add_arrow(
              start=end,
              end=normal_end,
              color=self.cfg.viz.normal_color,
              width=normal_width,
              label=f"{name}_normal_{i}",
            )

  # Private methods.

  def prepare_rays(self) -> None:
    """PRE-GRAPH: Read site positions and compute world-frame ray origins/dirs."""
    assert self._data is not None and self._model is not None
    assert self._ray_direction is not None
    assert self._sample_offsets is not None

    if self._num_sites == 0:
      return

    num_envs = self._data.nworld
    S = self._num_sites
    K = self._num_samples_per_site

    # Gather site positions [B, S, 3].
    site_ids = torch.tensor(
      self._site_ids, device=self._device, dtype=torch.long
    )
    site_pos = self._data.site_xpos[:, site_ids]  # [B, S, 3]

    # Expand site positions to include sample offsets → [B, S, K, 3].
    # offsets is [K, 3], broadcast to [1, 1, K, 3].
    offsets = self._sample_offsets.unsqueeze(0).unsqueeze(0)  # [1, 1, K, 3]
    site_pos_expanded = site_pos.unsqueeze(2) + offsets  # [B, S, K, 3]

    # Flatten to [B, S*K, 3] for raycasting.
    all_origins = site_pos_expanded.reshape(num_envs, S * K, 3)

    # Compute ray direction based on alignment.
    if self.cfg.ray_alignment == "world":
      # Fixed world-frame direction for all rays.
      world_rays = (
        self._ray_direction.unsqueeze(0)
        .unsqueeze(0)
        .expand(num_envs, S * K, 3)
        .clone()
      )
    elif self.cfg.ray_alignment == "base":
      # Rotate direction by each site's parent body rotation.
      site_mat = self._data.site_xmat[:, site_ids].view(
        num_envs, S, 3, 3
      )  # [B, S, 3, 3]
      world_rays_per_site = torch.einsum(
        "bsij,j->bsi", site_mat, self._ray_direction
      )  # [B, S, 3]
      # Repeat for each sample.
      world_rays = (
        world_rays_per_site.unsqueeze(2)
        .expand(num_envs, S, K, 3)
        .reshape(num_envs, S * K, 3)
      )
    elif self.cfg.ray_alignment == "yaw":
      # Use yaw-only rotation from each site's parent body.
      site_mat = self._data.site_xmat[:, site_ids].view(
        num_envs, S, 3, 3
      )  # [B, S, 3, 3]
      yaw_mat = self._extract_yaw_rotation_batched(site_mat)
      world_rays_per_site = torch.einsum(
        "bsij,j->bsi", yaw_mat, self._ray_direction
      )
      world_rays = (
        world_rays_per_site.unsqueeze(2)
        .expand(num_envs, S, K, 3)
        .reshape(num_envs, S * K, 3)
      )
    else:
      raise ValueError(f"Unknown ray_alignment: {self.cfg.ray_alignment}")

    # Write to warp arrays.
    assert self._ray_pnt is not None and self._ray_vec is not None
    pnt_torch = wp.to_torch(self._ray_pnt).view(num_envs, S * K, 3)
    vec_torch = wp.to_torch(self._ray_vec).view(num_envs, S * K, 3)
    pnt_torch.copy_(all_origins)
    vec_torch.copy_(world_rays)

    # Cache for postprocess.
    self._cached_world_origins = all_origins  # [B, S*K, 3]
    self._cached_world_rays = world_rays      # [B, S*K, 3]
    self._site_pos_w = site_pos               # [B, S, 3] (un-expanded)

  def raycast_kernel(self, rc: mjwarp.RenderContext) -> None:
    """IN-GRAPH: Execute BVH-accelerated raycast kernel."""
    if self._num_sites == 0:
      return
    rays(
      m=self._model.struct,  # type: ignore[attr-defined]
      d=self._data.struct,  # type: ignore[attr-defined]
      pnt=self._ray_pnt,
      vec=self._ray_vec,
      geomgroup=self._geomgroup,  # pyright: ignore[reportArgumentType]
      flg_static=True,
      bodyexclude=self._ray_bodyexclude,
      dist=self._ray_dist,
      geomid=self._ray_geomid,
      normal=self._ray_normal,
      rc=rc,
    )

  def postprocess_rays(self) -> None:
    """POST-GRAPH: Convert Warp outputs to PyTorch, compute heights, reduce."""
    if self._num_sites == 0:
      return

    assert self._cached_world_origins is not None
    assert self._cached_world_rays is not None

    num_envs = self._cached_world_origins.shape[0]
    S = self._num_sites
    K = self._num_samples_per_site

    assert self._ray_dist is not None and self._ray_normal is not None
    # Raw flat arrays [B, S*K].
    distances = wp.to_torch(self._ray_dist)
    normals_flat = wp.to_torch(self._ray_normal).view(num_envs, S * K, 3)
    distances[distances > self.cfg.max_distance] = -1.0

    hit_mask = distances >= 0  # [B, S*K]
    hit_pos_flat = self._cached_world_origins.clone()  # [B, S*K, 3]
    hit_pos_flat[hit_mask] = (
      self._cached_world_origins[hit_mask]
      + self._cached_world_rays[hit_mask]
      * distances[hit_mask].unsqueeze(-1)
    )

    # Zero out normals for misses.
    normals_flat[~hit_mask] = 0.0

    # Compute raw per-ray heights [B, S*K].
    raw_heights = torch.where(
      hit_mask,
      self._cached_world_origins[..., 2] - hit_pos_flat[..., 2],
      torch.tensor(-1.0, device=distances.device),
    )

    # Hit position for center ray (first ray per site) → [B, S, 3].
    hit_pos_per_site = hit_pos_flat.view(num_envs, S, K, 3)[:, :, 0, :]
    self._hit_pos_w = hit_pos_per_site

    # Apply reduction.
    if K == 1:
      # Single ray — no reduction needed.
      self._heights = raw_heights.view(num_envs, S)
      self._normals_w = normals_flat.view(num_envs, S, 3)
    elif self.cfg.reduction == "none":
      # Raw per-ray data: [B, S, K] and [B, S, K, 3].
      self._heights = raw_heights.view(num_envs, S, K)
      self._normals_w = normals_flat.view(num_envs, S, K, 3)
    else:
      heights_per_site = raw_heights.view(num_envs, S, K)  # [B, S, K]
      normals_per_site = normals_flat.view(num_envs, S, K, 3)

      # For reductions that need to handle misses (-1):
      # Replace misses with appropriate sentinel values.
      valid_mask = heights_per_site >= 0  # [B, S, K]

      self._heights = self._reduce_heights(
        heights_per_site, valid_mask
      )
      self._normals_w = self._reduce_normals(
        normals_per_site, valid_mask, heights_per_site
      )

  def _reduce_heights(
    self,
    heights: torch.Tensor,
    valid_mask: torch.Tensor,
  ) -> torch.Tensor:
    """Reduce [B, S, K] heights to [B, S] using configured reduction.

    Handles misses (height=-1) by excluding them from the aggregation.
    If all rays miss for a site, the result is -1.
    """
    B, S, K = heights.shape
    device = heights.device
    any_valid = valid_mask.any(dim=-1)  # [B, S]

    if self.cfg.reduction == "min":
      # Replace misses with +inf so they don't affect min.
      filled = torch.where(valid_mask, heights, torch.tensor(float("inf"), device=device))
      result = filled.min(dim=-1).values
    elif self.cfg.reduction == "max":
      # Replace misses with -inf so they don't affect max.
      filled = torch.where(valid_mask, heights, torch.tensor(float("-inf"), device=device))
      result = filled.max(dim=-1).values
    elif self.cfg.reduction == "mean":
      # Zero-fill misses and divide by count of valid rays.
      filled = torch.where(valid_mask, heights, torch.zeros_like(heights))
      count = valid_mask.float().sum(dim=-1).clamp(min=1)
      result = filled.sum(dim=-1) / count
    elif self.cfg.reduction == "median":
      # Replace misses with +inf, then take median.
      filled = torch.where(valid_mask, heights, torch.tensor(float("inf"), device=device))
      result = filled.median(dim=-1).values
    else:
      raise ValueError(f"Unknown reduction: {self.cfg.reduction}")

    # Where all rays missed, return -1.
    return torch.where(any_valid, result, torch.tensor(-1.0, device=device))

  def _reduce_normals(
    self,
    normals: torch.Tensor,
    valid_mask: torch.Tensor,
    heights: torch.Tensor,
  ) -> torch.Tensor:
    """Reduce [B, S, K, 3] normals to [B, S, 3] using configured reduction.

    For "mean", computes the average normal and re-normalizes.
    For "min"/"max"/"median", picks the normal at the ray selected by the
    corresponding height reduction (argmin/argmax/median index).
    """
    B, S, K, _ = normals.shape
    device = normals.device
    any_valid = valid_mask.any(dim=-1)  # [B, S]

    if self.cfg.reduction == "mean":
      filled = torch.where(
        valid_mask.unsqueeze(-1), normals, torch.zeros_like(normals)
      )
      count = valid_mask.float().sum(dim=-1, keepdim=True).clamp(min=1)
      avg = filled.sum(dim=2) / count
      # Re-normalize.
      norm = avg.norm(dim=-1, keepdim=True).clamp(min=1e-6)
      result = avg / norm
    else:
      # Emit one-time warning for non-mean normal reductions.
      if not self._warned_normal_reduction:
        logger.warning(
          "HeightSensor '%s': reduction='%s' selects the normal at the "
          "ray with the %s height. For geometrically meaningful terrain "
          "normals, consider using reduction='mean'.",
          self.cfg.name,
          self.cfg.reduction,
          self.cfg.reduction,
        )
        self._warned_normal_reduction = True

      # Compute the selection index from heights.
      if self.cfg.reduction == "min":
        filled = torch.where(
          valid_mask, heights,
          torch.tensor(float("inf"), device=device),
        )
        idx = filled.argmin(dim=-1)  # [B, S]
      elif self.cfg.reduction == "max":
        filled = torch.where(
          valid_mask, heights,
          torch.tensor(float("-inf"), device=device),
        )
        idx = filled.argmax(dim=-1)  # [B, S]
      elif self.cfg.reduction == "median":
        filled = torch.where(
          valid_mask, heights,
          torch.tensor(float("inf"), device=device),
        )
        # argsort and pick the middle index.
        sorted_idx = filled.argsort(dim=-1)
        mid = K // 2
        idx = sorted_idx[:, :, mid]  # [B, S]
      else:
        raise ValueError(f"Unknown reduction: {self.cfg.reduction}")

      # Gather normals at the selected index: [B, S, 3].
      idx_expanded = idx.unsqueeze(-1).unsqueeze(-1).expand(B, S, 1, 3)
      result = normals.gather(dim=2, index=idx_expanded).squeeze(2)

    # Zero out normals where all rays missed.
    result[~any_valid] = 0.0
    return result

  def _extract_yaw_rotation_batched(
    self, rot_mat: torch.Tensor
  ) -> torch.Tensor:
    """Extract yaw-only rotation from [B, S, 3, 3] rotation matrices."""
    B, S = rot_mat.shape[:2]
    device = rot_mat.device
    dtype = rot_mat.dtype

    # Project X-axis onto XY plane.
    x_axis = rot_mat[..., 0]  # [B, S, 3]
    x_proj = x_axis.clone()
    x_proj[..., 2] = 0
    x_norm = x_proj.norm(dim=-1)  # [B, S]

    # Handle singularity.
    threshold = 0.1
    singular = x_norm < threshold

    if singular.any():
      y_axis = rot_mat[..., 1]
      y_proj = y_axis.clone()
      y_proj[..., 2] = 0
      y_norm = y_proj.norm(dim=-1).clamp(min=1e-6)
      y_proj = y_proj / y_norm.unsqueeze(-1)
      x_from_y = torch.zeros_like(y_proj)
      x_from_y[..., 0] = y_proj[..., 1]
      x_from_y[..., 1] = -y_proj[..., 0]
      x_proj[singular] = x_from_y[singular]
      x_norm[singular] = 1.0

    x_norm = x_norm.clamp(min=1e-6)
    x_proj = x_proj / x_norm.unsqueeze(-1)

    yaw_mat = torch.zeros((B, S, 3, 3), device=device, dtype=dtype)
    yaw_mat[..., 0, 0] = x_proj[..., 0]
    yaw_mat[..., 1, 0] = x_proj[..., 1]
    yaw_mat[..., 0, 1] = -x_proj[..., 1]
    yaw_mat[..., 1, 1] = x_proj[..., 0]
    yaw_mat[..., 2, 2] = 1
    return yaw_mat
