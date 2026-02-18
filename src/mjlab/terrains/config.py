"""Terrain configuration presets and named terrain sets.

Each terrain type has a preset function with sensible defaults. Override any
parameter at the call site. Named configs (ROUGH, ALL, STAIRS) compose presets.

To add a new terrain, define a function decorated with @terrain_preset:

    @terrain_preset
    def my_new_terrain(**overrides) -> terrain_gen.SomeTerrainCfg:
      defaults: dict[str, Any] = dict(...)
      defaults.update(overrides)
      return terrain_gen.SomeTerrainCfg(**defaults)

It will be auto-included in ALL_TERRAIN_PRESETS and ALL_TERRAINS_CFG.
"""

from collections.abc import Callable
from typing import Any, TypeVar

import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg
from mjlab.terrains.terrain_importer import TerrainImporter, TerrainImporterCfg

# =============================================================================
# Auto-registration of terrain presets.
# =============================================================================

ALL_TERRAIN_PRESETS: dict[str, Callable[..., SubTerrainCfg]] = {}

_F = TypeVar("_F", bound=Callable[..., SubTerrainCfg])


def terrain_preset(fn: _F) -> _F:
  """Decorator that registers a terrain preset into ALL_TERRAIN_PRESETS."""
  ALL_TERRAIN_PRESETS[fn.__name__] = fn
  return fn


# =============================================================================
# Terrain presets — one function per terrain type, sensible defaults.
# Call with proportion + any overrides, e.g. pyramid_stairs(proportion=0.2).
# =============================================================================


@terrain_preset
def flat(**overrides) -> terrain_gen.BoxFlatTerrainCfg:
  return terrain_gen.BoxFlatTerrainCfg(**overrides)


@terrain_preset
def pyramid_stairs(**overrides) -> terrain_gen.BoxPyramidStairsTerrainCfg:
  defaults: dict[str, Any] = dict(
    step_height_range=(0.0, 0.2),
    step_width=0.3,
    platform_width=3.0,
    border_width=1.0,
  )
  defaults.update(overrides)
  return terrain_gen.BoxPyramidStairsTerrainCfg(**defaults)


@terrain_preset
def pyramid_stairs_inv(**overrides) -> terrain_gen.BoxInvertedPyramidStairsTerrainCfg:
  defaults: dict[str, Any] = dict(
    step_height_range=(0.0, 0.2),
    step_width=0.3,
    platform_width=3.0,
    border_width=1.0,
  )
  defaults.update(overrides)
  return terrain_gen.BoxInvertedPyramidStairsTerrainCfg(**defaults)


@terrain_preset
def hf_pyramid_slope(**overrides) -> terrain_gen.HfPyramidSlopedTerrainCfg:
  defaults: dict[str, Any] = dict(
    slope_range=(0.0, 0.7),
    platform_width=2.0,
    border_width=0.25,
  )
  defaults.update(overrides)
  return terrain_gen.HfPyramidSlopedTerrainCfg(**defaults)


@terrain_preset
def hf_pyramid_slope_inv(**overrides) -> terrain_gen.HfPyramidSlopedTerrainCfg:
  defaults: dict[str, Any] = dict(
    slope_range=(0.0, 0.7),
    platform_width=2.0,
    border_width=0.25,
    inverted=True,
  )
  defaults.update(overrides)
  return terrain_gen.HfPyramidSlopedTerrainCfg(**defaults)


@terrain_preset
def random_rough(**overrides) -> terrain_gen.HfRandomUniformTerrainCfg:
  defaults: dict[str, Any] = dict(
    noise_range=(0.02, 0.10),
    noise_step=0.02,
    border_width=0.25,
  )
  defaults.update(overrides)
  return terrain_gen.HfRandomUniformTerrainCfg(**defaults)


@terrain_preset
def wave_terrain(**overrides) -> terrain_gen.HfWaveTerrainCfg:
  defaults: dict[str, Any] = dict(
    amplitude_range=(0.0, 0.2),
    num_waves=4,
    border_width=0.25,
  )
  defaults.update(overrides)
  return terrain_gen.HfWaveTerrainCfg(**defaults)


@terrain_preset
def discrete_obstacles(**overrides) -> terrain_gen.HfDiscreteObstaclesTerrainCfg:
  defaults: dict[str, Any] = dict(
    obstacle_width_range=(0.3, 1.0),
    obstacle_height_range=(0.05, 0.3),
    num_obstacles=40,
    border_width=0.25,
  )
  defaults.update(overrides)
  return terrain_gen.HfDiscreteObstaclesTerrainCfg(**defaults)


@terrain_preset
def perlin_terrain_smooth(**overrides) -> terrain_gen.HfPerlinNoiseTerrainCfg:
  defaults: dict[str, Any] = dict(
    height_range=(0.0, 1.0),
    octaves=4,
    persistence=0.2,
    lacunarity=1.0,
    scale=5.0,
    horizontal_scale=0.1,
    resolution=0.1,
    base_thickness_ratio=1.0,
    border_width=0.25,
  )
  defaults.update(overrides)
  return terrain_gen.HfPerlinNoiseTerrainCfg(**defaults)


@terrain_preset
def perlin_terrain_rough(**overrides) -> terrain_gen.HfPerlinNoiseTerrainCfg:
  defaults: dict[str, Any] = dict(
    height_range=(0.0, 1.0),
    octaves=6,
    persistence=0.3,
    lacunarity=4.0,
    scale=10.0,
    horizontal_scale=0.1,
    resolution=0.1,
    base_thickness_ratio=1.0,
    border_width=0.25,
  )
  defaults.update(overrides)
  return terrain_gen.HfPerlinNoiseTerrainCfg(**defaults)


@terrain_preset
def box_random_grid(**overrides) -> terrain_gen.BoxRandomGridTerrainCfg:
  defaults: dict[str, Any] = dict(
    grid_width=0.4,
    grid_height_range=(0.0, 0.3),
    platform_width=2.0,
  )
  defaults.update(overrides)
  return terrain_gen.BoxRandomGridTerrainCfg(**defaults)


@terrain_preset
def box_random_grid_large(**overrides) -> terrain_gen.BoxRandomGridTerrainCfg:
  defaults: dict[str, Any] = dict(
    grid_width=0.8,
    grid_height_range=(0.0, 0.3),
    platform_width=2.0,
    holes=False,
    merge_similar_heights=False,
    height_merge_threshold=0.10,
    max_merge_distance=3,
  )
  defaults.update(overrides)
  return terrain_gen.BoxRandomGridTerrainCfg(**defaults)


@terrain_preset
def random_spread_boxes(**overrides) -> terrain_gen.BoxRandomSpreadTerrainCfg:
  defaults: dict[str, Any] = dict(
    num_boxes=80,
    box_width_range=(0.1, 1.0),
    box_length_range=(0.1, 2.0),
    box_height_range=(0.05, 0.3),
    platform_width=2.0,
    border_width=0.25,
  )
  defaults.update(overrides)
  return terrain_gen.BoxRandomSpreadTerrainCfg(**defaults)


@terrain_preset
def open_stairs(**overrides) -> terrain_gen.BoxOpenStairsTerrainCfg:
  defaults: dict[str, Any] = dict(
    step_height_range=(0.1, 0.2),
    step_width_range=(0.4, 0.8),
    platform_width=2.0,
    border_width=0.25,
    inverted=False,
  )
  defaults.update(overrides)
  return terrain_gen.BoxOpenStairsTerrainCfg(**defaults)


@terrain_preset
def inverted_open_stairs(**overrides) -> terrain_gen.BoxOpenStairsTerrainCfg:
  defaults: dict[str, Any] = dict(
    step_height_range=(0.1, 0.2),
    step_width_range=(0.4, 0.8),
    platform_width=2.0,
    border_width=0.25,
    inverted=True,
  )
  defaults.update(overrides)
  return terrain_gen.BoxOpenStairsTerrainCfg(**defaults)


@terrain_preset
def random_stairs(**overrides) -> terrain_gen.BoxRandomStairsTerrainCfg:
  defaults: dict[str, Any] = dict(
    step_width=0.4,
    step_height_range=(0.1, 0.3),
    platform_width=2.0,
    border_width=0.25,
  )
  defaults.update(overrides)
  return terrain_gen.BoxRandomStairsTerrainCfg(**defaults)


@terrain_preset
def stepping_stones(**overrides) -> terrain_gen.BoxSteppingStonesTerrainCfg:
  defaults: dict[str, Any] = dict(
    stone_size_range=(0.4, 0.8),
    stone_distance_range=(0.2, 0.5),
    stone_height=0.2,
    stone_height_variation=0.1,
    stone_size_variation=0.2,
    displacement_range=0.1,
    floor_depth=2.0,
    platform_width=2.0,
    border_width=0.25,
  )
  defaults.update(overrides)
  return terrain_gen.BoxSteppingStonesTerrainCfg(**defaults)


@terrain_preset
def narrow_beams(**overrides) -> terrain_gen.BoxNarrowBeamsTerrainCfg:
  defaults: dict[str, Any] = dict(
    num_beams=12,
    beam_width_range=(0.2, 0.8),
    beam_height=0.2,
    spacing=0.8,
    platform_width=2.0,
    border_width=0.25,
    floor_depth=2.0,
  )
  defaults.update(overrides)
  return terrain_gen.BoxNarrowBeamsTerrainCfg(**defaults)


@terrain_preset
def nested_rings(**overrides) -> terrain_gen.BoxNestedRingsTerrainCfg:
  defaults: dict[str, Any] = dict(
    num_rings=8,
    ring_width_range=(0.3, 0.6),
    gap_range=(0.1, 0.4),
    height_range=(0.1, 0.4),
    platform_width=2.0,
    border_width=0.25,
    floor_depth=2.0,
  )
  defaults.update(overrides)
  return terrain_gen.BoxNestedRingsTerrainCfg(**defaults)


@terrain_preset
def tilted_grid(**overrides) -> terrain_gen.BoxTiltedGridTerrainCfg:
  defaults: dict[str, Any] = dict(
    grid_width=1.0,
    tilt_range_deg=20.0,
    height_range=0.3,
    platform_width=2.0,
    border_width=0.25,
    floor_depth=2.0,
  )
  defaults.update(overrides)
  return terrain_gen.BoxTiltedGridTerrainCfg(**defaults)


# =============================================================================
# Named terrain sets — compose from presets.
# =============================================================================

# Registry of named configs — auto-discovered by the visualizer.
NAMED_TERRAIN_CONFIGS: dict[str, TerrainGeneratorCfg] = {}

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=len(ALL_TERRAIN_PRESETS),
  sub_terrains={
    "flat": flat(proportion=0.2),
    "pyramid_stairs": pyramid_stairs(proportion=0.2),
    "pyramid_stairs_inv": pyramid_stairs_inv(proportion=0.2),
    "hf_pyramid_slope": hf_pyramid_slope(proportion=0.1),
    "hf_pyramid_slope_inv": hf_pyramid_slope_inv(proportion=0.1),
    "random_rough": random_rough(proportion=0.1),
    "wave_terrain": wave_terrain(proportion=0.1),
    "box_random_grid": box_random_grid(proportion=0.1),
    "box_random_grid_large": box_random_grid_large(proportion=0.1),
    "perlin_terrain_smooth": perlin_terrain_smooth(proportion=0.1),
    "perlin_terrain_rough": perlin_terrain_rough(proportion=0.1),
    "random_spread_boxes": random_spread_boxes(proportion=0.05),
    "open_stairs": open_stairs(proportion=0.05),
    "inverted_open_stairs": inverted_open_stairs(proportion=0.05),
    "random_stairs": random_stairs(proportion=0.05),
    "tilted_grid": tilted_grid(proportion=0.05),
  },
  add_lights=True,
)

ALL_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=16,
  sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=1.0),
    "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=1.0,
      step_height_range=(0.0, 0.2),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=1.0,
      step_height_range=(0.0, 0.2),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=1.0,
      slope_range=(0.0, 0.7),
      platform_width=2.0,
      border_width=0.25,
    ),
    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
      proportion=1.0,
      noise_range=(0.02, 0.10),
      noise_step=0.02,
      border_width=0.25,
    ),
    "wave_terrain": terrain_gen.HfWaveTerrainCfg(
      proportion=1.0,
      amplitude_range=(0.0, 0.2),
      num_waves=4,
      border_width=0.25,
    ),
    "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
      proportion=1.0,
      obstacle_width_range=(0.3, 1.0),
      obstacle_height_range=(0.05, 0.3),
      num_obstacles=40,
      border_width=0.25,
    ),
    "perlin_noise": terrain_gen.HfPerlinNoiseTerrainCfg(
      proportion=1.0,
      height_range=(0.0, 1.0),
      octaves=4,
      persistence=0.3,
      lacunarity=2.0,
      scale=10.0,
      horizontal_scale=0.1,
      border_width=0.50,
    ),
    "box_random_grid": terrain_gen.BoxRandomGridTerrainCfg(
      proportion=1.0,
      grid_width=0.4,
      grid_height_range=(0.0, 0.3),
      platform_width=1.0,
    ),
    "random_spread_boxes": terrain_gen.BoxRandomSpreadTerrainCfg(
      proportion=1.0,
      num_boxes=80,
      box_width_range=(0.1, 1.0),
      box_length_range=(0.1, 2.0),
      box_height_range=(0.05, 0.3),
      platform_width=1.0,
      border_width=0.25,
    ),
    "open_stairs": terrain_gen.BoxOpenStairsTerrainCfg(
      proportion=1.0,
      step_height_range=(0.1, 0.2),
      step_width_range=(0.4, 0.8),
      platform_width=1.0,
      border_width=0.25,
    ),
    "random_stairs": terrain_gen.BoxRandomStairsTerrainCfg(
      proportion=1.0,
      step_width=0.8,
      step_height_range=(0.1, 0.3),
      platform_width=1.0,
      border_width=0.25,
    ),
    "stepping_stones": terrain_gen.BoxSteppingStonesTerrainCfg(
      proportion=1.0,
      stone_size_range=(0.4, 0.8),
      stone_distance_range=(0.2, 0.5),
      stone_height=0.2,
      stone_height_variation=0.1,
      stone_size_variation=0.2,
      displacement_range=0.1,
      floor_depth=2.0,
      platform_width=1.0,
      border_width=0.25,
    ),
    "narrow_beams": terrain_gen.BoxNarrowBeamsTerrainCfg(
      proportion=1.0,
      num_beams=12,
      beam_width_range=(0.2, 0.8),
      beam_height=0.2,
      spacing=0.8,
      platform_width=1.0,
      border_width=0.25,
      floor_depth=2.0,
    ),
    "nested_rings": terrain_gen.BoxNestedRingsTerrainCfg(
      proportion=1.0,
      num_rings=8,
      ring_width_range=(0.3, 0.6),
      gap_range=(0.1, 0.4),
      height_range=(0.1, 0.4),
      platform_width=1.0,
      border_width=0.25,
      floor_depth=2.0,
    ),
    "tilted_grid": terrain_gen.BoxTiltedGridTerrainCfg(
      proportion=1.0,
      grid_width=1.0,
      tilt_range_deg=20.0,
      height_range=0.3,
      platform_width=1.0,
      border_width=0.25,
      floor_depth=2.0,
    ),
  },
  add_lights=True,
)

STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=10,
  curriculum=True,
  sub_terrains={
    "flat": flat(proportion=0.25),
    "easy_stairs": pyramid_stairs(
      proportion=0.35,
      step_height_range=(0.02, 0.05),
      step_width=0.40,
    ),
    "moderate_stairs": pyramid_stairs(
      proportion=0.25,
      step_height_range=(0.05, 0.08),
      step_width=0.35,
      platform_width=2.5,
      border_width=0.8,
    ),
    "challenging_stairs": pyramid_stairs(
      proportion=0.15,
      step_height_range=(0.08, 0.10),
      step_width=0.30,
      platform_width=2.0,
      border_width=0.5,
    ),
  },
  add_lights=True,
)

ALL_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=len(ALL_TERRAIN_PRESETS),
  sub_terrains={name: fn(proportion=1.0) for name, fn in ALL_TERRAIN_PRESETS.items()},
  add_lights=True,
)

NAMED_TERRAIN_CONFIGS["All Terrains"] = ALL_TERRAINS_CFG
NAMED_TERRAIN_CONFIGS["Rough Terrains"] = ROUGH_TERRAINS_CFG
NAMED_TERRAIN_CONFIGS["Stairs Terrains"] = STAIRS_TERRAINS_CFG


if __name__ == "__main__":
  import mujoco.viewer
  import torch

  device = "cuda" if torch.cuda.is_available() else "cpu"

  terrain_cfg = TerrainImporterCfg(
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
  )
  terrain = TerrainImporter(terrain_cfg, device=device)
  mujoco.viewer.launch(terrain.spec.compile())
