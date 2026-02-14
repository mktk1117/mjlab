"""Tests for height_sensor.py."""

from __future__ import annotations

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import HeightSensorCfg, HeightSensorData, ObjRef
from mjlab.sensor.height_sensor import HeightSensor
from mjlab.sim.sim import Simulation, SimulationCfg


@pytest.fixture(scope="module")
def device():
  """Test device fixture."""
  return get_test_device()


def _make_scene_and_sim(
  device: str,
  xml: str,
  sensors: tuple,
  num_envs: int = 1,
  sim_cfg: SimulationCfg | None = None,
) -> tuple[Scene, Simulation]:
  """Create a scene and simulation with sensors wired up."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml))
  scene_cfg = SceneCfg(
    num_envs=num_envs,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=sensors,
  )
  scene = Scene(scene_cfg, device)
  model = scene.compile()
  if sim_cfg is None:
    sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  if scene.sensor_context is not None:
    sim.set_sensor_context(scene.sensor_context)
  return scene, sim


@pytest.fixture(scope="module")
def single_site_xml():
  """XML for a floating body with one site above a ground plane."""
  return """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"/>
        <body name="base" pos="0 0 2">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="5.0"/>
          <site name="bottom_site" pos="0 0 -0.1"/>
        </body>
      </worldbody>
    </mujoco>
  """


@pytest.fixture(scope="module")
def multi_site_xml():
  """XML with multiple sites at different heights."""
  return """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"/>
        <body name="base" pos="0 0 3">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="5.0"/>
          <site name="site_top" pos="0 0 0.5"/>
          <site name="site_mid" pos="0 0 0"/>
          <site name="site_bot" pos="0 0 -0.5"/>
        </body>
      </worldbody>
    </mujoco>
  """


@pytest.fixture(scope="module")
def no_floor_xml():
  """XML with a body but no floor geometry."""
  return """
    <mujoco>
      <worldbody>
        <body name="base" pos="0 0 2">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="5.0"/>
          <site name="bottom_site" pos="0 0 -0.1"/>
        </body>
      </worldbody>
    </mujoco>
  """


# =============================================================================
# Basic Height Measurement
# =============================================================================


class TestBasicHeight:
  """Test basic height measurement above flat ground."""

  def test_single_site_height(self, single_site_xml, device):
    """Body at z=2, site at z=-0.1 offset → site z=1.9, height ≈ 1.9."""
    sensor_cfg = HeightSensorCfg(
      name="height_test",
      sites=(ObjRef(type="site", name="bottom_site", entity="robot"),),
    )
    scene, sim = _make_scene_and_sim(device, single_site_xml, (sensor_cfg,))
    sim.forward()
    sim.sense()

    sensor: HeightSensor = scene["height_test"]
    data: HeightSensorData = sensor.data

    assert data.heights.shape == (1, 1)
    assert data.normals_w.shape == (1, 1, 3)
    assert data.hit_pos_w.shape == (1, 1, 3)
    assert data.site_pos_w.shape == (1, 1, 3)

    # Site is at z=1.9 (body at z=2, site offset z=-0.1), floor at z=0.
    height = data.heights[0, 0].item()
    assert height == pytest.approx(1.9, abs=0.05), (
      f"Expected height ≈ 1.9, got {height}"
    )

  def test_height_is_positive_above_ground(self, single_site_xml, device):
    """Height should be positive when site is above ground."""
    sensor_cfg = HeightSensorCfg(
      name="height_test",
      sites=(ObjRef(type="site", name="bottom_site", entity="robot"),),
    )
    scene, sim = _make_scene_and_sim(device, single_site_xml, (sensor_cfg,))
    sim.forward()
    sim.sense()

    sensor: HeightSensor = scene["height_test"]
    assert (sensor.data.heights > 0).all()


class TestMultipleSites:
  """Test height measurement with multiple sites."""

  def test_multiple_site_heights(self, multi_site_xml, device):
    """Sites at different offsets should have different heights."""
    sensor_cfg = HeightSensorCfg(
      name="height_test",
      sites=(
        ObjRef(type="site", name="site_top", entity="robot"),
        ObjRef(type="site", name="site_mid", entity="robot"),
        ObjRef(type="site", name="site_bot", entity="robot"),
      ),
    )
    scene, sim = _make_scene_and_sim(device, multi_site_xml, (sensor_cfg,))
    sim.forward()
    sim.sense()

    sensor: HeightSensor = scene["height_test"]
    data = sensor.data

    assert data.heights.shape == (1, 3)

    # Body at z=3. Sites at z=3.5, z=3.0, z=2.5.
    h_top = data.heights[0, 0].item()
    h_mid = data.heights[0, 1].item()
    h_bot = data.heights[0, 2].item()

    assert h_top == pytest.approx(3.5, abs=0.05)
    assert h_mid == pytest.approx(3.0, abs=0.05)
    assert h_bot == pytest.approx(2.5, abs=0.05)

    # Top site should have highest height.
    assert h_top > h_mid > h_bot

  def test_relative_height_differences(self, multi_site_xml, device):
    """Height differences between sites should match offsets."""
    sensor_cfg = HeightSensorCfg(
      name="height_test",
      sites=(
        ObjRef(type="site", name="site_top", entity="robot"),
        ObjRef(type="site", name="site_bot", entity="robot"),
      ),
    )
    scene, sim = _make_scene_and_sim(device, multi_site_xml, (sensor_cfg,))
    sim.forward()
    sim.sense()

    sensor: HeightSensor = scene["height_test"]
    h_top = sensor.data.heights[0, 0].item()
    h_bot = sensor.data.heights[0, 1].item()

    # Difference should be 1.0 (0.5 - (-0.5)).
    assert (h_top - h_bot) == pytest.approx(1.0, abs=0.05)


# =============================================================================
# Normals
# =============================================================================


class TestNormals:
  """Test surface normal output on flat ground."""

  def test_flat_ground_normals(self, single_site_xml, device):
    """Normals on flat ground should be (0, 0, 1)."""
    sensor_cfg = HeightSensorCfg(
      name="height_test",
      sites=(ObjRef(type="site", name="bottom_site", entity="robot"),),
    )
    scene, sim = _make_scene_and_sim(device, single_site_xml, (sensor_cfg,))
    sim.forward()
    sim.sense()

    sensor: HeightSensor = scene["height_test"]
    normals = sensor.data.normals_w[0, 0]

    assert normals[0].item() == pytest.approx(0.0, abs=0.01)
    assert normals[1].item() == pytest.approx(0.0, abs=0.01)
    assert normals[2].item() == pytest.approx(1.0, abs=0.01)


# =============================================================================
# Miss Handling
# =============================================================================


class TestMissHandling:
  """Test behavior when no ground is hit."""

  def test_upward_ray_returns_negative(self, single_site_xml, device):
    """Shooting upward (away from floor) should result in a miss (height=-1)."""
    sensor_cfg = HeightSensorCfg(
      name="height_test",
      sites=(ObjRef(type="site", name="bottom_site", entity="robot"),),
      direction=(0.0, 0.0, 1.0),  # Shoot UP — will miss the floor.
    )
    scene, sim = _make_scene_and_sim(device, single_site_xml, (sensor_cfg,))
    sim.forward()
    sim.sense()

    sensor: HeightSensor = scene["height_test"]
    assert sensor.data.heights[0, 0].item() < 0

  def test_miss_normals_are_zero(self, single_site_xml, device):
    """Normals should be zero when no surface is hit."""
    sensor_cfg = HeightSensorCfg(
      name="height_test",
      sites=(ObjRef(type="site", name="bottom_site", entity="robot"),),
      direction=(0.0, 0.0, 1.0),  # Shoot UP to miss floor.
    )
    scene, sim = _make_scene_and_sim(device, single_site_xml, (sensor_cfg,))
    sim.forward()
    sim.sense()

    sensor: HeightSensor = scene["height_test"]
    normals = sensor.data.normals_w[0, 0]
    assert torch.allclose(normals, torch.zeros(3, device=device))


# =============================================================================
# Multi-Environment
# =============================================================================


class TestMultiEnv:
  """Test height sensor with multiple environments."""

  def test_multi_env_shape(self, single_site_xml, device):
    """Output shapes should scale with num_envs."""
    num_envs = 4
    sensor_cfg = HeightSensorCfg(
      name="height_test",
      sites=(ObjRef(type="site", name="bottom_site", entity="robot"),),
    )
    scene, sim = _make_scene_and_sim(
      device, single_site_xml, (sensor_cfg,), num_envs=num_envs
    )
    sim.forward()
    sim.sense()

    sensor: HeightSensor = scene["height_test"]
    data = sensor.data

    assert data.heights.shape == (num_envs, 1)
    assert data.normals_w.shape == (num_envs, 1, 3)
    assert data.hit_pos_w.shape == (num_envs, 1, 3)
    assert data.site_pos_w.shape == (num_envs, 1, 3)

  def test_multi_env_heights_consistent(self, single_site_xml, device):
    """All envs should measure roughly the same height for identical setup."""
    num_envs = 4
    sensor_cfg = HeightSensorCfg(
      name="height_test",
      sites=(ObjRef(type="site", name="bottom_site", entity="robot"),),
    )
    scene, sim = _make_scene_and_sim(
      device, single_site_xml, (sensor_cfg,), num_envs=num_envs
    )
    sim.forward()
    sim.sense()

    sensor: HeightSensor = scene["height_test"]
    heights = sensor.data.heights

    # All envs should have very similar heights.
    for i in range(1, num_envs):
      assert heights[i, 0].item() == pytest.approx(
        heights[0, 0].item(), abs=0.1
      )


# =============================================================================
# Hit Positions
# =============================================================================


class TestHitPositions:
  """Test hit position correctness."""

  def test_hit_pos_on_floor(self, single_site_xml, device):
    """Hit position Z should be approximately 0 (floor level)."""
    sensor_cfg = HeightSensorCfg(
      name="height_test",
      sites=(ObjRef(type="site", name="bottom_site", entity="robot"),),
    )
    scene, sim = _make_scene_and_sim(device, single_site_xml, (sensor_cfg,))
    sim.forward()
    sim.sense()

    sensor: HeightSensor = scene["height_test"]
    hit_z = sensor.data.hit_pos_w[0, 0, 2].item()
    assert hit_z == pytest.approx(0.0, abs=0.05)


# =============================================================================
# Multi-Ray Sampling
# =============================================================================


class TestMultiRaySampling:
  """Test multi-ray sampling with ring pattern and reductions."""

  def test_sampling_output_shapes_with_reduction(self, single_site_xml, device):
    """With sampling + reduction, output should still be [B, S]."""
    sensor_cfg = HeightSensorCfg(
      name="height_test",
      sites=(ObjRef(type="site", name="bottom_site", entity="robot"),),
      sampling=HeightSensorCfg.SamplingCfg(
        radius=0.1,
        num_samples=8,
        include_center=True,
      ),
      reduction="mean",
    )
    scene, sim = _make_scene_and_sim(device, single_site_xml, (sensor_cfg,))
    sim.forward()
    sim.sense()

    sensor: HeightSensor = scene["height_test"]
    data = sensor.data

    # K=9 (8 ring + 1 center), but after reduction → [B, S].
    assert sensor.num_samples_per_site == 9
    assert data.heights.shape == (1, 1)
    assert data.normals_w.shape == (1, 1, 3)
    assert data.hit_pos_w.shape == (1, 1, 3)
    assert data.site_pos_w.shape == (1, 1, 3)

  def test_sampling_output_shapes_no_reduction(self, single_site_xml, device):
    """With reduction='none', output should be [B, S, K]."""
    sensor_cfg = HeightSensorCfg(
      name="height_test",
      sites=(ObjRef(type="site", name="bottom_site", entity="robot"),),
      sampling=HeightSensorCfg.SamplingCfg(
        radius=0.1,
        num_samples=4,
        include_center=True,
      ),
      reduction="none",
    )
    scene, sim = _make_scene_and_sim(device, single_site_xml, (sensor_cfg,))
    sim.forward()
    sim.sense()

    sensor: HeightSensor = scene["height_test"]
    data = sensor.data

    K = 5  # 4 ring + 1 center.
    assert sensor.num_samples_per_site == K
    assert data.heights.shape == (1, 1, K)
    assert data.normals_w.shape == (1, 1, K, 3)

  def test_sampling_no_center(self, single_site_xml, device):
    """With include_center=False, K = num_samples (no center)."""
    sensor_cfg = HeightSensorCfg(
      name="height_test",
      sites=(ObjRef(type="site", name="bottom_site", entity="robot"),),
      sampling=HeightSensorCfg.SamplingCfg(
        radius=0.1,
        num_samples=6,
        include_center=False,
      ),
      reduction="mean",
    )
    scene, sim = _make_scene_and_sim(device, single_site_xml, (sensor_cfg,))
    sim.forward()
    sim.sense()

    sensor: HeightSensor = scene["height_test"]
    assert sensor.num_samples_per_site == 6
    assert sensor.data.heights.shape == (1, 1)

  def test_mean_reduction_on_flat_ground(self, single_site_xml, device):
    """Mean reduction on flat ground ≈ single-ray result."""
    # Single ray.
    single_cfg = HeightSensorCfg(
      name="single",
      sites=(ObjRef(type="site", name="bottom_site", entity="robot"),),
    )
    # Multi-ray with mean.
    multi_cfg = HeightSensorCfg(
      name="multi",
      sites=(ObjRef(type="site", name="bottom_site", entity="robot"),),
      sampling=HeightSensorCfg.SamplingCfg(
        radius=0.3,
        num_samples=8,
      ),
      reduction="mean",
    )
    scene, sim = _make_scene_and_sim(
      device, single_site_xml, (single_cfg, multi_cfg)
    )
    sim.forward()
    sim.sense()

    single_h = scene["single"].data.heights[0, 0].item()
    multi_h = scene["multi"].data.heights[0, 0].item()

    # On flat ground, mean of ring samples ≈ center sample.
    assert multi_h == pytest.approx(single_h, abs=0.1)

  def test_min_reduction_all_hit(self, single_site_xml, device):
    """Min reduction on flat ground: min height ≈ single ray (all same z)."""
    sensor_cfg = HeightSensorCfg(
      name="height_test",
      sites=(ObjRef(type="site", name="bottom_site", entity="robot"),),
      sampling=HeightSensorCfg.SamplingCfg(radius=0.1, num_samples=4),
      reduction="min",
    )
    scene, sim = _make_scene_and_sim(device, single_site_xml, (sensor_cfg,))
    sim.forward()
    sim.sense()

    sensor: HeightSensor = scene["height_test"]
    height = sensor.data.heights[0, 0].item()
    assert height > 0  # Site is above ground.
    assert height == pytest.approx(1.9, abs=0.1)

  def test_max_reduction_all_hit(self, single_site_xml, device):
    """Max reduction on flat ground: max height ≈ single ray (all same z)."""
    sensor_cfg = HeightSensorCfg(
      name="height_test",
      sites=(ObjRef(type="site", name="bottom_site", entity="robot"),),
      sampling=HeightSensorCfg.SamplingCfg(radius=0.1, num_samples=4),
      reduction="max",
    )
    scene, sim = _make_scene_and_sim(device, single_site_xml, (sensor_cfg,))
    sim.forward()
    sim.sense()

    sensor: HeightSensor = scene["height_test"]
    height = sensor.data.heights[0, 0].item()
    assert height > 0
    assert height == pytest.approx(1.9, abs=0.1)

  def test_median_reduction_all_hit(self, single_site_xml, device):
    """Median reduction on flat ground: median height ≈ single ray."""
    sensor_cfg = HeightSensorCfg(
      name="height_test",
      sites=(ObjRef(type="site", name="bottom_site", entity="robot"),),
      sampling=HeightSensorCfg.SamplingCfg(radius=0.1, num_samples=4),
      reduction="median",
    )
    scene, sim = _make_scene_and_sim(device, single_site_xml, (sensor_cfg,))
    sim.forward()
    sim.sense()

    sensor: HeightSensor = scene["height_test"]
    height = sensor.data.heights[0, 0].item()
    assert height > 0
    assert height == pytest.approx(1.9, abs=0.1)

  def test_backward_compat_no_sampling(self, single_site_xml, device):
    """Without sampling config, behavior is identical to original single-ray."""
    sensor_cfg = HeightSensorCfg(
      name="height_test",
      sites=(ObjRef(type="site", name="bottom_site", entity="robot"),),
    )
    scene, sim = _make_scene_and_sim(device, single_site_xml, (sensor_cfg,))
    sim.forward()
    sim.sense()

    sensor: HeightSensor = scene["height_test"]
    assert sensor.num_samples_per_site == 1
    assert sensor.data.heights.shape == (1, 1)
    assert sensor.data.heights[0, 0].item() == pytest.approx(1.9, abs=0.05)

  def test_mean_normals_on_flat_ground(self, single_site_xml, device):
    """Mean-reduced normals on flat ground should still be (0, 0, 1)."""
    sensor_cfg = HeightSensorCfg(
      name="height_test",
      sites=(ObjRef(type="site", name="bottom_site", entity="robot"),),
      sampling=HeightSensorCfg.SamplingCfg(radius=0.2, num_samples=8),
      reduction="mean",
    )
    scene, sim = _make_scene_and_sim(device, single_site_xml, (sensor_cfg,))
    sim.forward()
    sim.sense()

    sensor: HeightSensor = scene["height_test"]
    normals = sensor.data.normals_w[0, 0]
    assert normals[0].item() == pytest.approx(0.0, abs=0.01)
    assert normals[1].item() == pytest.approx(0.0, abs=0.01)
    assert normals[2].item() == pytest.approx(1.0, abs=0.01)


# =============================================================================
# Concentric Ring Sampling
# =============================================================================


class TestConcentricRings:
  """Test multi-ring (concentric) sampling pattern."""

  def test_concentric_ring_sample_count(self, single_site_xml, device):
    """Ring sample count = center + sum(ring.num_samples)."""
    sensor_cfg = HeightSensorCfg(
      name="height_test",
      sites=(ObjRef(type="site", name="bottom_site", entity="robot"),),
      sampling=HeightSensorCfg.SamplingCfg(
        rings=(
          HeightSensorCfg.RingCfg(radius=0.05, num_samples=4),
          HeightSensorCfg.RingCfg(radius=0.10, num_samples=6),
          HeightSensorCfg.RingCfg(radius=0.20, num_samples=8),
        ),
        include_center=True,
      ),
      reduction="none",
    )
    scene, sim = _make_scene_and_sim(device, single_site_xml, (sensor_cfg,))
    sim.forward()
    sim.sense()

    sensor: HeightSensor = scene["height_test"]
    # 1 center + 4 + 6 + 8 = 19.
    assert sensor.num_samples_per_site == 19
    assert sensor.data.heights.shape == (1, 1, 19)

  def test_concentric_ring_no_center(self, single_site_xml, device):
    """Without center, count = sum(ring.num_samples)."""
    sensor_cfg = HeightSensorCfg(
      name="height_test",
      sites=(ObjRef(type="site", name="bottom_site", entity="robot"),),
      sampling=HeightSensorCfg.SamplingCfg(
        rings=(
          HeightSensorCfg.RingCfg(radius=0.1, num_samples=4),
          HeightSensorCfg.RingCfg(radius=0.2, num_samples=6),
        ),
        include_center=False,
      ),
      reduction="none",
    )
    scene, sim = _make_scene_and_sim(device, single_site_xml, (sensor_cfg,))
    sim.forward()
    sim.sense()

    sensor: HeightSensor = scene["height_test"]
    assert sensor.num_samples_per_site == 10
    assert sensor.data.heights.shape == (1, 1, 10)

  def test_concentric_ring_with_reduction(self, single_site_xml, device):
    """Concentric rings + mean reduction → [B, S] output."""
    sensor_cfg = HeightSensorCfg(
      name="height_test",
      sites=(ObjRef(type="site", name="bottom_site", entity="robot"),),
      sampling=HeightSensorCfg.SamplingCfg(
        rings=(
          HeightSensorCfg.RingCfg(radius=0.05, num_samples=4),
          HeightSensorCfg.RingCfg(radius=0.10, num_samples=6),
        ),
      ),
      reduction="mean",
    )
    scene, sim = _make_scene_and_sim(device, single_site_xml, (sensor_cfg,))
    sim.forward()
    sim.sense()

    sensor: HeightSensor = scene["height_test"]
    assert sensor.num_samples_per_site == 11  # 1 center + 4 + 6
    # After reduction: [B, S].
    assert sensor.data.heights.shape == (1, 1)
    assert sensor.data.heights[0, 0].item() == pytest.approx(1.9, abs=0.1)

  def test_five_ring_full_pattern(self, single_site_xml, device):
    """5-ring pattern (like docstring example) produces correct sample count."""
    sensor_cfg = HeightSensorCfg(
      name="height_test",
      sites=(ObjRef(type="site", name="bottom_site", entity="robot"),),
      sampling=HeightSensorCfg.SamplingCfg(
        rings=(
          HeightSensorCfg.RingCfg(radius=0.05, num_samples=4),
          HeightSensorCfg.RingCfg(radius=0.10, num_samples=6),
          HeightSensorCfg.RingCfg(radius=0.20, num_samples=8),
          HeightSensorCfg.RingCfg(radius=0.35, num_samples=10),
          HeightSensorCfg.RingCfg(radius=0.50, num_samples=12),
        ),
      ),
      reduction="none",
    )
    scene, sim = _make_scene_and_sim(device, single_site_xml, (sensor_cfg,))
    sim.forward()
    sim.sense()

    sensor: HeightSensor = scene["height_test"]
    # 1 center + 4 + 6 + 8 + 10 + 12 = 41.
    K = 41
    assert sensor.num_samples_per_site == K
    assert sensor.data.heights.shape == (1, 1, K)

    # All rays should hit flat ground — heights > 0.
    h = sensor.data.heights[0, 0]
    assert (h > 0).all()
    # All heights should be similar on flat ground.
    assert h.std().item() < 0.1

  def test_concentric_ring_flat_heights_consistent(
    self, single_site_xml, device
  ):
    """On flat ground, all ring rays should measure ~same height."""
    sensor_cfg = HeightSensorCfg(
      name="height_test",
      sites=(ObjRef(type="site", name="bottom_site", entity="robot"),),
      sampling=HeightSensorCfg.SamplingCfg(
        rings=(
          HeightSensorCfg.RingCfg(radius=0.1, num_samples=4),
          HeightSensorCfg.RingCfg(radius=0.3, num_samples=8),
        ),
      ),
      reduction="none",
    )
    scene, sim = _make_scene_and_sim(device, single_site_xml, (sensor_cfg,))
    sim.forward()
    sim.sense()

    sensor: HeightSensor = scene["height_test"]
    h = sensor.data.heights[0, 0]  # [K]
    # All heights ≈ 1.9 on flat ground.
    for i in range(h.shape[0]):
      assert h[i].item() == pytest.approx(1.9, abs=0.1)

