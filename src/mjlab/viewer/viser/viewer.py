"""mjlab play viewer based on Viser with simulation controls.

Adapted from an MJX visualizer by Chung Min Kim: https://github.com/chungmin99/
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import TYPE_CHECKING

import viser
from typing_extensions import override

from mjlab.sensor import CameraSensor
from mjlab.sim.sim import Simulation
from mjlab.viewer.base import BaseViewer, EnvProtocol, PolicyProtocol, VerbosityLevel
from mjlab.viewer.viser.camera_viewer import ViserCameraViewer
from mjlab.viewer.viser.scene import ViserMujocoScene
from mjlab.viewer.viser.reward_bar_panel import RewardBarPanel
from mjlab.viewer.viser.term_plotter import ViserTermPlotter

if TYPE_CHECKING:
  from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand


class ViserPlayViewer(BaseViewer):
  """Interactive Viser-based viewer with playback controls."""

  def __init__(
    self,
    env: EnvProtocol,
    policy: PolicyProtocol,
    frame_rate: float = 60.0,
    verbosity: VerbosityLevel = VerbosityLevel.SILENT,
  ) -> None:
    super().__init__(env, policy, frame_rate, verbosity)
    self._reward_plotter: ViserTermPlotter | None = None
    self._reward_bar_panel: RewardBarPanel | None = None
    self._metrics_plotter: ViserTermPlotter | None = None
    self._sim_lock = Lock()
    self._camera_viewers: list[ViserCameraViewer] = []

    # Virtual joystick state (populated in setup if velocity task).
    self._twist_command: UniformVelocityCommand | None = None
    self._joystick_enabled: viser.GuiInputHandle[bool] | None = None
    self._joystick_vx: viser.GuiInputHandle[float] | None = None
    self._joystick_vy: viser.GuiInputHandle[float] | None = None
    self._joystick_wz: viser.GuiInputHandle[float] | None = None

  @override
  def setup(self) -> None:
    """Setup the viewer resources."""
    sim = self.env.unwrapped.sim
    assert isinstance(sim, Simulation)

    self._server = viser.ViserServer(label="mjlab")
    self._threadpool = ThreadPoolExecutor(max_workers=1)
    self._counter = 0
    self._needs_update = False

    # Create ViserMujocoScene for all 3D visualization (with debug visualization enabled).
    self._scene = ViserMujocoScene.create(
      server=self._server,
      mj_model=sim.mj_model,
      num_envs=self.env.num_envs,
    )

    self._scene.env_idx = self.cfg.env_idx
    self._scene.debug_visualization_enabled = (
      True  # Enable debug visualization by default
    )

    # Create tab group.
    tabs = self._server.gui.add_tab_group()

    # Main tab with simulation controls and display settings.
    with tabs.add_tab("Controls", icon=viser.Icon.SETTINGS):
      # Status display.
      with self._server.gui.add_folder("Info"):
        self._status_html = self._server.gui.add_html("")

      # Simulation controls.
      with self._server.gui.add_folder("Simulation"):
        # Play/Pause button.
        self._pause_button = self._server.gui.add_button(
          "Play" if self._is_paused else "Pause",
          icon=viser.Icon.PLAYER_PLAY if self._is_paused else viser.Icon.PLAYER_PAUSE,
        )

        @self._pause_button.on_click
        def _(_) -> None:
          self.request_toggle_pause()
          self._needs_update = True

        # Reset button.
        reset_button = self._server.gui.add_button("Reset Environment")

        @reset_button.on_click
        def _(_) -> None:
          self.request_reset()
          self._needs_update = True

        # Speed controls.
        speed_buttons = self._server.gui.add_button_group(
          "Speed",
          options=["Slower", "Faster"],
        )

        @speed_buttons.on_click
        def _(event) -> None:
          if event.target.value == "Slower":
            self.request_speed_down()
          else:
            self.request_speed_up()

      # Camera feeds: collect all camera sensors and add to controls tab.
      camera_sensors = [
        sensor
        for sensor in self.env.unwrapped.scene.sensors.values()
        if isinstance(sensor, CameraSensor)
      ]
      if camera_sensors:
        with self._server.gui.add_folder("Camera Feeds"):
          self._camera_viewers = [
            ViserCameraViewer(self._server, sensor, sim.mj_model)
            for sensor in camera_sensors
          ]
      else:
        self._camera_viewers = []

      # Virtual joystick for velocity tasks.
      self._setup_joystick()

      # Add standard visualization options from ViserMujocoScene (Environment, Visualization, Contacts, Camera Tracking, Debug Visualization).
      self._scene.create_visualization_gui(
        camera_distance=self.cfg.distance,
        camera_azimuth=self.cfg.azimuth,
        camera_elevation=self.cfg.elevation,
      )

    self._prev_env_idx = self._scene.env_idx

    # Reward plots tab.
    if hasattr(self.env.unwrapped, "reward_manager"):
      with tabs.add_tab("Rewards", icon=viser.Icon.CHART_LINE):
        # Get reward term names and create reward plotter.
        term_names = [
          name
          for name, _ in self.env.unwrapped.reward_manager.get_active_iterable_terms(
            self._scene.env_idx
          )
        ]
        # Live bar panel (running-mean comparison).
        self._reward_bar_panel = RewardBarPanel(
          self._server,
          term_names,
          step_dt=self.env.unwrapped.step_dt,
        )
        self._reward_plotter = ViserTermPlotter(self._server, term_names, name="Reward")

    if hasattr(self.env.unwrapped, "metrics_manager"):
      term_names = [
        name
        for name, _ in self.env.unwrapped.metrics_manager.get_active_iterable_terms(
          self._scene.env_idx
        )
      ]
      if term_names:
        with tabs.add_tab("Metrics", icon=viser.Icon.CHART_BAR):
          # Get metrics term names and create metrics plotter.
          self._metrics_plotter = ViserTermPlotter(
            self._server, term_names, name="Metric"
          )

    # Groups tab (geoms and sites).
    self._scene.create_groups_gui(tabs)

  @override
  def _process_actions(self) -> None:
    """Process queued actions and sync UI state."""
    had_actions = bool(self._actions)
    super()._process_actions()
    if had_actions:
      self._sync_ui_state()

  def _sync_ui_state(self) -> None:
    """Sync UI elements to current state after action processing."""
    self._pause_button.label = "Play" if self._is_paused else "Pause"
    self._pause_button.icon = (
      viser.Icon.PLAYER_PLAY if self._is_paused else viser.Icon.PLAYER_PAUSE
    )
    self._update_status_display()

  @override
  def sync_env_to_viewer(self) -> None:
    """Synchronize environment state to viewer."""
    sim = self.env.unwrapped.sim
    assert isinstance(sim, Simulation)
    self._counter += 1
    if self._counter % 10 == 0:
      self._update_status_display()
      if self._scene.env_idx != self._prev_env_idx:
        self._prev_env_idx = self._scene.env_idx
        if self._reward_plotter:
          self._reward_plotter.clear_histories()
        if self._reward_bar_panel:
          self._reward_bar_panel.clear_histories()
        if self._metrics_plotter:
          self._metrics_plotter.clear_histories()
        # Clear debug visualizations when switching environments
        if self._scene.debug_visualization_enabled:
          self._scene.clear_debug_all()

      if (self._reward_plotter is not None or self._reward_bar_panel is not None) and not self._is_paused:
        terms = list(
          self.env.unwrapped.reward_manager.get_active_iterable_terms(
            self._scene.env_idx
          )
        )
        if self._reward_plotter is not None:
          self._reward_plotter.update(terms)
        if self._reward_bar_panel is not None:
          self._reward_bar_panel.update(terms)

      if self._metrics_plotter is not None and not self._is_paused:
        terms = list(
          self.env.unwrapped.metrics_manager.get_active_iterable_terms(
            self._scene.env_idx
          )
        )
        self._metrics_plotter.update(terms)

    # Update camera images
    if self._camera_viewers and (not self._is_paused or self._needs_update):
      for camera_viewer in self._camera_viewers:
        camera_viewer.update(sim.data, self._scene.env_idx)

    # Update debug visualizations if enabled
    if self._scene.debug_visualization_enabled and hasattr(
      self.env.unwrapped, "update_visualizers"
    ):
      self._scene.clear()  # Clear queued arrows from previous frame
      self.env.unwrapped.update_visualizers(self._scene)

    if self._counter % 2 != 0:
      return
    if self._is_paused and not self._needs_update and not self._scene.needs_update:
      return

    def update_scene() -> None:
      with self._sim_lock:
        with self._server.atomic():
          self._scene.update(sim.data)
          self._server.flush()

    self._threadpool.submit(update_scene)
    self._needs_update = False
    self._scene.needs_update = False

  @override
  def sync_viewer_to_env(self) -> None:
    """Synchronize viewer state to environment (e.g., perturbations)."""
    pass

  # ── Virtual joystick ──────────────────────────────────────────────

  def _setup_joystick(self) -> None:
    """Add virtual joystick controls if the task has a velocity command."""
    from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

    env = self.env.unwrapped
    if not hasattr(env, "command_manager"):
      return
    if "twist" not in env.command_manager.active_terms:
      return
    twist = env.command_manager.get_term("twist")
    if not isinstance(twist, UniformVelocityCommand):
      return

    self._twist_command = twist
    ranges = twist.cfg.ranges

    with self._server.gui.add_folder("Joystick"):
      self._joystick_enabled = self._server.gui.add_checkbox("Enable", initial_value=False)

      # Max velocity controls (adjusts slider ranges).
      max_vx = self._server.gui.add_number("Max lin_vel_x", initial_value=ranges.lin_vel_x[1], step=0.1, min=0.1, max=10.0)
      max_vy = self._server.gui.add_number("Max lin_vel_y", initial_value=ranges.lin_vel_y[1], step=0.1, min=0.1, max=10.0)
      max_wz = self._server.gui.add_number("Max ang_vel_z", initial_value=ranges.ang_vel_z[1], step=0.1, min=0.1, max=10.0)

      self._joystick_vx = self._server.gui.add_slider(
        "lin_vel_x", min=-ranges.lin_vel_x[1], max=ranges.lin_vel_x[1],
        step=0.05, initial_value=0.0,
      )
      self._joystick_vy = self._server.gui.add_slider(
        "lin_vel_y", min=-ranges.lin_vel_y[1], max=ranges.lin_vel_y[1],
        step=0.05, initial_value=0.0,
      )
      self._joystick_wz = self._server.gui.add_slider(
        "ang_vel_z", min=-ranges.ang_vel_z[1], max=ranges.ang_vel_z[1],
        step=0.05, initial_value=0.0,
      )

      @max_vx.on_update
      def _(_) -> None:
        self._joystick_vx.min = -max_vx.value
        self._joystick_vx.max = max_vx.value

      @max_vy.on_update
      def _(_) -> None:
        self._joystick_vy.min = -max_vy.value
        self._joystick_vy.max = max_vy.value

      @max_wz.on_update
      def _(_) -> None:
        self._joystick_wz.min = -max_wz.value
        self._joystick_wz.max = max_wz.value

      zero_btn = self._server.gui.add_button("Zero", icon=viser.Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        self._joystick_vx.value = 0.0
        self._joystick_vy.value = 0.0
        self._joystick_wz.value = 0.0

    # Monkey-patch _update_command so joystick values are injected DURING
    # the command update flow (before observations are computed), not after.
    original_update = twist._update_command

    def _patched_update_command() -> None:
      original_update()
      if self._joystick_enabled is not None and self._joystick_enabled.value:
        idx = self._scene.env_idx
        twist.vel_command_b[idx, 0] = self._joystick_vx.value
        twist.vel_command_b[idx, 1] = self._joystick_vy.value
        twist.vel_command_b[idx, 2] = self._joystick_wz.value

    twist._update_command = _patched_update_command

  @override
  def reset_environment(self) -> None:
    """Extend BaseViewer.reset_environment to clear reward and metrics histories."""
    with self._sim_lock:
      super().reset_environment()
    if self._reward_plotter:
      self._reward_plotter.clear_histories()
    if self._reward_bar_panel:
      self._reward_bar_panel.clear_histories()
    if self._metrics_plotter:
      self._metrics_plotter.clear_histories()

  @override
  def close(self) -> None:
    """Close the viewer and cleanup resources."""
    if self._reward_plotter:
      self._reward_plotter.cleanup()
    if self._reward_bar_panel:
      self._reward_bar_panel.cleanup()
    if self._metrics_plotter:
      self._metrics_plotter.cleanup()
    for camera_viewer in self._camera_viewers:
      camera_viewer.cleanup()
    self._threadpool.shutdown(wait=True)
    self._server.stop()

  @override
  def is_running(self) -> bool:
    """Check if viewer is running."""
    return True  # Viser runs until process is killed.

  def _update_status_display(self) -> None:
    """Update the HTML status display."""
    fps_display = f"{self._smoothed_fps:.1f}" if self._smoothed_fps > 0 else "—"
    self._status_html.content = f"""
      <div style="font-size: 0.85em; line-height: 1.25; padding: 0 1em 0.5em 1em;">
        <strong>Status:</strong> {"Paused" if self._is_paused else "Running"}<br/>
        <strong>Steps:</strong> {self._step_count}<br/>
        <strong>Speed:</strong> {self._time_multiplier:.0%}<br/>
        <strong>FPS:</strong> {fps_display}
      </div>
      """
