from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)
from mjlab.utils.controllers import ControllerType, create_controller

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _scale_asymmetric(value: float, limits: tuple[float, float]) -> float:
  """Scale a normalized [-1, 1] value to asymmetric limits."""
  if value > 0:
    return value * limits[1]
  return -value * limits[0]


class GamepadVelocityCommand(UniformVelocityCommand):
  """Velocity command driven by gamepad input.

  Reads input from a game controller and uses it to set velocity commands
  for a specific environment. Only the environment specified by `cfg.env_idx`
  responds to controller input. All other environments receive zero commands.
  """

  cfg: GamepadVelocityCommandCfg  # pyright: ignore[reportIncompatibleVariableOverride]

  def __init__(self, cfg: GamepadVelocityCommandCfg, env: ManagerBasedRlEnv):
    CommandTerm.__init__(self, cfg, env)

    self.robot = env.scene[cfg.entity_name]
    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    self._controller = create_controller(cfg.controller_type, deadzone=cfg.deadzone)
    self._controller_connected = self._controller.connect()
    if not self._controller_connected:
      print(
        "[WARNING]: No controller detected. Robot will remain stationary.\n"
        "          Connect a controller and restart to enable manual control."
      )

  def close(self) -> None:
    self._controller.disconnect()

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    del env_ids  # Unused.

  def _update_command(self) -> None:
    if not self._controller_connected:
      return

    self._controller.update()
    lin_x, lin_y, ang_z = self._controller.get_velocity_command()

    env_idx = self.cfg.env_idx
    if env_idx < self.num_envs:
      self.vel_command_b[env_idx, 0] = _scale_asymmetric(
        lin_x, self.cfg.ranges.lin_vel_x
      )
      self.vel_command_b[env_idx, 1] = _scale_asymmetric(
        lin_y, self.cfg.ranges.lin_vel_y
      )
      self.vel_command_b[env_idx, 2] = _scale_asymmetric(
        ang_z, self.cfg.ranges.ang_vel_z
      )


@dataclass(kw_only=True)
class GamepadVelocityCommandCfg(UniformVelocityCommandCfg):
  """Configuration for gamepad-driven velocity commands."""

  controller_type: ControllerType = "ps5"
  """Controller type to use."""

  env_idx: int = 0
  """Index of the environment to control with the controller."""

  deadzone: float = 0.15
  """Controller stick deadzone threshold (0 to 1)."""

  # Gamepad doesn't use these.
  heading_command: bool = False
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 0.0

  def build(self, env: ManagerBasedRlEnv) -> GamepadVelocityCommand:
    return GamepadVelocityCommand(self, env)
