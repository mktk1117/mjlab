"""Game controller interfaces for manual robot control."""

from typing import Literal

from mjlab.utils.controllers.base import BaseController as BaseController
from mjlab.utils.controllers.logitech_f710 import (
  LogitechF710Controller as LogitechF710Controller,
)
from mjlab.utils.controllers.ps5 import PS5Controller as PS5Controller

ControllerType = Literal["ps5", "f710"]


def create_controller(
  controller_type: ControllerType, deadzone: float = 0.15
) -> BaseController:
  """Factory function to create controller instances.

  Args:
    controller_type: Type of controller to create ("ps5" or "f710").
    deadzone: Threshold below which stick inputs are treated as zero.

  Returns:
    Controller instance of the specified type.

  Raises:
    ValueError: If controller_type is not recognized.
  """
  if controller_type == "ps5":
    return PS5Controller(deadzone=deadzone)
  elif controller_type == "f710":
    return LogitechF710Controller(deadzone=deadzone)
  else:
    raise ValueError(
      f"Unknown controller type: {controller_type}. Valid options: ps5, f710"
    )
