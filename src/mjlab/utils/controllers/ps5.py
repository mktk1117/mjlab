"""PlayStation 5 DualSense controller support."""

from __future__ import annotations

from mjlab.utils.controllers.base import BaseController


class PS5Controller(BaseController):
  """Interface to PS5 DualSense controller for robot velocity commands.

  PS5 DualSense axis mapping (via pygame):
    - Axis 0: Left stick X (left -1, right +1)
    - Axis 1: Left stick Y (up -1, down +1)
    - Axis 2: Right stick X (left -1, right +1)
    - Axis 3: Right stick Y (up -1, down +1)

  PS5 DualSense button mapping (via pygame):
    - Button 0: Cross (X)
    - Button 1: Circle
    - Button 2: Square
    - Button 3: Triangle
  """

  def get_velocity_command(self) -> tuple[float, float, float]:
    """Read controller sticks and return velocity command.

    Returns:
      Tuple of (lin_vel_x, lin_vel_y, ang_vel_z) in range [-1, 1].
      Returns (0, 0, 0) if controller is not connected.

    Stick mapping:
      - Left stick Y-axis: lin_vel_x (forward +1.0 / backward -1.0)
      - Left stick X-axis: lin_vel_y (right on stick = negative y in robot frame)
      - Right stick X-axis: ang_vel_z (turn right -1.0 / turn left +1.0)
    """
    if not self.is_connected():
      return (0.0, 0.0, 0.0)

    left_x = self._apply_deadzone(self._get_axis(0))
    left_y = self._apply_deadzone(self._get_axis(1))
    right_x = self._apply_deadzone(self._get_axis(2))

    # Invert axes: up/right on stick is negative in SDL, positive in robot frame.
    return (-left_y, -left_x, -right_x)

  def get_button_state(self, button: str) -> bool:
    """Check if a button is currently pressed (with debouncing).

    Args:
      button: Button name (cross, circle, square, triangle).

    Returns:
      True if button is pressed and debounce time has elapsed, False otherwise.
    """
    if not self.is_connected():
      return False

    button_map = {"cross": 0, "circle": 1, "square": 2, "triangle": 3}
    if button not in button_map:
      raise ValueError(f"Unknown button: {button}. Valid: {list(button_map.keys())}")
    return self._check_button_with_debounce(button_map[button], button)
