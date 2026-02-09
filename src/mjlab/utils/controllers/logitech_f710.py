"""Logitech F710 gamepad controller support."""

from __future__ import annotations

from mjlab.utils.controllers.base import BaseController


class LogitechF710Controller(BaseController):
  """Interface to Logitech F710 gamepad for robot velocity commands.

  IMPORTANT: The F710 has a physical switch on the back that toggles between
  DirectInput (D) and XInput (X) modes.
  - On macOS/Linux: Set switch to "D" (DirectInput/HID mode)
  - On Windows: Set switch to "X" (XInput mode)

  Logitech F710 axis mapping (DirectInput mode, via pygame on macOS):
    - Axis 0: Left stick X (left -1, right +1)
    - Axis 1: Left stick Y (up -1, down +1)
    - Axis 2: Left trigger (0 to 1)
    - Axis 3: Right stick X (left -1, right +1)
    - Axis 4: Right stick Y (up -1, down +1)
    - Axis 5: Right trigger (0 to 1)

  Logitech F710 button mapping (DirectInput mode, via pygame on macOS):
    - Button 0: A (bottom)
    - Button 1: B (right)
    - Button 2: X (left)
    - Button 3: Y (top)
    - Button 4: LB (left bumper)
    - Button 5: RB (right bumper)
    - Button 6: Back
    - Button 7: Start
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
    right_x = self._apply_deadzone(self._get_axis(3))  # Axis 3 (axis 2 is left trigger)

    # Invert axes: up/right on stick is negative in SDL, positive in robot frame.
    return (-left_y, -left_x, -right_x)

  def get_button_state(self, button: str) -> bool:
    """Check if a button is currently pressed (with debouncing).

    Args:
      button: Button name (a, b, x, y).

    Returns:
      True if button is pressed and debounce time has elapsed, False otherwise.
    """
    if not self.is_connected():
      return False

    button_map = {"a": 0, "b": 1, "x": 2, "y": 3}
    if button not in button_map:
      raise ValueError(f"Unknown button: {button}. Valid: {list(button_map.keys())}")
    return self._check_button_with_debounce(button_map[button], button)
