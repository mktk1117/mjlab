"""Test script for game controller input.

This script connects to a controller and displays real-time input from
joysticks and buttons. Useful for verifying controller connectivity and
testing axis/button mappings.

Example:
  uv run --extra controllers python scripts/test_controller.py --controller ps5
  uv run --extra controllers python scripts/test_controller.py --controller f710
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import tyro

from mjlab.utils.controllers import ControllerType, create_controller


@dataclass
class Args:
  """Test controller input and connectivity."""

  controller: ControllerType = "ps5"
  """Controller type to test."""

  deadzone: float = 0.15
  """Stick deadzone threshold (0 to 1)."""

  update_rate_hz: float = 30.0
  """Display update rate in Hz."""


def main(args: Args) -> None:
  """Run the controller test."""
  print(f"Initializing {args.controller} controller...")
  print(f"Deadzone: {args.deadzone}")
  print()

  controller = create_controller(args.controller, deadzone=args.deadzone)
  if not controller.connect():
    print("[ERROR]: Failed to connect to controller.")
    print("Please connect a controller and try again.")
    sys.exit(1)

  print("[SUCCESS]: Controller connected!")
  print()
  print("Instructions:")
  print("  - Move sticks to see velocity commands")
  print("  - Press buttons to see button states")
  print("  - Press Ctrl+C to exit")
  print()
  print("-" * 60)

  buttons = (
    ["cross", "circle", "square", "triangle"]
    if args.controller == "ps5"
    else ["a", "b", "x", "y"]
  )

  try:
    while True:
      controller.update()
      if not controller.is_connected():
        print("\n[WARNING]: Controller disconnected!")
        break

      lin_x, lin_y, ang_z = controller.get_velocity_command()
      button_states = {btn: controller.get_button_state(btn) for btn in buttons}
      pressed_buttons = [btn for btn, pressed in button_states.items() if pressed]
      status = (
        f"Velocity: "
        f"lin_x={lin_x:+.2f} lin_y={lin_y:+.2f} ang_z={ang_z:+.2f} | "
        f"Buttons: {', '.join(pressed_buttons) if pressed_buttons else 'none'}"
      )
      print(f"\r{status:<80}", end="", flush=True)

      time.sleep(1.0 / args.update_rate_hz)

  except KeyboardInterrupt:
    print("\n\nExiting...")
  finally:
    controller.disconnect()
    print("Controller disconnected.")


if __name__ == "__main__":
  import mjlab

  main(tyro.cli(Args, config=mjlab.TYRO_FLAGS))
