# Game Controller Support

This module provides interfaces for using game controllers to manually control robots in mjlab environments.

## Installation

Install the `controllers` extra to enable controller support:

```bash
uv pip install mjlab[controllers]
```

## Testing Your Controller

Use the test script to verify your controller is working correctly:

```bash
# Test PS5 controller
uv run --extra controllers python scripts/test_controller.py --controller ps5

# Test Logitech F710
uv run --extra controllers python scripts/test_controller.py --controller f710

# Custom deadzone
uv run --extra controllers python scripts/test_controller.py --controller ps5 --deadzone 0.2
```

The script displays real-time joystick input and button presses. Press Ctrl+C to exit.

## Supported Controllers

### Using the Factory Function (Recommended)

```python
from mjlab.utils.controllers import create_controller

# Create controller using factory (cleaner, avoids code duplication)
controller = create_controller("ps5", deadzone=0.15)  # or "f710"

if controller.connect():
    controller.update()  # Call every frame
    lin_x, lin_y, ang_z = controller.get_velocity_command()
```

### PlayStation 5 DualSense

```python
from mjlab.utils.controllers import PS5Controller

controller = PS5Controller(deadzone=0.15)
if controller.connect():
    controller.update()  # Call every frame
    lin_x, lin_y, ang_z = controller.get_velocity_command()

    # Check buttons (cross, circle, square, triangle)
    if controller.get_button_state("cross"):
        print("Cross button pressed!")
```

### Logitech F710 Gamepad

**IMPORTANT**: The F710 has a physical switch on the back:
- **macOS/Linux**: Set switch to **"D"** (DirectInput/HID mode)
- **Windows**: Set switch to **"X"** (XInput mode)

```python
from mjlab.utils.controllers import LogitechF710Controller

controller = LogitechF710Controller(deadzone=0.15)
if controller.connect():
    controller.update()  # Call every frame
    lin_x, lin_y, ang_z = controller.get_velocity_command()

    # Check buttons (a, b, x, y)
    if controller.get_button_state("a"):
        print("A button pressed!")
```

## Velocity Command Mapping

Both controllers use the same stick-to-velocity mapping:

- **Left stick Y-axis**: `lin_vel_x` (forward/backward)
  - Up = +1.0 (forward)
  - Down = -1.0 (backward)

- **Left stick X-axis**: `lin_vel_y` (strafe left/right)
  - Left = +1.0 (left in robot frame)
  - Right = -1.0 (right in robot frame)

- **Right stick X-axis**: `ang_vel_z` (turn left/right)
  - Left = +1.0 (turn left)
  - Right = -1.0 (turn right)

All values are normalized to the range `[-1, 1]` with deadzone filtering applied.
