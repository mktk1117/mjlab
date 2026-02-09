"""Base class for game controller interfaces.

macOS threading constraint
--------------------------
On macOS, SDL2/pygame requires the OS main thread for two things:

1. ``pygame.init()`` initializes the Cocoa video subsystem, which calls
   ``[NSApplication setMainMenu:]`` — crashes off the main thread.
2. ``pygame.event.pump()`` drives the Cocoa event loop via
   ``nextEventMatchingMask`` — also crashes off the main thread.

Both are needed: ``pygame.init()`` creates the Cocoa application context
that the HID manager depends on, and ``event.pump()`` triggers asynchronous
HID device enumeration (``SDL_NumJoysticks()`` returns 0 until events are
pumped). There is no combination of SDL hints (``SDL_VIDEODRIVER=dummy``,
``SDL_JOYSTICK_THREAD``, ``SDL_JOYSTICK_HIDAPI``) or direct
``SDL_JoystickUpdate()`` calls that can enumerate HID devices without the
Cocoa event loop on macOS.

Under ``mjpython``, user code runs on a secondary OS thread (the main
thread is reserved for the MuJoCo viewer). Python's
``threading.main_thread()`` is unreliable here because the interpreter
starts on the secondary thread, making it look like the "main" thread.
We use ``pthread_main_np()`` to check the actual OS main thread.

Solution: when running off the OS main thread, we spawn a child process
(via ``multiprocessing``) that has its own main thread. The child process
runs pygame normally and writes joystick state to shared memory arrays
that the parent reads lock-free.
"""

from __future__ import annotations

import ctypes
import multiprocessing
import queue
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  import pygame
else:
  try:
    import pygame
  except ImportError:
    pygame = None

_MAX_AXES = 16
_MAX_BUTTONS = 32


def _is_os_main_thread() -> bool:
  if sys.platform == "darwin":
    return ctypes.CDLL("libpthread.dylib").pthread_main_np() == 1
  return threading.current_thread() is threading.main_thread()


def _controller_worker(
  cmd_queue: multiprocessing.Queue,
  result_queue: multiprocessing.Queue,
  shared_axes: Any,
  shared_buttons: Any,
  shared_connected: Any,
) -> None:
  """Child process that reads controller input via pygame.

  Runs on its own main thread so Cocoa/SDL works normally on macOS.
  Writes joystick state to shared memory arrays.
  """
  import pygame

  pygame.init()
  pygame.joystick.init()

  joystick: pygame.joystick.JoystickType | None = None
  connected = False
  num_axes = 0
  num_buttons = 0

  while True:
    while not cmd_queue.empty():
      try:
        cmd = cmd_queue.get_nowait()
      except queue.Empty:
        break

      if cmd["type"] == "connect":
        for _ in range(20):
          pygame.event.pump()
          time.sleep(0.05)
          if pygame.joystick.get_count() > 0:
            break

        if pygame.joystick.get_count() > 0:
          joystick = pygame.joystick.Joystick(0)
          joystick.init()
          connected = True
          num_axes = min(joystick.get_numaxes(), _MAX_AXES)
          num_buttons = min(joystick.get_numbuttons(), _MAX_BUTTONS)
          shared_connected.value = 1
          result_queue.put({"success": True, "name": joystick.get_name()})
        else:
          result_queue.put({"success": False})

      elif cmd["type"] == "quit":
        if joystick is not None:
          joystick.quit()
        pygame.quit()
        return

    if connected and joystick is not None:
      pygame.event.pump()
      for i in range(num_axes):
        shared_axes[i] = joystick.get_axis(i)
      for i in range(num_buttons):
        shared_buttons[i] = joystick.get_button(i)

    time.sleep(1.0 / 120.0)


class BaseController(ABC):
  """Base interface for game controllers to provide robot velocity commands.

  Subclasses define axis indices and button mappings for specific controllers.
  See module docstring for the macOS threading design.

  Attributes:
    deadzone: Minimum absolute value for stick input to be considered non-zero.
  """

  def __init__(self, deadzone: float = 0.15):
    if not TYPE_CHECKING and pygame is None:
      raise ImportError(
        "pygame is required for controller support. "
        "Install with: uv pip install --group controllers"
      )

    self.deadzone = deadzone
    self._on_main_thread = _is_os_main_thread()
    self._last_button_time: dict[str, float] = {}
    self._button_debounce_ms = 200

    # Subprocess state (off main thread only).
    self._proc: multiprocessing.Process | None = None
    self._cmd_queue: multiprocessing.Queue | None = None
    self._shared_axes: Any = None
    self._shared_buttons: Any = None
    self._shared_connected: Any = None

    # Direct pygame state (main thread only).
    self._joystick: pygame.joystick.JoystickType | None = None
    self._lock = threading.RLock()

    if self._on_main_thread:
      pygame.init()
      pygame.joystick.init()

  def connect(self) -> bool:
    if self._on_main_thread:
      return self._connect_direct()
    return self._connect_subprocess()

  def _connect_direct(self) -> bool:
    with self._lock:
      if pygame.joystick.get_count() == 0:
        for _ in range(20):
          pygame.event.pump()
          time.sleep(0.05)
          if pygame.joystick.get_count() > 0:
            break

      if pygame.joystick.get_count() == 0:
        print("[WARNING]: No controllers detected.")
        return False

      self._joystick = pygame.joystick.Joystick(0)
      self._joystick.init()
      print(f"[INFO]: Connected to controller: {self._joystick.get_name()}")
      return True

  def _connect_subprocess(self) -> bool:
    self._cmd_queue = multiprocessing.Queue()
    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    self._shared_axes = multiprocessing.Array("d", _MAX_AXES)
    self._shared_buttons = multiprocessing.Array("b", _MAX_BUTTONS)
    self._shared_connected = multiprocessing.Value("b", 0)

    self._proc = multiprocessing.Process(
      target=_controller_worker,
      args=(
        self._cmd_queue,
        result_queue,
        self._shared_axes,
        self._shared_buttons,
        self._shared_connected,
      ),
      daemon=True,
    )
    self._proc.start()
    self._cmd_queue.put({"type": "connect"})

    try:
      result = result_queue.get(timeout=5.0)
    except Exception:
      print("[WARNING]: Controller subprocess timed out.")
      return False

    if result.get("success"):
      print(f"[INFO]: Connected to controller: {result['name']}")
      return True

    print("[WARNING]: No controllers detected.")
    return False

  def disconnect(self) -> None:
    if self._on_main_thread:
      with self._lock:
        if self._joystick is not None:
          self._joystick.quit()
          self._joystick = None
        pygame.quit()
    else:
      if self._cmd_queue is not None:
        self._cmd_queue.put({"type": "quit"})
      if self._proc is not None:
        self._proc.join(timeout=2.0)
        if self._proc.is_alive():
          self._proc.kill()
        self._proc = None

  def is_connected(self) -> bool:
    if self._on_main_thread:
      with self._lock:
        return self._joystick is not None and self._joystick.get_init()
    return (
      self._proc is not None
      and self._proc.is_alive()
      and self._shared_connected is not None
      and self._shared_connected.value == 1
    )

  def update(self) -> None:
    """Pump the event queue (main thread) or no-op (subprocess)."""
    if self._on_main_thread:
      pygame.event.pump()

  def _get_axis(self, index: int) -> float:
    if self._on_main_thread:
      assert self._joystick is not None
      return self._joystick.get_axis(index)
    if self._shared_axes is not None and index < _MAX_AXES:
      return self._shared_axes[index]
    return 0.0

  def _get_button(self, index: int) -> bool:
    if self._on_main_thread:
      assert self._joystick is not None
      return self._joystick.get_button(index) == 1
    if self._shared_buttons is not None and index < _MAX_BUTTONS:
      return self._shared_buttons[index] == 1
    return False

  @abstractmethod
  def get_velocity_command(self) -> tuple[float, float, float]:
    """Read sticks and return (lin_vel_x, lin_vel_y, ang_vel_z) in [-1, 1]."""
    pass

  @abstractmethod
  def get_button_state(self, button: str) -> bool:
    """Check if a named button is pressed (with debouncing)."""
    pass

  def _apply_deadzone(self, value: float) -> float:
    if abs(value) < self.deadzone:
      return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)

  def _check_button_with_debounce(self, button_idx: int, button_name: str) -> bool:
    is_pressed = self._get_button(button_idx)
    if is_pressed:
      now = int(time.monotonic() * 1000)
      last = self._last_button_time.get(button_name, 0)
      if now - last < self._button_debounce_ms:
        return False
      self._last_button_time[button_name] = now
    return is_pressed
