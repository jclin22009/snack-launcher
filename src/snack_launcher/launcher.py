"""Servo control for the snack launcher mechanism."""

from __future__ import annotations

from collections.abc import Callable
from time import sleep as _sleep
from typing import Protocol


PCA9685_ADDRESS = 0x40
SERVO_CHANNEL = 0
ELEVATION_SERVO_CHANNEL = 1
SINGLE_SHOT_THROTTLE = -0.35
SINGLE_SHOT_DURATION_S = 3.36
ZERO_SERVO_ANGLE_DEG = 0
VERT_ANGLE_MIN_DEG = 0
VERT_ANGLE_MAX_DEG = 90
ELEVATION_SERVO_ACTUATION_RANGE_DEG = 100
POSITIONAL_SERVO_MIN_PULSE_US = 1000
POSITIONAL_SERVO_MAX_PULSE_US = 2000


class _ContinuousServo(Protocol):
    throttle: float | None


class _PositionalServo(Protocol):
    angle: float | None
    actuation_range: int

    def set_pulse_width_range(self, min_pulse: int, max_pulse: int) -> None: ...


class _ServoKit(Protocol):
    continuous_servo: list[_ContinuousServo]
    servo: list[_PositionalServo]


def create_servo_kit() -> _ServoKit:
    """Create the PCA9685-backed ServoKit used by the launcher."""
    from adafruit_servokit import ServoKit

    return ServoKit(channels=16, address=PCA9685_ADDRESS)


def single_shot(
    kit: _ServoKit | None = None,
    *,
    sleep: Callable[[float], None] = _sleep,
) -> None:
    """Advance the continuous servo one calibrated launcher revolution.

    The timing is calibrated for the current mechanism: channel 0 at -0.35
    throttle for 3.36 seconds produces one 360-degree turn.
    """
    if kit is None:
        kit = create_servo_kit()

    servo = kit.continuous_servo[SERVO_CHANNEL]
    try:
        servo.throttle = SINGLE_SHOT_THROTTLE
        sleep(SINGLE_SHOT_DURATION_S)
    finally:
        servo.throttle = None


def zero_servo(kit: _ServoKit | None = None) -> None:
    """Move the channel-1 elevation servo to its calibrated zero position."""
    set_vert_angle(ZERO_SERVO_ANGLE_DEG, kit)


def set_vert_angle(deg: float, kit: _ServoKit | None = None) -> None:
    """Set the channel-1 elevation servo to a calibrated 0–90° angle."""
    if not VERT_ANGLE_MIN_DEG <= deg <= VERT_ANGLE_MAX_DEG:
        raise ValueError(
            f"angle must be between {VERT_ANGLE_MIN_DEG} and "
            f"{VERT_ANGLE_MAX_DEG} degrees"
        )
    if kit is None:
        kit = create_servo_kit()

    _elevation_servo(kit).angle = deg


def set_elevation_angle(angle_deg: float, kit: _ServoKit | None = None) -> None:
    """Compatibility alias for :func:`set_vert_angle`."""
    set_vert_angle(angle_deg, kit)


def _elevation_servo(kit: _ServoKit) -> _PositionalServo:
    """Return channel 1 with the SV-1260MG's calibrated 100° travel."""
    servo = kit.servo[ELEVATION_SERVO_CHANNEL]
    servo.set_pulse_width_range(
        POSITIONAL_SERVO_MIN_PULSE_US, POSITIONAL_SERVO_MAX_PULSE_US
    )
    servo.actuation_range = ELEVATION_SERVO_ACTUATION_RANGE_DEG
    return servo
