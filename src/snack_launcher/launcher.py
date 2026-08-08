"""Servo control for the snack launcher mechanism."""

from __future__ import annotations

from collections.abc import Callable
from time import sleep as _sleep
from typing import Protocol


PCA9685_ADDRESS = 0x40
FIRING_SERVO_CHANNEL = 0
TILT_SERVO_CHANNEL = 1
PAN_SERVO_CHANNEL = 2
SINGLE_SHOT_THROTTLE = -0.35
SINGLE_SHOT_DURATION_S = 3.36
SERVO_HOME_ANGLE_DEG = 0
TILT_ANGLE_MIN_DEG = 0
TILT_ANGLE_MAX_DEG = 90
TILT_SERVO_ACTUATION_RANGE_DEG = 100
PAN_ANGLE_MIN_DEG = -50
PAN_ANGLE_MAX_DEG = 50
PAN_CENTER_SERVO_ANGLE_DEG = 50
PAN_SERVO_ACTUATION_RANGE_DEG = 100
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

    servo = kit.continuous_servo[FIRING_SERVO_CHANNEL]
    try:
        servo.throttle = SINGLE_SHOT_THROTTLE
        sleep(SINGLE_SHOT_DURATION_S)
    finally:
        servo.throttle = None


def zero_tilt_servo(kit: _ServoKit | None = None) -> None:
    """Move the channel-1 tilt servo to its calibrated home position."""
    set_tilt_angle(SERVO_HOME_ANGLE_DEG, kit)


def zero_pan_servo(kit: _ServoKit | None = None) -> None:
    """Move the channel-2 pan servo to its centered (straight-ahead) command."""
    set_pan_angle(0, kit)


def set_pan_angle(deg: float, kit: _ServoKit | None = None) -> None:
    """Set pan angle: negative is left, zero is centered, positive is right."""
    if not PAN_ANGLE_MIN_DEG <= deg <= PAN_ANGLE_MAX_DEG:
        raise ValueError(
            f"angle must be between {PAN_ANGLE_MIN_DEG} and "
            f"{PAN_ANGLE_MAX_DEG} degrees"
        )
    if kit is None:
        kit = create_servo_kit()

    _pan_servo(kit).angle = PAN_CENTER_SERVO_ANGLE_DEG + deg


def home_gimbal(kit: _ServoKit | None = None) -> None:
    """Home pan and tilt, then stop the channel-0 firing servo."""
    if kit is None:
        kit = create_servo_kit()

    zero_tilt_servo(kit)
    zero_pan_servo(kit)
    kit.continuous_servo[FIRING_SERVO_CHANNEL].throttle = None


def set_tilt_angle(deg: float, kit: _ServoKit | None = None) -> None:
    """Set the channel-1 tilt servo to a calibrated 0–90° angle."""
    if not TILT_ANGLE_MIN_DEG <= deg <= TILT_ANGLE_MAX_DEG:
        raise ValueError(
            f"angle must be between {TILT_ANGLE_MIN_DEG} and "
            f"{TILT_ANGLE_MAX_DEG} degrees"
        )
    if kit is None:
        kit = create_servo_kit()

    _tilt_servo(kit).angle = deg


def _tilt_servo(kit: _ServoKit) -> _PositionalServo:
    """Return channel 1 with the SV-1260MG's calibrated 100° travel."""
    servo = kit.servo[TILT_SERVO_CHANNEL]
    servo.set_pulse_width_range(
        POSITIONAL_SERVO_MIN_PULSE_US, POSITIONAL_SERVO_MAX_PULSE_US
    )
    servo.actuation_range = TILT_SERVO_ACTUATION_RANGE_DEG
    return servo


def _pan_servo(kit: _ServoKit) -> _PositionalServo:
    """Return channel 2 with the SV-1260MG's calibrated 100° travel."""
    servo = kit.servo[PAN_SERVO_CHANNEL]
    servo.set_pulse_width_range(
        POSITIONAL_SERVO_MIN_PULSE_US, POSITIONAL_SERVO_MAX_PULSE_US
    )
    servo.actuation_range = PAN_SERVO_ACTUATION_RANGE_DEG
    return servo
