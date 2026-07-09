"""Low-level flight controller for the DJI Mavic 2 PRO.

The paper's action space commands a forward *linear velocity* and a *yaw rate*
while altitude is held constant (2-D motion). Gazebo provided that velocity
interface directly; in Webots we must synthesise it from the 4 propellers.

This module wraps the PID stabilisation from Webots' stock `mavic2pro`
controller and adds two outer loops:
  * a P velocity loop  (forward-speed error  -> pitch disturbance)
  * a P yaw-rate loop  (yaw-rate error       -> yaw disturbance)
Roll is always driven to zero, so the drone never strafes.
"""
import math
import config as C


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


class FlightController:
    def __init__(self, robot, timestep):
        self.robot = robot
        self.timestep = timestep

        self.imu = robot.getDevice("inertial unit")
        self.gps = robot.getDevice("gps")
        self.gyro = robot.getDevice("gyro")
        self.compass = robot.getDevice("compass")
        self.imu.enable(timestep)
        self.gps.enable(timestep)
        self.gyro.enable(timestep)
        self.compass.enable(timestep)

        names = ["front left propeller", "front right propeller",
                 "rear left propeller", "rear right propeller"]
        self.motors = [robot.getDevice(n) for n in names]
        for m in self.motors:
            m.setPosition(float("inf"))
            m.setVelocity(1.0)


        # Prime the sensors: right after enable() they return NaN until the
        # simulation has stepped at least once. Step a few times (motors idle)
        # so the first apply() reads valid IMU/GPS/gyro data instead of NaN.
        for _ in range(4):
            if robot.step(timestep) == -1:
                break

    # ------------------------------------------------------------------ #
    # sensor helpers                                                     #
    # ------------------------------------------------------------------ #
    def position(self):
        x, y, _ = self.gps.getValues()
        return x, y

    def altitude(self):
        return self.gps.getValues()[2]

    def yaw(self):
        return self.imu.getRollPitchYaw()[2]

    def roll_pitch(self):
        rpy = self.imu.getRollPitchYaw()
        return rpy[0], rpy[1]

    def forward_speed(self):
        """Horizontal velocity projected onto the current heading (m/s)."""
        vx, vy, _ = self.gps.getSpeedVector()
        yaw = self.yaw()
        return vx * math.cos(yaw) + vy * math.sin(yaw)

    def yaw_rate(self):
        return self.gyro.getValues()[2]

    # ------------------------------------------------------------------ #
    # actuation                                                          #
    # ------------------------------------------------------------------ #
    def apply(self, target_speed, target_yaw_rate):
        """Run ONE control tick toward the given high-level command."""
        roll, pitch, _ = self.imu.getRollPitchYaw()
        roll_v, pitch_v, _ = self.gyro.getValues()
        altitude = self.altitude()
        fwd_speed = self.forward_speed()
        yaw_rate = self.yaw_rate()

        # Defensive guard: if any sensor is still NaN, hold a neutral hover
        # thrust instead of pushing NaN into the motors.
        if any(not math.isfinite(v) for v in
               (roll, pitch, roll_v, pitch_v, altitude, fwd_speed, yaw_rate)):
            t = C.K_VERTICAL_THRUST
            for mot, o in zip(self.motors, [t, -t, -t, t]):
                mot.setVelocity(o)
            return

        # outer loops: velocity -> pitch, yaw-rate -> yaw
        speed_err = target_speed - fwd_speed
        pitch_disturbance = _clamp(-C.PITCH_PER_MPS * speed_err, -2.5, 2.5)
        yaw_err = target_yaw_rate - yaw_rate
        yaw_disturbance = _clamp(C.YAW_RATE_GAIN * yaw_err, -1.5, 1.5)
        roll_disturbance = 0.0  # never strafe

        # Clamp the gyro rates that feed the P terms: when the drone briefly
        # tumbles they can spike and make the motor command diverge to
        # thousands (physics blow-up). Bounding them keeps control stable.
        roll_v = _clamp(roll_v, -3.0, 3.0)
        pitch_v = _clamp(pitch_v, -3.0, 3.0)

        # inner PID (identical to the stock Webots Mavic controller)
        roll_input = C.K_ROLL_P * _clamp(roll, -1.0, 1.0) + roll_v + roll_disturbance
        pitch_input = C.K_PITCH_P * _clamp(pitch, -1.0, 1.0) + pitch_v + pitch_disturbance
        yaw_input = yaw_disturbance
        diff_alt = _clamp(C.TARGET_ALTITUDE - altitude + C.K_VERTICAL_OFFSET, -1.0, 1.0)
        vertical_input = C.K_VERTICAL_P * diff_alt ** 3.0

        base = C.K_VERTICAL_THRUST + vertical_input
        m = C.MAX_MOTOR_VELOCITY
        fl = _clamp(base - roll_input + pitch_input - yaw_input, 0.0, m)
        fr = _clamp(base + roll_input + pitch_input + yaw_input, 0.0, m)
        rl = _clamp(base - roll_input - pitch_input + yaw_input, 0.0, m)
        rr = _clamp(base + roll_input - pitch_input - yaw_input, 0.0, m)
        outs = [fl, -fr, -rl, rr]
        # final bullet-proof guard: never pass NaN/inf to a motor, whatever the
        # physics did this step.
        if any(not math.isfinite(o) for o in outs):
            t = C.K_VERTICAL_THRUST
            outs = [t, -t, -t, t]
        for mot, o in zip(self.motors, outs):
            mot.setVelocity(o)

    def hover(self):
        """Hold position (zero forward speed, zero yaw rate)."""
        self.apply(0.0, 0.0)
