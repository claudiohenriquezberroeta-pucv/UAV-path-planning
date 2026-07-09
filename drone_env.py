"""Gymnasium environment wrapping the Webots Mavic 2 PRO supervisor.
"""
import math
import random

import numpy as np
#Gymnasium: Estándar API para el aprendizaje por refuerzo con una colección diversa de entornos de referencia.
import gymnasium as gym
from gymnasium import spaces

import config as C
from flight_controller import FlightController

# obstacle centres in the stage-2 world (must match indoor_with_obstacles.wbt)
STAGE2_OBSTACLES = [(6, 6), (-7, 5), (8, -8), (-8, -7), (0, 10), (-2, -2)]
OBSTACLE_CLEARANCE = 2.5


def _wrap(angle):
    """Wrap to (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class MavicPathPlanningEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, supervisor, stage=1, reward_model=2, continuous=False):
        super().__init__()
        self.sup = supervisor
        self.timestep = int(supervisor.getBasicTimeStep())
        self.stage = stage
        self.reward_model = reward_model
        self.continuous = continuous     # True for TD3 (continuous actions)

        self.fc = FlightController(supervisor, self.timestep)
        self.lidar = supervisor.getDevice("lidar")
        self.lidar.enable(self.timestep)
        self.n_rays = self.lidar.getHorizontalResolution()

        self.self_node = supervisor.getSelf()
        self.trans_field = self.self_node.getField("translation")
        self.rot_field = self.self_node.getField("rotation")

        # optional floor markers for start (blue) and goal (green) visualisation
        self.start_marker = supervisor.getFromDef("START_MARKER")
        self.goal_marker = supervisor.getFromDef("GOAL_MARKER")

        # candidate start/goal coordinates ("red squares" of Fig. 1)
        self.coords = self._build_coordinate_set()

        # --- spaces ---------------------------------------------------- #
        if self.continuous:
            # Box([forward_speed, yaw_rate]) in [-1, 1]^2 for TD3.
            self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(
                len(C.FORWARD_SPEEDS) * len(C.YAW_RATES))      # 15 (PPO/paper)
        max_dist = math.hypot(C.ARENA_SIZE, C.ARENA_SIZE)
        low = np.array([-1.0, 0.0] + [0.0] * self.n_rays, dtype=np.float32)
        high = np.array([1.0, 1.0] + [1.0] * self.n_rays, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self._max_dist = max_dist

        self.goal = (0.0, 0.0)
        self.prev_dist = 0.0
        self.steps = 0
        self.difficulty = 0.0 if C.CURRICULUM else 1.0

    def set_difficulty(self, value):
        """Called by the training callback to make the task harder over time."""
        self.difficulty = float(max(0.0, min(1.0, value)))
        return self.difficulty

    def _sample_start_goal(self):
        """Difficulty-scaled start.

        difficulty 0 -> nearby goal, drone already facing it (easy);
        difficulty 1 -> far goal, fully random initial heading (paper task).
        """
        lo, hi = -C.HALF + C.MARGIN, C.HALF - C.MARGIN
        d = self.difficulty
        dist_cap = C.CURR_START_DIST + d * (C.CURR_MAX_DIST - C.CURR_START_DIST)
        heading_spread = C.CURR_START_HEADING + d * (math.pi - C.CURR_START_HEADING)

        for _ in range(200):
            sx, sy = random.uniform(lo, hi), random.uniform(lo, hi)
            if self.stage == 2 and not self._is_free(sx, sy):
                continue
            dist = random.uniform(C.CURR_START_DIST, max(C.CURR_START_DIST, dist_cap))
            ang = random.uniform(-math.pi, math.pi)
            gx, gy = sx + dist * math.cos(ang), sy + dist * math.sin(ang)
            if not (lo <= gx <= hi and lo <= gy <= hi):
                continue
            if self.stage == 2 and not self._is_free(gx, gy):
                continue
            goal_dir = math.atan2(gy - sy, gx - sx)
            yaw = _wrap(goal_dir + random.uniform(-heading_spread, heading_spread))
            return (sx, sy), (gx, gy), yaw
        # fallback: any two grid points
        start, goal = random.sample(self.coords, 2)
        yaw = random.uniform(-math.pi, math.pi)
        return start, goal, yaw

    # ------------------------------------------------------------------ #
    # setup helpers                                                      #
    # ------------------------------------------------------------------ #
    def _build_coordinate_set(self):
        lo, hi = -C.HALF + C.MARGIN, C.HALF - C.MARGIN
        pts, x = [], lo
        while x <= hi + 1e-6:
            y = lo
            while y <= hi + 1e-6:
                if self.stage == 1 or self._is_free(x, y):
                    pts.append((x, y))
                y += C.COORD_GRID_STEP
            x += C.COORD_GRID_STEP
        return pts

    @staticmethod
    def _is_free(x, y):
        for ox, oy in STAGE2_OBSTACLES:
            if math.hypot(x - ox, y - oy) < OBSTACLE_CLEARANCE:
                return False
        return True

    def _decode(self, action):
        if self.continuous:
            # action in [-1, 1]^2 -> (forward speed in [0, MAX], yaw in [-MAX, MAX])
            a = np.clip(np.asarray(action, dtype=np.float32).ravel(), -1.0, 1.0)
            speed = (float(a[0]) + 1.0) * 0.5 * C.MAX_FORWARD_SPEED
            yaw_rate = float(a[1]) * C.MAX_YAW_RATE
            return speed, yaw_rate
        si, yi = divmod(int(action), len(C.YAW_RATES))
        return C.FORWARD_SPEEDS[si], C.YAW_RATES[yi]

    # ------------------------------------------------------------------ #
    # observation & reward                                               #
    # ------------------------------------------------------------------ #
    def _lidar_ranges(self):
        out = np.array(self.lidar.getRangeImage(), dtype=np.float32)
        np.nan_to_num(out, copy=False, posinf=15.0, neginf=15.0, nan=15.0)
        return np.clip(out, 0.0, 15.0)

    def _heading_error(self):
        x, y = self.fc.position()
        goal_angle = math.atan2(self.goal[1] - y, self.goal[0] - x)
        return _wrap(goal_angle - self.fc.yaw())

    def _distance(self):
        x, y = self.fc.position()
        return math.hypot(self.goal[0] - x, self.goal[1] - y)

    def _observation(self, lidar):
        heading = self._heading_error()
        dist = self._distance()
        obs = np.empty(2 + self.n_rays, dtype=np.float32)
        obs[0] = heading / math.pi                 # normalised to [-1, 1]
        obs[1] = min(dist / self._max_dist, 1.0)   # normalised to [0, 1]
        obs[2:] = lidar / 15.0                      # normalised to [0, 1]
        return obs

    def _progress_heading(self, heading, linear_speed):
        """Reward term of Section 3.3 (differs between reward models)."""
        h = abs(heading)
        if h < C.HEADING_THRESHOLD:
            return linear_speed * 5.0
        coef = C.HEADING_SPEED_COEF[self.reward_model]
        linear_fn = (45.0 / 17.0) * (h / math.pi - 1.0 / 18.0)
        return linear_fn * (-(1.0 + coef * linear_speed))

    # ------------------------------------------------------------------ #
    # gym API                                                            #
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        start, goal, yaw = self._sample_start_goal()
        self.goal = goal

        self.trans_field.setSFVec3f([start[0], start[1], C.TARGET_ALTITUDE])
        self.rot_field.setSFRotation([0, 0, 1, yaw])
        # Force linear+angular velocity to zero. resetPhysics() alone does not
        # always clear residual spin/velocity, which can leave the drone in a
        # corrupted state that never recovers (persistent crashing).
        self.self_node.setVelocity([0.0] * 6)
        self.self_node.resetPhysics()
        # global physics reset: a per-node resetPhysics does not clear a NaN that
        # has propagated into the ODE solver; this does, so the sim recovers
        # instead of collapsing into permanent spawn-time crashes.
        self.sup.simulationResetPhysics()

        # move the visual markers to the current start / goal
        if self.start_marker is not None:
            self.start_marker.getField("translation").setSFVec3f(
                [start[0], start[1], 0.1])
        if self.goal_marker is not None:
            self.goal_marker.getField("translation").setSFVec3f(
                [goal[0], goal[1], 0.1])

        # Settle until the drone hovers stably (near target altitude, low spin)
        # instead of a fixed count -- guarantees every episode starts clean.
        for i in range(80):
            self.fc.hover()
            if self.sup.step(self.timestep) == -1:
                break
            if i >= 20:
                alt_ok = abs(self.fc.altitude() - C.TARGET_ALTITUDE) < 0.3
                spin_ok = abs(self.fc.yaw_rate()) < 0.1
                if alt_ok and spin_ok:
                    break

        self.steps = 0
        self.prev_dist = self._distance()
        obs = self._observation(self._lidar_ranges())
        return obs, {}

    def step(self, action):
        speed, yaw_rate = self._decode(action)

        # grace period: the spawn point is verified obstacle-free, so any
        # "collision" on the very first step is a Lidar transient, not a real
        # hit -> ignore it to avoid spurious spawn-time crashes.
        check_collision = self.steps >= 1
        collided = False
        tumbled = False
        for _ in range(C.ACTION_REPEAT):
            self.fc.apply(speed, yaw_rate)
            if self.sup.step(self.timestep) == -1:
                break
            # catch a tumble the instant it starts, before the coarse-timestep
            # integration diverges to NaN and corrupts the physics engine.
            roll, pitch = self.fc.roll_pitch()
            if not (math.isfinite(roll) and math.isfinite(pitch)) or \
               abs(roll) > C.TILT_LIMIT or abs(pitch) > C.TILT_LIMIT:
                tumbled = True
                break
            if check_collision and float(np.min(self._lidar_ranges())) < C.COLLISION_RANGE:
                collided = True
                break

        self.steps += 1
        lidar = self._lidar_ranges()
        x, y = self.fc.position()
        # physics-glitch guard: if the pose became non-finite, end the episode
        # cleanly as a failure instead of propagating NaN into obs/reward/PPO.
        if not (math.isfinite(x) and math.isfinite(y)):
            self.self_node.resetPhysics()
            safe_obs = np.zeros(2 + self.n_rays, dtype=np.float32)
            safe_obs[2:] = 1.0                      # lidar all clear (far)
            info = {"is_success": False, "collision": True}
            return safe_obs, float(C.R_FAIL), True, False, info

        # tumble detected mid-loop -> end the episode as a crash before the
        # physics diverges.
        if tumbled:
            info = {"is_success": False, "collision": True}
            return self._observation(lidar), float(C.R_FAIL), True, False, info
        heading = self._heading_error()
        dist = self._distance()
        out_of_bounds = abs(x) > C.HALF or abs(y) > C.HALF

        # --- reward (Section 3.3) ------------------------------------- #
        shaped = C.R_TIME                                        # time penalty
        shaped += C.PROGRESS_DIST_GAIN * (self.prev_dist - dist)  # progress-dist
        shaped += self._progress_heading(heading, speed)        # progress-heading
        self.prev_dist = dist
        # guard against physics-glitch spikes (e.g. a controller blow-up making
        # the position jump): clamp the per-step shaping so one bad step can't
        # poison the PPO value function.
        reward = float(np.clip(shaped, -60.0, 60.0))

        terminated = False
        success = dist < C.GOAL_RADIUS
        if success:
            reward += C.R_SUCCESS
            terminated = True
        elif collided or out_of_bounds:
            reward += C.R_FAIL
            terminated = True

        truncated = self.steps >= C.MAX_STEPS
        info = {"is_success": success, "collision": collided or out_of_bounds}
        obs = self._observation(lidar)
        return obs, float(reward), terminated, truncated, info
