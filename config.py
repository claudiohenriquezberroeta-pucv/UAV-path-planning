import math

# --------------------------------------------------------------------------- #
# Environment geometry                                                        #
# --------------------------------------------------------------------------- #
ARENA_SIZE = 30.0            # 30 m x 30 m indoor room
HALF = ARENA_SIZE / 2.0
MARGIN = 2.0                 # keep start/goal away from the walls
TARGET_ALTITUDE = 1.5        # fixed flight height -> motion stays 2-D
GOAL_RADIUS = 1.0            # distance at which the goal counts as reached
COLLISION_RANGE = 0.6        # Lidar range below which we declare a collision
TILT_LIMIT = 1.0             # rad (~57 deg): end the episode if the drone
                             # tilts past this, BEFORE the physics diverges to NaN
                             # (Lidar minRange is 0.5 so the drone's own body,
                             #  <0.5 m, is ignored -> no spurious self-collisions)

# The UAV randomly picks two of these as start and goal every episode. 
COORD_GRID_STEP = 4.0        # spacing of the candidate-coordinate grid

# --------------------------------------------------------------------------- #
# Difficulty curriculum 
# --------------------------------------------------------------------------- #

CURRICULUM = True
CURR_START_DIST = 3.0        # min start->goal distance at difficulty 0 (m)
CURR_MAX_DIST = 22.0         # start->goal distance cap at difficulty 1 (m)
CURR_START_HEADING = 0.30    # initial heading spread at difficulty 0 (rad, ~17 deg)
CURR_WINDOW = 80             # sliding episodes used to measure the promotion rate
CURR_PROMOTE_RATE = 0.60     # promote only when robustly good (avoids promoting on
                             # a lucky peak that then collapses at the next level)
CURR_STEP = 0.10             # difficulty increment per promotion

# --------------------------------------------------------------------------- #
# Action space  -> Discrete(15) = 3 forward speeds x 5 yaw rates              #
# --------------------------------------------------------------------------- #
FORWARD_SPEEDS = [1.0, 0.5, 0.0]                       # m/s   
YAW_RATES = [-0.25, -0.12, 0.0, 0.12, 0.25]            # rad/s 

# --------------------------------------------------------------------------- #
# Reward model                                                                #
# --------------------------------------------------------------------------- #
R_SUCCESS = 2000.0           # terminal reward, task success
R_FAIL = -500.0              # terminal reward, collision
R_TIME = -1.0                # time penalty, every step
PROGRESS_DIST_GAIN = 40.0    # progress-distance multiplier
HEADING_THRESHOLD = math.radians(20.0)   # 20 deg boundary of progress-heading
# progress-heading factor for |heading| > 20 deg differs between the two models:
#   model 1 -> -(1 + 1 * linear_speed)
#   model 2 -> -(1 + 3 * linear_speed)
HEADING_SPEED_COEF = {1: 1.0, 2: 3.0}

# --------------------------------------------------------------------------- #
# Episode / step timing                                                       #
# --------------------------------------------------------------------------- #
ACTION_REPEAT = 8            # [PORT] low-level control ticks held per RL step
                             # 8 * 16 ms = 128 ms per decision (basicTimeStep now 16)
MAX_STEPS = 300              # [PORT] episode truncation; enough steps to turn and
                             # cross the room (300 * 128 ms = 38 s at up to 1 m/s)

# --------------------------------------------------------------------------- #
# PPO hyper-parameters                                                        #
# --------------------------------------------------------------------------- #

LEARNING_RATE = 1.5e-4       # lowered for curriculum stability (was 3e-4;
                             # 3e-4 caused the policy to collapse after promotions)
GAE_LAMBDA = 0.95            # SB3 PPO default (paper/RLlib used 1.0)
KL_TARGET = 0.2              # RLlib "kl_coeff" (adaptive KL) -> SB3 target_kl
TRAIN_BATCH_SIZE = 10000     # RLlib "train_batch_size" (= 1 iteration)
GAMMA = 0.99                 # not stated in paper; RLlib/SB3 default

# Curriculum schedule                                                         
STAGE1_ITERATIONS = 50       # demonstrative run: enough for the difficulty
                             # curriculum to climb toward 1.0 (paper used 200)
STAGE2_ITERATIONS = 50       # obstacle stage (paper used 100)
# total timesteps per stage = iterations * train_batch_size
STAGE1_TIMESTEPS = STAGE1_ITERATIONS * TRAIN_BATCH_SIZE   # 2,000,000
STAGE2_TIMESTEPS = STAGE2_ITERATIONS * TRAIN_BATCH_SIZE   # 1,000,000

# --------------------------------------------------------------------------- #
# Algorithm selection: "PPO" (paper, discrete actions) or "TD3" (variant)      #
# --------------------------------------------------------------------------- #

ALGO = "PPO"

MAX_FORWARD_SPEED = 1.0       # m/s  (paper's max forward speed)
MAX_YAW_RATE = 0.25           # rad/s (matches the widened discrete YAW_RATES)

# TD3 hyper-parameters (SB3 defaults, tuned lightly for this task).
TD3_LEARNING_RATE = 3e-4
TD3_BUFFER_SIZE = 200_000
TD3_LEARNING_STARTS = 5_000   # random exploration steps before learning
TD3_BATCH_SIZE = 256
TD3_TAU = 0.005               # target network soft-update rate
TD3_TRAIN_FREQ = 64           # env steps between gradient-update bursts
TD3_GRADIENT_STEPS = 64       # gradient updates per burst
TD3_ACTION_NOISE = 0.1        # std of Gaussian exploration noise (per action dim)
TD3_POLICY_DELAY = 2          # TD3 delayed policy/target updates

# --------------------------------------------------------------------------- #
# Low-level flight controller (from Webots' own mavic2pro example)  [PORT]    #
# --------------------------------------------------------------------------- #
K_VERTICAL_THRUST = 68.5
K_VERTICAL_OFFSET = 0.6
K_VERTICAL_P = 3.0
K_ROLL_P = 50.0
K_PITCH_P = 30.0
MAX_MOTOR_VELOCITY = 150.0   # clamp motor command (hover ~68.5) to keep the
                             # PID from diverging to thousands and blowing up physics
# High-level command -> attitude-disturbance gains (tuned for the Mavic model).
PITCH_PER_MPS = 1.6          # pitch disturbance produced per 1 m/s forward speed
YAW_RATE_GAIN = 3.2          # yaw disturbance produced per 1 rad/s yaw-rate error

# --------------------------------------------------------------------------- #
# Files                                                                       #
# --------------------------------------------------------------------------- #
MODEL_DIR = "models"
LOG_DIR = "logs"             # per-iteration goal-rate CSV + TensorBoard events
STAGE1_MODEL = "ppo_stage1_no_obstacles"
STAGE2_MODEL = "ppo_stage2_with_obstacles"
