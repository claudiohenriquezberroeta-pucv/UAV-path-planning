import csv
import os
import time

import numpy as np
# Librería SB3 de implementación de RL con Pytorch
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import config as C
from drone_env import MavicPathPlanningEnv


# --------------------------------------------------------------------------- #
# algorithm helpers                                                           #
# --------------------------------------------------------------------------- #
def _algo_class(algo):
    return TD3 if algo.upper() == "TD3" else PPO


def _is_continuous(algo):
    return algo.upper() == "TD3"


def _model_base(algo, stage, reward_model):
    env = "with_obstacles" if stage == 2 else "no_obstacles"
    return f"{algo.lower()}_stage{stage}_{env}_rm{reward_model}"


def _vec_path(algo, reward_model):
    return os.path.join(C.MODEL_DIR, f"vecnormalize_{algo.lower()}_rm{reward_model}.pkl")


class GoalRateCallback(BaseCallback):
    
    def __init__(self, csv_path, best_model_path=None, best_vec_path=None,
                 window=50, heartbeat_steps=2000, verbose=1):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.best_model_path = best_model_path
        self.best_vec_path = best_vec_path
        self.best_score = float("-inf")
        self.window = window
        self.heartbeat_steps = heartbeat_steps
        self.results = []                 
        self._t0 = None
        self._next_heartbeat = heartbeat_steps
        self._next_iter_step = C.TRAIN_BATCH_SIZE
        self._iter_success = []           
        self._iter_rates = []
        self._iteration = 0
        self._csv = None
        self._writer = None
        self.difficulty = 0.0 if C.CURRICULUM else 1.0
        self._promote_window = []

    def _on_training_start(self):
        self._t0 = time.time()
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self._csv = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._csv)
        self._writer.writerow(["iteration", "timesteps", "episodes",
                               "goal_rate", "goal_rate_ma5", "difficulty"])
        self._csv.flush()
        print(f"training started: target {self.locals.get('total_timesteps', '?')} "
              f"timesteps (1 curve point = {C.TRAIN_BATCH_SIZE} steps)")
        print(f"learning curve -> {self.csv_path}")

    def _on_step(self):
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is None:
                continue
            success = 1.0 if info.get("is_success") else 0.0
            outcome = ("GOAL " if success else
                       "CRASH" if info.get("collision") else "TIMEOUT")
            self.results.append(success)
            self._iter_success.append(success)
            recent = self.results[-self.window:]
            rate = 100.0 * sum(recent) / len(recent)
            print(f"  ep {len(self.results):5d} | {outcome} | "
                  f"len={int(ep['l']):3d} | reward={ep['r']:8.1f} | "
                  f"goal-rate(last {len(recent)})={rate:5.1f}% | diff={self.difficulty:.1f}")
            self._maybe_promote(success)

        if self.num_timesteps >= self._next_heartbeat:
            self._next_heartbeat += self.heartbeat_steps
            dt = max(time.time() - self._t0, 1e-6)
            print(f"... {self.num_timesteps:7d} steps | "
                  f"{self.num_timesteps / dt:5.0f} steps/s | "
                  f"{len(self.results)} episodes done | diff={self.difficulty:.1f}")

        if self.num_timesteps >= self._next_iter_step:
            self._next_iter_step += C.TRAIN_BATCH_SIZE
            self._record_iteration()
        return True

    def _maybe_promote(self, success):
        if not C.CURRICULUM or self.difficulty >= 1.0:
            return
        self._promote_window.append(success)
        if len(self._promote_window) > C.CURR_WINDOW:
            self._promote_window.pop(0)
        if len(self._promote_window) < C.CURR_WINDOW:
            return
        rate = sum(self._promote_window) / len(self._promote_window)
        if rate >= C.CURR_PROMOTE_RATE:
            self.difficulty = min(1.0, self.difficulty + C.CURR_STEP)
            self.training_env.env_method("set_difficulty", self.difficulty)
            print(f">>> PROMOTED difficulty -> {self.difficulty:.2f} "
                  f"(sliding goal rate {100 * rate:.0f}%) <<<")
            self._promote_window = []

    def _record_iteration(self):
        self._iteration += 1
        n = len(self._iter_success)
        rate = 100.0 * sum(self._iter_success) / n if n else 0.0
        self._iter_rates.append(rate)
        ma5 = sum(self._iter_rates[-5:]) / len(self._iter_rates[-5:])
        self._writer.writerow([self._iteration, self.num_timesteps, n,
                               f"{rate:.2f}", f"{ma5:.2f}", f"{self.difficulty:.2f}"])
        self._csv.flush()
        self.logger.record("rollout/goal_rate", rate)
        self.logger.record("rollout/goal_rate_ma5", ma5)
        self.logger.record("rollout/difficulty", self.difficulty)
        print(f"== iteration {self._iteration:3d} done | episodes={n:3d} | "
              f"goal rate={rate:5.1f}% | MA5={ma5:5.1f}% | difficulty={self.difficulty:.2f} ==")
        self._iter_success = []
        self._save_if_best(ma5)

    def _save_if_best(self, ma5):
        if not self.best_model_path:
            return
        score = self.difficulty * 1000.0 + ma5
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.best_model_path)
            if self.best_vec_path:
                self.training_env.save(self.best_vec_path)
            print(f"    * new BEST saved (difficulty={self.difficulty:.2f}, "
                  f"MA5={ma5:.1f}%) -> {self.best_model_path}.zip")

    def _on_training_end(self):
        if self._csv:
            self._csv.close()


# --------------------------------------------------------------------------- #
# model                                                               #
# --------------------------------------------------------------------------- #
def _make_ppo(env, tensorboard=None):
    return PPO(
        "MlpPolicy", env,
        learning_rate=C.LEARNING_RATE, n_steps=C.TRAIN_BATCH_SIZE,
        batch_size=125, n_epochs=10, gamma=C.GAMMA, gae_lambda=C.GAE_LAMBDA,
        target_kl=C.KL_TARGET, verbose=1, tensorboard_log=tensorboard,
    )


def _make_td3(env, tensorboard=None):
    n_actions = env.action_space.shape[-1]
    noise = NormalActionNoise(mean=np.zeros(n_actions),
                              sigma=C.TD3_ACTION_NOISE * np.ones(n_actions))
    return TD3(
        "MlpPolicy", env,
        learning_rate=C.TD3_LEARNING_RATE, buffer_size=C.TD3_BUFFER_SIZE,
        learning_starts=C.TD3_LEARNING_STARTS, batch_size=C.TD3_BATCH_SIZE,
        tau=C.TD3_TAU, gamma=C.GAMMA, train_freq=C.TD3_TRAIN_FREQ,
        gradient_steps=C.TD3_GRADIENT_STEPS, action_noise=noise,
        policy_delay=C.TD3_POLICY_DELAY, verbose=1, tensorboard_log=tensorboard,
    )


def _make_model(algo, env, tensorboard=None):
    return _make_td3(env, tensorboard) if _is_continuous(algo) \
        else _make_ppo(env, tensorboard)


def _build_venv(supervisor, stage, reward_model, continuous, monitor_path=None):
    def _init():
        env = MavicPathPlanningEnv(supervisor, stage=stage,
                                   reward_model=reward_model, continuous=continuous)
        return Monitor(env, filename=monitor_path)
    return DummyVecEnv([_init])


# --------------------------------------------------------------------------- #
# train / evaluate                                                            #
# --------------------------------------------------------------------------- #
def train(supervisor, stage, reward_model, algo=None):
    algo = (algo or C.ALGO).upper()
    continuous = _is_continuous(algo)
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    os.makedirs(C.LOG_DIR, exist_ok=True)
    tag = f"stage{stage}_rm{reward_model}_{algo.lower()}"
    monitor_path = os.path.join(C.LOG_DIR, f"monitor_{tag}")
    csv_path = os.path.join(C.LOG_DIR, f"goal_rate_{tag}.csv")
    vec_path = _vec_path(algo, reward_model)

    base = _model_base(algo, stage, reward_model)
    best_model_path = os.path.join(C.MODEL_DIR, f"{base}_best")
    best_vec_path = os.path.join(C.MODEL_DIR, f"vecnormalize_{algo.lower()}_rm{reward_model}_best.pkl")

    venv = _build_venv(supervisor, stage, reward_model, continuous, monitor_path)
    if stage == 2 and os.path.exists(vec_path):
        print(f"curriculum: loading normalization stats {vec_path}")
        venv = VecNormalize.load(vec_path, venv)
        venv.training = True
        venv.norm_reward = True
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, gamma=C.GAMMA)
    cb = GoalRateCallback(csv_path=csv_path, best_model_path=best_model_path,
                          best_vec_path=best_vec_path)

    if stage == 1:
        print(f"=== STAGE 1 (no obstacles) | {algo} | reward model {reward_model} ===")
        model = _make_model(algo, venv, tensorboard=C.LOG_DIR)
        total = C.STAGE1_TIMESTEPS
    else:
        print(f"=== STAGE 2 (with obstacles) | {algo} | reward model {reward_model} ===")
        prior = os.path.join(C.MODEL_DIR, f"{_model_base(algo, 1, reward_model)}.zip")
        if os.path.exists(prior):
            print(f"curriculum transfer: loading policy {prior}")
            model = _algo_class(algo).load(prior[:-4], env=venv, tensorboard_log=C.LOG_DIR)
        else:
            print("WARNING: stage-1 model not found, training stage 2 from scratch")
            model = _make_model(algo, venv, tensorboard=C.LOG_DIR)
        total = C.STAGE2_TIMESTEPS

    model.learn(total_timesteps=total, callback=cb, tb_log_name=tag)
    path = os.path.join(C.MODEL_DIR, base)
    model.save(path)
    venv.save(vec_path)
    print(f"saved -> {path}.zip  (+ {vec_path})")


def evaluate(supervisor, stage, reward_model, episodes=100, algo=None):
    """Measure the goal rate of a trained policy (paper's headline metric)."""
    algo = (algo or C.ALGO).upper()
    continuous = _is_continuous(algo)
    base = _model_base(algo, stage, reward_model)
    best = os.path.join(C.MODEL_DIR, f"{base}_best.zip")
    final = os.path.join(C.MODEL_DIR, f"{base}.zip")
    path = best if os.path.exists(best) else final
    if not os.path.exists(path):
        print(f"no trained model at {final}")
        return
    print(f"evaluating: {path}")
    venv = _build_venv(supervisor, stage, reward_model, continuous)
    venv.env_method("set_difficulty", 1.0)    # evaluate on the full paper task
    best_vec = os.path.join(C.MODEL_DIR,
                            f"vecnormalize_{algo.lower()}_rm{reward_model}_best.pkl")
    vec_path = best_vec if os.path.exists(best_vec) else _vec_path(algo, reward_model)
    if os.path.exists(vec_path):
        venv = VecNormalize.load(vec_path, venv)
        venv.training = False
        venv.norm_reward = False
    model = _algo_class(algo).load(path[:-4])

    # PPO: stochastic (matches training measure). TD3: deterministic actor.
    det = _is_continuous(algo)
    obs = venv.reset()

    # Adaptive warm-up: discard cold-start episodes until the physics is
    # demonstrably healthy (3 consecutive real flights), capped so it can't loop.
    print("warming up until physics is stable (not counted)...")
    healthy, warm = 0, 0
    while healthy < 3 and warm < 200:
        action, _ = model.predict(obs, deterministic=det)
        obs, _, dones, infos = venv.step(action)
        if dones[0]:
            warm += 1
            length = int(infos[0].get("episode", {}).get("l", 0))
            healthy = healthy + 1 if length > 10 else 0
    print(f"warm-up done after {warm} episodes; measuring {episodes}...")

    goals = 0
    ep = 0
    while ep < episodes:
        action, _ = model.predict(obs, deterministic=det)
        obs, _, dones, infos = venv.step(action)
        if dones[0]:
            ep += 1
            info = infos[0]
            success = bool(info.get("is_success", False))
            goals += int(success)
            outcome = ("GOAL" if success else
                       "crash" if info.get("collision") else "timeout")
            length = int(info.get("episode", {}).get("l", 0))
            print(f"episode {ep:3d}: {outcome:7s} len={length}")
    print(f"\n[{algo}] Goal rate over {episodes} episodes: {100.0 * goals / episodes:.1f}%")
