
"""
algoritmos          ppo | td3             (default: config.ALGO)
Stage 1 :worlds/indoor_no_obstacles.wbt,
Stage 2 :worlds/indoor_with_obstacles.wbt.
"""
import argparse
import sys


try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass

from controller import Supervisor

import config as C
import train as trainer


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--reward-model", type=int, choices=[1, 2], default=2)
    p.add_argument("--stage", type=int, choices=[1, 2], default=1)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--algo", choices=["ppo", "td3"], default=C.ALGO.lower())
    return p.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    supervisor = Supervisor()

    print(f"UAV curriculum controller: mode={args.mode} algo={args.algo} "
          f"stage={args.stage} reward_model={args.reward_model}")

    if args.mode == "train":
        trainer.train(supervisor, args.stage, args.reward_model, algo=args.algo)
    else:
        trainer.evaluate(supervisor, args.stage, args.reward_model,
                         args.episodes, algo=args.algo)

    print("controller finished.")


if __name__ == "__main__":
    main()
