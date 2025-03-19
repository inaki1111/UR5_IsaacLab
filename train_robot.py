import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="UR5 with Differential IK Controller")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn")
parser.add_argument("--train", action="store_true", help="Enable training mode")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from ur5_cfg import UR5_CFG
from scene import scene_config
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from stable_baselines3 import PPO

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    scene_cfg = scene_config(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    env = Sb3VecEnvWrapper(scene)
    
    if args_cli.train:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_ur5_log")
        model.learn(total_timesteps=100000)
        model.save("ppo_ur5_model")
    else:
        model = PPO.load("ppo_ur5_model")
        obs = env.reset()
        while simulation_app.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _ = env.step(action)
            sim.step()
    
    simulation_app.close()

if __name__ == "__main__":
    main()
