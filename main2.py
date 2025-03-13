import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="UR5")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import math
import numpy as np
import torch
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import CameraCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from ur5_cfg import UR5_CFG
from scene import MyInteractiveSceneCfg

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    while simulation_app.is_running():
        ur5 = scene["ur5"]
        if count % 200 == 0:
            count = 0
            root_state = ur5.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            ur5.write_root_pose_to_sim(root_state[:, :7])
            ur5.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos = ur5.data.default_joint_pos.clone()
            joint_vel = ur5.data.default_joint_vel.clone()
            ur5.write_joint_state_to_sim(joint_pos, joint_vel)
            ur5.reset()
            print("reset ur5 state")
        joint_pos_target = ur5.data.default_joint_pos + torch.randn_like(ur5.data.joint_pos) * 0.1
        joint_pos_target = joint_pos_target.clamp_(ur5.data.soft_joint_pos_limits[..., 0],
                                                    ur5.data.soft_joint_pos_limits[..., 1])
        ur5.set_joint_position_target(joint_pos_target)
        ur5.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        sim_time += sim_dt
        count += 1
        ur5.update(sim_dt)

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    scene_cfg = MyInteractiveSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()
