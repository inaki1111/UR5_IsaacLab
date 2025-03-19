import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="UR5 with Differential IK Controller")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn")
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
from controller import apply_diff_ik_controller
from controller import control_gripper

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    
    robot_entity_cfg = SceneEntityCfg(
        "ur5",
        joint_names=[
            "shoulder_pan_joint", 
            "shoulder_lift_joint", 
            "elbow_joint", 
            "wrist_1_joint", 
            "wrist_2_joint", 
            "wrist_3_joint"
        ],
        body_names=["wrist_3_link"]
    )
    robot_entity_cfg.resolve(scene)
    
    ee_goals = [
        [0.5,  0.0, 0.5, 0.0, 0.0, 0.0, 1.0],  # (x,y,z,qz,qy,qz,qw)
        [0.8, -0.2, 0.55, 0.0, 0.0, 0.0, 1.0],  
        [0.4,  0.4, 0.65, 0.0, 0.0, 0.0, 1.0],  
        [0.6, -0.3, 0.7, 0.0, 0.0, 0.0, 1.0],   
        [0.3, -0.1, 0.45, 0.0, 0.0, 0.0, 1.0]  
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    current_goal_idx = 0
    goal = ee_goals[current_goal_idx]
    
    count = 0
    while simulation_app.is_running():
        ur5 = scene["ur5"]
        if count % 150 == 0:
            count = 0
            joint_pos = ur5.data.default_joint_pos.clone()
            joint_vel = ur5.data.default_joint_vel.clone()
            ur5.write_joint_state_to_sim(joint_pos, joint_vel)
            ur5.reset()
            diff_ik_controller.reset()
            print("Cambio de goal")
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
            goal = ee_goals[current_goal_idx]
        else:
            # Llamada a la función que aplica el controlador diferencial IK con el goal actual
            apply_diff_ik_controller(ur5, diff_ik_controller, robot_entity_cfg, goal)
            
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        count += 1
        ur5.update(sim_dt)

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    scene_cfg = scene_config(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()
