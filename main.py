import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="training.")
parser.add_argument("--num_envs", type=int, default=9, help="number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import math
import copy
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from ur5_cfg import UR5_CFG

def design_scene(num_envs: int) -> tuple[dict, list]:
    scene_entities = {}
    origins = []
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    dome_light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    dome_light_cfg.func("/World/Light", dome_light_cfg)
    for i in range(num_envs):
        prim_path = f"/World/Robot_{i}"
        robot_cfg = copy.deepcopy(UR5_CFG)
        robot_cfg.prim_path = prim_path
        robot = Articulation(cfg=robot_cfg)
        scene_entities[f"robot_{i}"] = robot
        origins.append([i * 2.0, 0.0, 0.0])
    return scene_entities, origins

def run_simulation(sim: sim_utils.SimulationContext, entities: dict, origins: torch.Tensor):
    sim_dt = sim.get_physics_dt()
    count = 0
    while sim.app.is_running():
        if count % 500 == 0:
            count = 0
            for i, key in enumerate(sorted(entities.keys())):
                robot = entities[key]
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[i]
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                desired_joint_pos = robot.data.default_joint_pos.clone() + torch.randn_like(robot.data.default_joint_pos) * 0.01
                joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(desired_joint_pos, joint_vel)
                robot.reset()
            print("Robots reset")
        for i, key in enumerate(sorted(entities.keys())):
            robot = entities[key]
            efforts = torch.zeros_like(robot.data.joint_pos)
            robot.set_joint_effort_target(efforts)
            robot.write_data_to_sim()
            robot.update(sim_dt)
        sim.step()
        count += 1

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    scene_entities, origins = design_scene(args_cli.num_envs)
    origins_tensor = torch.tensor(origins, device=sim.device)
    sim.reset()
    print("Starting simulation...")
    run_simulation(sim, scene_entities, origins_tensor)
    simulation_app.close()

if __name__ == "__main__":
    main()
