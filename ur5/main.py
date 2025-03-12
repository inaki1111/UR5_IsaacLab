import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="training.")
parser.add_argument("--num_envs", type=int, default=9, help="number of enviroments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import math
import torch
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets import Articulation
from ur5_cfg import UR5_CFG

def design_scene() -> tuple[dict, list[float]]:
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    dome_light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    dome_light_cfg.func("/World/Light", dome_light_cfg)
    prim_utils.create_prim("/World/Robot", "Xform", translation=[0.0, 0.0, 0.0])
    robot = Articulation(cfg=UR5_CFG)
    scene_entities = {"robot": robot}
    origins = [0.0, 0.0, 0.0]
    return scene_entities, origins

def run_simulation(sim: sim_utils.SimulationContext, entities: dict, origins: torch.Tensor):
    robot = entities["robot"]
    sim_dt = sim.get_physics_dt()
    count = 0
    while sim.app.is_running():
        if count % 500 == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # Se asigna la pose inicial con un pequeño ruido (error mínimo)
            desired_joint_pos = robot.data.default_joint_pos.clone() + torch.randn_like(robot.data.default_joint_pos) * 0.01
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(desired_joint_pos, joint_vel)
            robot.reset()
            print("robot reseted")
        # No se aplican esfuerzos; se mantiene la pose
        efforts = torch.zeros_like(robot.data.joint_pos)
        robot.set_joint_effort_target(efforts)
        robot.write_data_to_sim()
        sim.step()
        count += 1
        robot.update(sim_dt)

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    scene_entities, origins = design_scene()
    origins_tensor = torch.tensor(origins, device=sim.device)
    sim.reset()
    print("Starting simulation...")
    run_simulation(sim, scene_entities, origins_tensor)
    simulation_app.close()

if __name__ == "__main__":
    main()
