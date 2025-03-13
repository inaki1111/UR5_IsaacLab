import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="This script demonstrates different single-arm manipulators."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import math
import os
import numpy as np
import torch

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# import ur5 description
from ur5_cfg import UR5_CFG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def design_scene() -> tuple[dict, torch.Tensor]:
    # ground plane
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    # light config
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # origin of the scene
    origin = [0.0, 0.0, 0.0]
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origin)

    # table 
    table_cfg = sim_utils.UsdFileCfg(
        usd_path = os.path.join(BASE_DIR, "assets","table.usd"),
    )
    table_cfg.func("/World/Origin1/Table", table_cfg, translation=(0.0, 0.0, 0.0))


    ur5_config = UR5_CFG.replace(prim_path="/World/Origin1/Robot")
    # Initial position of the robot
    ur5_config.init_state.pos = (-0.7, 0.0, 0.63)
    custom_robot = Articulation(cfg=ur5_config)

    scene_entities = {"custom_robot": custom_robot}
    origins_tensor = torch.tensor([origin], dtype=torch.float32)
    return scene_entities, origins_tensor


def run_simulator(
    sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor
):
    """main loop of the sim"""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        # reset robot state after 200 steps
        if count % 200 == 0:
            sim_time = 0.0
            count = 0
            for index, robot in enumerate(entities.values()):
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[index]
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])

                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.reset()
            print("reset robot state")

        # apply random force to the robot
        for robot in entities.values():
            joint_pos_target = robot.data.default_joint_pos + torch.randn_like(robot.data.joint_pos) * 0.1
            joint_pos_target = joint_pos_target.clamp_(
                robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
            )
            robot.set_joint_position_target(joint_pos_target)
            robot.write_data_to_sim()

        sim.step()
        sim_time += sim_dt
        count += 1

        for robot in entities.values():
            robot.update(sim_dt)


def main():
    """main simulation function"""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    scene_entities, scene_origins = design_scene()
    scene_origins = scene_origins.to(sim.device)

    sim.reset()
    print("config complete")
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
