import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="UR5 ")
parser.add_argument("--num_envs", type=int, default=9, help="Number of environments to spawn")
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@configclass
class MyInteractiveSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(BASE_DIR, "assets", "table.usd"))
    )
    ur5: ArticulationCfg = UR5_CFG.replace(prim_path="{ENV_REGEX_NS}/ur5")
    ur5.init_state.pos = (-0.7, 0.0, 0.63)
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ur5/wrist_3_link/camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=11.2, focus_distance=2.2, horizontal_aperture=6.4,
            clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.14729, -0.01019, -0.02615),
            rot=(0.5, 0.5, -0.5, 0.5),
            convention="ros"
        )
    )

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
        joint_pos_target = joint_pos_target.clamp_(ur5.data.soft_joint_pos_limits[..., 0], ur5.data.soft_joint_pos_limits[..., 1])
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
    #print("config complete")
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()
