import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Training.")
parser.add_argument("--num_envs", type=int, default = 1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import math
import os
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg, CameraCfg
import isaaclab.envs.mdp as mdp
from isaaclab.assets.articulation import ArticulationCfg

# =============================================================================
# arm configuration
# =============================================================================

MANIPULATOR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        #usd_path = os.environ['HOME'] + "/Downloads/moveit2_UR5/ur5_moveit.usd",
        usd_path=os.environ['HOME'] + "/ros2_ws/src/moveit_ur_config/ur5_isaacsim.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=10.0,
            max_angular_velocity=10.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pos of the robot
        pos=(-0.7, 0.0, 0.63),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "shoulder_pan_joint": math.radians(132.0),
            "shoulder_lift_joint": math.radians(-8.9),
            "elbow_joint": math.radians(-86.3),
            "wrist_1_joint": math.radians(-104.0),
            "wrist_2_joint": math.radians(-1.0),
            "wrist_3_joint": math.radians(33.0),
            "robotiq_85_left_knuckle_joint": math.radians(26.0),
            "robotiq_85_right_knuckle_joint": math.radians(26.0),
        },

    ),
    actuators={
        "arm_actuator": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            effort_limit=87.0,
            velocity_limit=100.0,
            stiffness=800.0,
            damping=40.0,
        ),
                "gripper_actuator": ImplicitActuatorCfg(
            joint_names_expr=[
                "robotiq_85_left_knuckle_joint",
            ],
            effort_limit=100.0,
            velocity_limit=3.0,  
            stiffness=0.0,
            damping=0.1,
        ),

    },
)


@configclass
class ManipulatorSceneCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.environ['HOME'] + "/ros2_ws/src/moveit_ur_config/table.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )


    # load the robot
    robot: ArticulationCfg = MANIPULATOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # camera
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ur5/wrist_3_link/camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
        focal_length=11.2, focus_distance=2.2, horizontal_aperture=6.4, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.14729, -0.01019, -0.02615), rot=(0.5, 0.5, -0.5, 0.5), convention="ros"),
)



    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )


@configclass
class ActionsCfg:
    #define velocity action for the joints
    joint_velocities = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[
            "elbow_joint",
            "robotiq_85_left_knuckle_joint",
            #"robotiq_85_right_knuckle_joint",
            "shoulder_lift_joint",
            "shoulder_pan_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        scale=1.0,
    )


@configclass
class DummyCfg:
    pass

@configclass
class ManipulatorEnvCfg(ManagerBasedRLEnvCfg):
    scene: ManipulatorSceneCfg = ManipulatorSceneCfg(num_envs=1, env_spacing=4.0)
    observations: DummyCfg = DummyCfg()
    actions: ActionsCfg = ActionsCfg()
    events: DummyCfg = DummyCfg()
    rewards: DummyCfg = DummyCfg()
    terminations: DummyCfg = DummyCfg()
    commands: DummyCfg = DummyCfg()
    curriculum: DummyCfg = DummyCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.sim.dt = 0.005  # Timestep 200 Hz
        self.episode_length_s = 10
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.render_interval = self.decimation




def main():

    env_cfg = ManipulatorEnvCfg()
    
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    count = 0
    while simulation_app.is_running():
        """single_action = torch.tensor([10.0, 10.0, 10.0, -1.0, -1.0, -1.0, -1.0])
        joint_vel = single_action.unsqueeze(0).repeat(args_cli.num_envs, 1)
        obs, rew, terminated, truncated, info = env.step(joint_vel)
        if count != 0 and count % 300 == 0:
            #env.reset()
            count = 0
            
        count += 1"""
        no_op_action = torch.zeros(7)
        joint_vel = no_op_action.unsqueeze(0).repeat(args_cli.num_envs, 1)
        obs, rew, terminated, truncated, info = env.step(joint_vel)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
