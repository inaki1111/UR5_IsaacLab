import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Training.")
parser.add_argument("--num_envs", type=int, default = 9, help="Number of environments to spawn.")

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


# =============================================================================
# Configuración del brazo manipulador
# =============================================================================
# Nota: Actualiza 'usd_path' con la ruta a tu USD.
MANIPULATOR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path = os.environ['HOME'] + "/ros2_ws/src/moveit_ur_config/config/ur5/ur5.usd",
        #usd_path = os.environ['HOME'] + "/Downloads/moveit2_UR5/ur5_moveit.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=10.0,
            max_angular_velocity=10.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Posición base del robot (modifícala si es necesario)
        pos=(0.0, 0.0, 0.0),
        # Quaternion identidad
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "elbow_joint": math.radians(132.0),
            "robotiq_85_left_knuckle_joint": 0.0,
            "robotiq_85_right_knuckle_joint": 0.0,
            "shoulder_lift_joint": math.radians(-8.9),
            "shoulder_pan_joint": math.radians(-86.3),
            "wrist_1_joint": math.radians(-104.0),
            "wrist_2_joint": math.radians(-1.0),
            "wrist_3_joint": math.radians(33.0),
        },
        joint_vel={
            "elbow_joint": 0.0,
            "robotiq_85_left_knuckle_joint": 0.0,
            "robotiq_85_right_knuckle_joint": 0.0,
            "shoulder_lift_joint": 0.0,
            "shoulder_pan_joint": 0.0,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm_actuator": ImplicitActuatorCfg(
            joint_names_expr=[
                "elbow_joint",
                "robotiq_85_left_knuckle_joint",
                "robotiq_85_right_knuckle_joint",
                "shoulder_lift_joint",
                "shoulder_pan_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            effort_limit=100.0,
            velocity_limit=3.0,  # Valor común; ajústalo si es necesario
            stiffness=0.0,
            damping=0.1,
        ),
    },
)

# =============================================================================
# Configuración de la escena
# =============================================================================
@configclass
class ManipulatorSceneCfg(InteractiveSceneCfg):
    # Plano del suelo
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    # Carga del robot manipulador
    robot: ArticulationCfg = MANIPULATOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Iluminación básica
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )

# =============================================================================
# Configuración mínima para acciones (dummy) y entorno
# =============================================================================
@configclass
class ActionsCfg:
    # Se define una acción de velocidad articular para las 8 articulaciones
    joint_velocities = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[
            "elbow_joint",
            "robotiq_85_left_knuckle_joint",
            "robotiq_85_right_knuckle_joint",
            "shoulder_lift_joint",
            "shoulder_pan_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        scale=1.0,
    )

# Configuraciones "dummy" para componentes no utilizados
@configclass
class DummyCfg:
    pass

@configclass
class ManipulatorEnvCfg(ManagerBasedRLEnvCfg):
    scene: ManipulatorSceneCfg = ManipulatorSceneCfg(num_envs=1, env_spacing=2.0)
    observations: DummyCfg = DummyCfg()
    actions: ActionsCfg = ActionsCfg()
    events: DummyCfg = DummyCfg()
    rewards: DummyCfg = DummyCfg()
    terminations: DummyCfg = DummyCfg()
    commands: DummyCfg = DummyCfg()
    curriculum: DummyCfg = DummyCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.sim.dt = 0.005  # Timestep de simulación: 200 Hz
        self.episode_length_s = 10
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.render_interval = self.decimation

# =============================================================================
# Función principal
# =============================================================================
def main():
    # Se crea la configuración del entorno y se actualiza el número de entornos
    env_cfg = ManipulatorEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # Se instancia el entorno
    env = ManagerBasedRLEnv(cfg=env_cfg)

    count = 0
    while simulation_app.is_running():
        if count % 300 == 0: #300
            count = 0
            env.reset()
            print("-" * 80)
            print("[INFO]: Reiniciando entorno...")
        # Se aplican velocidades nulas (dummy) para cada articulación
        #joint_vel = torch.zeros_like(env.action_manager.action)
  # Define la acción para una sola instancia (8 articulaciones)
        single_action = torch.tensor([-100.0, 0.0, 0.0, 100.0, -100.0, -100.0, -100.0, -100.0])
# Agrega una dimensión para el batch y replica para cada entorno
        joint_vel = single_action.unsqueeze(0).repeat(args_cli.num_envs, 1)
        obs, rew, terminated, truncated, info = env.step(joint_vel)
        count += 1

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
