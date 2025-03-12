import argparse
import os
import math
import torch

from isaaclab.app import AppLauncher
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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
        usd_path=os.environ['HOME'] + "/ros2_ws/src/moveit_ur_config/ur5_isaacsim.usd",
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
        # Posición base del robot 
        pos=(-0.7, 0.0, 0.63),
        # Quaternion identidad
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "elbow_joint": 0.0,
            "robotiq_85_left_knuckle_joint": 0.0,
            "robotiq_85_right_knuckle_joint": 0.0,
            "shoulder_lift_joint": 0.0,
            "shoulder_pan_joint": 0.0,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
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

    # Mesa: Agrega el asset de la mesa
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.environ['HOME'] + "/ros2_ws/src/moveit_ur_config/table.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),  # La parte superior de la mesa a 0.75 m de altura
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # Carga del robot manipulador
    robot: ArticulationCfg = MANIPULATOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # camera
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_3_joint",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=5.25, focus_distance=400.0, horizontal_aperture=6.4, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.01092, 0.00541, 0.1), rot=(0.5, -0.5, -0.5, 0.5), convention="ros"),
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
        # Aplicar la acción (dummy) en cada paso
        single_action = torch.tensor([-100.0, -100.0, 0.0, -100.0, -100.0, -100.0, -100.0, -100.0])
        joint_vel = single_action.unsqueeze(0).repeat(args_cli.num_envs, 1)
        obs, rew, terminated, truncated, info = env.step(joint_vel)
        
        # Cada 300 steps se reinicia la escena y se resetea la pose del robot
        if count % 300 == 0:
            count = 0
            # --- Reset personalizado de la escena ---
            # Se asume que la escena ya está instanciada y que el robot se encuentra
            # registrado en ella con la clave "robot"
            robot_sim = env.scene["robot"]
            # Restablecer el estado raíz (posición y orientación base)
            root_state = robot_sim.data.default_root_state.clone()
            # Se añade un offset si se están usando múltiples entornos
            root_state[:, :3] += env.cfg.scene.env_origins
            robot_sim.write_root_pose_to_sim(root_state[:, :7])
            robot_sim.write_root_velocity_to_sim(root_state[:, 7:])
            # Restablecer el estado articular, añadiendo un pequeño ruido para evitar estados idénticos
            joint_pos = robot_sim.data.default_joint_pos.clone()
            joint_vel_state = robot_sim.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot_sim.write_joint_state_to_sim(joint_pos, joint_vel_state)
            # Reiniciar los buffers internos de la escena
            env.scene.reset()
            print("[INFO]: Resetting robot state...")
        
        count += 1
        print("Step:", count)

    env.close()

if __name__ == "__main__":
    # add argparse arguments
    parser = argparse.ArgumentParser(description="Training.")
    parser.add_argument("--num_envs", type=int, default=9, help="Number of environments to spawn.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    main()
    simulation_app.close()
