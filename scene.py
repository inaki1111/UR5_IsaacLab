import os
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from ur5_cfg import UR5_CFG
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@configclass
class scene_config(InteractiveSceneCfg):
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

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0, 0.64], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,  
            ),
        ),
    )
    
    def __post_init__(self):
        super().__post_init__()
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.07, 0.07, 0.07)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.cube_transform = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/ur5/base",
            target_frames=[FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Object"
            )],
            debug_vis=False,
            visualizer_cfg=marker_cfg
        )


        self.ee_transform = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/ur5/base",
            target_frames=[FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/ur5/wrist_3_link",
                offset=OffsetCfg(pos=[0.0, 0.23, 0])  # offset for the gripper frame
            )],
            debug_vis=False,
            visualizer_cfg=marker_cfg
        )
#rot=[0.0, 0.0, 0.70711, 0.70711]