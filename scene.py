import os
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.sensors import CameraCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from ur5_cfg import UR5_CFG

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

    # sensors
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ur5/wrist_3_link/camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.0, focus_distance=0.9, horizontal_aperture=4.96, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.1), rot=(-0.7, 0.7, 0.0, 0.0), convention="ros"),
    )    
