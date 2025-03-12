import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import os
import math
# configuration

UR5_CFG = ArticulationCfg(
    prim_path="/ur5",
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.environ['HOME'] + "/ros2_ws/src/moveit_ur_config/config/ur5/ur5.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "shoulder_pan_joint": math.radians(132.0),
            "shoulder_lift_joint": math.radians(-8.9),
            "elbow_joint": math.radians(-86.3),
            "wrist_1_joint": math.radians(-104.0),
            "wrist_2_joint": math.radians(-1.0),
            "wrist_3_joint": math.radians(33.0),
            "robotiq_85_left_knuckle_joint": math.radians(26.0),
            "robotiq_85_right_knuckle_joint": math.radians(26.0),
        }
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
            joint_names_expr=["robotiq_85_left_knuckle_joint"],
            effort_limit=100.0,
            velocity_limit=3.0,
            stiffness=0.0,
            damping=0.1,
        ),
    }
)
