import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

class UR5DifferentialIK:
    def __init__(self, scene, device):
        self.scene = scene
        self.device = device

        # Configuración del controlador IK
        self.diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls"
        )
        self.diff_ik_controller = DifferentialIKController(self.diff_ik_cfg, num_envs=scene.num_envs, device=device)

        # Configuración de las articulaciones del UR5
        self.robot_entity_cfg = SceneEntityCfg(
            "ur5",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint"
            ],
            body_names=["wrist_3_link"]
        )
        self.robot_entity_cfg.resolve(scene)
        self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1  # Índice del Jacobiano

    def compute_ik(self, ur5, ee_goals):
        """Calcula y aplica la cinemática inversa para el UR5"""
        jacobian = ur5.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
        ee_pose_w = ur5.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = ur5.data.root_state_w[:, 0:7]
        joint_pos = ur5.data.joint_pos[:, self.robot_entity_cfg.joint_ids]

        # Transformar la pose del efector final al sistema de referencia base
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7],
            ee_pose_w[:, :3], ee_pose_w[:, 3:7]
        )

        # Calcular nueva posición de articulaciones
        joint_pos_des = self.diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        return joint_pos_des

    def reset(self):
        """Resetea el controlador"""
        self.diff_ik_controller.reset()
