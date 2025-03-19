import torch
from isaaclab.utils.math import subtract_frame_transforms

def apply_diff_ik_controller(ur5, diff_ik_controller, robot_entity_cfg, goal):
    # Establecemos el comando del controlador con el goal
    diff_ik_controller.set_command(goal)
    
    # Se asume que el índice del jacobiano del ee es: (body_id - 1)
    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1

    # Obtenemos el jacobiano del end-effector
    jacobian = ur5.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
    
    # Obtenemos el estado actual del ee y la raíz
    ee_pose_w = ur5.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
    root_pose_w = ur5.data.root_state_w[:, 0:7]
    joint_pos = ur5.data.joint_pos[:, robot_entity_cfg.joint_ids]
    
    # Calculamos la transformación del ee en el frame del robot
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, :3], root_pose_w[:, 3:7],
        ee_pose_w[:, :3], ee_pose_w[:, 3:7]
    )
    
    # Calculamos las nuevas posiciones de las articulaciones usando el controlador diferencial IK
    joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
    
    # Aplicamos las posiciones calculadas al robot
    ur5.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)



# gripper controller
def control_gripper(ur5, open=True): # robot_joints #state
    gripper_joint_name = "robotiq_85_left_knuckle_joint"
    gripper_joint_index = ur5.data.joint_names.index(gripper_joint_name) #position of the gripper articulation
    pos_open = 35.0 
    pos_close = 0.0 
    target_pos = ur5.data.joint_pos.clone() 
    target_pos[:, gripper_joint_index] = pos_open if open else pos_close    
    ur5.set_joint_position_target(target_pos)

