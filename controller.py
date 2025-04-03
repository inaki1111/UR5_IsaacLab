import torch
from isaaclab.utils.math import subtract_frame_transforms

# gripper controller
def control_gripper(ur5, open=True): # robot_joints #state
    gripper_joint_name = "robotiq_85_left_knuckle_joint"
    gripper_joint_index = ur5.data.joint_names.index(gripper_joint_name) #position of the gripper articulation
    pos_open = 35.0 
    pos_close = 0.0 
    target_pos = ur5.data.joint_pos.clone() 
    target_pos[:, gripper_joint_index] = pos_open if open else pos_close    
    ur5.set_joint_position_target(target_pos)

