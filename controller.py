import torch
from isaaclab.utils.math import subtract_frame_transforms
import math
# gripper controller
def control_gripper(ur5, open=True):
    # Nombres de las articulaciones de la pinza
    left_joint_name = "robotiq_85_left_knuckle_joint"
    right_joint_name = "robotiq_85_right_knuckle_joint"  # Ajusta si es diferente

    # Índices en el vector de posiciones
    left_index = ur5.data.joint_names.index(left_joint_name)
    right_index = ur5.data.joint_names.index(right_joint_name)

    # Valores de apertura y cierre (en grados o radianes según tu sistema)
    pos_open = 0.0
    pos_close = math.radians(35) 

    # Target para ambos lados
    left_target = pos_open if open else pos_close
    right_target = left_target  # Espejo del otro lado

    # Clonar el estado actual y modificar solo los de la pinza
    target_pos = ur5.data.joint_pos.clone()
    target_pos[:, left_index] = left_target
    target_pos[:, right_index] = right_target

    # Aplicar el nuevo objetivo
    ur5.set_joint_position_target(target_pos)
