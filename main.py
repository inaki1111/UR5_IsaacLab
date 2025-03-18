import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR5")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from ur5_cfg import UR5_CFG
from scene import MyInteractiveSceneCfg

def control_gripper(ur5, open=True):
    """
    Controla la apertura y cierre de la garra Robotiq 2F-85 modificando el target position.
    """
    gripper_joint_name = "robotiq_85_left_knuckle_joint"

    if gripper_joint_name not in ur5.data.joint_names:
        print(f"ERROR: No se encontró el joint {gripper_joint_name}")
        return

    gripper_joint_index = ur5.data.joint_names.index(gripper_joint_name)
    
    # Define valores de posición que correspondan a abierta y cerrada
    pos_open = 40.0   # Ajusta este valor según tu configuración
    pos_close = 0.0 # Ajusta este valor según tu configuración
    
    target_pos = ur5.data.joint_pos.clone()  # Copia el estado actual
    target_pos[:, gripper_joint_index] = pos_open if open else pos_close
    
    ur5.set_joint_position_target(target_pos)
    print(f"Garra {'abierta' if open else 'cerrada'}")

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    gripper_open = True  # Estado inicial de la garra
    cycle_count = 0      # Contador para saber cuándo reiniciar la simulación

    while simulation_app.is_running():
        ur5 = scene["ur5"]

        # Si es el inicio del ciclo, "spawneamos" al robot en la posición fija
        if cycle_count == 0:
            ur5.reset()
            root_state = ur5.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins  # Se ubica en la posición del entorno
            ur5.write_root_pose_to_sim(root_state[:, :7])
            ur5.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos = ur5.data.default_joint_pos.clone()
            joint_vel = ur5.data.default_joint_vel.clone()
            ur5.write_joint_state_to_sim(joint_pos, joint_vel)
            print("Robot spawneado en posición fija.")

        # Control de la garra: se alterna el estado y se aplica el torque
        control_gripper(ur5, gripper_open)
        gripper_open = not gripper_open

        # Espera unos segundos (aquí 2 segundos) para ver el cambio en la simulación
        steps_to_wait = int(2.0 / sim_dt)
        for _ in range(steps_to_wait):
            ur5.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            ur5.update(sim_dt)

        # Se cuenta el ciclo (cada ciclo es un cambio, por ejemplo: apertura y luego cierre)
        cycle_count += 1

        # Cuando se completa un ciclo de apertura y cierre (2 cambios), se reinicia la simulación
        if cycle_count >= 2:
            cycle_count = 0
            print("Reiniciando simulación...\n")

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    scene_cfg = MyInteractiveSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()
