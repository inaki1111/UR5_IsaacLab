# main.py
import argparse
from turtle import delay
import torch
from isaaclab.app import AppLauncher

# Parseo de argumentos
parser = argparse.ArgumentParser(description="UR5 with Differential IK Controller")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Inicializar la aplicación
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports de IsaacLab
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

# Importar nuestra escena y el controlador IK
from scene import MyInteractiveSceneCfg
from differential_ik_controller import UR5DifferentialIK

def run_simulator(sim, scene, ik_controller):
    sim_dt = sim.get_physics_dt()

    # Metas del efector final
    ee_goals = torch.tensor([
        [0.5,  0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
        [0.8, -0.2, 0.55, 0.0, 0.0, 0.0, 1.0],
        [0.4,  0.4, 0.65, 0.0, 0.0, 0.0, 1.0],
        [0.6, -0.3, 0.7, 0.0, 0.0, 0.0, 1.0],
        [0.3, -0.1, 0.45, 0.0, 0.0, 0.0, 1.0]
    ], device=sim.device)

    current_goal_idx = 0
    ik_commands = torch.zeros(
        scene.num_envs, 
        ik_controller.diff_ik_controller.action_dim, 
        device=sim.device
    )
    ik_commands[:] = ee_goals[current_goal_idx]

    count = 0
    while simulation_app.is_running():
        # Obtenemos el UR5 que se registró en la escena
        ur5 = scene["ur5"]

        # Cada 150 pasos, reiniciamos y cambiamos de objetivo
        if count % 150 == 0:
            count = 0
            joint_pos = ur5.data.default_joint_pos.clone()
            joint_vel = ur5.data.default_joint_vel.clone()
            # Mandamos al robot a la pos. por defecto
            ur5.write_joint_state_to_sim(joint_pos, joint_vel)
            ur5.reset()

            # Reseteamos el controlador IK
            ik_controller.reset()
            ik_controller.diff_ik_controller.set_command(ik_commands)
            print("Change position")

            # Pasamos al siguiente objetivo
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
            ik_commands[:] = ee_goals[current_goal_idx]

        else:
            # Calcular posiciones articulares
            joint_pos_des = ik_controller.compute_ik(ur5, ik_commands)
            # Aplicar esas posiciones como target
            ur5.set_joint_position_target(
                joint_pos_des, 
                joint_ids=ik_controller.robot_entity_cfg.joint_ids
            )

        # Avances típicos de la simulación
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        count += 1
        ur5.update(sim_dt)

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    # Crear la config de la escena y la escena
    scene_cfg = MyInteractiveSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)

    # --- FORZAR UNO O DOS PASOS DE SIM PARA INICIALIZAR LA ESCENA ---
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())

    # Si todavía sigue sin inicializar, puedes repetir:
    # scene.write_data_to_sim()
    # sim.step()
    # scene.update(sim.get_physics_dt())

    # Ahora creamos el IK (el robot ya debería tener root_physx_view)
    ik_controller = UR5DifferentialIK(scene, sim.device)

    # Reset de física, etc.
    sim.reset()

    # Correr la simulación
    run_simulator(sim, scene, ik_controller)

if __name__ == "__main__":
    main()
    simulation_app.close()
