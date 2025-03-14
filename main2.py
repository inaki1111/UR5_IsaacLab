import argparse
import torch
from isaaclab.app import AppLauncher
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from scene import MyInteractiveSceneCfg


parser = argparse.ArgumentParser(description="UR5 with Differential IK Controller")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from differential_ik_controller import UR5DifferentialIK
def run_simulator(sim, scene, ik_controller):
    sim_dt = sim.get_physics_dt()

    # Definir metas del efector final (posiciones deseadas)
    ee_goals = torch.tensor([
        [0.5,  0.0, 0.5, 0.0, 0.0, 0.0, 1.0],  
        [0.8, -0.2, 0.55, 0.0, 0.0, 0.0, 1.0],  
        [0.4,  0.4, 0.65, 0.0, 0.0, 0.0, 1.0],  
        [0.6, -0.3, 0.7, 0.0, 0.0, 0.0, 1.0],   
        [0.3, -0.1, 0.45, 0.0, 0.0, 0.0, 1.0]  
    ], device=sim.device)

    current_goal_idx = 0
    ik_commands = torch.zeros(scene.num_envs, ik_controller.diff_ik_controller.action_dim, device=sim.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    count = 0
    while simulation_app.is_running():
        ur5 = scene["ur5"]
        if count % 150 == 0:
            count = 0
            joint_pos = ur5.data.default_joint_pos.clone()
            joint_vel = ur5.data.default_joint_vel.clone()
            ur5.write_joint_state_to_sim(joint_pos, joint_vel)
            ur5.reset()
            ik_controller.reset()
            ik_controller.diff_ik_controller.set_command(ik_commands)
            print("Change position")

            # Actualizar objetivo
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
            ik_commands[:] = ee_goals[current_goal_idx]

        else:
            joint_pos_des = ik_controller.compute_ik(ur5, ik_commands)
            ur5.set_joint_position_target(joint_pos_des, joint_ids=ik_controller.robot_entity_cfg.joint_ids)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        count += 1
        ur5.update(sim_dt)

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    # Crear la escena y el controlador IK
    scene_cfg = MyInteractiveSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    ik_controller = UR5DifferentialIK(scene, sim.device)

    sim.reset()
    run_simulator(sim, scene, ik_controller)

if __name__ == "__main__":
    main()
    simulation_app.close()
