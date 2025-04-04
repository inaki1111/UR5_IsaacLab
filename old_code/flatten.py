from pxr import Usd, Sdf
from omni.usd import get_context

# Ruta donde guardarás el archivo flattenizado
output_path = "/home/inaki/Desktop/gripper_flattened.usd"

# Obtener el stage actual desde Isaac Sim
stage = get_context().get_stage()
root_layer = stage.GetRootLayer()

# Crear una nueva capa vacía
flattened_layer = Sdf.Layer.CreateNew(output_path)

# Abrir un nuevo stage sobre la nueva capa
flattened_stage = Usd.Stage.Open(flattened_layer.identifier)

# Exportar la escena actual a la nueva capa como flattened
stage.Flatten().Export(output_path)

print(f"✅ Escena flattenizada correctamente en: {output_path}")
