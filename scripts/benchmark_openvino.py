from openvino.runtime import Core

core=Core()

model=core.read_model(
    "yolov8n_openvino_model/yolov8n.xml"
)

compiled_model=core.compile_model(
    model,
    "CPU"
)