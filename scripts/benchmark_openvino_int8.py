from ultralytics import YOLO

model=YOLO("/home/mayur/Desktop/Benchmark/yolov8n.pt")

model.export(
    format="openvino",
    int8=True
)