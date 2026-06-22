from ultralytics import YOLO

model1=YOLO("yolov8n.pt")
model2=YOLO("yolov8s.pt")


model1.export(format="onnx")
model1.export(format="openvino")

model2.export(format="onnx")
model2.export(format="openvino")