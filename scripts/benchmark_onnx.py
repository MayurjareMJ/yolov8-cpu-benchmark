import onnxruntime as ort
import numpy as np
import cv2
import time



session=ort.InferenceSession(
    "yolov8n.onnx",
    providers=['CPUExecutionProvider']
)