import tensorflow as tf
# import torch
import onnx
import onnx_tf

print("TensorFlow version:", tf.__version__)
# print("PyTorch version:", torch.__version__)
print("ONNX version:", onnx.__version__)
print("ONNX-TF version:", onnx_tf.__version__)


onnx_model = onnx.load("/home/athena/Documents/GitHub/Dog-NosePrint-Recognition/model-pet.onnx")