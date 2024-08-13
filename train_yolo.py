from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')

results = model.train(data = '400_classes_dataset', batch = 16, epochs = 1)