from ultralytics import YOLO

model = YOLO('yolov10n.yaml')

result = model.train(data='data.yaml', epochs = 1)
