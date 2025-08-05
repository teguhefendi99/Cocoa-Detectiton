from ultralytics import YOLO
model = YOLO("yolo8m.pt")
train_results = model.train(
 data="coco8.yaml",
 epochs=100,
 imgsz=640,
 device="0",
)
metrics = model.val()
