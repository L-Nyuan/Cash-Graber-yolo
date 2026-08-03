from ultralytics import YOLO

model=YOLO(
"/root/yolo/result/exp_2/best.pt"
)

result=model.predict(
"/root/dataset_yolo/images/val/4501_view_0_dir.jpg",
imgsz=640,
conf=0.25,
save=True,
retina_masks=True
)

print(result[0].names)