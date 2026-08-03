from ultralytics import YOLO

model=YOLO(
"/root/yolo/result/exp_2/best.pt"
)

result=model.predict(
"/root/yolo/dataset_real/01/frame_000600.jpg",
imgsz=640,
conf=0.25,
save=True,
retina_masks=True
)

print(result[0].names)