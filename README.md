# Cash-Graber-yolo
2026埃斯顿机器人抓取比赛的视觉部分代码仓库

`clean_dataset`是对不同视角和不同光照条件数据集的清理，因为在训练过程中发现数据集有冗余，遂添加

`debug_centers`现在的`Seg_To_Txt`的K-means聚类存在问题，所以添加该调试模块，显示经过颜色压缩后的聚类的中心

`interacitve_color_picker`因为K-means聚类的问题和数据集关系比较大，懒得解决，所以直接手动取颜色作为聚类中心

`preview_letterbox`防止yolo网络内部对图像压缩太狠（目前是640*640），导致无法看到小物品，所以提前预览

`visualize_labels`绘制txt数据集和中包围点segmentation，检查制作成果

`Seg_To_Txt`从官方数据集到Yolo可用数据格式（文件格式需要另外调整）

`yolo_train`训练主函数


官方数据集中有很多多余的深度图像，删掉就行了

python -c "
import cv2, sys
sys.path.insert(0, '/root/yolo/scripts/inference')
from yolo_inference import YOLOSegInference
from visualization_utils import save_debug_image

yolo = YOLOSegInference('/root/yolo/result/final/last.pt')

img_bgr = cv2.imread('/root/ros2_ws/dataset_real/images/val/37_010_jpg.rf.NiwQfaRabHYHhxKNzA3m.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
print(f'图像尺寸: {img_rgb.shape}')

r = yolo.predict(img_rgb)
print(f\"推理 {r['inference_time_ms']:.1f}ms, 检测到 {len(r['objects'])} 个物体\")
for o in r['objects']:
    print(f\"  {o['class_name']:20s} conf={o['confidence']:.2f}  pixels={o['mask'].sum()}\")

save_debug_image(img_rgb, r['objects'], '/root/yolo/debug_test4501.jpg')
print('可视化已保存: /root/yolo/debug_test.jpg')
"

python yolo_inference_node.py --ros-args -p mode:=debug