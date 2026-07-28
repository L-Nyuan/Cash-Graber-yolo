import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import re
import cv2
from loguru import logger
from PIL import Image
from dataclasses import dataclass, field
from numpy.typing import NDArray
from sklearn.cluster import KMeans

@dataclass
class ObjectGT:
    """单个物体的GT yaml信息"""
    class_name: str
    cx: float               # 2D质心 x（列坐标）
    cy: float               # 2D质心 y（行坐标），这个坐标可以作为k-mens聚类的中心
    visibility: float      # 物体可见度

@dataclass
class SceneGT:
    """单个场景的GT yaml信息"""
    scene_id: int
    objects: list   # list[ObjectGT]

@dataclass
class OutputPath:
    """
    处理结束后输出数据集的地址，默认文件夹结构为\n
    datasets/\n
    ├── images/\n
    │   ├── train/\n
    │   ├── val/\n
    │   └── test/\n
    ├── labels/\n
    │   ├── train/\n
    │   ├── val/\n
    │   └── test/\n
    └── dataset.yaml\n
    Attributes:
        image: str, 输出图片的地址
        labels: str, 输出标签的地址
    """
    root:str
    images:str = field(init=False)
    labels:str = field(init=False)

    def __post_init__(self):
        self.images = os.path.join(self.root, 'images/')
        self.labels = os.path.join(self.root, 'labels/')

@dataclass
class ViewPoint:
    """
    存放每个视角的yaml文件和各个rgb图像与seg图像的路径
    """
    index:int
    root:str
    view_point:int           # 0: left, 1: middle, 2: right    
    yaml_path:str = field(init=False)
    seg_path:str = field(init=False)
    dir_light_rgb:str = field(init=False)
    point_light_rgb:str = field(init=False)
    spot_light_rgb:str = field(init=False)
    gt:SceneGT = field(init=False) # 该视角的GT信息

    def __post_init__(self):
        if self.view_point == 0:
            self.yaml_path = os.path.join(self.root, f'GT_left_camera_{self.index}.yaml')
            self.seg_path = os.path.join(self.root, f'left_camera_scene_{self.index}_segmentation.png')
            self.dir_light_rgb = os.path.join(self.root, f'left_camera_directional_light_scene_{self.index}_rgb.png')
            self.point_light_rgb = os.path.join(self.root, f'left_camera_point_light_scene_{self.index}_rgb.png')
            self.spot_light_rgb = os.path.join(self.root, f'left_camera_spot_light_scene_{self.index}_rgb.png')
        elif self.view_point == 1:
            self.yaml_path = os.path.join(self.root, f'GT_middle_camera_{self.index}.yaml')
            self.seg_path = os.path.join(self.root, f'middle_camera_scene_{self.index}_segmentation.png')
            self.dir_light_rgb = os.path.join(self.root, f'middle_camera_directional_light_scene_{self.index}_rgb.png')
            self.point_light_rgb = os.path.join(self.root, f'middle_camera_point_light_scene_{self.index}_rgb.png')
            self.spot_light_rgb = os.path.join(self.root, f'middle_camera_spot_light_scene_{self.index}_rgb.png')
        elif self.view_point == 2:
            self.yaml_path = os.path.join(self.root, f'GT_right_camera_{self.index}.yaml')
            self.seg_path = os.path.join(self.root, f'right_camera_scene_{self.index}_segmentation.png')
            self.dir_light_rgb = os.path.join(self.root, f'right_camera_directional_light_scene_{self.index}_rgb.png')
            self.point_light_rgb = os.path.join(self.root, f'right_camera_point_light_scene_{self.index}_rgb.png')
            self.spot_light_rgb = os.path.join(self.root, f'right_camera_spot_light_scene_{self.index}_rgb.png')
        else:
            raise ValueError(f"视角索引必须是 0/1/2，收到: {self.view_point}")

        """读取yaml, 提取2D_centroid + visibility + class_name"""
        with open(self.yaml_path, 'r') as f:
            # with是上下文管理器，确保文件在使用后正确关闭
            # 'r'表示以只读模式打开文件
            data = yaml.safe_load(f)
            # 解析yaml数据，保存为Python对象，同时safe_load方法可以防止执行不安全的代码
        objects = []
        for key, info in data['objects'].items(): # data['objects']是字典取值，.items()相当于降维了，data里面每一项都是字典（其实就一项），.items()能把字典变成可迭代的元组
            # 现在提取出来的key是'10-meat_can' -> 去掉数字前缀
            class_name = re.sub(r'^\d+-', '', key)# re.sub是替换函数,re.sub(规则, 替换成什么, 原始字符串)
            # r表示正则表达式，\不要转义，^表示开头，\d+表示匹配一个或多个数字，-表示匹配一个连字符
        
            x, y = info['2D_centroid'] # 取出2D质心坐标
            vis = info['visibility'][0] # 取出可见度，[0]是因为visibility是一个只有一个元素的列表
            objects.append(ObjectGT(class_name=class_name, cx=x, cy=y, visibility=vis))
        
        logger.debug(f"读取yaml文件: {self.yaml_path}, 提取到{len(objects)}个物体信息\n{objects}")
        self.gt = SceneGT(scene_id=self.index, objects=objects)

        logger.debug(f"视角{self.view_point}的yaml路径: {self.yaml_path}\nseg路径: {self.seg_path}\ndir_light_rgb路径: {self.dir_light_rgb}\npoint_light_rgb路径: {self.point_light_rgb}\nspot_light_rgb路径: {self.spot_light_rgb}")
            


class CEPBProcessor:
    """
    处理CEPB数据集的类
    Attributes:
        index: int, 数据集编号
        input_path: str, 输入数据集的地址（以/结尾）
        output_path: str, 输出数据集的地址（以/结尾）
    """
    CLASS_NAMES = sorted([
        'Cheez-it', 'Starkist_Tuna', 'Scissors', 'Frenchs_Mustard', 'Tomato_Soup',
        'Foam_Brick', 'Clamp', 'Plastic_Banana', 'Mug', 'meat_can',
    ])


    def __init__(self, index: int, input_dir: str, output:OutputPath):
        """
        构造了当前索引的三个视角yaml文件路径, seg图片路径, 并且读取了相关的信息
        """

        # 一些全局变量
        self.step = 10 #图像量化步长
        self.h = 1536 # 图像高度
        self.w = 2048 # 图像宽度
        self.vis = 0.2 # 可见度阈值，低于该阈值的物体不参与聚类
        self.contour_step = 0.001 # 轮廓近似精度，越小越精确，单位是像素长度的比例

        self.index = index
        self.output = output
        self.input_dir = input_dir

        # 构造文件路径
        self.left_view = ViewPoint(index=index, root=input_dir, view_point=0)
        self.middle_view = ViewPoint(index=index, root=input_dir, view_point=1)
        self.right_view = ViewPoint(index=index, root=input_dir, view_point=2)




    def seg_to_cluster(self, view: ViewPoint)-> tuple[NDArray, dict]:
        """
        读取seg图像并作像素量化和聚类
        """
        img = np.array(Image.open(view.seg_path)) # 读取seg图像

        rgb = img[:, :, :3] # 取RGB通道
        # 进行像素量化/感觉叫离散化更合适一点
        quantized = (rgb // self.step) * self.step

        # 先对颜色去重，减少计算量
        colors, inverse = np.unique(
            quantized.reshape(-1, 3),       # 展开成二维数组，具体展开方式没看。每一行都是一个rgb组合
            axis=0,                         # 按照行去重，一模一样的行就只保留一个
            return_inverse=True             # 压缩索引，表示原来的每个像素在去重后的数组中的索引位置，quantized.reshape(-1, 3)[i] == colors[inverse[i]]
        )
        # 取背景色（大概率可以省却掉，因为背景色都是白色。而且，因为聚类发生在量化后的图形中，所以取色也应该在量化后的图形中
        corners = np.array([quantized[0, 0, :3], quantized[0, self.w - 1, :3], quantized[self.h - 1, 0, :3], quantized[self.h - 1, self.w - 1, :3]])
        bg_black = np.mean(corners, axis=0).astype(np.uint8) # 取四个角的平均值作为背景色

        # 取白色盒子色
        white_box = (255//self.step) * self.step
        bg_white = np.array([white_box, white_box, white_box], dtype=np.uint8)

        centers = [bg_black, bg_white] # 先把背景色加入到聚类中心列表中
        labels = ['__background__', '__white_box__']

        # 取每个物体的质心色,因为聚类发生在量化后的图形中，所以取色也应该在量化后的图形中,虽然大概率不会取到过度色，但是还是更加合理一些
        for obj in view.gt.objects:
            if obj.visibility < self.vis:
                continue
            # 取出质心的颜色
            color = quantized[int(obj.cy), int(obj.cx)]
            centers.append(color)
            labels.append(obj.class_name)

        logger.debug(f"聚类中心颜色: {centers}\n聚类中心标签: {labels}")

        # 进行k-means聚类
        K = len(centers) # 聚类中心数量(还有盒子本身的颜色，等待修改)
        kmeans = KMeans(n_clusters=K, init=np.array(centers), n_init=1, random_state=42) # n_init=1表示只进行一次初始化，random_state=0表示随机种子固定
        kmeans.fit(colors) # 训练模型，输入是二维数组，每一行是一个像素的rgb值
        logger.debug(f"聚类结果: {kmeans.labels_}")

        # kmeans.labels_[inverse]表示的是某个像素所属的簇的编号，这个编号和聚类中心存在一一对应的联系
        cluster_map = kmeans.labels_[inverse].reshape(self.h, self.w)

        # 构建label_map, label_map的每个像素值是对应的标签索引
        label_map = {i:labels[i] for i in range(K)}

        self._debug_visualize(cluster_map, label_map, quantized_img=quantized, view_name=f"scene_{view.index}_view_{view.view_point}")

        return cluster_map, label_map


    def cluster_to_outline(self, cluster_map: NDArray, label_map: dict)->dict:
        """将聚类结果转换为轮廓图"""
        outlines = {} # 不考虑背景轮廓
        for idx, label in label_map.items():
            if label == '__background__':
                continue
            if label == '__white_box__':
                logger.debug(f"物体 {label} 是白色盒子，已忽略该物体。")
                continue
            mask = (cluster_map == idx).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.RETR_EXTERNAL 只压缩最外层轮廓，忽略孔洞，压缩水平/垂直/对角线段
            if contours:
                # 找到最大轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                peri = cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon=self.contour_step * peri, closed=True)
                pts = approx.reshape(-1, 2)  # 去掉多余的维度 pts.shape = (N, 2)

                if len(pts) >= 3:  # 至少需要3个点才能构成多边形
                    outlines[idx] = pts.astype(np.float32)  # 转换为float32类型，方便后续处理
                else:
                    logger.warning(f"物体 {label} 的轮廓点数不足3，无法构成多边形，已忽略该物体。")
            else:
                logger.warning(f"物体 {label} 没有找到轮廓，可能是因为聚类结果不理想或物体被遮挡。")
        return outlines



    def save_files(self, outlines:dict, view:ViewPoint, label_map:dict):
        """
        输出修正完的数据到目标地址
        """
        light_types = {
            'dir': view.dir_light_rgb,
            'point': view.point_light_rgb,
            'spot': view.spot_light_rgb
        }

        for light_name, light_path in light_types.items():
            # 完成图像的保存
            img = np.array(Image.open(light_path))  # 读取图像
            img_rgb = img[:, :, :3]  # 取RGB通道

            os.makedirs(self.output.images, exist_ok=True)
            out_name = f"{self.index}_view_{view.view_point}_{light_name}.jpg"
            out_path = os.path.join(self.output.images, out_name)
            Image.fromarray(img_rgb).save(out_path)
            logger.debug(f"保存RGB图像到: {out_path}")

            # 保存轮廓点到txt文件
            os.makedirs(self.output.labels, exist_ok=True)
            out_name = f"{self.index}_view_{view.view_point}_{light_name}.txt"
            out_path = os.path.join(self.output.labels, out_name)

            with open(out_path, 'w') as f:
                for cluster_id, pts in outlines.items():
                    if len(pts) < 3:
                        logger.warning(f"物体 {label_map[cluster_id]} 的轮廓点数不足3，无法构成多边形，已忽略该物体。")
                        continue
                    class_name = label_map[cluster_id]
                    class_id = self.CLASS_NAMES.index(class_name) if class_name in self.CLASS_NAMES else -1
                    if class_id == -1:
                        logger.warning(f"物体 {class_name} 不在预定义类别列表中，已忽略该物体。")
                        continue

                    pts_norm = pts.copy()
                    pts_norm[:, 0] /= self.w  # x坐标归一化
                    pts_norm[:, 1] /= self.h  # y坐标归一化


                    parts = [str(class_id)]
                    for x, y in pts_norm:
                        parts.append(f"{x:.6f}")
                        parts.append(f"{y:.6f}")
                    line = ' '.join(parts)
                    f.write(line + '\n')


    def process(self):
        """
        处理每个视角的函数
        """
        for view in [self.left_view, self.middle_view, self.right_view]:
            cluster_map, label_map = self.seg_to_cluster(view)
            outlines = self.cluster_to_outline(cluster_map, label_map)
            self.save_files(outlines, view, label_map)


    def _debug_visualize(self, cluster_map, label_map, quantized_img, view_name="scene"):
        """
        调试可视化：左图 = 量化后图像，右图 = 聚类结果（固定颜色 + 类名标注）。
        不弹窗、不渲染 GUI 框，直接保存 PNG 到磁盘。
        """

        if logger._core.min_level > 10:          # 只有 DEBUG 模式才执行
            return

        from matplotlib.colors import ListedColormap

        K = cluster_map.max() + 1
        logger.debug(f"渲染调试可视化，视角：{view_name}，类别数：{K}")

        # ================================================================
        # 固定颜色表 —— 每种物体一个专属颜色，跨场景不变，方便肉眼对比
        # 颜色值域 [0, 1]，分别对应 (R, G, B)
        # ================================================================
        FIXED_COLORS = {
            '__background__':  (1.0, 1.0, 1.0),   # 白色 — 背景
            '__white_box__':   (1.0, 1.0, 1.0),   # 白色 — 白色盒子
            'Cheez-it':        (0.89, 0.10, 0.11),   # 红   — 芝士饼干
            'Starkist_Tuna':   (0.22, 0.49, 0.72),   # 蓝   — 金枪鱼罐头
            'Scissors':        (0.30, 0.69, 0.29),   # 绿   — 剪刀
            'Frenchs_Mustard': (1.00, 0.80, 0.00),   # 金黄 — 芥末酱
            'Tomato_Soup':     (0.90, 0.40, 0.00),   # 橙   — 番茄汤
            'Foam_Brick':      (0.55, 0.24, 0.60),   # 紫   — 泡沫砖
            'Clamp':           (0.00, 0.75, 0.75),   # 青   — C 型夹
            'Plastic_Banana':  (0.96, 0.87, 0.30),   # 淡黄 — 塑料香蕉
            'Mug':             (0.50, 0.50, 0.55),   # 灰蓝 — 马克杯
            'meat_can':        (0.93, 0.28, 0.51),   # 粉红 — 肉罐头
        }
        UNKNOWN = (1.0, 1.0, 1.0)                   # 白色 — 未知（异常信号）

        # ================================================================
        # 按 cluster_id 顺序构建 ListedColormap
        # label_map = {0: '__background__', 1: 'Cheez-it', 2: 'Clamp', ...}
        # ================================================================
        color_list = [FIXED_COLORS.get(label_map.get(i, ''), UNKNOWN)
                    for i in range(K)]
        cmap = ListedColormap(color_list, name='CEPB_fixed')

        # ================================================================
        # 画布：1 行 2 列
        # ================================================================
        fig, axes = plt.subplots(1, 2, figsize=(22, 9))

        # ---- 左：量化图像 ----
        ax0 = axes[0]
        ax0.imshow(quantized_img)
        ax0.set_title(f"quantized_img  step={self.step}  |  {view_name}",
                    fontsize=12, fontweight='bold')
        ax0.axis('off')

        # ---- 右：聚类结果 ----
        ax1 = axes[1]
        im = ax1.imshow(cluster_map, cmap=cmap, vmin=0, vmax=K - 1,
                        interpolation='nearest')
        ax1.set_title(f"K-Means_images  K={K}  |  {view_name}",
                    fontsize=12, fontweight='bold')
        ax1.axis('off')

        # ---- 颜色条：显示 cluster_id + 类名 + 备注 ----
        cbar = plt.colorbar(im, ax=ax1, ticks=range(K), fraction=0.046)
        tick_labels = [f"{i}: {label_map.get(i, '?')}" for i in range(K)]
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=7)

        # ---- 保存到磁盘，关闭画布释放内存 ----
        debug_dir = os.path.join(self.output.root, 'debug_vis')
        os.makedirs(debug_dir, exist_ok=True)
        save_path = os.path.join(debug_dir, f"{view_name}_cluster.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)                              # 不弹窗，不占内存
        logger.debug(f"调试图像已保存至：{save_path}")


def main():
    # 设置日志记录器
    logger.add("logs/Seg_To_Txt_{time}.log", rotation="10 MB", level="DEBUG", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")
    test = CEPBProcessor(index=1, input_dir="/root/dataset/", output=OutputPath(root="/root/yolo/test_output"))
    test.process()
    return 0

if __name__ == "__main__":
    main()