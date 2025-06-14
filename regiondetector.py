"""
屏幕区域检测模块

此模块提供了用于检测和处理图像中特定区域的功能。
主要用于检测图像中符合特定条件的矩形区域。
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from action import ActionControl
from myapriltag import AprilTagDetector
from PIL import Image
from pynq import Overlay, Xlnk


class RegionDetector:
    """屏幕区域检测器类"""

    def __init__(self):
        """初始化区域检测器"""
        self.camera_index = 2
        self.camera_width = 1920
        self.camera_height = 1080
        self.min_area = 1500  # 最小区域面积
        self.solidity_threshold = 0.8  # 凸度阈值
        self.aspect_ratio_range = (0.35, 0.8)  # 长宽比范围
        self.actioncontroller = ActionControl()

    def capture_image(self, save_path: str) -> bool:
        """
        从摄像头捕获图像并保存

        参数:
            save_path: 保存图像的路径

        返回:
            是否成功捕获图像
        """
        img = self.actioncontroller.head_capture_for_detector()

        if img is not None:
            cv2.imwrite(save_path, img)
            print(f"图像已保存到 {save_path}")
            return True
        else:
            print("无法读取图像")
            return False

    def preprocess_image(
        self, img_path: str, tag_min_area: int = 500
    ) -> Tuple[List[np.ndarray], List[List[Dict[str, Any]]], List[int]]:
        """
        预处理图像并检测区域（基于AprilTag左下角ROI）

        参数:
            img_path: 输入图像路径
            tag_min_area: AprilTag最小面积阈值

        返回:
            result_imgs: 每个tag对应的处理后图像列表
            regions_list: 每个tag对应的区域列表
            tag_ids: 每个tag的id列表
        """
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            return [], [], []

        height, width = img.shape[:2]
        result_imgs = []
        regions_list = []
        tag_ids = []

        # 1. 检测AprilTag
        tag_detector = AprilTagDetector()
        tags = tag_detector.detect_tags(img)

        for tag in tags:
            tag_id = int(tag.tag_id)
            if tag_id <= 0 or tag_id > 36:
                print(f"Tag {tag_id} 不是柱上tag，跳过")
                continue
            # 计算tag面积
            corners = tag.corners  # shape: (4,2)
            tag_w = np.linalg.norm(corners[1] - corners[0])
            tag_h = np.linalg.norm(corners[3] - corners[0])
            tag_area = tag_w * tag_h
            if tag_area < tag_min_area:
                print(f"Tag {tag.tag_id} 面积过小，跳过")
                continue
            print(f"Tag{tag_id}面积合适，开始处理")
            # 以tag中心为基准，取4倍tag长宽的区域
            center_x = int(np.mean(corners[:, 0]))
            center_y = int(np.mean(corners[:, 1]))
            roi_x = center_x
            roi_y = center_y
            roi_w = int(tag_w * 4)
            roi_h = int(tag_h * 4)
            if roi_x <0.05*1920 :
                print(f"tag{tag_id}不在中心")
                continue
            # 保证ROI不越界
            roi_x = max(0, roi_x)
            roi_y = max(0, roi_y)
            if roi_x - roi_w < 1:
                roi_w=roi_x-1
            if roi_y + roi_h > height:
                roi_h=height-roi_y-1
            roi_img = img[
                roi_y : roi_y + roi_h, roi_x - roi_w : roi_x - int(tag_w)
            ].copy()
            if roi_img.size == 0:
                print(f"Tag{tag_id}对应区域不在摄像头内")
                continue
            
            # 对ROI区域执行原有的区域检测逻辑
            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            erode_kernel = np.ones((3, 3), np.uint8)
            binary = cv2.erode(binary, erode_kernel, iterations=1)
            dilate_kernel = np.ones((4, 4), np.uint8)
            binary = cv2.dilate(binary, dilate_kernel, iterations=3)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[-2:]

            result_img = roi_img.copy()
            detected_regions = []
            roi_height, roi_width = roi_img.shape[:2]

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    
                    continue
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                if len(approx) >= 4:
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        solidity = float(area) / hull_area
                        if solidity > self.solidity_threshold:
                            x, y, w, h = cv2.boundingRect(contour)
                            aspect_ratio = float(w) / h
                            if (
                                self.aspect_ratio_range[0]
                                < aspect_ratio
                                < self.aspect_ratio_range[1]
                            ):
                                detected_regions.append(
                                    {
                                        "x": x,
                                        "y": y,
                                        "width": w,
                                        "height": h,
                                        "area": area,
                                    }
                                )
            # 选择面积最大的区域
            if detected_regions:
                print(f"Tag{tag_id}找到了对应屏幕")
                max_region = max(detected_regions, key=lambda x: x["area"])
                # 在原图上画出矩形
                cv2.rectangle(
                    result_img,
                    (max_region["x"], max_region["y"]),
                    (
                        max_region["x"] + max_region["width"],
                        max_region["y"] + max_region["height"],
                    ),
                    (0, 255, 0),
                    2,
                )
            else:
                print(f"{tag_id}cannot find a screen")
                continue

            result_imgs.append(result_img)
            regions_list.append([max_region])
            tag_ids.append(tag_id)

        return result_imgs, regions_list, tag_ids

    def extract_regions(
        self,
        img_path: str,
        regions: List[Dict[str, Any]],
        output_dir: str,
        prefix: str = "",
    ) -> None:
        """
        从原图中提取指定区域并分别保存

        参数:
            img_path: 原图路径
            regions: 检测到的区域列表
            output_dir: 输出文件夹路径
            prefix: 文件名前缀
        """
        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"无法读取图像: {img_path}")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, region in enumerate(regions):
            x, y, w, h = region["x"], region["y"], region["width"], region["height"]
            screen_region = original_img[y : y + h, x : x + w]
            region_path = os.path.join(output_dir, f"{prefix}region_{i + 1}.jpg")
            cv2.imwrite(region_path, screen_region)
            print(f"  区域已保存: {region_path}")
            print(f"  位置: ({x}, {y}) 尺寸: {w}x{h} 面积: {region['area']}")

    def process_and_save(
        self, input_path: str, output_dir: str, prefix: str = ""
    ) -> List[Dict[str, Any]]:
        """
        处理图像并保存所有tag对应的区域

        参数:
            input_path: 输入图像路径
            output_dir: 输出文件夹路径
            prefix: 文件名前缀

        返回:
            检测到的所有区域信息列表
        """
        result_imgs, regions_list, tag_ids = self.preprocess_image(input_path)
        all_regions = []
        for idx, (result_img, regions, tag_id) in enumerate(
            zip(result_imgs, regions_list, tag_ids)
        ):
            result_img_path = os.path.join(
                output_dir, f"{prefix}tag_{tag_id}_result.jpg"
            )
            cv2.imwrite(result_img_path, result_img)
            print(f"处理后图像已保存: {result_img_path}")
            # 保存每个tag下的所有区域
            self.extract_regions(
                input_path, regions, output_dir, prefix=f"{prefix}tag_{tag_id}_"
            )
            for region in regions:
                region["tag_id"] = tag_id
                all_regions.append(region)
        return all_regions

    def process_multi_images(self, image_paths, brightness_threshold=210):
        #196 198
        """
        处理多张图片，合并所有图片的tag检测结果，只保留每个tag_id亮度最低且低于阈值的那组。

        参数:
            image_paths: 图片路径列表
            brightness_threshold: 亮度阈值，只有低于该值的区域才会被保留

        返回:
            tagid_to_best: dict, key为tag_id，value为{"result_img", "regions", "brightness"}
        """
        all_result_imgs = []
        all_regions_list = []
        all_tag_ids = []
        all_brightness = []

        for image_path in image_paths:
            print(f"正在处理{image_path}")
            result_imgs, regions_list, tag_ids = self.preprocess_image(image_path)
            for tag_idx, (regions, tag_id) in enumerate(zip(regions_list, tag_ids)):
                if not regions or regions[0]["width"] == 0 or regions[0]["height"] == 0:
                    continue
                region = regions[0]
                x, y, w, h = region["x"], region["y"], region["width"], region["height"]
                roi_img = result_imgs[tag_idx][y : y + h, x : x + w]
                gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
                brightness = np.median(gray)
                #if brightness > brightness_threshold:
                #    print(f"{image_path} de {tag_id} too birght")
                #    continue  # 跳过亮度过高的区域
                all_result_imgs.append(result_imgs[tag_idx])
                all_regions_list.append([region])
                all_tag_ids.append(tag_id)
                all_brightness.append(brightness)

        tagid_to_best = {}
        for i, tag_id in enumerate(all_tag_ids):
            if (
                tag_id not in tagid_to_best
                or all_brightness[i] < tagid_to_best[tag_id]["brightness"]
            ):
                tagid_to_best[tag_id] = {
                    "result_img": all_result_imgs[i],
                    "regions": all_regions_list[i],
                    "brightness": all_brightness[i],
                }
        print(f"将对总共{len(tagid_to_best)}个合理屏幕进行分类通信")
        return tagid_to_best
