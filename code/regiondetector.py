"""
屏幕区域检测模块

此模块提供了用于检测和处理图像中特定区域的功能。
主要用于检测图像中符合特定条件的矩形区域。
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional,Any
from PIL import Image
import matplotlib.pyplot as plt
from pynq import Overlay
from pynq import Xlnk
from action import ActionControl
class RegionDetector:
    """屏幕区域检测器类"""
    
    def __init__(self):
        """初始化区域检测器"""
        self.camera_index = 2
        self.camera_width = 1920
        self.camera_height = 1080
        self.min_area = 1000  # 最小区域面积
        self.solidity_threshold = 0.8  # 凸度阈值
        self.aspect_ratio_range = (0.3, 0.8)  # 长宽比范围
        self.actioncontroller = ActionControl()
    
    def capture_image(self, save_path: str) -> bool:
        """
        从摄像头捕获图像并保存
        
        参数:
            save_path: 保存图像的路径
            
        返回:
            是否成功捕获图像
        """
        img = self.actioncontroller.head_capture()
        
        if img is not None:
            cv2.imwrite(save_path, img)
            print(f"图像已保存到 {save_path}")
            return True
        else:
            print("无法读取图像")
            return False
    
    def preprocess_image(self, img_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        预处理图像并检测区域
        
        参数:
            img_path: 输入图像路径
            
        返回:
            处理后的图像和检测到的区域列表
        """
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            return np.array([]), []
            
        height, width = img.shape[:2]
        
        # 定义中心区域范围
        center_x_start = int(width * 0.05)  # 从5%开始
        center_x_end = int(width * 0.95)    # 到95%结束
        center_y_start = int(height * 0.3)  # 从30%开始
        center_y_end = int(height * 0.7)    # 到70%结束
        
        # 转换为灰度图并进行高斯模糊
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 使用自适应阈值处理，突出黑色区域
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
       
        # 使用形态学操作清理噪点
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 先进行腐蚀操作，去除小的噪点
        erode_kernel = np.ones((3, 3), np.uint8)
        binary = cv2.erode(binary, erode_kernel, iterations=1)
        
        # 然后进行膨胀操作，连接双层框
        dilate_kernel = np.ones((4, 4), np.uint8)
        binary = cv2.dilate(binary, dilate_kernel, iterations=3)
        
        # 查找轮廓
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        
        # 在原图上标记找到的方框
        result_img = img.copy()
        detected_regions = []
        
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 过滤太小的区域
            if area < self.min_area:
                continue
                
            # 获取轮廓的近似多边形
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # 如果是四边形
            if len(approx) >= 4:
                # 计算轮廓的凸包
                hull = cv2.convexHull(contour)
                # 计算凸包的面积
                hull_area = cv2.contourArea(hull)
                # 计算轮廓的凸性缺陷
                if hull_area > 0:
                    solidity = float(area) / hull_area
                    # 如果solidity接近1，说明轮廓接近凸形
                    if solidity > self.solidity_threshold:
                        # 获取边界框
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # 计算中心点
                        center_x = x + w/2
                        center_y = y + h/2
                        
                        # 检查是否在中心区域内
                        if (center_x_start <= center_x <= center_x_end and 
                            center_y_start <= center_y <= center_y_end):
                            # 计算长宽比
                            aspect_ratio = float(w) / h
                            
                            # 过滤不合适的长宽比
                            if (self.aspect_ratio_range[0] < aspect_ratio < 
                                self.aspect_ratio_range[1]):
                                # 保存检测到的区域信息
                                detected_regions.append({
                                    'x': x,
                                    'y': y,
                                    'width': w,
                                    'height': h,
                                    'area': area
                                })
        
        # 选择面积最大的区域
        if detected_regions:
            max_region = max(detected_regions, key=lambda x: x['area'])
            # 在原图上画出矩形
            cv2.rectangle(
                result_img, 
                (max_region['x'], max_region['y']), 
                (max_region['x'] + max_region['width'], 
                 max_region['y'] + max_region['height']), 
                (0, 255, 0), 2
            )
            return result_img, [max_region]
        else:
            return result_img, []

    def extract_regions(self, img_path: str, regions: List[Dict[str, Any]], 
                       output_path: str) -> None:
        """
        从原图中提取指定区域并保存
        
        参数:
            img_path: 原图路径
            regions: 检测到的区域列表
            output_path: 输出路径
        """
        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"无法读取图像: {img_path}")
            return
            
        for i, region in enumerate(regions):
            x, y, w, h = region['x'], region['y'], region['width'], region['height']
            # 从原图中提取区域
            screen_region = original_img[y:y+h, x:x+w]
            # 保存提取的区域
            cv2.imwrite(output_path, screen_region)

            print(f"  检测到区域:")
            print(f"  位置: ({x}, {y})")
            print(f"  尺寸: {w} x {h}")
            print(f"  面积: {region['area']}")
            print("  -------------------")
    
    def process_and_save(self, input_path: str, output_path: str) -> List[Dict[str, Any]]:
        """
        处理图像并保存结果
        
        参数:
            input_path: 输入图像路径
            output_path: 输出图像路径
            
        返回:
            检测到的区域列表
        """
        # 调用函数进行预处理
        result_image, detected_regions = self.preprocess_image(input_path)
        
        if result_image.size == 0:
            return []
            
        # 保存处理后的图像
        cv2.imwrite(output_path, result_image)
        
        # 提取并保存区域
        if detected_regions:
            self.extract_regions(input_path, detected_regions, output_path)
            
        return detected_regions

