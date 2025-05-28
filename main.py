"""
机器人导航和屏幕识别主程序

此程序实现了机器人导航到指定位置，识别屏幕，进行图像分类，然后与上位机通信的功能。
"""

# 导入必要的模块
import os

import cv2
import numpy as np
from action import ActionControl
from classfier import Classifier
from communication import Channel

# 导入自定义模块
from map2 import load_tag_pos
from navigation import RobotNavigator
from PIL import Image
from regiondetector import RegionDetector

def classify_and_communicate(
    tagid_to_best, classifier, channel, flower_types, target, idx, prefix="nav"
):
    for tag_id, data in tagid_to_best.items():
        regions = data["regions"]
        if not regions or regions[0]["width"] == 0 or regions[0]["height"] == 0:
            continue
        region = regions[0]
        region_img_path = f"{prefix}_region_{idx + 1}_tag{tag_id}.jpg"
        x, y, w, h = region["x"], region["y"], region["width"], region["height"]
        roi_img = data["result_img"][y : y + h, x : x + w]
        cv2.imwrite(region_img_path, roi_img)
        print(f"屏幕区域已提取并保存至: {region_img_path}")

        # 分类阶段
        print("\n=== 开始分类阶段 ===")
        try:
            from PIL import Image

            screen_img = Image.open(region_img_path).resize((28, 28))
        except Exception as e:
            print(f"无法打开屏幕区域图像: {e}，跳过该区域")
            continue

        class_result = classifier.wrap_classify(screen_img)
        if class_result is None:
            print("分类失败，跳过该区域")
            continue

        print(f"分类结果: {class_result}")
        if 0 <= class_result < len(flower_types):
            flower_type = flower_types[class_result]
            print(f"检测到的花朵类型: {flower_type}")
        else:
            print(f"未知分类结果: {class_result}，跳过该区域")
            continue

        # 通信阶段
        print("\n=== 开始通信阶段 ===")
        if target == flower_type:
            print("已经是目标花，跳过通信")
            continue
        try:
            score = channel.change_flower(tag_id, flower_type, target)
            if score is not None:
                print(f"成功改变花朵！分数: {score}")
            else:
                print("改变花朵失败")
        except Exception as e:
            print(f"通信过程出错: {e}，跳过该区域")

