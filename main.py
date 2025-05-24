"""
机器人导航和屏幕识别主程序

此程序实现了机器人导航到指定位置，识别屏幕，进行图像分类，然后与上位机通信的功能。
"""

# 导入必要的模块
import os

import cv2
import numpy as np
from PIL import Image

from action import ActionControl
from classfier import Classifier
from communication import Channel

# 导入自定义模块
from map2 import load_tag_pos
from navigation import RobotNavigator
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
        try:
            score = channel.change_flower(tag_id, flower_type, target)
            if score is not None:
                print(f"成功改变花朵！分数: {score}")
            else:
                print("改变花朵失败")
        except Exception as e:
            print(f"通信过程出错: {e}，跳过该区域")


def main():
    """主函数，执行导航、识别、分类和通信功能"""

    # 相机参数
    camera_matrix = np.array(
        [[187.4419, 0, 0], [0, 187.2186, 0], [327.5147, 230.3991, 1]], dtype=np.float32
    ).T

    dist_coeffs = np.array([0.1797, -0.1798, 0, 0], dtype=np.float32)

    # 加载AprilTag位置数据
    tag_poses = load_tag_pos()
    # 巡航目标点列表
    goal_positions = [
        np.array([180.0, 50.0]),
        np.array([90, 50]),
        # 可以继续添加更多目标点
    ]

    # 初始化动作控制器
    action_controller = ActionControl()
    # 初始化通信通道（只初始化一次）
    channel = Channel("192.168.1.254", "team1", "password1")  # 使用正确的IP和凭据
    # 初始化团队，获取目标
    try:
        target = channel.initialize_team()
        print(f"目标花朵类型: {target}")
    except Exception as e:
        print(f"通信初始化出错: {e}")
        return

    # 分类器只需初始化一次
    classifier = Classifier("design_1.bit")
    # 区域检测器只需初始化一次
    detector = RegionDetector()

    flower_types = [
        "bailianhua",
        "chuju",
        "hehua",
        "juhua",
        "lamei",
        "lanhua",
        "meiguihua",
        "shuixianhua",
        "taohua",
        "yinghua",
        "yuanweihua",
        "zijinghua",
    ]

    for idx, goal_pos in enumerate(goal_positions):
        print(f"\n=== 巡航到第{idx + 1}个目标点: {goal_pos} ===")
        navigator = RobotNavigator(camera_matrix, dist_coeffs, tag_poses, goal_pos)
        navigator.navigate()
        print(f"=== 到达第{idx + 1}个目标点 ===")

        # 屏幕识别阶段
        print("\n=== 开始屏幕识别阶段 ===")
        image_paths = [
            f"screen_{idx + 1}_front.jpg",
            f"screen_{idx + 1}_left.jpg",
            f"screen_{idx + 1}_right.jpg",
        ]
        # 拍照顺序：正面、左、右
        detector.capture_image(image_paths[0])
        action_controller.turn_head_left()
        detector.capture_image(image_paths[1])
        action_controller.turn_head_back()
        action_controller.turn_head_right()
        detector.capture_image(image_paths[2])
        action_controller.turn_head_back()

        tagid_to_best = detector.process_multi_images(image_paths)
        # 处理完所有图像后，进行分类和通信
        classify_and_communicate(
            tagid_to_best, classifier, channel, flower_types, target, idx
        )


if __name__ == "__main__":
    main()
