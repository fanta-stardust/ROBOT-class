"""
机器人导航和屏幕识别主程序

此程序实现了机器人导航到指定位置，识别屏幕，进行图像分类，然后与上位机通信的功能。
"""

# 导入必要的模块
import numpy as np
import os
import cv2
from PIL import Image
import time

# 导入自定义模块
from map2 import load_tag_pos
from navigation import RobotNavigator
from regiondetector import RegionDetector
from classfier import Classifier
from communication import Channel
from action import ActionControl


def main():
    """主函数，执行导航、识别、分类和通信功能"""
    
    # 相机参数
    camera_matrix = np.array([
        [187.4419, 0, 0],
        [0, 187.2186, 0],
        [327.5147, 230.3991, 1]
    ], dtype=np.float32).T
        
    dist_coeffs = np.array([0.1797, -0.1798, 0, 0], dtype=np.float32)
        
    # 加载AprilTag位置数据并设置目标位置
    tag_poses = load_tag_pos()
    goal_pos = np.array([90.0, 150.0])  # 目标点坐标
    
    # 初始化动作控制器
    action_controller = ActionControl()
    
    # 1. 导航阶段
    print("=== 开始导航阶段 ===")
    # 创建导航器并执行导航
    navigator = RobotNavigator(camera_matrix, dist_coeffs, tag_poses, goal_pos)
    navigator.navigate()
    print("=== 导航完成 ===")
    
    # 导航完成后，等待一段时间以使机器人稳定
    time.sleep(1)
    
    # 2. 屏幕识别阶段
    print("\n=== 开始屏幕识别阶段 ===")
    # 初始化区域检测器
    detector = RegionDetector()
    
    # 拍照
    image_path = 'screen.jpg'
    success = detector.capture_image(image_path)
    
    if not success:
        print("无法捕获图像，退出程序")
        return
    
    # 处理图像并识别屏幕区域
    result_img, detected_regions = detector.preprocess_image(image_path)
    
    if not detected_regions:
        print("未检测到屏幕区域，尝试再次拍照")
        # 转头尝试再次拍照
        action_controller.turn_head_left()
        success = detector.capture_image(image_path)
        action_controller.turn_head_back()
        if success:
            result_img, detected_regions = detector.preprocess_image(image_path)
        
        if not detected_regions:
            action_controller.turn_head_right()
            success = detector.capture_image(image_path)
            action_controller.turn_head_back()
            if success:
                result_img, detected_regions = detector.preprocess_image(image_path)
    
    if not detected_regions:
        print("多次尝试后仍未检测到屏幕区域，退出程序")
        return
    
    # 提取屏幕区域并保存
    region_path = 'screen_region.jpg'
    detector.extract_regions(image_path, detected_regions, region_path)
    print("屏幕区域已提取并保存至:", region_path)
    
    # 3. 分类阶段
    print("\n=== 开始分类阶段 ===")
    # 加载分类器
    classifier = Classifier('design_1.bit')
    
    # 打开提取的屏幕区域图像
    try:
        screen_img = Image.open(region_path).resize((28, 28))
    except Exception as e:
        print(f"无法打开屏幕区域图像: {e}")
        return
    
    # 执行分类
    class_result = classifier.wrap_classify(screen_img)
    
    if class_result is None:
        print("分类失败")
        return
    
    print(f"分类结果: {class_result}")
    
    # 将数字结果转换为花朵类型
    flower_types = ['bailianhua',
        'chuju',
        'hehua',
        'juhua',
        'lamei',
        'lanhua',
        'meiguihua',
        'shuixianhua',
        'taohua',
        'yinghua',
        'yuanweihua',
        'zijinghua']
    
    if class_result in flower_types:
        flower_type = flower_types[class_result]
        print(f"检测到的花朵类型: {flower_type}")
    else:
        print(f"未知分类结果: {class_result}")
        return
    
    # 4. 通信阶段
    print("\n=== 开始通信阶段 ===")
    # 初始化通信通道
    channel = Channel('10.0.0.10', 'team1', 'password1')  # 使用正确的IP和凭据
    
    try:
        # 初始化团队，获取目标
        target = channel.initialize_team()
        print(f"目标花朵类型: {target}")
        
        # 获取当前位置附近的AprilTag ID
        current_pos, _, detected_ids = navigator.localizer.get_pose()
        if not detected_ids:
            print("无法获取当前位置附近的AprilTag ID")
            return
            
        # 使用第一个检测到的AprilTag ID
        tag_id = int(detected_ids[0])
        print(f"使用AprilTag ID: {tag_id}")
        
        # 将花朵改变为目标类型
        score = channel.change_flower(tag_id, flower_type, target)
        
        if score is not None:
            print(f"成功改变花朵！分数: {score}")
        else:
            print("改变花朵失败")
    
    except Exception as e:
        print(f"通信过程出错: {e}")


if __name__ == "__main__":
    main()

