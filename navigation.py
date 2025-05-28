import math

import cv2
import numpy as np

from action import ActionControl
from myapriltag import AprilTagDetector


class AStarPlanner:
    def __init__(self, ox, oy, resolution, rr,turn_penalty=0.1):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)
        self.turn_penalty = turn_penalty

    class Node:
        def __init__(self, x, y, cost, parent_index,direction=None):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index
            self.direction = direction

        def __str__(self):
            return (
                str(self.x)
                + ","
                + str(self.y)
                + ","
                + str(self.cost)
                + ","
                + str(self.parent_index)
                + ","
                +str(self.direction)
            )
        

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """
        
        start_node = self.Node(
            self.calc_xy_index(sx, self.min_x),
            self.calc_xy_index(sy, self.min_y),
            0.0,
            -1,
            direction=None
        )
        goal_node = self.Node(
            self.calc_xy_index(gx, self.min_x),
            self.calc_xy_index(gy, self.min_y),
            0.0,
            -1,
            direction=None
        )

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                return None,None

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost
                + self.calc_heuristic(goal_node, open_set[o]),
            )
            current = open_set[c_id]

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, m in enumerate(self.motion):
                new_direction=(m[0],m[1])
                turn_cost=0
                if current.direction is not None and current.direction != new_direction:
                    turn_cost = self.turn_penalty
                node = self.Node(
                    current.x + self.motion[i][0],
                    current.y + self.motion[i][1],
                    current.cost + self.motion[i][2] + turn_cost,
                    c_id,
                    direction = new_direction
                )
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = (
            [self.calc_grid_position(goal_node.x, self.min_x)],
            [self.calc_grid_position(goal_node.y, self.min_y)],
        )
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return int(round((position - min_pos) / self.resolution))

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):
        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [
            [False for _ in range(int(self.y_width))] for _ in range(int(self.x_width))
        ]
        for ix in range(int(self.x_width)):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(int(self.y_width)):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],[1,1,math.sqrt(2)],[1,-1,math.sqrt(2)],[-1,-1,math.sqrt(2)],[-1,1,math.sqrt(2)]]

        return motion


class AprilTagLocalizer:
    """AprilTag定位器类"""

    def __init__(self, camera_matrix, dist_coeffs, tag_poses):
        """
        初始化定位器

        参数:
            camera_matrix: 相机内参矩阵
            dist_coeffs: 畸变系数
            tag_poses: AprilTag的3D位置字典
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.tag_poses = tag_poses
        self.detector = AprilTagDetector()
        self.actioncontroller = ActionControl()

    def calculate_pose(self, detections, margin_ratio=0.2):
        """根据检测到的AprilTag计算机器人位姿"""
        if not detections:
            return None, None, []

        objectPoints = []
        imagePoints = []
        detected_ids = []

        # 获取图像尺寸
        height, width = 1080, 1920  # 根据实际相机分辨率设置
        valid_area = [
            margin_ratio * width,
            margin_ratio * height,
            (1 - margin_ratio) * width,
            (1 - margin_ratio) * height,
        ]

        # 收集有效的特征点
        for detection in detections:
            tag_id = str(detection.tag_id)
            corners = detection.corners

            if tag_id in self.tag_poses:
                cx = (corners[0, 0] + corners[2, 0]) / 2
                cy = (corners[0, 1] + corners[2, 1]) / 2

                if (valid_area[0] <= cx <= valid_area[2]) and (
                    valid_area[1] <= cy <= valid_area[3]
                ):
                    objectPoints.append(self.tag_poses[tag_id])
                    for corner in corners:
                        imagePoints.append(corner)
                    detected_ids.append(tag_id)

        if not objectPoints:
            return None, None, []

        # 转换为numpy数组
        objectPoints = np.vstack(objectPoints).astype(np.float32)
        imagePoints = np.array(imagePoints).reshape(-1, 1, 2).astype(np.float32)

        # 求解PnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints.reshape(-1, 1, 3),
            imagePoints.reshape(-1, 1, 2),
            self.camera_matrix,
            self.dist_coeffs,
            # flags=cv2.SOLVEPNP_IPPE,
            confidence=0.99,
            reprojectionError=3.0,
        )

        if not success:
            return None, None, detected_ids

        # 计算位置和朝向
        R, _ = cv2.Rodrigues(rvec)
        pos = -np.linalg.inv(R) @ tvec
        direction = np.linalg.inv(R) @ (np.array([[0], [0], [1]]) - tvec)
        direction = (direction - pos)[:2]
        if pos[0]<7 or pos[1]<7 or pos[2]>45 or pos[2]<25 or pos[0]>287 or pos[1]>287:
            print("位置定位不合理")
            return None, None, detected_ids

        return pos.flatten(), direction, detected_ids

    def get_pose(
        self,
        region_detector=None,
        classifier=None,
        channel=None,
        flower_types=None,
        target=None,
        idx=0,
    ):
        """获取当前位姿,包含转头搜索，并可进行识别与通信"""
        # 正向拍照尝试定位
        img = self.actioncontroller.head_capture_for_detector()
        cv2.imwrite(f"nav_{idx+1}_front.jpg", img)

        # 识别与通信
        if region_detector and classifier and channel and flower_types and target:
            tagid_to_best = region_detector.process_multi_images(
                [f"nav_{idx+1}_front.jpg"]
            )
            try:
                from main import classify_and_communicate
            except ImportError:
                from .main import classify_and_communicate
            classify_and_communicate(
                tagid_to_best,
                classifier,
                channel,
                flower_types,
                target,
                idx,
                prefix="nav_front",
            )
        # 定位
        #img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        detections = self.detector.detect_tags(img)
        pos, direction, ids = self.calculate_pose(detections)
        if pos is not None:
            print(f"当前方向：{direction}")
            return pos, direction, ids

        # 左转头拍照
        self.actioncontroller.turn_head_left()
        img = self.actioncontroller.head_capture_for_detector()
        cv2.imwrite(f"nav_{idx+1}_left.jpg", img)
        # 识别与通信
        if region_detector and classifier and channel and flower_types and target:
            tagid_to_best = region_detector.process_multi_images(
                [f"nav_{idx+1}_left.jpg"]
            )
            try:
                from main import classify_and_communicate
            except ImportError:
                from .main import classify_and_communicate
            classify_and_communicate(
                tagid_to_best,
                classifier,
                channel,
                flower_types,
                target,
                idx,
                prefix="nav_left",
            )

        #img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        detections = self.detector.detect_tags(img)
        pos, direction, ids = self.calculate_pose(detections)
        self.actioncontroller.turn_head_back()
        print(f"左转头前方向{direction} pos{pos} ")
        if pos is not None:
            direction = self.inverse_rotation(
                np.array([0, 1]), np.array([-0.96, 0.29]), direction#-0.57 0.82
            )
            print(f"转头后方向：{direction} pos{pos}")
            return pos, direction, ids

        # 右转头拍照
        self.actioncontroller.turn_head_right()
        img = self.actioncontroller.head_capture_for_detector()
        cv2.imwrite(f"nav_{idx+1}_right.jpg", img)

        # 识别与通信
        if region_detector and classifier and channel and flower_types and target:
            tagid_to_best = region_detector.process_multi_images(
                [f"nav_{idx+1}_right.jpg"]
            )
            try:
                from main import classify_and_communicate
            except ImportError:
                from .main import classify_and_communicate
            classify_and_communicate(
                tagid_to_best,
                classifier,
                channel,
                flower_types,
                target,
                idx,
                prefix="nav_right",
            )
        #img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        detections = self.detector.detect_tags(img)
        pos, direction, ids = self.calculate_pose(detections)
        self.actioncontroller.turn_head_back()
        print(f"右转头前方向{direction} pos{pos} ")
        if pos is not None:
            direction = self.inverse_rotation(
                np.array([0, 1]), np.array([0.92, 0.38]), direction#0.59 0.81
            )
            print(f"转头后方向：{direction} pos{pos}")
            return pos, direction, ids

        return None, None, []

    @staticmethod
    def inverse_rotation(v1, v2, v_rotated):
        """计算方向向量的逆旋转"""
        dot_product = np.dot(v1, v2)
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        cos_theta = dot_product / (norm_v1 * norm_v2)
        sin_theta = cross_product / (norm_v1 * norm_v2)

        R_inv = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])

        return np.dot(R_inv, v_rotated)


class PathPlanner:
    """路径规划器类"""

    def __init__(self, grid_size=5.0, robot_radius=12.0):
        """
        初始化路径规划器

        参数:
            grid_size: 栅格大小(cm)
            robot_radius: 机器人半径(cm)
        """
        self.grid_size = grid_size
        self.robot_radius = robot_radius
        self.setup_obstacle_map()

    def setup_obstacle_map(self):
        """设置障碍物地图"""
        # 设置障碍物位置
        ox, oy = [], []
        # 添加柱子障碍物
        lr = np.array(
            [
                [196, 5],
                [116.5, 6.5],
                [14.5, 54.5],
                [69.5, 181],
                [92, 263],
                [154.5, 184],
                [235, 246.5],
                [235.5, 145.8],
                [158, 85],
            ]
        )

        # 为每个柱子添加四边形边界
        for pos in lr:
            width = height = 25
            # 添加四条边
            for i in range(int(width)):
                ox.extend(
                    [pos[0] + i, pos[0] + i, pos[0] + width - i, pos[0] + width - i]
                )
                oy.extend([pos[1], pos[1] + height, pos[1], pos[1] + height])
            for i in range(int(height)):
                ox.extend([pos[0], pos[0] + width, pos[0], pos[0] + width])
                oy.extend(
                    [pos[1] + i, pos[1] + i, pos[1] + height - i, pos[1] + height - i]
                )

        # 添加场地边界
        for x in range(0, 295):
            ox.extend([x, x])
            oy.extend([0, 294])
        for y in range(0, 295):
            ox.extend([0, 294])
            oy.extend([y, y])

        self.ox = ox
        self.oy = oy
        self.planner = AStarPlanner(ox, oy, self.grid_size, self.robot_radius)

    def plan(self, start_pos, goal_pos):
        """规划路径"""
        rx, ry = self.planner.planning(
            float(start_pos[0]),
            float(start_pos[1]),
            float(goal_pos[0]),
            float(goal_pos[1]),
        )
        if rx == None:
            return None,None
        return rx[::-1], ry[::-1]  # 反转使起点在前

    def get_direction_changes(self, rx, ry):
        """计算路径的方向变化点"""
        direction_changes = []
        if not rx or not ry:
            return direction_changes

        current_direction = None
        for i in range(1, len(rx)):
            dx = rx[i] - rx[i - 1]
            dy = ry[i] - ry[i - 1]

            direction = self.calculate_direction(dx, dy)
            if direction != current_direction:
                direction_changes.append((rx[i - 1], ry[i - 1], direction))
                current_direction = direction

        # 添加终点
        direction_changes.append((rx[-1], ry[-1], current_direction))
        return direction_changes

    def calculate_direction(self, dx, dy):
        """计算方向类型"""
        dx = round(dx / self.grid_size)
        dy = round(dy / self.grid_size)

        if dx == 0 and dy > 0:
            return "up"
        elif dx > 0 and dy == 0:
            return "right"
        elif dx == 0 and dy < 0:
            return "down"
        elif dx < 0 and dy == 0:
            return "left"
        elif dx > 0 and dy > 0:
            return "up-right"
        elif dx > 0 and dy < 0:
            return "down-right"
        elif dx < 0 and dy < 0:
            return "down-left"
        elif dx < 0 and dy > 0:
            return "up-left"
        return "unknown"


class RobotNavigator:
    """机器人导航器类"""

    def __init__(self, camera_matrix, dist_coeffs, tag_poses, goal_pos,
                 region_detector=None, classifier=None, channel=None, flower_types=None, target=None):
        self.localizer = AprilTagLocalizer(camera_matrix, dist_coeffs, tag_poses)
        self.planner = PathPlanner()
        self.controller = ActionControl()
        self.goal_pos = goal_pos
        self.direction_changes = []
        # 保存参数
        self.region_detector = region_detector
        self.classifier = classifier
        self.channel = channel
        self.flower_types = flower_types
        self.target = target

    def navigate(self):
        """执行导航任务"""
        idx = 0
        while True:
            pos, direction, detected_ids = self.localizer.get_pose(
                region_detector=self.region_detector,
                classifier=self.classifier,
                channel=self.channel,
                flower_types=self.flower_types,
                target=self.target,
                idx=idx
            )
            idx += 1

            if pos is None:
                print("定位失败")
                #self.controller.run_back()
                #self.controller.run_back()
                self.controller.run_str("turn004L")
                self.controller.run_str("turn004L")
                continue
            print(pos)
            # 2. 检查是否到达目标
            if (
                np.sqrt(
                    (pos[0] - self.goal_pos[0]) ** 2 + (pos[1] - self.goal_pos[1]) ** 2
                )
                <= 30
            ):
                print("到达目标点")
                break

            # 3. 检查是否需要重新规划
            if (
                not self.direction_changes
                or np.sqrt(
                    (pos[0] - self.direction_changes[0][0]) ** 2
                    + (pos[1] - self.direction_changes[0][1]) ** 2
                )
                > 15
            ):
                # 重新规划路径
                print("重新规划路径")
                rx, ry = self.planner.plan(pos, self.goal_pos)
                if rx == None:
                    self.controller.run_back()
                    self.controller.run_back()
                    continue
                print(f"路径{rx}")
                print(f"路径{ry}")
                if np.sqrt((rx[0] - pos[0]) ** 2 + (ry[0] - pos[1]) ** 2) >= 15:
                    rx.insert(0, round(pos[0] / 10) * 10)
                    ry.insert(0, round(pos[1] / 10) * 10)
                self.direction_changes = self.planner.get_direction_changes(rx, ry)

            # 4. 执行运动
            if self.direction_changes:
                current_point = self.direction_changes[0]
                target_direction = self.controller.directions[current_point[2]]
                print(f"当前位置{current_point} 方向{direction} 目标方向{target_direction}")
                # 转向
                self.controller.turn_to_direction(direction, target_direction)

                # 前进
                if len(self.direction_changes) > 1:
                    next_point = self.direction_changes[1]
                    distance = np.sqrt(
                        (next_point[0] - current_point[0]) ** 2
                        + (next_point[1] - current_point[1]) ** 2
                    )
                    self.controller.go_straight(distance)

                # 移除已完成的路径点
                self.direction_changes.pop(0)
