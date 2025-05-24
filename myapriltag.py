from typing import Any, Dict, List
import apriltag
import logging
import cv2
MatLike = Any

class AprilTagDetector:
    """
    Class for detecting AprilTags in images.
    """
    def __init__(self, tag_family: str = "tag36h11") -> None:
        """
        Initialize the AprilTagDetector.

        Args:
            tag_family (str): The family of AprilTags to detect.
        """
        self.logger = logging.getLogger(__name__)
        self.tag_family = tag_family
        # 创建 DetectorOptions 对象
        options = apriltag.DetectorOptions(families=self.tag_family)
        # 使用 options 初始化 Detector
        self.detector = apriltag.Detector(options)

    def detect_tags(self, image: MatLike) -> List[Dict[str, Any]]:
        """
        Detect AprilTags in an image.

        Args:
            image (MatLike): The image to detect AprilTags in.
        """
        # 将图像转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 使用 Detector 检测 AprilTags
        results = self.detector.detect(gray)
        return results
    def draw_tags(self, image: MatLike, tags: List[Any]) -> MatLike:
        """
        在图像上绘制AprilTag的边框和ID。

        Args:
            image (MatLike): 原始图像。
            tags (List[Any]): 检测到的tag对象列表。

        Returns:
            MatLike: 标注后的图像。
        """
        img_draw = image.copy()
        for tag in tags:
            corners = tag.corners.astype(int)  # apriltag.Detection对象的属性
            # 绘制四条边
            for i in range(4):
                pt1 = tuple(corners[i])
                pt2 = tuple(corners[(i+1)%4])
                cv2.line(img_draw, pt1, pt2, (0, 255, 0), 2)
        return img_draw