from regiondetector import RegionDetector
import os
def test_region_detector():
    # 本地已有的测试图片路径
    img_path = "../pic/lanhua/test_color_21.jpg"  # 请确保该图片在当前目录或填写绝对路径
    output_dir = "region_output"

    detector = RegionDetector()
    print("开始测试 preprocess_image ...")
    result_imgs, regions_list, tag_ids = detector.preprocess_image(img_path)

    print(f"检测到 {len(tag_ids)} 个tag")
    for idx, (result_img, regions, tag_id) in enumerate(zip(result_imgs, regions_list, tag_ids)):
        print(f"Tag {tag_id}: 区域数量 {len(regions)}")
        for region in regions:
            print(f"  区域信息: {region}")

        # 保存处理后图像
        result_img_path = f"{output_dir}/tag_{tag_id}_result.jpg"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        import cv2
        cv2.imwrite(result_img_path, result_img)
        print(f"处理后图像已保存: {result_img_path}")

        # 保存区域图像
        detector.extract_regions(img_path, regions, output_dir, prefix=f"tag_{tag_id}_")

if __name__ == "__main__":
    test_region_detector()