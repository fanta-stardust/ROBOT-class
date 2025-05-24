import binascii
import numpy as np
import serial
import os
import cv2
import logging
import math

def run_action(cmd):
    ser = serial.Serial("/dev/ttyPS0", 9600, timeout=5)
    cnt_err = 0
    while 1:
        test_read = ser.read()
        #print('test_read', test_read)
        cnt_err += 1
        if test_read== b'\xa3' or cnt_err == 50:
            break
    
    if cnt_err == 50:
        print('can not get REQ')
    else:
        #print('read REQ finished!')
        ser.write(cmd2data(cmd))
        #print('send action ok!')
    ser.close()

def crc_calculate(package):
    crc = 0
    for hex_data in package:

        b2 = hex_data.to_bytes(1, byteorder='little')
        crc = binascii.crc_hqx(b2, crc)

    return [(crc >> 8), (crc & 255)]    # 校验位两位

def cmd2data(cmd):
    cnt=0
    cmd_list=[]
    for i in cmd:
        cnt+=1
        cmd_list+=[ord(i)]
    cmd_list=[0xff,0xff]+[(cnt+5)>>8,(cnt+5)&255]+[0x01,(cnt+1)&255,0x03]+cmd_list
    cmd_list=cmd_list+crc_calculate(cmd_list)
    return cmd_list

def wait_req():
    ser = serial.Serial("/dev/ttyPS0", 9600, timeout=5)
    while 1:
        test_read=ser.read()
        if test_read== b'\xa3' :
            #print('read REQ finished!') 
            break


class ActionControl:
    """
    Class representing the control of actions for a robot.
    """

    def __init__(self) -> None:
        self.logger = self.logger_init()

    def logger_init(self) -> logging.Logger:
        """
        Initialize the logger for logging actions.

        Returns:
            logging.Logger: The logger object.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(f"{__name__}.log", mode='w')
        formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def head_capture(self):
        """
        Capture an image from the head camera.

        Returns:
            Union[MatLike, None]: The captured image if successful, None otherwise.
        """
        headCam = cv2.VideoCapture(2)  #参数表示摄像头编号
        headCam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        headCam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, frame = headCam.read()
        headCam.release()
        if ret:
            return frame
        else:
            self.logger.error("cannot take a photo")
            return None
        
    def head_capture_for_detector(self):
        """
        屏幕识别使用高分辨率拍摄
        """
        headCam = cv2.VideoCapture(2)  #参数表示摄像头编号
        headCam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        headCam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        ret, frame = headCam.read()
        headCam.release()
        if ret:
            return frame
        else:
            self.logger.error("cannot take a photo")
            return None
        
        
    def run_str(self, command: str) -> None:
        """
        Run a command and log the action.

        Args:
            command (str): The command to run.
        """
        self.logger.debug("run command %s"%(command, ))
        run_action(command)
        wait_req()
        self.logger.debug("command done")

    def turn_head_left(self):
        self.run_str("HeadTurn140")

    def turn_head_right(self):
        self.run_str("HeadTurn060")

    def turn_head_back(self):
        self.run_str("HeadTurnMM")

    def run_back(self):
        self.run_str("Back2Run")

    def turn_to_direction(self, current_direction, target_direction):
        """转向到目标方向"""
        turn_direction, angle = self.calculate_turn(current_direction, target_direction)
        if angle >= 20:
            num = round(angle/20)
            cmd = 'turn004R' if turn_direction == "顺时针" else 'turn004L'
            for _ in range(num):
                self.run_str(cmd)

    def go_straight(self, distance):
        """直线行走指定距离"""
        if distance > 25:
            # 长距离使用快速前进
            num = int(np.floor(distance / 25))
            print(f"执行fastForward03 {num}次")
            for _ in range(num):
                    self.run_str("fastForward03")
        else:
            # 短距离使用普通前进
            num = int(np.floor(distance / 6))
            print(f"执行Forwalk02 {num}次")
            for _ in range(num):
                self.run_str("Forwalk02")
    
    @staticmethod
    def calculate_turn(current_direction, target_direction):
        """计算转向方向和角度"""
        current_angle = math.atan2(current_direction[1], current_direction[0])
        target_angle = math.atan2(target_direction[1], target_direction[0])
        
        turn_angle = math.degrees(target_angle - current_angle)
        
        # 调整到[-180,180]范围
        if turn_angle > 180:
            turn_angle -= 360
        elif turn_angle < -180:
            turn_angle += 360
            
        turn_direction = "逆时针" if turn_angle > 0 else "顺时针"
        return turn_direction, abs(turn_angle)

    directions = {
        "up": (0, 1),
        "right": (1, 0),
        "down": (0, -1),
        "left": (-1, 0),
        "up-right": (1, 1),
        "down-right": (1, -1),
        "down-left": (-1, -1),
        "up-left": (-1, 1)
    }
