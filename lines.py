import cv2
import numpy as np
from PIL import ImageGrab
from pyfirmata import Arduino, util
import serial
import time
import math
#########################
#'''
ser = serial.Serial()
ser.baudrate = 9600  # 设置波特率
ser.port = 'COM4'  # 端口是COM3
print(ser)
ser.open()  # 打开串口
print(ser.is_open)  # 检验串口是否打开
#'''
#########################
ju = cv2.imread("Resources/j.png", 0)
jw, jh = ju.shape[::-1]
po = cv2.imread("Resources/point1.png", 0)
pw, ph = po.shape[::-1]
imgs = cv2.imread("Resources/jump1.png.png", 0)
W, H = ju.shape[1], ju.shape[0]
threshold = 0.9
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
count = 0
start_t = time.time()


# 图像并排显示
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


# 处理图像
def processimg(img):
    global start_t
    W, H = img.shape[1], img.shape[0]
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
    # imgBlur = cv2.bilateralFilter(imgGray, 9, 75, 75)
    imgCanny = cv2.Canny(imgBlur, 10, 45)
    dilation = cv2.dilate(imgCanny, kernel)
    loc_ju = jumper(imgGray)
    if loc_ju is not None and 50<loc_ju[0]<W-50 and 50<loc_ju[1]<H-100:
        cutted, nts = llc(img, loc_ju, dilation, W)
        if nts is not None and (time.time()-start_t) > 4:
            start_t = time.time()
            tap(loc_ju, nts)
        cv2.imshow("sb", cutted)
    return dilation, imgGray


# 判断状态并获
def llc(img, loc_ju, dilation, W):
    cutted = dilation.copy()
    if loc_ju[0] < W // 2:
        # 棋子在屏幕左侧，取棋子坐标右侧的图
        cutted = cutted[200:loc_ju[1]+2, loc_ju[0]+jw//2:-110]
        rows, cols = cutted.shape
        points = np.array([[(0, rows), (0, rows - jh), (jw*4, rows), (0, rows)]])
        side = "l"
    else:
        cutted = cutted[200:loc_ju[1]+2, 100:loc_ju[0]-jw//2]
        rows, cols = cutted.shape
        points = np.array([[(cols, rows), (cols, rows - jh), (cols-jw*4, rows), (cols, rows)]])
        side = "r"
    cutted = roi_mask(cutted, points, 1)
    try:
        nexts = nextstep(img, cutted, side, loc_ju)
    except:
        return cutted, None
    return cutted, nexts


# 得下一个跳台的位置
def nextstep(img, cutted, side, loc_ju):
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgp = imggray.copy()
    rows, cols = imgp.shape
    points = np.array([[(0, 200), (0, rows), (cols, rows), (cols, 200)]])
    img_roi = roi_mask(imgp, points, 0)
    res = cv2.matchTemplate(img_roi, po, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val > 0.9:
        cv2.rectangle(img, max_loc, (max_loc[0] + pw, max_loc[1] + ph), (0, 0, 255), 2)
    else:
        crop_h, crop_w = cutted.shape
        center_x, center_y = 0, 0
        max_x = 0
        for y in range(crop_h):
            for x in range(crop_w):
                if cutted[y, x] == 255:
                    if center_x == 0:
                        if side == "l":
                            center_x = x + loc_ju[0] + jw//2
                        else:
                            center_x = x+100
                    if x > max_x:
                        center_y = y+200
                        max_x = x
        cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), -1)
        max_loc = np.array([center_x, center_y])
    return max_loc


# 获得小人的位置
def jumper(img):
    res = cv2.matchTemplate(img, ju, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    cv2.rectangle(img, max_loc, (max_loc[0] + jw, max_loc[1] + jh), (0, 0, 255), 2)
    cv2.circle(img, (max_loc[0] + jw // 2, max_loc[1] + jh), 2, (0, 255, 0), 10)
    # print(imgs[max_loc[0]-10][max_loc[1]-10])
    return np.array([max_loc[0] + jw // 2, max_loc[1] + jh])


# 获得轮廓坐标
def getContours(img, imgContour):

    countours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in countours:
        area = cv2.contourArea(cnt)
        if area > 800:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            (x, y, radius) = np.int0((x, y, radius))
            return [x, y, radius]


# roi掩膜
def roi_mask(img, corner_points, contrary):
    # 创建掩膜
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, corner_points, 255)
    if contrary:
        mask = cv2.bitwise_not(mask, mask)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


# 发送信息给Arduino
def tap(ju, nts):
    d = ju-nts
    dis = math.hypot(d[0],d[1])
    sec = 3.65*dis/1000
    ser.write(b"l")
    time.sleep(sec+0.16)
    ser.write(b"r")

# 主函数读图截屏
def main():
    time.sleep(2)
    while True:
        img = np.array(ImageGrab.grab(bbox=(72, 135, 610, 860)))
        imgp = img.copy()
        dis, result = processimg(imgp)
        stc = stackImages(0.3, [imgp, result])
        cv2.imshow('window', stc)
        if cv2.waitKey(25) & 0xff == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
