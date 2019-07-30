import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# 基本图的w,h  Sample original input
Sample_raw_x = 128
Sample_raw_y = 128

rpn_stride = 8  # RPN8倍下采样  8 times downsampling

Feature_size_X = Sample_raw_x / rpn_stride  # feature map的size
Feature_size_Y = Sample_raw_y / rpn_stride

scales = [1,2,4]  # 锚框的尺寸
ratios = [0.5, 1, 2]  # 锚框的长宽比


def anchor(Feature_size_X, Feature_size_Y, rpn_stride, scales, ratios):
    # 组合尺寸和比例 scales ratio
    scales, ratios = np.meshgrid(scales, ratios)
    scales, ratios = scales.flatten(), ratios.flatten()

    # 计算anchor尺寸
    scalesX = scales * np.sqrt(ratios)  # 宽度
    scalesY = scales / np.sqrt(ratios)  # 长度

    # 因为anchor是在feature map上进行的
    # anchor 映射原图
    ShiftX = np.arange(0, Feature_size_X) * rpn_stride
    ShiftY = np.arange(0, Feature_size_Y) * rpn_stride

    # anchor point 在原图的位置
    ShiftX, ShiftY = np.meshgrid(ShiftX, ShiftY)  # x, y 是anchor的中心点

    # 每个anchor 点 上 需要有9个尺寸的anchor框
    centerX, anchorX = np.meshgrid(ShiftX, scalesX)  # 意思就是每一个anchor中心点x坐标对应9种宽度
    centerY, anchorY = np.meshgrid(ShiftY, scalesY)

    # stack 各种尺寸， 各种比例 对应各种长度
    anchor_center = np.stack([centerY, centerX], axis=2).reshape(-1, 2)
    anchor_size = np.stack([anchorY, anchorX], axis=2).reshape(-1, 2)

    # 左上 右下 的坐标点输出
    boxes = np.concatenate([anchor_center - 0.5 * anchor_size, anchor_center + 0.5 * anchor_size], axis=1)
    return boxes



anchors = anchor(Feature_size_X, Feature_size_Y, rpn_stride, scales, ratios)
print(anchors.shape)


plt.figure(figsize=(10,10))
image = Image.open('test.jpg')  # 128 128 3
plt.imshow(image)

asx = plt.gca()  # get current axs

for i in range(anchors.shape[0]):
    box = anchors[i]
    rec = patches.Rectangle((box[0],box[1]), box[2] - box[0], box[3] - box[1], edgecolor='r', facecolor='None')
    asx.add_patch(rec)
plt.show()