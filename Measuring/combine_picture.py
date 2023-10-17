# coding=utf-8
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


def jigsaw(imgs, gap=0):
    imgs = [Image.fromarray(img) for img in imgs]
    w, h = imgs[0].size
    result = Image.new(imgs[0].mode, ((w+gap)*5-gap, (h+gap)*5-gap))
    for i in range(5):
        for j in range(5):
            result.paste(imgs[i*5+j], box=((w+gap)*j, (h+gap)*i))
    return np.array(result)


if __name__ == '__main__':
    
    imgs = []
    example = 3
    for i in range(25):
        imge = cv2.imread("picture/probeTree_l{tom}-{jerry}.png".format(tom=i, jerry=example))
        imgs.append(imge)
        
    img = jigsaw(imgs)
    cv2.imwrite("picture/probeTree-{jerry}.png".format(jerry=example), img)
