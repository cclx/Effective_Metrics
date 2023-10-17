# coding=utf-8
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


def jigsaw(imgs, gap=0):
    imgs = [Image.fromarray(img) for img in imgs]
    #print(imgs)
    w, h = imgs[0].size
    result = Image.new(imgs[0].mode, ((w+gap)*5-gap, (h+gap)*2-gap))
    for i in range(2):
        for j in range(5):
            result.paste(imgs[i*5+j], box=((w+gap)*j, (h+gap)*i))
    return np.array(result)


if __name__ == '__main__':
    
    imgs = []
    #example = 3
    for i in range(10):
        imge = cv2.imread("picture/Evaluated_probe_cola_unfreeze-seed{:}.png".format(1+4*i))
        #print(imge)
        imgs.append(imge)   
    img = jigsaw(imgs)
    cv2.imwrite("picture/unfreeze-probe_cola.png", img)
