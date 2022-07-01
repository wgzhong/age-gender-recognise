import cv2
import numpy as np

def data_resize(image, image_size=[256,128]):
    def calresizezone(img, radio, v, is_h=False):
        if is_h:
            v = int(radio*v)
            img=cv2.resize(img, (v, image_size[0]))
            return img
        else:
            v = int(radio*v)
            img=cv2.resize(img, (image_size[1], v))
            return img
    
    def fullpix(v, img_size):
        a1=a2=0
        if v<img_size:
            offset=img_size-v
            if offset%2>0:
                a1=int((offset-1)/2)
                a2=int((offset-1)/2)+1
            else:
                a1=a2=int(offset/2)
        return a1,a2

    (h,w,_) = image.shape
    if h > image_size[0] or w > image_size[1]:
        h_offset = h-image_size[0]
        w_offset = w-image_size[1]
        h_radio = image_size[0] / h
        w_radio = image_size[1] / w
        if h_offset>0 or w_offset>0:
            if h_radio > w_radio:
                image=calresizezone(image, w_radio, w, True)
            else:
                image=calresizezone(image, h_radio, h)
    (h,w,_) = image.shape
    h1,h2 = fullpix(h, image_size[0])
    w1,w2 = fullpix(w, image_size[1])
    image = cv2.copyMakeBorder(image, h1,h2,w1,w2,cv2.BORDER_CONSTANT,value=[0,0,0])
    image = image / 255.0
    return image

def cirte(logits):
    info=""
    gender = np.where(logits[:,0]>0.5, 1, 0)
    age = np.argmax(logits[0:,1:], axis=-1)
    if gender == 1:
        info = info + "female, "
    else:
        info = info + "male, "
    if age == 0:
        info = info + "AgeOver60"
    elif age == 1:
        info = info + "Age18-60"
    else:
        info = info + "AgeLess18"
    print(info)


