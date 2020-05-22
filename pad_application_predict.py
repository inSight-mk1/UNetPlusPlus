from segmentation_models import Unet, Nestnet, Xnet
from PIL import Image
import numpy as np
import random
import copy
import os

import time

random.seed(0)
class_colors = [[0, 0, 0], [0, 255, 0]]
NCLASSES = 2
HEIGHT = 544
WIDTH = 544

model = Xnet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose', classes=NCLASSES)
# model.load_weights("logs/ep010-loss1.375-val_loss0.657.h5 ")
model.load_weights('logs/ep011-loss0.007-val_loss0.010.h5')

imgs = os.listdir("./img")
print(imgs)

"""
def iou(y_true, y_pred, label: int):

    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(y_true, label), K.floatx())
    y_pred = K.cast(K.equal(y_pred, label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise,return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)

Ious = [[], []]
sess = tf.Session()

"""

for jpg in imgs:
    name = jpg.split(".")[0]
    img = Image.open("./img/" + jpg)
    old_img = copy.deepcopy(img)
    orininal_h = np.array(img).shape[0]
    orininal_w = np.array(img).shape[1]

    img = img.resize((WIDTH, HEIGHT))
    img = np.array(img)
    img = img/255
    img = img.reshape(-1, HEIGHT, WIDTH, 3)
    """
    png = Image.open("./png/"+name+".png")
    png = png.resize((WIDTH,HEIGHT))
    png = np.array(png)
    seg_labels = png[:, :, 0]
"""
    start_t = time.time()
    pr = model.predict(img)[0]
    pr = pr.reshape((HEIGHT, WIDTH, NCLASSES)).argmax(axis=-1)
    print(jpg, 'time: ', time.time() - start_t)
    """
    for i in range(NCLASSES):
        x= iou(seg_labels, pr, i)
        x= x.eval(session=sess)
        Ious[i].append(x)
"""
    seg_img = np.zeros((HEIGHT, WIDTH,3))
    colors = class_colors

    for c in range(NCLASSES):
        seg_img[:, :, 0] += ((pr[:, :] == c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c)*(colors[c][2])).astype('uint8')

    seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))

    image = Image.blend(old_img, seg_img, 0.3)
    image.save("./img_out/" + jpg)


#print(np.mean(Ious[0]))
#print(np.mean(Ious[1]))