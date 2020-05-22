from segmentation_models import Unet, Nestnet, Xnet
from data.load_data import load_data
import numpy as np
from PIL import Image
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


NCLASSES = 2
HEIGHT = 544
WIDTH = 544


def generate_arrays_from_file(lines, batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for _ in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = Image.open("C:\\Users\\admin\\dongwei\\workspace\\dataset\\defeat_seg\\jpg" + '\\' + name)
            img = img.resize((WIDTH, HEIGHT))
            img = np.array(img)
            img = img / 255
            X_train.append(img)

            name = (lines[i].split(';')[1]).replace("\n", "")
            # 从文件中读取图像
            img = Image.open("C:\\Users\\admin\\dongwei\\workspace\\dataset\\defeat_seg\\png" + '\\' + name)
            img = img.resize((WIDTH, HEIGHT))
            img = np.array(img)
            seg_labels = np.zeros((WIDTH, HEIGHT, NCLASSES))
            for c in range(NCLASSES):
                seg_labels[:, :, c] = (img[:, :, 0] == c).astype(int)
            # seg_labels = np.reshape(seg_labels, (-1, NCLASSES))
            Y_train.append(seg_labels)

            # 读完一个周期后重新开始
            i = (i + 1) % n
        yield (np.array(X_train), np.array(Y_train))


# prepare data
dataset_path = 'C:\\Users\\admin\\dongwei\\workspace\\dataset\\defeat_seg'
# range in [0,1], the network expects input channels of 3
# x, y = load_data(root_dir=dataset_path, contents=['jpg', 'png'])

# prepare model
# build UNet++
model = Xnet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose', classes=NCLASSES)
# model = Unet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose') # build U-Net
# model = NestNet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose') # build DLA

model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])

# train model
# model.fit(x, y)
batch_size = 2
with open("C:\\Users\\admin\\dongwei\\workspace\\dataset\\defeat_seg\\train.txt", "r") as f:
    lines = f.readlines()
# 90%用于训练，10%用于估计。
num_val = int(len(lines) * 0.1)
num_train = len(lines) - num_val
log_dir = "logs/"
checkpoint_period = ModelCheckpoint(
    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    monitor='val_loss',
    save_weights_only=True,
    save_best_only=True,
    period=1)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1)
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=10,
    verbose=1)
model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                    steps_per_epoch=max(1, num_train // batch_size),
                    validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                    validation_steps=max(1, num_val // batch_size),
                    epochs=50,
                    initial_epoch=0,
                    callbacks=[checkpoint_period, reduce_lr, early_stopping])
