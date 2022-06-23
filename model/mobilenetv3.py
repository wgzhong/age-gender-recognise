import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
import yaml

class hswish(Model):
    def __init__(self):
        super(hswish, self).__init__()
        self.relu6 = ReLU(max_value=6.0)
    def call(self, x):
        out = x * self.relu6(x + 3) / 6
        return out

class hsigmoid(Model):
    def __init__(self):
        super(hsigmoid, self).__init__()
        self.relu6 = ReLU(max_value=6.0)
    def call(self, x):
        out = self.relu6(x + 3) / 6
        return out

# def hsigmoid(x):
#     return tf.nn.relu6(x + 3) / 6

# def hswish(x):
#     return x * hsigmoid(x)

class SeModule(Model):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = Sequential([
            GlobalAveragePooling2D(keepdims=True),
            Conv2D(in_size // reduction, kernel_size=1, strides=1, padding="valid", use_bias=False),
            BatchNormalization(),
            ReLU(),
            Conv2D(in_size, kernel_size=1, strides=1, padding="valid", use_bias=False),
            BatchNormalization(),
            hsigmoid()]
        )

    def call(self, x):
        return x * self.se(x)

class Block(Model):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = Conv2D(expand_size, kernel_size=1, strides=1, padding="valid", use_bias=False)
        self.bn1 = BatchNormalization()
        self.nolinear1 = nolinear
        self.pad1 = ZeroPadding2D(padding=kernel_size//2)
        self.conv2 = Conv2D(expand_size, kernel_size=kernel_size, strides=stride, padding="valid", groups=expand_size, use_bias=False)
        self.bn2 = BatchNormalization()
        self.nolinear2 = nolinear
        self.conv3 = Conv2D(out_size, kernel_size=1, strides=1, padding="valid", use_bias=False)
        self.bn3 = BatchNormalization()
        if stride == 1:
            self.shortcut = Sequential()
            self.shortcut.add(Conv2D(out_size, kernel_size=1, strides=1, padding="valid", use_bias=False))
            self.shortcut.add(BatchNormalization())

    def call(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(self.pad1(out))))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class mobilenetv3_small(Model):
    def __init__(self, cfg):
        super(mobilenetv3_small, self).__init__()
        self.pad1 = ZeroPadding2D(padding=1)
        self.conv1 = Conv2D(filters=16, kernel_size=3, strides=2, padding="valid", use_bias=False)
        self.bn1 = BatchNormalization()
        self.hs1 = hswish()

        self.bneck = Sequential([
            Block(3, 16, 16, 16, ReLU(), SeModule(16), 2),
            Block(3, 16, 72, 24, ReLU(), None, 2),
            Block(3, 24, 88, 24, ReLU(), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1)
            ]
        )

        self.conv2 = Conv2D(576, kernel_size=1, strides=1, padding="valid", use_bias=False)
        self.bn2 = BatchNormalization()
        self.hs2 = hswish()
        self.linear3 = Dense(1280, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.bn3 = BatchNormalization()
        self.hs3 = hswish()
        self.gender = Dense(cfg["train"]["gender_num"], kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.age = Dense(cfg["train"]["age_num"], kernel_initializer=tf.random_normal_initializer(stddev=0.01))

    def call(self, x):
        out = self.hs1(self.bn1(self.conv1(self.pad1(x))))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = AveragePooling2D((7,3))(out)
        out = self.hs3(self.bn3(self.linear3(out)))
        gender = self.gender(out)
        age = self.age(out)
        return (gender, age)


class mobilenetv3_large(Model):
    def __init__(self, cfg):
        super(mobilenetv3_large, self).__init__()
        self.conv1 = Conv2D(16, kernel_size=3, strides=2, padding="same", use_bias=False)
        self.bn1 = BatchNormalization(16)
        self.hs1 = hswish()

        self.bneck = Sequential(
            Block(3, 16, 16, 16, ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, ReLU(inplace=True), SeModule(40), 2),
            Block(5, 40, 120, 40, ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )


        self.conv2 = Conv2D(960, kernel_size=1, strides=1, padding="valid", use_bias=False)
        self.bn2 = BatchNormalization()
        self.hs2 = hswish()
        self.linear3 = Dense(1280, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.bn3 = BatchNormalization()
        self.hs3 = hswish()
        self.linear4 = Dense(cfg.num_classes, kernel_initializer=tf.random_normal_initializer(stddev=0.01))

    def call(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = AveragePooling2D(7)(out)
        out = Reshape([out.shape[-1]])(out)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out


def test():
    file = open("../config/config.yaml", 'r', encoding="utf-8")
    cfg = yaml.safe_load(file)  
    model = mobilenetv3_small(cfg)
    model.build(input_shape=(12,224,112,3))
    model.summary()
if __name__=="__main__":
    test()