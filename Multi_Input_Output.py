from keras.models import Model
from keras.layers import Lambda,Activation,Dense,Conv2D,Input,BatchNormalization,MaxPool2D,Flatten
import keras.backend as K
import numpy as np

input_tensor_1 = Input([32,32,3])  # keras里面的Input是定义shape，并不是真正的input
input_tensor_2 = Input([4,])
input_target = Input([2,])

# 网络结构
# 第一组
x = BatchNormalization(axis=1)(input_tensor_1)

x = Conv2D(filters=32, kernel_size=(3,3), padding='same')(x)
x = Activation('relu')(x)
x = MaxPool2D(pool_size=(2,2))(x)

x = Conv2D(filters=32, kernel_size=(3,3), padding='same')(x)
x = Activation('relu')(x)
x = MaxPool2D(pool_size=(2,2))(x)

x = Flatten()(x)
x = Dense(units=16)(x)
out2 = Dense(units=2)(x)

# 第二组
y = Dense(units=32)(input_tensor_2)
out1 = Dense(units=2)(y)

# 第三组
z = Dense(units=8)(input_target)
out3 = Dense(units=2)(z)


model = Model(input=[input_tensor_1,input_tensor_2,input_target], output=[out1,out2,out3])


# 画出model
from keras.utils.vis_utils import plot_model
plot_model(model=model, to_file='model_plain.png', show_shapes=True)
