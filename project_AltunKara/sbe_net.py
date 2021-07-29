from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten
from tensorflow.python.keras.models import Model
import tensorflow as tf

def loss1(y_true,y_pred):
  if(y_true == 0):
    return -tf.math.log(1-y_pred + 0.00001)
  else:#(y_true == 0)
    return -tf.math.log(y_pred + 0.00001)

def sbe_net(sz=(256,256, 2)):
  x = Input(sz)
  inp = x

  f = 32
  for i in range(0, 4):
    f *= 2
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = MaxPooling2D()(x)

  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = GlobalAveragePooling2D()(x)
  x = Flatten()(x)
  out = Dense(1, activation='sigmoid')(x)

  #model creation 
  model = Model(inputs=[inp], outputs=[out])
  model.compile(optimizer = 'Adam', loss=[loss1])

  return model
