import tensorflow as tf
from tensorflow.python.keras.layers import Input, Conv2D, Concatenate, Multiply, Add, MaxPooling2D, Conv2DTranspose
from tensorflow.python.keras.models import Model
from sbe_net import sbe_net

from image_files_reader import train_x, train_y
import numpy as np

def boundary_map_loss_function(y_true, y_pred):
  res = -y_true * tf.math.log(y_pred) - (1-tf.math.log(y_pred)) * (1-y_true)
  return res

sbe_ = sbe_net()
def sbe_loss_true(y_true, y_pred):
  MGT = y_true[:,:,:,0:1]
  Spred = y_true[:,:,:,1:2]  sbe_.fit(x = x, y= np.array([tf.ones(1), tf.zeros(1)]),  epochs = 1, batch_size=1, validation_split=0.05, verbose = 0)

  SGT = y_pred
  real_inp  = Concatenate(axis=3)([MGT, SGT])
  fake_inp = Concatenate(axis=3)([MGT, Spred])
  x = Concatenate(axis = 0)([real_inp, fake_inp])
  val = sbe_.outputs[0]
  return -tf.math.log(val + 0.00001)

# model ve callback methods
# unet + boundary preserving block map
def unet_plus_bpbm_net(sz = ( 256, 256, 3)):
  x = Input(sz)
  inputs = x
  
  f = 8
  layers = []
  outs_arr = []
  for i in range(0, 6):
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2D(f, 3, activation='relu', padding='same') (x)

    out1 = Conv2D(f, 1, dilation_rate=1, padding='same', activation='relu')(x)
    out2 = Conv2D(f, 3, dilation_rate=1, padding='same', activation='relu')(x)
    out3 = Conv2D(f, 3, dilation_rate=2, padding='same', activation='relu')(x)
    out4 = Conv2D(f, 3, dilation_rate=4, padding='same', activation='relu')(x)
    out5 = Conv2D(f, 3, dilation_rate=6, padding='same', activation='relu')(x)

    temp = Concatenate()([out1, out2, out3, out4, out5])

    out  = Conv2D(1, 1, dilation_rate=1, padding='same', activation='sigmoid')(temp)
    outs_arr.append(out)
    temp = Multiply()([x, out])
    x = Add()([temp, x])
    layers.append(x)
    x = MaxPooling2D() (out)
    f = f*2
  ff2 = 64 


  #bottleneck 
  j = len(layers) - 1
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  
  out1 = Conv2D(f, 1, dilation_rate=1, padding='same', activation='relu')(x)
  out2 = Conv2D(f, 3, dilation_rate=1, padding='same', activation='relu')(x)
  out3 = Conv2D(f, 3, dilation_rate=2, padding='same', activation='relu')(x)
  out4 = Conv2D(f, 3, dilation_rate=4, padding='same', activation='relu')(x)
  out5 = Conv2D(f, 3, dilation_rate=6, padding='same', activation='relu')(x)

  temp = Concatenate()([out1, out2, out3, out4, out5])

  out  = Conv2D(1, 1, dilation_rate=1, padding='same', activation='sigmoid')(temp)
  outs_arr.append(out)
  temp = Multiply()([x, out])
  x = Add()([temp, x])
  x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
  x = Concatenate(axis=3)([x, layers[j]])
  j = j -1 

  #upsampling 
  for i in range(0, 5):
    ff2 = ff2//2
    f = f // 2 
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    out1 = Conv2D(f, 1, dilation_rate=1, padding='same', activation='relu')(x)
    out2 = Conv2D(f, 3, dilation_rate=1, padding='same', activation='relu')(x)
    out3 = Conv2D(f, 3, dilation_rate=2, padding='same', activation='relu')(x)
    out4 = Conv2D(f, 3, dilation_rate=4, padding='same', activation='relu')(x)
    out5 = Conv2D(f, 3, dilation_rate=6, padding='same', activation='relu')(x)

    temp = Concatenate()([out1, out2, out3, out4, out5])

    out  = Conv2D(1, 1, dilation_rate=1, padding='same', activation='sigmoid')(temp)
    outs_arr.append(out)
    temp = Multiply()([x, out])
    x = Add()([temp, x])
    x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
    x = Concatenate(axis=3)([x, layers[j]])
    j = j -1 

  #classification 
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  outputs = Conv2D(1, 1, activation='sigmoid') (x)
  outs_arr.append(outputs)
  outs_arr.append(outputs)
  # sbe_inp1 = Concatenate()([outputs, tf.convert_to_tensor(train_y, tf.float32)])
  # sbe_ = sbe_net(sbe_inp1)

  # outs_arr.append(sbe_)

  # sbe_inp2 = np.concatenate([all_MGT, train_y], axis = 3)
  # sbe_ = sbe_net(sbe_inp2)

  #model creation 
  model = Model(inputs=[inputs], outputs=outs_arr)
  model.compile(optimizer = 'Adam', loss = [boundary_map_loss_function] * 12 + ['binary_crossentropy'] + [sbe_loss_true])

  return model

def build_callbacks():
  checkpointer = ModelCheckpoint(filepath='unet.h5', verbose=0, save_best_only=True, save_weights_only=True)
  callbacks = [checkpointer, PlotLearning()]
  return callbacks

# inheritance for training process plot 
class PlotLearning(tf.keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.i = 0
    self.x = []
    self.losses = []
    self.val_losses = []
    self.acc = []
    self.val_acc = []
    #self.fig = plt.figure()
    self.logs = []
  def on_epoch_end(self, epoch, logs={}):
    print(logs)
