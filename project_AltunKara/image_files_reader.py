import numpy as np
try:
    import Image
except ImportError:
    from PIL import Image
import os

isbi2016_ground_truth_path = "../../501/ISBI2016_ISIC_Part1_Training_GroundTruth/"
isbi2016_training_path = "../../501/ISBI2016_ISIC_Part1_Training_Data/"


training_files = os.listdir(isbi2016_ground_truth_path)[20:30]

def image_files_reader(sz = (256, 256)):
  files = np.array(training_files)
  #files.resize((int(len(files) / batch_size), batch_size)) 
  x_files = []
  y_files = []

  for f in files:
    #get the masks. Note that masks are png files 
    mask = Image.open(isbi2016_ground_truth_path+ f )
    
    # mask = np.array(mask)
    mask = np.array(mask.resize(sz))
    # print(mask[mask>0])
    y_files.append(mask)

    #preprocess the raw images 
    raw = Image.open(isbi2016_training_path + f'{f[:-17]}.jpg')
    # print(isbi2016_training_path + f'{f[:-17]}.jpg')
    raw = raw.resize(sz)
    raw = np.array(raw)
    #check the number of channels because some of the images are RGBA or GRAY
    if len(raw.shape) == 2:
      raw = np.stack((raw,)*3, axis=-1)
    else:
      raw = raw[:,:,0:3]
    x_files.append(raw)

  #preprocess a batch of images and masks 
  x_files = np.array(x_files)/255.
  y_files = np.array(y_files)
  y_files = np.expand_dims(y_files,3)
  return (x_files, y_files)

(train_x, train_y) = image_files_reader()

