try:
    import Image
except ImportError:
    from PIL import Image
import cv2 as cv
import numpy as np
import tensorflow as tf
from PIL import ImageDraw
from image_files_reader import train_y
import matplotlib.pyplot as plt
from skimage.measure import block_reduce


T = 400 #number of iterations in bpb
n = 10 # number of points in bpb

#from Stackoverflow
def ccw_sort(p):
    p = np.array(p)
    mean = np.mean(p,axis=0)
    d = p-mean
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def mean_iou(yt0, yp0):
  inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
  union = tf.math.count_nonzero(tf.add(yt0, yp0))
  iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
  return iou


#boundary presserving block map creator
def bpbm_creator(images, sz=(256,256)):
  result_arr =[]
  counter = 0
  for SGT in images:
    edges = cv.Canny(SGT, 100, 200)
    edges[edges != 0] = 1

    nonzero_edges = np.nonzero(edges)

    contours = [ (nonzero_edges[1][i], nonzero_edges[0][i]) for i in range(len(nonzero_edges[1])) ]
    iou_best_image = None
    iou_best_image_tensor = None
    iou_best_val = tf.constant([0.0])
    for j in range(T):
      random_indices = np.random.randint(len(contours), size=n)
      points = np.zeros((n,2))
      for i in range(n):
        index = random_indices[i]
        points[i] = contours[index]

      points = ccw_sort(points)
      
      points = [tuple(point) for point in points]
      img = Image.new("RGB", sz, '#000')
      img1 = ImageDraw.Draw(img)
      img1.polygon(points, fill ="#fff", outline ="#fff") 

      y_pred = tf.keras.preprocessing.image.img_to_array(img)
      
      y_true = SGT
      y_true[y_true!=0] = 1
      y_pred[y_pred!=0] = 1
      iou_val = mean_iou(y_pred, y_true)

      if tf.math.greater(iou_val, iou_best_val):
        iou_best_image = img
        iou_best_val = iou_val
        iou_best_image_tensor = y_pred 
    result_arr.append(iou_best_image_tensor)
  return result_arr

created_images = bpbm_creator(train_y)

down_samples = [np.zeros((900, 256, 256, 1)), 
                np.zeros((900, 128, 128, 1)),
                np.zeros((900, 64, 64, 1)),
                np.zeros((900, 32, 32, 1)),
                np.zeros((900, 16, 16, 1)),
                np.zeros((900, 8, 8, 1)),
                np.zeros((900, 4, 4, 1)),
                np.zeros((900, 8, 8, 1)),
                np.zeros((900, 16, 16, 1)),
                np.zeros((900, 32, 32, 1)),
                np.zeros((900, 64, 64, 1)),
                np.zeros((900, 128, 128, 1))]

for j, created_image in enumerate(created_images):
  created_image = created_image[:,:,0]
  for i in range(6):
    down_samples[i][j,:,:,0] = created_image
    created_image = block_reduce(created_image, block_size=(2,2), func=np.max, cval=np.max(created_image))
    down_samples[11-i][j,:,:,0] = created_image
  down_samples[6][j,:,:,0] = created_image

all_MGT = np.array(created_images)[:,:,:,0:1]

