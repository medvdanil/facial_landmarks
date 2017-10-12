import numpy as np
from skimage.color import gray2rgb, rgb2gray
from skimage import img_as_float
import cv2
from os import path
from collections import namedtuple
from matplotlib import pyplot as plt
from skimage.transform import AffineTransform
from glob import iglob
from dlib import get_frontal_face_detector
from matplotlib import pyplot as plt

n_landmarks = 68
img_input_shape = 48, 48
rescale_k = 0.15

    
def imshow(image):
    fig, ax = plt.subplots(figsize=(5, 4))
    if len(image.shape) == 2:
        ax.imshow(image, plt.cm.Greys_r)
    else:
        ax.imshow(image)
    ax.axis('off')
    plt.show()


def batch_show(batch):
    shape = batch.shape[1:]
    batch = np.reshape(batch, (-1, *shape))
    n = int(np.ceil(np.sqrt(len(batch))))
    img = np.zeros((n * shape[0], n * shape[1], shape[2]), dtype=batch.dtype)
    for k in range(len(batch)):
        i, j = k // n, k % n
        img[i * shape[0]:(i+1) * shape[0], j * shape[1]:(j+1) * shape[1]] = batch[k]
    imshow(img)


def landmarks_show(image, points, color=[0, 0, 0]):
    img = image.copy()
    for p in points:
        r, c = int(p[1] + 0.5) - 1, int(p[0] + 0.5) - 1
        if 0 <= r < img_input_shape[0] and 0 <= c < img_input_shape[1]:
            img[r, c] = color
    imshow(img)
    
    
def draw_rect(image, roi_unbound, color=[0, 0, 0]):
    roi = max(0, roi_unbound[0]), max(0, roi_unbound[1]), \
          min(image.shape[1] - 1, roi_unbound[2]), \
          min(image.shape[0] - 1, roi_unbound[3]) 
    image[roi[1]:roi[3], roi[0]] = color
    image[roi[1]:roi[3], roi[2]] = color
    image[roi[1], roi[0]:roi[2]] = color
    image[roi[3], roi[0]:roi[2]] = color
    
class BatchGenerator:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.order = np.arange(len(self.data))
        self.offset = len(self.data)
    
    def next_batch(self, batch_size):
        assert(batch_size <= len(self.data))
        if self.offset + batch_size <= len(self.data):
            self.offset += batch_size
            return  self.data[self.order[self.offset - batch_size:self.offset]], \
                    self.labels[self.order[self.offset - batch_size:self.offset]]
        p = self.order[self.offset:]
        self.order = np.random.permutation(len(self.data))
        self.offset += batch_size - len(self.data)
        p = np.hstack((p, self.order[:self.offset]))
        return self.data[p], self.labels[p]


def load_data(data_dir, is_train=True):
    x = []
    y = []
    names = []
    
    detector = get_frontal_face_detector()
    patt = path.join(data_dir, '*', 'train' if is_train else 'test', '*')
    for img_file in iglob(patt):
        img = cv2.imread(img_file)
        if img is None:
            continue
        img = img[:,:,::-1] # bgr to rgb
        
        pts_file = img_file[:img_file.rfind('.')] + '.pts'
        f = open(pts_file, 'r')
        s = f.readline() # version: 1
        s = f.readline() # n_points: 68
        if s != 'n_points:  %d\n' % n_landmarks:
            print("Must be %d point in file" % n_landmarks)
            continue
        s = f.readline() # {
        points = []
        for _ in range(n_landmarks):
            points.append(list(map(float, f.readline().split())))
        points = np.array(points, dtype=np.float32)
        points -= 1 # left top corner must be 0, 0
        roi = np.concatenate((points.min(axis=0), points.max(axis=0) + 1)).astype(np.int32)
        f.close()
        
        dets, scores, idx = detector.run(img, 1, -1)
        
        best_d = -1
        best_iou = -1
        if is_train:
            #rects_img = img.copy()
            #draw_rect(rects_img, roi, color=[0, 0, 255])
            for i, d in enumerate(dets):
                if scores[i] < -0.5:
                    continue
                x0, y0 = d.left(), d.top()
                w, h = d.width(), d.height()
                x1, y1 = x0 + w, y0 + h
                #draw_rect(rects_img, (x0, y0, x1, y1))
                iroi = max(roi[0], x0), max(roi[1], y0), min(roi[2], x1), min(roi[3], y1)
                if not (iroi[0] < iroi[2] and iroi[1] < iroi[3]):
                    continue
                isect = (iroi[2] - iroi[0]) * (iroi[3] - iroi[1])
                union = w * h + (roi[2] - roi[0]) * (roi[3] - roi[1]) - isect
                iou = isect / union
                if iou < 0.3:
                    continue
                if iou > best_iou:
                    best_iou = iou
                    best_d = i
            print("%s best det: %f" % (img_file, scores[best_d]), 
                  dets[best_d], "iou %.3f" % best_iou)
        else:
            best_d = 0 # highest score
        if best_d < 0:
            print("Can't detect face on image %s" % img_file)
            continue
        d = dets[best_d] 
        x0, y0 = d.left(), d.top()
        w, h = d.width(), d.height()
        x1, y1 = x0 + w, y0 + h
        
        x0 = max(int(x0 - w * rescale_k + 0.5), 0)
        y0 = max(int(y0 - h * rescale_k + 0.5), 0)
        
        x1 = min(int(x1 + w * rescale_k + 0.5), img.shape[1])
        y1 = min(int(y1 + h * rescale_k + 0.5), img.shape[0])   
        img = img[y0:y1, x0:x1]
        img = cv2.resize(img, img_input_shape[::-1], cv2.INTER_CUBIC)
        
        if is_train:
            points[:, 0] = (points[:, 0] - x0) / (x1 - x0) * img_input_shape[1]
            points[:, 1] = (points[:, 1] - y0) / (y1 - y0) * img_input_shape[0]
            y.append(points)
        x.append(img)
        names.append(img_file)

    return np.stack(x), np.stack(y), np.array(names)
