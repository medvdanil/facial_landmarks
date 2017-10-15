import numpy as np
from skimage.color import gray2rgb, rgb2gray
from skimage import img_as_float, io
#import cv2
from os import path
from collections import namedtuple
from matplotlib import pyplot as plt
from skimage.transform import AffineTransform, warp
from glob import iglob
from dlib import get_frontal_face_detector, shape_predictor
from matplotlib import pyplot as plt
import joblib

n_landmarks = 68
img_input_shape = 64, 64
border_k = 0.15
jit_k = 3
jit_ranges = (0.2, 0.1, 0.1)
dlib_predictor_path = '/files/data/face/shape_predictor_68_face_landmarks.dat'
five_best_pts = [36, 45, 30, 48, 54]
    
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
    n = int(np.ceil(np.sqrt(len(batch))))
    img = np.zeros((n * shape[0], n * shape[1], shape[2]), dtype=batch.dtype)
    for k in range(len(batch)):
        i, j = k // n, k % n
        img[i * shape[0]:(i+1) * shape[0], j * shape[1]:(j+1) * shape[1]] = batch[k]
    imshow(img)


def draw_landmarks(img, points, color):
    for p in points:
        r, c = int(p[1] + 0.5) - 1, int(p[0] + 0.5) - 1
        if 0 <= r < img_input_shape[0] and 0 <= c < img_input_shape[1]:
            img[r, c] = color
    return img
    

def landmarks_show(image, points, color=[0, 0, 0]):
    img = image.copy()
    draw_landmarks(img, points, color)
    imshow(img)
    

def landmarks_batch_show(batch, points, color=[0, 0, 0]):
    tmp = batch.copy()
    for i, img in enumerate(tmp):
        draw_landmarks(img, points[i], color)
    batch_show(tmp)
    
    
def draw_rect(image, roi_unbound, color=[0, 0, 0]):
    roi = max(0, roi_unbound[0]), max(0, roi_unbound[1]), \
          min(image.shape[1] - 1, roi_unbound[2]), \
          min(image.shape[0] - 1, roi_unbound[3]) 
    image[roi[1]:roi[3], roi[0]] = color
    image[roi[1]:roi[3], roi[2]] = color
    image[roi[1], roi[0]:roi[2]] = color
    image[roi[3], roi[0]:roi[2]] = color


def jittering(img, p0, p1, points, is_train):
    jit_params = (np.random.rand(jit_k if is_train else 0, 
                        len(jit_ranges)) * 2 - 1) * jit_ranges
    jit_params = np.vstack((np.zeros(len(jit_ranges)), jit_params))
    imgs, labels = [], []
    fr_size = p1 - p0
    for ang, ofx, ofy in jit_params:
        t = AffineTransform(translation=-(p0 + p1) / 2) + \
            AffineTransform(rotation=ang) + \
            AffineTransform(translation=fr_size * (ofx, ofy)) + \
            AffineTransform(scale=img_input_shape[::-1] / fr_size) + \
            AffineTransform(translation=np.array(img_input_shape[::-1]) / 2.)
        imgs.append(warp(img, t._inv_matrix, output_shape=img_input_shape, 
                         order=3, mode='edge'))
        labels.append(t(points))
    #print(jit_params)
    #landmarks_batch_show(np.array(imgs), np.array(labels))
    return imgs, labels


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
    wh = []
    dlib_err = []
    
    detector = get_frontal_face_detector()
    predictor = shape_predictor(dlib_predictor_path)
    patt = path.join(data_dir, 'Menpo', 'train' if is_train else 'test', '*')
    
    for img_file in iglob(patt):
        try:
            #img = cv2.imread(img_file)
            img = io.imread(img_file)
        except (FileNotFoundError, OSError):
            img = None
        if img is None:
            continue
        if len(img.shape) == 2: # grayscale
            img = np.stack([img] * 3, axis=2)
            
        #img = img[:,:,::-1] # bgr to rgb
        
        pts_file = img_file[:img_file.rfind('.')] + '.pts'
        f = open(pts_file, 'r')
        s = f.readline() # version: 1
        s = f.readline() # n_points: 68
        if s.replace(' ', '') != 'n_points:%d\n' % n_landmarks:
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
        if best_d < 0:
            print("Can't detect face on image %s" % img_file)
            continue
        print("%s best det: %f" % (img_file, scores[best_d]), 
                dets[best_d], "iou %.3f" % best_iou)
        d = dets[best_d] 
        x0, y0 = d.left(), d.top()
        w, h = d.width(), d.height()
        x1, y1 = x0 + w, y0 + h
        
        x0 = max(int(x0 - w * border_k + 0.5), 0)
        y0 = max(int(y0 - h * border_k + 0.5), 0)
        
        x1 = min(int(x1 + w * border_k + 0.5), img.shape[1])
        y1 = min(int(y1 + h * border_k + 0.5), img.shape[0]) 
        """
        five_lm = points[five_best_pts].ravel()
        f_labels.write(img_file + ' %.3f' * 14 % ((x0, x1, y0, y1) + tuple(five_lm)) + '\n')
        f_labels.flush()
        """
        
        xj, yj = jittering(img, np.array((x0, y0)), np.array((x1, y1)), points, is_train)
        y += yj
        x += xj
        names += [img_file] * (1 + jit_k * int(is_train))
        wh += [(x1 - x0, y1 - y0)] * (1 + jit_k * int(is_train))
        
        
        
        if not is_train: # predict with dlib                
            shape = predictor(img, d)
            dpred = np.zeros(points.shape)
            for i, p in enumerate(shape.parts()):
                dpred[i] = p.x, p.y   
            
            err = (dpred - points) ** 2
            for line in np.hstack((dpred, points, err)):
                print(", ".join(map(lambda x: "%.3f" % x, line)))
            err = np.mean(np.sqrt(np.sum(err, axis=1)))
            err /= np.sqrt(w * h)
            dlib_err.append(err)
            print("err = %.5f" % err)
            
    joblib.dump((x, y, wh, names), "tmp2.dump", compress=9)
    s = 'train' if is_train else 'test'
    data = {s + '_x': np.stack(x), s + '_y': np.stack(y), 
            s + '_wh': np.array(wh), s + '_names': np.array(names)}
    if not is_train:
        data['dlib_err'] = dlib_err
    return data


def draw_CED(*args, err_max=0.08):
    for err_arr in args:
        x = np.zeros(len(err_arr) + 1)
        y = np.zeros(len(err_arr) + 1)
        x[1:] = np.sort(err_arr)
        y[1:] = np.arange(len(err_arr)) / len(err_arr)
        trh_i = np.searchsorted(x, err_max)
        plt.plot(x[:trh_i], y[:trh_i]) 
    plt.show()
