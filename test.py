from train import *
from os import listdir

samples_dir = './samples/'

def test():
    conv_net = build_graph()
    saver = tf.train.Saver()
     
    detector = get_frontal_face_detector()
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, './models/conv_net.ckpt')
        print("Model restored.")
        
        for img_file in listdir(samples_dir):
            try:
                #img = cv2.imread(img_file)
                img = io.imread(path.join(samples_dir, img_file))
            except (FileNotFoundError, OSError):
                img = None
            if img is None:
                continue
            if len(img.shape) == 2: # grayscale
                img = np.stack([img] * 3, axis=2)
            print("Image %s" % img_file)
            dets, scores, idx = detector.run(img, 1, -1)
             
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.imshow(img)
            ax.axis('off')
                
            for i, d in enumerate(dets):
                if scores[i] < -0.1:
                    continue
                x0, y0 = d.left(), d.top()
                w, h = d.width(), d.height()
                x1, y1 = x0 + w, y0 + h
                
                x0 = max(int(x0 - w * border_k + 0.5), 0)
                y0 = max(int(y0 - h * border_k + 0.5), 0)
                
                x1 = min(int(x1 + w * border_k + 0.5), img.shape[1])
                y1 = min(int(y1 + h * border_k + 0.5), img.shape[0])   
                xj, yj = jittering(img, np.array((x0, y0)), np.array((x1, y1)), np.zeros((n_landmarks, 2)), False)
                xj = np.array(xj)
                loss, pred = sess.run([conv_net['cost'], conv_net['pred']], feed_dict={
                    conv_net['x']: xj, conv_net['y']: yj, conv_net['keep_prob']: 1.0})
                points = pred[0] * (x1 - x0, y1 - y0) + (x0, y0)
                ax.scatter(points[:,0], points[:, 1], marker='o', s=2)
                
            plt.show()
if __name__ == '__main__':
    test()




