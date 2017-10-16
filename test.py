from train import *
from os import listdir, path

samples_dir = './samples/'
data_test_dump = 'data_test.npz'

def test():
    conv_net = build_graph()
    saver = tf.train.Saver()

    detector = get_frontal_face_detector()
    data_f = None
    if path.isfile(data_dump):
        data_f = data_dump
    elif path.isfile(data_test_dump):
        data_f = data_test_dump
    if data_f is not None:
        data = np.load(data_f)    
        test_x, test_y = data['test_x'], data['test_y']
        test_y[..., 0] /= img_input_shape[1]
        test_y[..., 1] /= img_input_shape[0]
        test_wh = data['test_wh']
        dlib_err = data['dlib_err']
        names = data['test_names']
        msk300w = np.array([True if '300W' in n else False for n in names], dtype=np.bool)
    
    
    
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, './models/conv_net.ckpt')
        print("Model restored.")
        
        if data_f is not None:
            loss, err_arr = calc_score(sess, conv_net, test_x, test_y, test_wh)
            loss, err_300w = calc_score(sess, conv_net, test_x[msk300w], 
                                        test_y[msk300w], test_wh[msk300w])
            loss, err_menpo = calc_score(sess, conv_net, test_x[~msk300w], 
                                        test_y[~msk300w], test_wh[~msk300w])
            print('err 300W: %.6f' % np.mean(err_300w))
            print('err menpo: %.6f' % np.mean(err_menpo))
            areas = draw_CED(err_arr, err_300w, err_menpo, dlib_err,
                            labels=['300W+Menpo', '300W', 'Menpo', 'DLIB'])
            ar_norm = areas / 0.08
            print(areas)
            print(ar_norm)
        
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




