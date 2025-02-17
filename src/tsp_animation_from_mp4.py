"""
・Thin-Plate-Spline-Motion-Model-mainフォルダに置いて実行する

python tsp_animation_from_mp4.py (画像ファイル)

"""

import cv2, os, sys, yaml
import numpy as np
import torch
from demo import relative_kp, load_checkpoints, make_animation, find_best_frame

video_path  = './assets/driving.mp4'
img_shape   = (256,256)
checkpoint  = './checkpoints/vox.pth.tar'
config      = './config/vox-256.yaml' 
mode        = 'relative'
output      = 'result'

def main():

    argv = sys.argv
    argc = len(argv)

    print('%s animate image using driving video' % argv[0])
    print('[usage] python %s <image>' % argv[0])

    if argc < 2:
        quit()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if not os.path.exists(output):
        os.mkdir(output)

    source_image = cv2.imread(argv[1])
    source_image = cv2.resize(source_image, img_shape)
    source_image = cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR)
    source_image = source_image.astype(np.float32) / 255.0

    cap = cv2.VideoCapture(video_path)
    driving_video = []
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, img_shape)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        driving_video.append(frame)
    
    cap.release()
    
    inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(config_path = config, checkpoint_path = checkpoint, device = device)

    predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = mode)
  
    no = 1
    for imgRGB in predictions:

        imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
        print(imgBGR.shape, imgBGR.dtype)
        cv2.imshow('result', imgBGR)
        dst_path = os.path.join(output, '%04d.png' % no)
        imgBGR = np.clip(imgBGR * 255, 0, 255)
        imgBGR = imgBGR.astype(np.uint8)
        cv2.imwrite(dst_path, imgBGR)
        no += 1
        cv2.waitKey(10)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

