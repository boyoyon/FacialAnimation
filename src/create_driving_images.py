import numpy as np
import sys, glob, os, cv2

LAST_FRAME_NO = 101

def usage(progName):
    print('%s extracts reference images from animate.gif' % progName)
    print('[usage] python %s' % progName)

def main():
    argv = sys.argv
    argc = len(argv)

    usage(argv[0])

    folder_name = 'driving_images'

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    cap = cv2.VideoCapture('animate.gif')

    frameNo  = 1
    ret = True
    while ret :
        
        ret, frame = cap.read()

        if not ret:
            continue

        if frameNo > LAST_FRAME_NO:
            break

        H, W = frame.shape[:2]
        left = W * 2 // 3
        right = W
        dst = frame[:, left:right, :]
        dst = cv2.resize(dst, (256, 256))
         
        dst_path = os.path.join(folder_name, '%04d.png' % frameNo)        
        cv2.imwrite(dst_path, dst)
        print('save %s' % dst_path)
        frameNo += 1

        key = cv2.waitKey(1)
        if key != -1:
            break

    cap.release()

if __name__ == '__main__':
    main()
