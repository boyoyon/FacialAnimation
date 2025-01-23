import torch
from _animate import normalize_kp
from _demo import load_checkpoints
import numpy as np
import cv2, glob, os, sys

cpu = False if torch.cuda.is_available() else True

checkpoint_path = os.path.join(os.path.dirname(__file__), 'vox-cpk.pth.tar')

argv = sys.argv
argc = len(argv)

print('%s changing facial expression of the face within the image' % argv[0])
print('[usage] python %s <face image>' % argv[0])

if argc < 2:
    quit()

source_image = cv2.imread(argv[1])
source_image = cv2.resize(source_image, (256, 256))
source_image = source_image.astype(np.float32) / 255.0

generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml', checkpoint_path=checkpoint_path, cpu=cpu)

output_folder = 'result'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

relative=True
adapt_movement_scale=True

paths = glob.glob('driving_images/*.png')

with torch.no_grad() :
    predictions = []
    source = torch.tensor(source_image[np.newaxis]).permute(0, 3, 1, 2)
    
    if not cpu:
        source = source.cuda()
    kp_source = kp_detector(source)
    count = 1

    for path in paths:
        
        frame = cv2.imread(path)
        frame = cv2.flip(frame,1)
        frame = frame.astype(np.float32) / 255.0
            
        if count == 1:
            source_image1 = frame
            source1 = torch.tensor(source_image1[np.newaxis]).permute(0, 3, 1, 2)
            kp_driving_initial = kp_detector(source1)
        
        frame_test = torch.tensor(frame[np.newaxis]).permute(0, 3, 1, 2)

        driving_frame = frame_test
        
        if not cpu:
            driving_frame = driving_frame.cuda()
        
        kp_driving = kp_detector(driving_frame)
        kp_norm = normalize_kp(kp_source=kp_source,
                            kp_driving=kp_driving,
                            kp_driving_initial=kp_driving_initial, 
                            use_relative_movement=relative,
                            use_relative_jacobian=relative, 
                            adapt_movement_scale=adapt_movement_scale)
        
        out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
        
        im = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
        predictions.append(im)
        
        cv2.imshow('Facial Expression',im)

        dst_path = os.path.join(output_folder, '%04d.png' % count)
        dst = im * 255
        dst = np.clip(dst, 0, 255)
        dst = dst.astype(np.uint8)
        cv2.imwrite(dst_path, dst)
        print('save %s' % dst_path)

        count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()
