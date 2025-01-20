import sys, cv2, glob

def main():

    argv = sys.argv
    argc = len(argv)

    if argc < 2:
        print('%s creates video from images' % argv[0])
        print('%s <wildcard for images>' % argv[0])
        quit()

    paths = glob.glob(argv[1])
    nrData = len(paths)

    src = cv2.imread(paths[0])

    height, width = src.shape[:2]

    # encoder(for mp4)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # output file name, encoder, fps, size(fit to image size)
    video = cv2.VideoWriter('output.mp4',fourcc, 20.0, (width, height))

    if not video.isOpened():
        print("can't be opened")
        sys.exit()

    for i, path in enumerate(paths):

        print('processing %d/%d: %s' % ((i+1), nrData, path))

        img = cv2.imread(path)

        # can't read image, escape
        if img is None:
            print("can't read")
            break

        # add
        video.write(img)

    video.release()
    print('save output.mp4')

if __name__ == '__main__':
    main()
