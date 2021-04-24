import numpy as np
import cv2
import glob
import tqdm

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 60.0, (640, 480))
images = glob.glob('./RESULTS/images/*')
with tqdm.tqdm(total=len(images)) as pbar:
    for idx in range(len(images)):
        a = 1
        if idx > 400:
            a = 2
        if idx % a == 0:
            frame = cv2.imread(images[idx])
            out.write(frame)
            pbar.update()

out.release()
cv2.destroyAllWindows()
