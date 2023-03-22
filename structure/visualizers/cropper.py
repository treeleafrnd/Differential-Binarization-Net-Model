import cv2
import numpy as np

def image_crop(original_image, cnt, i):
    [X, Y, W, H] = cv2.boundingRect(np.int0(cnt))
    print([X, Y, W, H])
    cropped_image = original_image[Y:Y + H, X:X + W]
    cv2.imwrite(f'./demo_results/crops/contour{i}.png', cropped_image)
