import cv2

import numpy as np




rect_r = 5
rect_h_num = 20
rect_w_num = 20
left_pad_w = 80
top_pad_h = 80
pitch = 20

im_height = 600
im_width = 600

img = np.zeros((im_height, im_width), dtype=np.uint8)
img[:] = 255

color = (0, 0, 0) # black

for j in range(rect_h_num):
    for i in range(rect_w_num):
        center = [i * pitch + left_pad_w, j * pitch + top_pad_h]
        begin = (center[0] - rect_r, center[1] - rect_r )
        end = (center[0] + rect_r, center[1] + rect_r )
        cv2.rectangle(img, begin, end, color, -1)
        
cv2.imwrite('pattern.png', img)