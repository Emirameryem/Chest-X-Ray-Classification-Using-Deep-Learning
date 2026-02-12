import os
import cv2
import numpy as np

print("OpenCV Version:", cv2.__version__)
try:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    success = cv2.imwrite("test_image.png", img)
    print("Write Success:", success)
    if success:
        read_img = cv2.imread("test_image.png")
        print("Read Shape:", read_img.shape if read_img is not None else "None")
        os.remove("test_image.png")
except Exception as e:
    print("Error:", e)
