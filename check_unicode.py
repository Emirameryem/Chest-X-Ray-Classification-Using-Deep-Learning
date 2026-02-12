import os
import cv2
import numpy as np

# Test directory with unciode characters
test_dir = "test_ümîr_öçşğü"
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

print(f"Testing write to: {test_dir}")

try:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Try writing
    fname = os.path.join(test_dir, "test.png")
    success = cv2.imwrite(fname, img)
    print(f"Write Success to {fname}: {success}")
    
    # Try reading
    if success:
        read_img = cv2.imread(fname)
        print("Read Shape:", read_img.shape if read_img is not None else "None")
except Exception as e:
    print("Error:", e)
