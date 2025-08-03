
import cv2

img = cv2.imread('/usr/share/backgrounds/NVIDIA_Logo.png')  # Or any existing image
cv2.imshow('Test Window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

