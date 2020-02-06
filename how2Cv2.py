import cv2
import numpy as np

im = np.zeros([512, 512, 3], dtype=np.uint8)

fgn = 'figure'

win = cv2.namedWindow(fgn)

def my_callback(event, x, y, flags, param):
    print('hello')
    print(event)
    print(x)
    print(y)
    print(flags)
    print(param)
    print('')

cv2.setMouseCallback(fgn, my_callback)

while(1):
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()