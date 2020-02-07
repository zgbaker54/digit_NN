import cv2
import numpy as np

global status
status = {}
status['down'] = False

def my_callback(event, x, y, flags, param):
    
    global status
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
        status['down'] = True

    elif event == cv2.EVENT_LBUTTONUP:
        print(x,y)
        print('up')
        status['down'] = False
    
    if status['down']:
        print('down...')

im = np.zeros([512, 512, 3], dtype=np.uint8)

fgn = 'figure'

win = cv2.namedWindow(fgn)

cv2.setMouseCallback(fgn, my_callback)


def nothing(x):
    pass
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, fgn,0,1,nothing)

while(1):
#    cv2.imshow('im', im)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()