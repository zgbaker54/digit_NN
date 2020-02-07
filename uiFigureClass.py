import cv2
import numpy as np

class UIFigure:
    
    figure_ctr = 0
    
    
    def __init__(self):
        
        UIFigure.figure_ctr += 1
        
        self.fig_id = 'figure' + str(UIFigure.figure_ctr)
        self.fig = cv2.namedWindow(self.fig_id)
        cv2.setMouseCallback(self.fig_id, self.mouse_callback)
        self.reset_mouse()
        
        self.im = np.zeros([512, 512, 3], dtype=np.uint8)
        cv2.imshow(self.fig_id, self.im)
        
        self.pen_size = 10
    
    
    def mouse_callback(self, event, x, y, flags, param):
        self.mouse['loc'] = [x, y]
        # left mouse click is initiated
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse['l_down'] = [x, y]
            self.mouse['l_click'] = True
        # left mouse click is released
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse['l_up'] = [x, y]
            self.mouse['l_click'] = False
            self.resolve_mouse_click()
            self.reset_mouse()
        # left mouse click is ongoing
        elif self.mouse['l_click'] == True:
            self.mouse['l_drag'].append([x, y])
            self.resolve_mouse_click()
    
    def reset_mouse(self):
        # information regarding mouse clicks and movement
        self.mouse = {}
        self.mouse['loc'] = None # [x, y] location of the mouse at the callback
        self.mouse['l_down'] = None # [x, y] location where left mouse click is initiated
        self.mouse['l_up'] = None # [x, y] location where left mouse click is released
        self.mouse['l_click'] = False # bool noting if mouse is currently left-clicked
        self.mouse['l_drag'] = [] # list of [x, y] locations where ouse travels during a left click
    
    
    def resolve_mouse_click(self):
        # after the mouse is clicked down, possibly moved, and released; the
        # click must be resolved
        
        # get pen indices
        idxs = []
        idxs.append(self.mouse['loc'][1] - self.pen_size)
        idxs.append(self.mouse['loc'][1] + self.pen_size)
        idxs.append(self.mouse['loc'][0] - self.pen_size)
        idxs.append(self.mouse['loc'][0] + self.pen_size)
        # draw
        self.im[idxs[0]:idxs[1], idxs[2]:idxs[3], :] = 255
        cv2.imshow(self.fig_id, self.im)
#        print('resolving!')
#        print(self.mouse)
    
    
    def run(self):
        print('Running ' + self.fig_id + '...')
        while(1):
            if cv2.waitKey(0) & (cv2.getWindowProperty(self.fig_id, 0) < 0):
                break
        print('Finished running ' + self.fig_id)


## test classes
#myFig = UIFigure()
#myFig.run()