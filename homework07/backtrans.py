import numpy as np
from PIL import Image
from FFD import *
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  

def load_image(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    return image_array

def bilinear_interpolation(backmap: np.array, original_array: np.array) -> np.array:
    r"""
    Args:
        backmap: shape (height, width, 2), coordinates information
        original_array: shape (height, width), coordinates with intensity
    Return:
        trans_array: shape (height, width), backmap coordinates with intensity
    """
    height, width = original_array.shape
    trans_array = np.zeros((height, width))
    # shape (height, width)
    coords_x = backmap[:, :, 0]
    coords_y = backmap[:, :, 1]
    # shape (height, width)
    min_x = np.floor(coords_x).astype(int)
    min_y = np.floor(coords_y).astype(int)
    max_x = np.minimum(min_x + 1, height - 1)
    max_y = np.minimum(min_y + 1, width - 1)
    # shape (height, width)
    u = coords_x - min_x
    v = coords_y - min_y
    # shape (height, width)
    value00 = original_array[min_x, min_y]
    value01 = original_array[min_x, max_y]
    value10 = original_array[max_x, min_y]
    value11 = original_array[max_x, max_y]
    # shape (height, width)
    trans_array = (1-u)*(1-v)*value00 + (1-u)*v*value01 + u*(1-v)*value10 + u*v*value11

    return trans_array 

class ControlPointDragger:
    def __init__(self, array, m, n):
        self.array = array
        self.height, self.width = array.shape
        self.controlShiftDist, self.lx, self.ly = construct_controlShiftDist(self.height, self.width, m, n)
        self.dragged_point = None
        self.points = {}
        self.plot_to_key = {}
        self._setup_gui()
    
    def _setup_gui(self):
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
        self.ax.imshow(self.array, cmap='gray')
        self.ax.set_title('Drag control points and close window to finish', fontsize=12)
        self.ax.axis('off')
        
        for (i, j) in self.controlShiftDist.keys():
            initX, initY = i * self.lx, j * self.ly
            # ax.plot returns a list
            plot, = self.ax.plot(initY, initX, 'ro', markersize=5, picker=10)
            self.points[(i, j)] = {
                'plot': plot,
                'currentX': initX,
                'currentY': initY,
                'initialX': initX,
                'initialY': initY
            }
            self.plot_to_key[plot] = (i, j)
        
        plt.tight_layout()
        
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
    
    def on_pick(self, event):
        # event: click the control point, dragged_point is the current point
        if event.artist in self.plot_to_key:
            self.dragged_point = self.plot_to_key[event.artist]
    
    def on_motion(self, event):
        # examine if any control point has been clicked, also examine if the mouse is inside the image
        if self.dragged_point is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        
        # drag the point and clip
        newX = np.clip(event.ydata, 0, self.height - 1)
        newY = np.clip(event.xdata, 0, self.width - 1)
        
        p = self.points[self.dragged_point]
        # update the visual location of the point 
        p['plot'].set_data([newY], [newX])
        p['currentX'] = newX
        p['currentY'] = newY
        self.fig.canvas.draw()
    
    def on_release(self, event):
        self.dragged_point = None
    
    def get_controlShiftDist(self):
        for (i, j), p in self.points.items():
            dx = p['currentX'] - p['initialX']
            dy = p['currentY'] - p['initialY']
            self.controlShiftDist[(i, j)] = np.array([dx, dy])
        return self.controlShiftDist
    
    def show(self):
        plt.show()

def backtrans(array: np.array, m: int, n: int) -> np.array:
    r"""
    Back-Transform  

    Args:
        array: the array of original image
        m: number of intervals in height direction
        n: number of intervals in width direction
           choose m and n by which the size of image is divisible
    Return:
        transarray: the back-transformed array
    """
    dragger = ControlPointDragger(array, m, n)
    dragger.show()
    
    controlShiftDist = dragger.get_controlShiftDist()
    backmap = ffd(dragger.height, dragger.width, dragger.lx, dragger.ly, controlShiftDist)
    transarray = bilinear_interpolation(backmap, array)
    
    return transarray

if __name__ == "__main__":

    image_dir = "image"
    image_name = "putin.jpg"
    image_path = os.path.join(image_dir, image_name)

    array = load_image(image_path)
    print(f"Size of image {image_path}: {array.shape}")
    m = int(input("Enter m (number of intervals in height direction): "))
    n = int(input("Enter n (number of intervals in width direction): "))

    # back transformation
    transarray = backtrans(array, m, n)

    # comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(array, cmap='gray')
    axes[0].set_title('Original image')
    axes[0].axis('off')

    axes[1].imshow(transarray, cmap='gray')
    axes[1].set_title('Transformed image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, "FFDbacktrans.png"), bbox_inches='tight', dpi=150)
    plt.show()