import pydicom
from pydicom import dcmread
import os
import numpy as np
import cv2
from PIL import Image
from skimage import exposure
import matplotlib.pyplot as plt 


def dcmToImage(ds):
    image = ds.pixel_array

    # show the negative image
    image = 255 - image

    # Comress the image
    image = cv2.resize(image, (0,0), fx=0.3, fy=0.3)

    image = exposure.equalize_adapthist(image/np.max(image), clip_limit=0.03)
    return image

def colorRangeGraph(image):
    # add all of the y values together
    y = np.sum(image, axis=0)
    
    # add total amount of x value entries
    x = len(y)

    # sum all y values
    ySum = np.sum(y)

    mean = ySum / x

    # display graph of the image color range with mean
    
    plt.hist(image.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    plt.axvline(mean, color='r', linestyle='dashed', linewidth=2)
    
    plt.show()




def main():
    inputDir = 'dcmHoldingFolder'

    file = os.path.join(inputDir, 'xray.dcm')

    ds = pydicom.dcmread(file)

    image = dcmToImage(ds)
    #cv2.imshow('image', image)
    #cv2.waitKey(0)
    colorRangeGraph(image)

main()

