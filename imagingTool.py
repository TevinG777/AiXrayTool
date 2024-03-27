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
    # find the avevage xvalue of the image
    meanIndex = np.mean(image.ravel())

    # display graph of the image color range with mean
    
    plt.hist(image.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    plt.axvline(meanIndex, color='r', linestyle='dashed', linewidth=2)
    
    plt.show()

    return meanIndex




def main():
    inputDir = 'dcmHoldingFolder'

    file = os.path.join(inputDir, 'xray.dcm')

    ds = pydicom.dcmread(file)

    image = dcmToImage(ds)
    #cv2.imshow('image', image)
    #cv2.waitKey(0)
    thresh = colorRangeGraph(image)

    print(thresh)

    # use cv2 to create a color mask to only select pixels that are above the threshold
    mask = cv2.inRange(image, thresh, 1.0)

    # apply the mask to the image
    #result = cv2.bitwise_and(image, image, mask=mask)

    

    # display the result
    cv2.imshow('result', mask)
    cv2.waitKey(0)


main()

