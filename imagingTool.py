import pydicom
from pydicom import dcmread
import os
import numpy as np
import cv2
from PIL import Image
from skimage import exposure
import numpy as np
import matplotlib.pyplot as plt 


def dcmToImage(ds):
    image = ds.pixel_array

    # show the negative image
    image = 255 - image

    # Comress the image
    image = cv2.resize(image, (0,0), fx=0.3, fy=0.3)

    image = exposure.equalize_adapthist(image/np.max(image), clip_limit=0.03)
    return image

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def colorRangeGraph(image):
    # find the avevage xvalue of the image
    meanIndex = np.mean(image.ravel())

    # move the mean slightly to the left using standard deviation
    meanIndex = meanIndex - 0.35*np.std(image.ravel())
    # display graph of the image color range with mean
    
    #plt.hist(image.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    #plt.axvline(meanIndex, color='r', linestyle='dashed', linewidth=2)
    #plt.show()

    return meanIndex

def TEfinder(image):
    rectLocation = []

     # Convert the image to grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert the image to 8-bit
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # apply thresholding to the image
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # find the contours in the image
    countours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find ovals in the image
    oval_contours = []
    for cnt in countours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)  
        if len(approx) >= 5:  
            area = cv2.contourArea(cnt)  
            if area <200:  
                oval_contours.append(cnt)
                

    # Convert image to color
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    xList = [0]
    yList = [0]
    
    # draw the contours on the image
    cv2.drawContours(image, oval_contours, -1, (0, 255, 0), 3)
    plt.imshow(image)
    plt.show()

    # grab x, y values of the oval contours
    for cnt in oval_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        xList.append(x)
        yList.append(y)

    # strip outliers from the x, y values and any 0 values
    xList = np.array(xList)
    yList = np.array(yList)
    xList = reject_outliers(xList)
    yList = reject_outliers(yList)
    
    xList = xList[xList != 0]
    yList = yList[yList != 0]
    
    print(xList, yList)

    rectLocation.append(np.max(xList)) 
    rectLocation.append(np.min(xList))

    rectLocation.append(np.max(yList))
    rectLocation.append(np.min(yList))

    # display the image

    return rectLocation


def main():
    # Read the dicom file
    inputDir = 'dcmHoldingFolder'
    file = os.path.join(inputDir, 'xray1.dcm')

    # Convert the dicom file to an image
    ds = pydicom.dcmread(file)
    image = dcmToImage(ds)


    # Iterate the mask to refine the image
    for i in range(3):
        thresh = colorRangeGraph(image)
        mask = cv2.inRange(image, thresh, 1.0)
        image = cv2.bitwise_and(image, image, mask=mask)

    
    # Pass image into TE finder
    rect = TEfinder(image)
        
    xMax, xMin, yMax, yMin = rect
    
    print(xMax, xMin, yMax, yMin)
    
    # convert colorspace so that rectangle can be overlayed on the image
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # create a rectangle using dimensions and overlay it on the image
    cv2.rectangle(image, (xMin, yMin), (xMax, yMax), (255, 0, 0), 2)
    cv2.imshow('image', image)
    cv2.waitKey(0)

   
    



main()

