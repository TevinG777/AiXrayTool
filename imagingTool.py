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
            if area <250:  
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

    xList = np.array(xList)
    yList = np.array(yList)

    # remove values if they are withing 1/32 of the max x value
    imageLength = image.shape[1]
    xList = xList[xList < 27/30*imageLength]
    xList = xList[xList > 1/10*imageLength]

    # strip outliers from the x, y values and any 0 values
    xList = reject_outliers(xList)
    yList = reject_outliers(yList)

    rectLocation.append(np.max(xList)) 
    rectLocation.append(np.min(xList))

    rectLocation.append(np.max(yList))
    rectLocation.append(np.min(yList))

    # display the image

    return rectLocation

def TurbFinder(image, rectPoly):
    rectLocation = []

     # Convert the image to grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert the image to 8-bit
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Turn the area inside of rectPoly to black
    xMax, xMin, yMax, yMin = rectPoly
    image[yMin:yMax, xMin:xMax] = 0

    

    # if pixel value is less than 50, turn it black
    #image[image < 135] = 0
    
    # convert image back to color
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # convert the image to 8-bit
    gray_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # erode and dilate the image
    kernel = np.ones((5, 5), np.uint8)
    gray_image = cv2.erode(gray_image, kernel, iterations=3)
    gray_image = cv2.dilate(gray_image, kernel, iterations=3)
    
    # if pixels around the image are less than 50, turn them black
    gray_image[gray_image < 120] = 0


    
    # find the contours in the grayscale image
    contours, _ = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # convert the image back to color
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    
    # display the image with contours
   #cv2.drawContours(gray_image, contours, -1, (0, 255, 0), 3)
   #cv2.imshow('grey', gray_image)
   #cv2.waitKey(0)

    # draw rectangles around each contour
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # if area is less than 1000, ignore it
        if w*h < 1000:
            continue
        rectLocation.append((x, y, w, h))

    return rectLocation

def removeIntersectingRects(rects):
    # remove intersecting rectangles
    i = 0
    while i < len(rects):
        j = i + 1
        while j < len(rects):
            x1, y1, w1, h1 = rects[i]
            x2, y2, w2, h2 = rects[j]
            if x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2:
                if w1 * h1 > w2 * h2:
                    rects.pop(j)
                else:
                    rects.pop(i)
                    i -= 1
                    break
            j += 1
        i += 1
    return rects
    

def main():
    # Read the dicom file
    inputDir = 'dcmHoldingFolder'
    file = os.path.join(inputDir, 'xrayTest.dcm')

    # Convert the dicom file to an image
    ds = pydicom.dcmread(file)
    image = dcmToImage(ds)
    originalImage = image.copy()


    # Iterate the mask to refine the image
    for i in range(3):
        thresh = colorRangeGraph(image)
        mask = cv2.inRange(image, thresh, 1.0)
        image = cv2.bitwise_and(image, image, mask=mask)

    
    # Pass image into TE finder
    rect = TEfinder(image)
    partsRect = TurbFinder(image, rect)

    print(partsRect)
    
    xMax, xMin, yMax, yMin = rect

    
    # convert colorspace so that rectangle can be overlayed on the image
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    

    # create a rectangle using dimensions and overlay it on the image
    cv2.rectangle(image, (xMin-10, yMin-10), (xMax+10, yMax+10), (0, 0, 255), 2)

    # remove intersecting rectangles
    partsRect = removeIntersectingRects(partsRect)
    print(partsRect)
    # create a rectangle using dimensions and overlay it on the image
    for part in partsRect:
        x, y, w, h = part
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    plt.imshow(image)
    plt.show()

    cv2.waitKey(0)
    
        
    
   
    



main()
