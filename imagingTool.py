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
    meanIndex = meanIndex - 0.7*np.std(image.ravel())
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

    coords= []
    
    # draw the contours on the image
    cv2.drawContours(image, oval_contours, -1, (0, 255, 0), 3)
    plt.imshow(image)
    plt.show()

    #pull all (x,y) sets from the contours
    for cnt in oval_contours:
        for point in cnt:
            x, y = point[0]
            coords.append((x, y))

    coords = np.array(coords)
    # remove values if they are within 1/32 of the max x value
    imageLength = image.shape[1]
    coords = coords[coords[:,0] < imageLength - imageLength/32]

    #TODO: Return the polygon giving the xy coordinates of the points at the corners of the array and then pass them to the TurbFinder and main method

    return 0

def TurbFinder(image, rectPoly):
    rectLocation = []

     # Convert the image to grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert the image to 8-bit
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Turn area inside the polygon black
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(rectPoly)], 255)

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
    file = os.path.join(inputDir, 'waxTE.dcm')

    # Convert the dicom file to an image
    ds = pydicom.dcmread(file)
    image = dcmToImage(ds)
    originalImage = image.copy()


    # Iterate the mask to refine the image
    for i in range(2):
        thresh = colorRangeGraph(image)
        mask = cv2.inRange(image, thresh, 1.0)
        image = cv2.bitwise_and(image, image, mask=mask)

    
    # Pass image into TE finder
    rectPoly = TEfinder(image)
    partsRect = TurbFinder(image, rectPoly)
    
    # convert colorspace so that rectangle can be overlayed on the image
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # create a polygon using the values returned from the TEfinderMethod
    #cv2.polylines(image, [np.array(rectPoly)], True, (0, 0, 255), 2)
    
    # print different parts of the image
    
    #print(rectPoly)

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

