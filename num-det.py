import cv2
import numpy as np

def search_number(image):
    gauss_blur = cv2.GaussianBlur(image,(5,5),0)
    median_blur = cv2.medianBlur(image,5)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_gauss=cv2.cvtColor(gauss_blur,cv2.COLOR_BGR2GRAY)
    gray_median=cv2.cvtColor(median_blur,cv2.COLOR_BGR2GRAY)
    minVal=50
    maxVal=100
    canny=cv2.Canny(gray,minVal,maxVal,True)
    canny_gauss=cv2.Canny(gray_gauss,minVal,maxVal,True)
    canny_median=cv2.Canny(gray_median,minVal,maxVal,True)
    
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('image', 900,600)
    #cv2.imshow('image',canny)
    #cv2.waitKey()
    #
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('image', 900,600)
    #cv2.imshow('image',canny_gauss)
    #cv2.waitKey()
    #
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('image', 900,600)
    #cv2.imshow('image',canny_median)
    #cv2.waitKey()
    
    contours=cv2.findContours(canny,cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours[1]))

    if len(contours[1])>1:
        for contorno in contours[1]:
            epsilon = 0.1*cv2.arcLength(contorno,True)
            approx = cv2.approxPolyDP(contorno,epsilon,True)
            #print(len(approx))
            area=cv2.contourArea(contorno)
            if area>50:
                copy=np.full(image.shape,255,dtype=np.uint8)
                cv2.drawContours(copy, contorno, -1, (0,0,255), 3)
                
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('image', 900,600)
                cv2.imshow('image',copy)
                cv2.waitKey()


    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('image', 900,600)
    #cv2.imshow('image',image)
    #cv2.waitKey()


def main():
    img = cv2.imread('test.jpg')
    search_number(img)

if __name__ == '__main__':
    main()
