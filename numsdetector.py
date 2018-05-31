# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
from glob import glob
import numpy as np

global res
global dilated


def dilate(dilation_size):
	dilation_size = 2*dilation_size+1
	kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilation_size,dilation_size))
	dilated = cv2.dilate(res,kernel)
	return dilated
	#cv2.imshow('dilation demo',dilated)


erosion_size = 0   # initial kernel size  = 1
dilation_size = 0
max_kernel_size = 21 # maximum kernel size = 43


# define the dictionary of digit segments so we can identify
# each digit on the thermostat
DIGITS_LOOKUP = {
(1, 1, 1, 0, 1, 1, 1): 0,
(0, 0, 1, 0, 0, 1, 0): 1,
(1, 0, 1, 1, 1, 1, 0): 2,
(1, 0, 1, 1, 0, 1, 1): 3,
(0, 1, 1, 1, 0, 1, 0): 4,
(1, 1, 0, 1, 0, 1, 1): 5,
(1, 1, 0, 1, 1, 1, 1): 6,
(1, 0, 1, 0, 0, 1, 0): 7,
(1, 1, 1, 1, 1, 1, 1): 8,
(1, 1, 1, 1, 0, 1, 1): 9
}


lower_black = np.array([64,64,56], dtype=np.uint8)
upper_black= np.array([212,183,98], dtype=np.uint8)

print("hola")

for fn in glob('/home/charlesdickens/Documents/vantTEC/NUMBER_DETECTION/VANTEC_number_detection/test_images/*.png'):


	print("hola")
	# load the example image

	image = cv2.imread(fn)
	cv2.imshow('original',image)
	imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	#cv2.imshow('hsv',imgHSV)
	res =  cv2.inRange(imgHSV , lower_black, upper_black)
	cv2.imshow('filter black',res)

	dilated = dilate(9)

	image, cnts, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE )

	'''
	print(len(contours))
	maxContour = 0
	for contour in contours:
	contourSize = cv2.contourArea(contour)
	if contourSize > maxContour:
		maxContour = contourSize
		maxContourData = contour
	print(maxContour)
	'''

	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	cnts =  cnts[0:3]

	# loop over the contours
	for c in cnts:
		# compute the center of the contour
		M = cv2.moments(c)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
	 
		# draw the contour and center of the shape on the image
		cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
		cv2.circle(image, (cX, cY), 7, (0, 255, 255), -1)
		cv2.putText(image, "center", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
	 
	# show the image
	cv2.imshow("Image", image)


	#edged = cv2.Canny(res, 0, 255, 255)






	# pre-process the image by resizing it, converting it to
	# graycale, blurring it, and computing an edge map
	#image = imutils.resize(image, height=500)
	#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#cv2.imshow('gray',gray)


	#ret,thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
	#cv2.imshow('thresh1',thresh1)
	#erode(0)

	#if cv2.waitKey(0) == 27:
	#	cv2.destroyAllWindows()

	#blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	#cv2.imshow('blurred',blurred)



	#edged = cv2.Canny(blurred, 50, 200, 255)
	#cv2.imshow('edged',edged)



	# find contours in the edge map, then sort them by their
	# size in descending order
	'''
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)


	for i,contour in enumerate(cnts):
	if(cv2.contourArea(contour) == 0.0):
		del cnts[i]


	arr = []
	for i,contour in enumerate(cnts):
	arr.append(cv2.contourArea(contour))

	"""MINIMUM DIFFERENCE"""
	# Initialize difference as infinite
	n = len(arr) 
	global diff 
	diff = 10**10.0
	arr = sorted(arr)

	# Find the min diff by comparing adjacent
	# pairs in sorted array
	for i in range(n-1):
	if (arr[i+1] - arr[i]) < diff:
		diff = arr[i+1] - arr[i]

	print("Minimum difference is " + str(diff))



	print("Perimeters")


	peris = []

	for i,c in enumerate(cnts):
	# approximate the contour
	peris.append(cv2.arcLength(c, True))
	print(peris[i])




	displayCnt = None

	black  = image

	h = len(black)
	w = len(black[0])

	for y in range(h):
	for x in range(w):
		black[y,x] = [255,255,255]


	for i,contour in enumerate(cnts):	

	print(cv2.contourArea(contour))
	if(cv2.contourArea(contour) > 60.0 and cv2.contourArea(contour) < 70.00):
		cv2.drawContours(black, cnts, i, (255,255,0), 3)

	cv2.imshow('draws',black)


	# loop over the contours
	for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# if the contour has four vertices, then we have found
	# the thermostat display
	if len(approx) == 4:
		displayCnt = approx
		break


	'''
	

	k = cv2.waitKey(0) & 0xFF
	if k == 27:
		exit()
		cv2.destroyAllWindows()


