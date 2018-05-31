# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
 


global thresh1


def erode(erosion_size):
    erosion_size = 2*erosion_size+1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(erosion_size,erosion_size))
    eroded = cv2.erode(thresh1,kernel)
    cv2.imshow('erosion demo',eroded)

def dilate(dilation_size):
	dilation_size = 2*dilation_size+1
	kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilation_size,dilation_size))
	dilated = cv2.dilate(thresh1,kernel)
	cv2.imshow('dilation demo',dilated)


erosion_size = 0   # initial kernel size  = 1
dilation_size = 0

max_kernel_size = 21 # maximum kernel size = 43



cv2.namedWindow('erosion demo')
cv2.namedWindow('dilation demo')

# Creating trackbar for kernel size
cv2.createTrackbar('Size: 2n+1','erosion demo',erosion_size,max_kernel_size,erode)
cv2.createTrackbar('Size: 2n+1','dilation demo',dilation_size,max_kernel_size,dilate)





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

# load the example image
image = cv2.imread("example.jpg")
 
# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)


#ret,thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
#cv2.imshow('thresh1',thresh1)
#erode(0)
#dilate(0)
#if cv2.waitKey(0) == 27:
#	cv2.destroyAllWindows()

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow('blurred',blurred)

edged = cv2.Canny(blurred, 50, 200, 255)
cv2.imshow('edged',edged)



# find contours in the edge map, then sort them by their
# size in descending order
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

k = cv2.waitKey(0) & 0xFF
if k == 27:
    exit()

cv2.destroyAllWindows()
