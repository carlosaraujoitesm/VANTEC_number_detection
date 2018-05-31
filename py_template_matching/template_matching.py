import cv2
import numpy as np
from matplotlib import pyplot as plt
from glob import glob

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF_NORMED']


for fn in glob('/home/charlesdickens/Documents/vantTEC/NUMBER_DETECTION/VANTEC_number_detection/py_template_matching/test_images/*.png'):
 
	img = cv2.imread(fn,0)
	img2 = img.copy()
	template = cv2.imread('1.png',0)
	w, h = template.shape[::-1]

	for meth in methods:
		img = img2.copy()
		method = eval(meth)

		# Apply template Matching
		res = cv2.matchTemplate(img,template,method)
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

		# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
		if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
		    top_left = min_loc
		else:
		    top_left = max_loc
		bottom_right = (top_left[0] + w, top_left[1] + h)

		cv2.rectangle(img,top_left, bottom_right, 255, 2)
		cv2.imshow('detection',img)
		k = cv2.waitKey(0) & 0xFF
		if k == 27:
			cv2.destroyAllWindows()
