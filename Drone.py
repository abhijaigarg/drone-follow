import numpy as np
import cv2
import math,time

def init_mean_coordinates(feed):
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
	counter = 0
	mean_x = 0
	mean_y = 0
	mean_w = 0
	mean_h = 0

	while(counter<5):
		ret, img = feed.read()
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray,1.3,5)
	
		for(x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			mean_x+=(x)
			mean_y+=(y)
			mean_w+=(w)
			mean_h+=(h)
			counter+=1
			print counter
			
		cv2.imshow('Init',img)

		if cv2.waitKey(5)==27:
			break
	
	mean_x/=counter
	mean_y/=counter
	print("Time:"+str(time.clock()))

	return mean_x,mean_y,w,h

min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([255,173,127],np.uint8)


def skin_color_detect(sourceImage):

	imageYCrCb = cv2.cvtColor(sourceImage,cv2.COLOR_BGR2YCR_CB)

	# Find region with skin tone in YCrCb image
	skinRegion = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)

	#Do contour detection on skin region
	img, contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	# Draw the contour on the source image

	for i, c in enumerate(contours):
		area = cv2.contourArea(c)
		if area > 1000:
			cv2.drawContours(sourceImage, contours, i, (0, 255 , 0), -1)

	return sourceImage


def meanshift(feed,mean_x,mean_y,mean_w,mean_h): #camshift_x,camshift_y,w,h):#feed,x,y,w,h):
	# setup initial location of window
	ret, frame = feed.read()
	r,h,c,w = 3*mean_y,mean_h/2,mean_x,mean_w/2#,200,125  # simply hardcoded the values
	track_window = (c,r,w,h)

	# set up the ROI for tracking
	roi = frame[r:r+h, c:c+w]
	hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
	roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
	cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
	x,y,w,h = -1,-1,-1,-1

	move = ""

	# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
	term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

	while(1):
	    ret ,frame = feed.read()
	    #frame = skin_color_detect(frame)
	    move = ""

	    if ret == True:
	        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

	        # apply meanshift to get the new location
	        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

	        
	        if((math.fabs(track_window[0]-x)>20) or \
	        (math.fabs(track_window[1]-y)>5)):# or (math.fabs(track_window[1]-1)>50) or (math.fabs(track_window[2]-w)>50) or (math.fabs(track_window[3]-h)>50)):
	        	#move="left/right"
	        	if (track_window[0]-x)>0:
		        	move="right"
		        	x,y,w,h = track_window
		       	elif (track_window[0]-x)<0:
					move="left"
					x,y,w,h = track_window

	        	"""
	        	if ((track_window[1]-y)>0):
	        		
				elif (track_window[1]-y)<0:
					if (track_window[0]-x)>0:
						move="up-right"
		        		x,y,w,h = track_window
		        	elif (track_window[0]-x)<0:
						move="up-left"
						x,y,w,h = track_window
					
					#x,y,w,h = track_window
	        	"""
					
	        cv2.putText(frame,"Move = " + move, (10,15), cv2.FONT_HERSHEY_PLAIN, 0.8, 255,1)
	        cv2.putText(frame,"(X,Y,W,H) = (" + str(x) + "," + str(y) + "," + str(w) + "," + str(h) + ")", (10,30), cv2.FONT_HERSHEY_PLAIN, 0.8, 255,1)
	        cv2.putText(frame,"(X,Y,W,H) = (" + str(track_window[0]) + "," + str(track_window[1]) + "," + str(track_window[2]) + "," + str(track_window[3]) + ")", (10,50), cv2.FONT_HERSHEY_PLAIN, 0.8, 255,1)
	        img2 = cv2.rectangle(frame,(x,y),(x+w,y+h),255,2)
	        
	        cv2.imshow('img2',frame)

	        k = cv2.waitKey(60) & 0xff
	        if k == 27:
	            break
	    else:
	        break

def calculate_centroid(pts):
	centroid  = []
	centroid.append((pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0])/4)
	centroid.append((pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1])/4)
	return centroid
def calculate_area(pts):
	pts = pts.tolist()
	area = 0
	print pts
	for i in range(4):
		if i==3:
			area += (pts[i][0]*pts[0][1]) - (pts[i][1]*pts[0][0])
		else:
			area += (pts[i][0]*pts[i+1][1]) - (pts[i][1]*pts[i+1][0])
	area /= 2
	return area

def camshift(cap,mean_x,mean_y,mean_w,mean_h):

	# take first frame of the video
	ret,frame = cap.read()
	print ret
	# setup initial location of window
	r,h,c,w = mean_y,mean_h,(mean_x),mean_w#250,90,400,125  # simply hardcoded the values
	track_window = (c,r,w,h)

	# set up the ROI for tracking
	roi = frame[r:r+h, c:c+w]
	hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
	roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
	cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
	area = 0
	centroid = [0,0]
	# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
	term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
	
	while(1):
		#move_back = 0
		#move_left = 0
	    ret ,frame = cap.read()
	    #frame = skin_color_detect(frame)
	    move = ""
	    if ret == True:
	        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

	        # apply camshift to get the new location
	        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

	        # Draw it on image
	        pts = cv2.boxPoints(ret)
	        #cv2.putText(frame,pts, (10,50), cv2.FONT_HERSHEY_PLAIN, 0.8, 255,1)
	        
	        pts = np.int0(pts)
	        print pts.tolist()
	        img2 = cv2.polylines(frame,[pts],True, 255,2)
	        
	        #make movement decisions
	        if(math.fabs(area-calculate_area(pts))>10):
	        	if area > calculate_area(pts):
	        		move_front = -1
	        	if area < calculate_area(pts):
	        		move_front = 1
	        	area = calculate_area(pts)

	        if(math.fabs(centroid[0]-calculate_centroid(pts)[0])>10):
	        	if centroid[0] > calculate_centroid(pts)[0]:
	        		move_left = -1
	        	if centroid[0] < calculate_centroid(pts)[0]:
	        		move_left = 1
	        	centroid = calculate_centroid(pts)

	        if move_left == 1:
	        	if move_front == 1:
	        		move = "left-front"
	        	elif move_front == -1:
	        		move = "left-back"
	        	else:
	        		move = "left"
	        elif move_left == -1:
	        	if move_front == 1:
	        		move = "right-front"
	        	elif move_front == -1:
	        		move = "right-back"
	        	else:
	        		move = "right"
	        else:
	        	if move_front == 1:
	        		move = "front"
	        	elif move_front == -1:
	        		move = "back"
	        	else:
	        		move = ""

	        cv2.putText(img2,"Move = " + move, (10,15), cv2.FONT_HERSHEY_PLAIN, 0.8, 255,1)
	        cv2.imshow('img2',img2)

	        k = cv2.waitKey(60) & 0xff
	        if k == 27:
	            break
	    else:
	        break



def main():
	
	feed = cv2.VideoCapture('tcp://192.168.1.1:5555/')

	mean_x, mean_y,mean_w,mean_h = init_mean_coordinates(feed)
	
	#meanshift(feed,mean_x,mean_y,mean_w,mean_h)#,camshift_x,camshift_y,w,h)#feed,camshift_x,camshift_y,w,h)
	camshift(feed,mean_x,mean_y,mean_w,mean_h)
	feed.release()
	cv2.destroyAllWindows()


main()
