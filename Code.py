# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 10:19:54 2016

@author: himanshug
"""
import threading
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pytesseract import image_to_string
img = cv2.imread('new.jpg',0)
height, width = img.shape[:2]
#equ = cv2.equalizeHist(img)

#blur=cv2.medianBlur(img,5)
#thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#            cv2.THRESH_BINARY,11,2)
edges = cv2.Canny(img,50,100)
       
#cv2.imshow('edges',edges)
#cv2.waitKey(1)

kernel =cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
kernel1 =cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
kernel2 =cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
dilation = cv2.dilate(edges,kernel,iterations = 1)
erosion = cv2.erode(dilation,kernel1,iterations =1)

closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel1)

vis = img.copy()
mser = cv2.MSER_create()
regions = mser.detectRegions(erosion, None)

mask = np.zeros(img.shape,np.uint8)
i=0
for p in regions:
    x,y,w,h = cv2.boundingRect(p)
    #area1 = cv2.contourArea(p)
    #rect_area=w*h
    if w*h<(height*width)/3 :
        cv2.rectangle(mask,(x,y),(x+w,y+h),(255,255,255),-1)
    
    #extent.append(float(area1)/rect_area)
    #print float(area1)/rect_area
    i=i+1
    #print w*h

print i

kernel_mask =cv2.getStructuringElement(cv2.MORPH_RECT,(6,6))
kernel_mask_1=cv2.getStructuringElement(cv2.MORPH_RECT,(5,1))

erosion_mask = cv2.erode(mask,kernel_mask,iterations =1)
dilation_mask= cv2.dilate(erosion_mask,kernel_mask_1,iterations = 7)

_,contours, hierarchy = cv2.findContours(dilation_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

j=1
start_time=time.time()
font=cv2.FONT_HERSHEY_SIMPLEX
dd=0;
exitFlag = 0

class myThread (threading.Thread):
    def __init__(self, threadID, name, scounter,encounter,contours):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.scounter = scounter
        self.encounter=encounter
    def run(self):
        print "Starting " + self.name
        recognize(self.name, self.scounter,self.encounter, contours)
        global dd
        dd+=1
        print "Exiting " + self.name

def recognize(threadName, scounter,encounter, contours):
    i=scounter
    while i<=encounter :
        if exitFlag:
            threadName.exit()
        #        print "%s: %s" % (threadName, time.ctime(time.time()))
        x,y,w,h = cv2.boundingRect(contours[i])
        #cv2.rectangle(img,(x,y-h/2),(x+w,y+h+h/2),(0,255,0),2)
        word=img[y-h:y+h+h/2,x:x+w]
        
        imp_img = Image.fromarray(word)
        #start_time=time.time()
        #s=image_to_string(imp_img,config="-psm 7")
        #print s
        find_word="possible"
        if w*h>0:
           s=image_to_string(imp_img)
           print s
           if s.find(find_word)!=-1:
           #cv2.imwrite('ss.png',img[y:y+h,x:x+w])
              cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),2)
        #cv2.putText(vis,"i",(x+w,y+h),font,1,(0,255,0),2,cv2.LINE_AA)
#        j=j+1
        #print time.time()-start_time
            
        i+= 1
l=len(contours)
l1=int(l/4)-1
l2=l1+1
l3=l2+int(l/4)
l4=l3+int(l/4)        
thread1 = myThread(1, "Thread-1", 0,l1,contours)
thread2 = myThread(2, "Thread-2", l2,l2+int(l/4)-1,contours)
thread3= myThread(3, "Thread-3", l3,l3+int(l/4)-1,contours)
thread4= myThread(4, "Thread-4", l4,l-1,contours)
thread1.start()
thread2.start()
thread3.start()
thread4.start()
start_time=time.time()
        

#for p in contours:
#    x,y,w,h = cv2.boundingRect(p)
#    word=img[y-h:y+h+h/2,x:x+w]
#    imp_img = Image.fromarray(word)
#    #start_time=time.time()
#    #s=image_to_string(imp_img,config="-psm 7")
#    #print s
#    find_word="hard"
#    if w*h>0:
#        s=image_to_string(imp_img)
#        print s
#        if s.find(find_word)!=-1:
#        #cv2.imwrite('ss.png',img[y:y+h,x:x+w])
#            cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),2)
#    #cv2.putText(vis,"i",(x+w,y+h),font,1,(0,255,0),2,cv2.LINE_AA)
#    j=j+1
#    #print time.time()-start_time
#print j
#print time.time()-start_time

while 1:
  if dd==4:
   print '\n'   
   print time.time()-start_time   
   plt.imshow(vis, cmap = 'gray', interpolation = 'bicubic')
   plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
   plt.show()
   break
cv2.imwrite("possible.jpg",vis)
