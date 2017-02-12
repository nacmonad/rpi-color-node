#! /usr/bin/python
import sys
import numpy as np
import cv2
import time
from sklearn.cluster import KMeans
from json_tricks.np import dumps

#set up webcam
cap = cv2.VideoCapture(0)
num_c = 4

#setup cam resolution
cap.set(3,320)
cap.set(4,240)

def recreate_image(colorcodes, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = 3
    image = np.zeros((w, h, d)).astype('uint8')
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = colorcodes[labels[label_idx]]
            label_idx += 1
    return image

def genColorsFrame(num_clusts, colorcodes, w, h):

    sect_size = w/num_clusts
    col_chart = np.zeros((frame.shape[0],frame.shape[1],3), dtype='uint8')

    for i in range(0,w,sect_size):
        col_chart[i:i+sect_size] = (quant[i/sect_size][0],quant[i/sect_size][1],quant[i/sect_size][2])

    return col_chart

#EVENT LOOP
while(True):
    #Capture Frame
    ret, frame = cap.read()
    (w,h,d) = frame.shape

    if(ret):
        #Do something with frame
        try:
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            #cv2.imshow('ds',ds)
            #start = time.time()
	    try:
                # k-means and plot
                # downsample first!
                r = 50.0 / frame.shape[1]
                dim = (50, int(frame.shape[0] * r))
                ds = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                ds = ds.reshape((ds.shape[0]*ds.shape[1]),3)
                # get kmeans!
                kmeans = KMeans(n_clusters=num_c, random_state=0).fit(ds)
                quant = sorted(kmeans.cluster_centers_.astype('uint8'), key=lambda x: x[1])
        	#quant = sorted(quant,key=lambda x: x[1])
                #quant = np.argsort(quant,axis=1)
		#labels = kmeans.predict(frame.reshape((frame.shape[0]*frame.shape[1]),3))  
                #generate new frame from quantized colors and labels
                #clustered = recreate_image(quant,labels,w,h)
           	#time.sleep(1)     
                
            except Exception as err:
                print(err)

	
            #cv2.imshow('kmeans clustering', np.hstack([frame,clustered]))
            cv2.imshow('frame',np.hstack([frame, genColorsFrame(num_c, quant, w,h)]))
	    print dumps({'mydata':quant})
	    #sys.stdout.flush()
	    #time.sleep(2)
	    #end = time.time()
            #seconds = end-start
            #print "Time taken : {0} seconds".format(seconds)

	except Exception as err:
            print(err)

    
    # key-listener
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# REFERENCES
# K-MEANS:
# http://stackoverflow.com/questions/13613573/how-to-speed-up-color-clustering-in-opencv
# http://www.pyimagesearch.com/2014/07/07/color-quantization-opencv-using-k-means-clustering/

