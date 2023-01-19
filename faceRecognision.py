from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import subprocess
import telegram
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(320, 240))
face_cascade = cv2.CascadeClassifier('/home/mbilla/opencv-4.0.0/data/haarcascades/haarcascade_frontalface_default.xml')
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('trainer/trainer.yml')
id = 0
font = cv2.FONT_HERSHEY_SIMPLEX
time.sleep(0.1)
names=['None','Maha','Achyuth','Santosh']
for i in names:
	if i!='None':
		subprocess.call(['chmod','0000',i])
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	image = frame.array
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for(x, y, w, h) in faces:
		image = cv2.rectangle(image,(x, y), (x+w, y+h), (255, 0, 0), 2)
		id, conf = rec.predict(gray[y:y+h, x:x+w])
		print(id)
		if(id >= len(names)):
			id = "Unknown"
		else:
			id=names[id]
		subprocess.call(['chmod','0777',id])
		#if(id == 1):
		#	id = "negar"
		#elif(id == 2):
		#	id = "professoraerabi"
        #cv2.putText(image, str(id), (x,y+h), font, 2, (255,255,255), 3)
              #n()
        # Check if confidence is less them 100 ==> "0" is perfect match
		if (conf < 100):
			#id = names[id]
			conf = "  {0}%".format(round(100 - conf))
		else:
			#id = "unknown"
			conf = "  {0}%".format(round(100 - conf))
		cv2.putText(image, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
		cv2.putText(image, str(conf), (x+5,y+h-5), font, 1, (255,255,0), 1)
		
	cv2.imshow("Frame", image)
	rawCapture.truncate(0)
	if(cv2.waitKey(1)&0xFF==27):
		break


