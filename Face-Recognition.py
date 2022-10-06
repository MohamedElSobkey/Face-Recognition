import cv2
import numpy as np
import face_recognition
import os

# creating variable : path to the images
path = 'persons'
images = []
classNames = []
personsList = os.listdir(path)



for cl in personsList:
    curPersonn = cv2.imread(f'{path}/{cl}')
    images.append(curPersonn)
    #to print the of images only without jbg
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

#this function for encoding the images & putting specific points around the face
def findEncodeings(image):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

#putting all images in the list
encodeListKnown = findEncodeings(images)
#print(encodeListKnown)
print('Encoding Complete.')

# for openning the cam 
cap = cv2.VideoCapture(0)


#for reading the video frames
while True:
    _, img = cap.read()
    
    # to decrease the size of image to save our time
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    # for encoding the small size of images
    faceCurentFrame = face_recognition.face_locations(imgS)
    #to recognition the position of the current image
    encodeCurentFrame = face_recognition.face_encodings(imgS, faceCurentFrame)
    
    #comparing
    for encodeface, faceLoc in zip(encodeCurentFrame, faceCurentFrame):
        #matching names with the images
        matches = face_recognition.compare_faces(encodeListKnown, encodeface)
        #less distance for the current image
        faceDis = face_recognition.face_distance(encodeListKnown, encodeface)
        #print(faceDis)
        #gave the least value which refers to the person on the cam 
        matchIndex = np.argmin(faceDis)
        
        
        #for painting a square around the face and writting person's name
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,0,255), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
    
    
    
    
    cv2.imshow('Face Recogntion', img)
    #1 for video 
    #0 for image
    cv2.waitKey(1)

