import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file("ImagesBasic/elon.jpg")
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("ImagesBasic/elon_test.jpg")
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

# imgBill = face_recognition.load_image_file("ImagesBasic/billgates.jpg")
# imgBill = cv2.cvtColor(imgBill,cv2.COLOR_BGR2RGB)
# faceLocBill = face_recognition.face_locations(imgBill)[0]
# encodeBill = face_recognition.face_encodings(imgBill)[0]
# cv2.rectangle(imgBill,(faceLocBill[3],faceLocBill[0]))
# cv2.imshow('billgates', imgBill)

faceloc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

facelocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),2)


results = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('elon', imgElon)
cv2.imshow('elon_test', imgTest)
cv2.waitKey(0)