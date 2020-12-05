imagePath = '2.jpg'
cascPath = 'haarcascade_frontalface_default.xml'

import cv2

image = cv2.imread(imagePath)
faceCascade = cv2.CascadeClassifier(cascPath)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Gray Scale Image", gray)
cv2.waitKey(0)


faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30,30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

print(f"Found {len(faces)} faces!")

for (x, y, w, h) in faces:
    cv2.rectangle(
        image, 
        (x, y), 
        (x+w, y+h), 
        (0, 255, 0), 2
    )

cv2.imshow("Faces found", image)
cv2.waitKey(0)