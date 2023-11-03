import cv2

image = cv2.imread('KakaoTalk_20231102_215131373_03.jpg')
image = cv2.resize(image, dsize=(1920, 1080), interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img,(5,5),0)
img= cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3,6)
#img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5,9)
cv2.imshow('as', img)
cv2.waitKey(0)
k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

#img = cv2.morphologyEx(img, cv2.MORPH_OPEN, k)
#img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
#img = cv2.dilate(img, k)
contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    area = cv2.contourArea(contour)
    if area > 4 and area < 10000:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow('as', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
