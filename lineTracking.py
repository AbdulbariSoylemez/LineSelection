import cv2
import numpy as np
import math

line=cv2.imread("foto/park.jpg")
copy=line.copy()
grey=cv2.cvtColor(line,cv2.COLOR_BGR2GRAY)
kenar=cv2.Canny(grey,100,150)
lines=cv2.HoughLinesP(kenar,1, np.pi / 180, 50) # burda bize çizgileri yakalamamızı sağlıyor


for i in lines:
    x1,y1,x2,y2=i[0]
    cv2.line(line,(x1,y1),(x2,y2),(0,0,255),5) # burda ana resimde çizgiyi çiziyoruz


cv2.imshow("Line",line)
cv2.imshow("Orjinal",copy)

cv2.waitKey(0)
cv2.destroyAllWindows()