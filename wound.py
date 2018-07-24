import numpy as np
import cv2

# load the image
image = cv2.imread("tutorial1.jpg", 1)

# red color boundaries (R,B and G)
lower = [1, 0, 20]
upper = [60, 40, 200]

# create NumPy arrays from the boundaries
lower = np.array(lower, dtype="uint8")
upper = np.array(upper, dtype="uint8")

# find the colors within the specified boundaries and apply
# the mask
mask = cv2.inRange(image, lower, upper)
cv2.imshow('Mask',mask)

output = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow('sadas',output)




def cont():
    mask = cv2.inRange(image, lower, upper)
    #cv2.imshow('Mask', mask)
    ret, thresh = cv2.threshold(mask, 40, 255, 0)
    im2, conto, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return conto


contours=cont()

for cnt in contours:
    # draw in blue the contours that were founded
    #cv2.drawContours(output, contours, -1, 255, 3)



    x,y,w,h = cv2.boundingRect(cnt)
    print(str(x))
    if x<20:
        print("top")
        for i in range(x,w+x):
            for j in range(y,h+y):
                output[j][i][0]=0
                output[j][i][1]=0
                output[j][i][2]=0
                image[j][i][0] = 0
                image[j][i][1] = 0
                image[j][i][2] = 0
        contours=cont()





#find the biggest area
c = max(contours, key = cv2.contourArea)
x,y,w,h = cv2.boundingRect(c)

# draw the book contour (in green)
cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)

# show the images
cv2.imshow("Result", np.hstack([image, output]))

cv2.waitKey(0)
