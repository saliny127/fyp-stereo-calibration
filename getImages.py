import cv2

cap = cv2.VideoCapture(1)

num = 0

while cap.isOpened():

    succes, img = cap.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'):  # wait for 's' key to save and exitssssss
        cv2.imwrite('images/img' + str(num) + '.png', img)
        height, width, channels = img.shape
        imgL = img[0:height, 0:int(width/2)]
        imgR = img[0:height, int(width/2):width+1]
        cv2.imwrite('images/stereoLeft/imageL' + str(num) + '.png', imgL)
        cv2.imwrite('images/stereoright/imageR' + str(num) + '.png', imgR)
        print("image saved!")
        num += 1

    height, width, channels = img.shape
    imS = cv2.resize(img, (int(width/2), int(height/2)))
    cv2.imshow('Img', imS)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()
