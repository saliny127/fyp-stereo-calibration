import cv2
import numpy as np
import heapq

# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
R = cv_file.getNode('rot').mat()
T = cv_file.getNode('trans').mat()

image_size = (1280, 960)

# Open both cameras
cap = cv2.VideoCapture('videos/1.mp4')

# output video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 10.0, (2560,  960))


def find_objects(frame_left, frame_right):
    left_rectified = cv2.remap(
        frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(
        frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LINEAR)

    # Create the stereo object
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

    left_img_new = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    right_img_new = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

    # Compute the disparity map
    disparity = stereo.compute(left_img_new, right_img_new)

    # Normalize the disparity map
    disparity_norm = cv2.normalize(
        disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Normalize the disparity map
    # min = disparity.min()
    # max = disparity.max()
    # disparity = np.uint8(255 * (disparity - min) / (max - min))

    focal_length = 8
    baseline = 60
    # depth_map = (focal_length * baseline) / disparity

    # Convert to grayscale
    # gray = cv2.cvtColor(disparity, cv2.COLOR_BGR2GRAY)

    # Object detection using contours
    gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Threshold the image to detect the object
    # _, thresholded = cv2.threshold(
    #     disparity, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow('thresholded', thresh)

    # # Find contours
    # contours, _ = cv2.findContours(
    #     thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original image
    cv2.drawContours(left_rectified, contours, -1, (0, 255, 0), 2)

    # Find the largest contourq
    # largest_contour = max(contours, key=cv2.contourArea)
    contours = heapq.nlargest(3, contours, key=cv2.contourArea)

    # loop throgh contours
    for contour in contours:
        # Draw a rectangle around the largest contour
        (x, y, w, h) = cv2.boundingRect(contour)

        cv2.rectangle(left_rectified, (x, y),
                      (x + w, y + h), (255, 0, 0), 2)

        # Focal length and baseline distance
        focal_length = 1200  # meters
        baseline_distance = 0.54  # meters

        # Calculate the distance to the object
        perceived_width = w
        distance = focal_length * baseline_distance / perceived_width

        # Display the distance
        cv2.putText(left_rectified, "Distance: {:.2f}m".format(
            distance), (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    height, width, channels = left_rectified.shape
    imS = cv2.resize(left_rectified, (int(width/2), int(height/2)))
    cv2.imshow('left_rectified', imS)


while (cap.isOpened()):

    succes, img = cap.read()
    height, width, channels = img.shape
    frame_left = img[0:height, 0:int(width/2)]
    frame_right = img[0:height, int(width/2):width+1]

    vis = np.concatenate((frame_right, frame_left), axis=1)
    height, width, channels = vis.shape
    imS = cv2.resize(vis, (int(width/2), int(height/2)))
    cv2.imshow('frame', imS)

    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('a'):
        find_objects(frame_left, frame_right)

    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
