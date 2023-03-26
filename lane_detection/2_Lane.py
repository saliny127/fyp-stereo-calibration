# I have given two ways (one in comments). is another way of detecting the lanes
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt


# def roi(image, vertices):
#     mask = np.zeros_like(image)
#     mask_color = 260
#     cv2.fillPoly(mask, vertices, mask_color)
#     cropped_img = cv2.bitwise_and(image, mask)
#     return cropped_img


# def draw_lines(image, hough_lines):
#     for line in hough_lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(image, (x1, y1), (x2, y2), (0, 260, 0), 2)

#     return image

# def process(img):
#     height = img.shape[0]
#     width = img.shape[1]
#     roi_vertices = [
#         (0, 680),
#         (2*width/3, 2*height/3),
#         (width, 1020)
#     ]

#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray_img = cv2.dilate(gray_img, kernel=np.ones((3, 3), np.uint8))

#     canny = cv2.Canny(gray_img, 130, 220)

#     roi_img = roi(canny, np.array([roi_vertices], np.int32))

#     lines = cv2.HoughLinesP(roi_img, 1, np.pi / 180, threshold=15, minLineLength=15, maxLineGap=2)

#     final_img = draw_lines(img, lines)

#     return final_img


# cap = cv2.VideoCapture("C:/Users/Steven/Downloads/roadlane0.mp4")

# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# fourcc = cv2.VideoWriter_fourcc(*"XVID")

# saved_frame = cv2.VideoWriter("lane_detection.avi", fourcc, 30.0, (frame_width, frame_height))

# while cap.isOpened():
#     ret, frame = cap.read()

#     try:
#         frame = process(frame)

#         saved_frame.write(frame)
#         cv2.imshow("frame", frame)

#         if cv2.waitKey(1) & 0xFF == 27:
#             break

#     except Exception:
#         break

# cap.release()

# saved_frame.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
import pandas as pd


def roi(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def draw_lines_new(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    if lines is not None:
        img_shape = img.shape
        slope_right = []
        slope_left = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                np.seterr(divide='ignore', invalid='ignore')
                # calculating slope of the line
                slope = (y2 - y1) / float(x2 - x1)
                if slope > 0.4:  # if slope is positive and above this tweeked value 04 its left lane
                    # store both ends of the line in a list
                    slope_right.append([x1, y1])
                    slope_right.append([x2, y2])
                elif slope < -0.4:  # else its right lane
                    # store both ends of the line in a list
                    slope_left.append([x1, y1])
                    slope_left.append([x2, y2])
        # y=mx+b
        # initializing a dictionary to hold longest line length
        longest_left = {'dist': 0, 'm': None, 'b': None}
        longest_right = {'dist': 0, 'm': None,
                         'b': None}  # slope and intercept
        # Draw right line
        if len(slope_right) >= 2:
            for x1, y1 in slope_right[:-1]:
                # for each combination of both ends of the line
                for x2, y2 in slope_right[1:]:
                    distance = ((x1 - x2) ** 2 + (y1 - y2) **
                                2) ** 0.5  # compute its length
                    # maximum length is stored
                    if distance > longest_left['dist']:
                        # along with its slope and intercept
                        longest_left['dist'] = distance
                        longest_left['m'] = abs(float(y1 - y2) / (x1 - x2))
                        longest_left['b'] = y1 - longest_left['m'] * x1

            # in the end we calculate both x1, y1 and x2,y2  using slope and intercept
            y1 = img_shape[0]
            x1 = pd.to_numeric(
                (y1 - longest_left['b']) / longest_left['m'], errors='coerce')
            x1 = x1.astype(int)
            y2 = int(y1 / 1.6)
            x2 = pd.to_numeric(
                (y2 - longest_left['b']) / longest_left['m'], errors='coerce')
            x2 = x2.astype(int)
            cv2.line(img, (int(x1), y1), (int(x2), y2),
                     color, thickness)  # and plot the line

        # Similarly draw left line
        if len(slope_left) >= 2:
            for x1, y1 in slope_left[:-1]:
                for x2, y2 in slope_left[1:]:
                    distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                    if distance > longest_right['dist']:
                        longest_right['dist'] = distance
                        longest_right['m'] = -abs(float(y1 - y2) / (x1 - x2))
                        longest_right['b'] = y1 - longest_right['m'] * x1
            y1 = img_shape[0]
            x1 = pd.to_numeric(
                (y1 - longest_right['b']) / longest_right['m'], errors='coerce')
            x1 = x1.astype(int)
            y2 = int(y1 / 1.6)
            x2 = pd.to_numeric(
                (y2 - longest_right['b']) / longest_right['m'], errors='coerce')
            x2 = x2.astype(int)
            cv2.line(img, (int(x1), y1), (int(x2), y2), color, thickness)


def draw_lines(image, lines):
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
    return image


def process(image):
    height, width = image.shape[:2]
    region_of_interest_vertices = [
        (0, height),
        (0, height/1.2),
        (width/2.2, height/4),
        (width/1.6, height/4),
        (width, height/2),
        (width, height)
    ]
    # region_of_interest_vertices = [
    #     (0, height),
    #     (width/2, height/2),
    #     (width, height)
    # ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_image = cv2.Canny(blur_image, threshold1=50, threshold2=150)
    cropped_image = roi(canny_image, np.array(
        [region_of_interest_vertices], np.int32))

    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/180,
                            threshold=20, lines=np.array([]), minLineLength=40, maxLineGap=2)

    line_image = np.zeros_like(image)
    if lines is not None:
        line_image = draw_lines(line_image, lines)

     # lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
    #                         maxLineGap=max_line_gap)
    # line_img = np.zeros((cropped_image.shape[0], cropped_image.shape[1], 1), dtype=np.uint8)
    # line_img = cv2.cvtColor(line_img, cv2.COLOR_GRAY2RGB)
    # draw_lines_new(line_image, lines)

    return cv2.addWeighted(image, 0.8, line_image, 1, 0)


cap = cv2.VideoCapture("./videos/1.mp4")

while cap.isOpened():
    ret, frame = cap.read()

    try:

        height, width = frame.shape[:2]
        imS = cv2.resize(frame, (int(width / 2), int(height / 2)))
        midpoint = width // 4
        # Extract the left half of the frame
        left_half_frame = imS[:, :midpoint, :]
        new_frame = process(left_half_frame)
        # frame = process(frame)

        cv2.imshow("Road Lane Detection", new_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except Exception:
        break

cap.release()
cv2.destroyAllWindows()
