import cv2 as cv
import numpy as np

# Funkcje do filtrowania kolorów, redukcji szumu i adaptacyjnych parametrów HoughCircles
def filter_by_color(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_bound = np.array([0, 100, 100])
    upper_bound = np.array([10, 255, 255])
    mask = cv.inRange(hsv, lower_bound, upper_bound)
    return cv.bitwise_and(frame, frame, mask=mask)

def reduce_noise(frame):
    kernel = np.ones((5, 5), np.uint8)
    return cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)

def hough_circles(frame):
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurFrame = cv.GaussianBlur(grayFrame, (17, 17), 0)
    circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.4, 100,
                              param1=50, param2=20, minRadius=5, maxRadius=100)
    return circles


video = cv.VideoCapture('movingball.mp4')
prevCircle = None

distance = lambda x1, y1, x2, y2: np.sqrt((float(x1) - x2) ** 2 + (float(y1) - y2) ** 2)

width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

scaling_factor = 0.5

cv.namedWindow('circles', cv.WINDOW_NORMAL)
cv.resizeWindow('circles', int(width * scaling_factor), int(height * scaling_factor))


while True:
    ret, frame = video.read()

    if not ret: break
    original_frame = frame.copy()

    frame = filter_by_color(frame)
    frame = reduce_noise(frame)
    circles = hough_circles(frame)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = None
        for i in circles[0, :]:
            if chosen is None: chosen = i
            if prevCircle is not None:
                if distance(chosen[0], chosen[1], prevCircle[0], prevCircle[1]) >= distance(i[0], i[1], prevCircle[0],
                                                                                            prevCircle[1]):
                    chosen = i
        cv.circle(original_frame, (chosen[0], chosen[1]), 1, (0, 255, 100), 3)
        cv.circle(original_frame, (chosen[0], chosen[1]), chosen[2], (0, 255, 100), 3)
        prevCircle = chosen

    cv.imshow("circles", original_frame)

    if cv.waitKey(1) & 0xFF == ord('q'): break

video.release()
cv.destroyAllWindows()