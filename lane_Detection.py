import cv2
import numpy as np
import matplotlib.pyplot as plt


# img = cv2.imread("test_image.jpg")
# lane_image = np.copy(img)


def make_coordinate(image, line_parametres):
    try:
        slope, intercept = line_parametres
    except TypeError:
        slope, intercept = 0, 0

    y1 = image.shape[0] + 20
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def average_resim(image, lines):
    global left_line
    global right_line
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        print(left_fit_average, 'left')
        left_line = make_coordinate(image, left_fit_average)
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        print(right_fit_average, 'right')
        right_line = make_coordinate(image, right_fit_average)

    return np.array([left_line, right_line])


def kenar(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 45, 150)
    return canny


def ex_serit(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1,), (x2, y2), (255, 255, 0), 10)
    return line_image


def incelenecek_bolge(image):
    height = image.shape[0]
    triangle = np.array([
        [(30, height), (1200, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


path = "C:\\Users\\Hasan\\Desktop\\YapayZeka\\Open_CV\\Opencv_Kods\\serit_takip\\test2.mp4"
vid = cv2.VideoCapture(path)

while (vid.isOpened()):
    _, frame = vid.read()

    kenar_inceleme = kenar(frame)
    bolge = incelenecek_bolge(kenar_inceleme)

    lines = cv2.HoughLinesP(bolge, 2, np.pi / 180, 100, np.array([]), minLineLength=50, maxLineGap=5)
    averaged_lines = average_resim(frame, lines)
    line_image = ex_serit(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    cv2.imshow("Result", combo_image)
    cv2.waitKey(1)

vid.release()
cv2.destroyAllWindows()
