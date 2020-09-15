from processing import zeros_like, equalizeHist, CLAHE, line_trans, gama
import math
import cv2
import numpy as np

# if the point is above the line, return True.
def above_line(y, x, k, b):
    return (x * k + b) <= y

def cut(img, k1, b1, k2, b2):
    height, width = img.shape
    down = np.zeros((height, width, 3), dtype='uint8')

    for i in range(height):
        for j in range(width):
            if above_line(i, j, k2, b2):
                down[i, j] = 255
            else:
                down[i, j] = img[i,j]

    for i in range(height):
        for j in range(width):
            if above_line(i, j, k1, b1) and down[i,j,0]<255:
                down[i, j] = img[i, j]
            else:
                down[i, j] = 255
    return np.array(down)
if __name__ == '__main__':
    image_name = '404'
    image1 = cv2.imread("data/390/test/" + image_name +".jpeg",0)
    image = equalizeHist(image1)
    # image = zeros_like(image1)

    img = image
    img = cv2.GaussianBlur(img, (9,9), 0)
    # edges = cv2.Canny(img,20,50, apertureSize=3)
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 70)
    edges = cv2.Canny(img,40,60, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 55)
    result = image1.copy()

    h = image1.shape[0]
    w = image1.shape[1]
    points = []
    for line in lines:
        rho = line[0][0]
        theta = line[0][1]

        # eliminate the theta
        # if (rho > 300 and rho < 1300 and theta > 1.1 and theta < 1.6):
        if(rho > 300 and rho < 1300 and theta > 1.38 and theta < 1.40):
            pt1 = (int(rho / np.cos(theta)), 0)
            pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])

            k = (pt1[1] - pt2[1]) / (pt1[0] - pt2[0])
            b = pt1[1] - k * pt1[0]
            p = [pt1,pt2,k,b]

            points.append(p)
            # print(k,b)
            # draw edges
            cv2.line(result, pt1, pt2, (255),4)

    cv2.imwrite("./data/390/test/edge_detect/hough_edges.png",edges)
    cv2.imwrite("./data/390/test/edge_detect/hough_result.png",result)

    img = image1
    k = [i[2] for i in points]
    b = [i[3] for i in points]

    h, w = image1.shape
    b_medium = h/2
    points.sort()

    # find the boundary of melted pool, one is upper than medium line, another is lower than medium line
    for i in range(0,len(points)-1):
        if(points[i][3] - b_medium) * (points[i+1][3] - b_medium) < 0:
            # print(points[i][3], points[i+1][3])
            i1 = i
            i2 = i+1

    b1 = points[i1][3]
    b2 = points[i2][3]

    k1 = points[i1][2]
    k2 = points[i2][2]

    d = abs((b1 - b2) * math.cos(math.atan(k1)))
    print("picture {}: the width of melted pool is {:d} pixel".format(image_name,int(d)))

    # set the offset to the edge.
    offset = 300
    b1_left = points[i1][3] - offset
    b1_right = w * k1 + b1_left

    b2_left = points[i2][3] + offset
    b2_right = w * k2 + b2_left

    res = cut(img, k1, b1_left, k2, b2_left)

    upper_bound =int(max(min(b1_left,b1_right),0))
    lower_bound = int(min(max(b2_left,b2_right),h))
    cut = res[upper_bound:lower_bound,:]

    cv2.imwrite('./data/390/test/res_'+str(offset)+image_name+'.jpg', res)
    cv2.imwrite('./data/390/test/cut_'+str(offset)+image_name+'.jpg', cut)

