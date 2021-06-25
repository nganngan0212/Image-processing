import cv2
import numpy as np
import math

def find_corners(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(img, (9,6), None)

    if ret:
        cv2.drawChessboardCorners(img, (9,6), corners, ret)
        corners = np.resize(corners,(9, 6, 2))

    return img, corners

def find_pixel_distance(p1, p2):
    dis = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    dis = np.ceil(dis)
    return dis

def find_pixel_angle(p1, p2):
    dy = p1[1] - p2[1]
    dx = p2[0] - p1[0]

    rads = math.atan2(dy,dx)
    degs = math.degrees(rads)
    return degs

def find_ratio(corners, real):
    rows, cols = corners.shape[:2]
    distances = []
    for i in range(rows-1):
        for j in range(cols-1):
            dis1 = find_pixel_distance(corners[i,j], corners[i,j+1])
            dis2 = find_pixel_distance(corners[i,j], corners[i+1,j])
            if(dis1<100):
                distances.append(dis1)
            if dis2<100:
                distances.append(dis2)
    distances = np.round(distances)
    ave = np.ceil(np.sum(distances)/len(distances))
    # ave = distances[2]
    print("Average pixel: ", ave)
    return ave/real

def find_edge(img):

    edges = cv2.Canny(img,50,150,apertureSize = 3)

    lines = cv2.HoughLinesP(edges,1,np.pi/360,50,minLineLength=30,maxLineGap=10)

    lengths = []
    for line in lines:
        x1,y1,x2,y2 = line[0]
        pt1 = (x1, y1)
        pt2 = (x2, y2)
        dis = find_pixel_distance(pt1, pt2)
        deg = find_pixel_angle(pt1, pt2)
        # print("Distance: ", dis)
        # print("Deg: ", deg)
        if deg<-10 and deg>-80:
            lengths.append(dis)
            # print("Distance: ", dis)
            # print("Deg: ", deg)
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
            cv2.circle(img, pt1, radius=0, color=(0, 0, 255), thickness=3)
            cv2.circle(img, pt2, radius=0, color=(0, 0, 255), thickness=3)

    return img, lengths

img = cv2.imread("hop.jpg")
rows, cols = img.shape[:2]
img = cv2.resize(img,(int(cols/2), int(rows/2)), interpolation = cv2.INTER_AREA) 
cv2.imshow('Origin', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_find_corners, corners = find_corners(img)
cv2.imshow('Corners', img_find_corners)
real = 2.95 # cm per square
ratio = find_ratio(corners, real) #pixel per cm
print("Ratio: ",ratio)

img_find_lines, lengths = find_edge(gray)

cv2.imshow("Test",img_find_lines)

m_dis = lengths[0]/ratio

r_dis = 7 # Real length of the ruler in cm
print("Real distance: {} cm".format(r_dis))
print("Measurement distance: {} cm".format(m_dis))

cv2.waitKey(0)
cv2.destroyAllWindows()