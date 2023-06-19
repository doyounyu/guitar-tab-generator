import cv2
import numpy as np
import math
from matplotlib.path import Path
import time

last_chord_time = None
temp_chord = None
Chord_Time = []
detected_time = time.time()


num_width = 7
num_height = 6
lower_color_red = np.array([0, 0, 233])
upper_color_red = np.array([255, 163, 255])
lower_color_green = np.array([109, 130, 0])
upper_color_green = np.array([255, 230, 115])

tab_note = np.array([]) #악보 배열

chord_list = ['C', 'G', 'Am', 'F', 'Em']
last_chord = None

# define the dictionary of chord representations
chord_repr = {
    'C': ['0-', '1-', '0-', '2-', '3-', 'x-'],
    'G': ['3-', '2-', '0-', '0-', '0-', '3-'],
    'Am': ['x-', '0-', '2-', '2-', '1-', '0-'],
    'F': ['1-', '1-', '2-', '3-', '3-', '1-']
}

def update_chord(idx_pressed):
    global last_chord
    global last_chord_time
    global temp_chord
    global detected_time
    chord_index = find_chord(idx_pressed)
    

    if chord_index != -1:  # 코드 감지됨
        if temp_chord is None: #
            temp_chord = chord_list[chord_index]
            detected_time = time.time()  # Store the current time
        else:
            if chord_list[chord_index] == temp_chord and time.time() - detected_time >= 0.5:
                if last_chord is not None and last_chord != temp_chord:
                    Chord_Time.append((last_chord, time.time() - detected_time))  
                last_chord = temp_chord
                temp_chord = None
            elif chord_list[chord_index] != temp_chord:
                temp_chord = chord_list[chord_index]

    print("chord_time: ", Chord_Time)
    


def detect_points(image, lower, upper, single_point=False):
    
    mask = cv2.inRange(image, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    points = []

    for contour in contours:
        if cv2.contourArea(contour) > 300:  # ignore small contours to reduce noise
            M = cv2.moments(contour)
            if M["m00"] != 0:  # avoid division by zero
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                points.append((cX, cY))  # add the center of the colored point to the list

    if single_point:
        if points:  # make sure at least one point was found
            # return the first point in the list
            return [points[0]]
        else:
            return []  # return an empty list if no point was found
    else:
        return points  # return all detected points


def arrange_rectangle_points(points):
    # 주어진 점들을 x 좌표에 따라 정렬합니다.
    points.sort(key = lambda x: x[0])

    # 처음 두 점과 마지막 두 점으로 나누어 각각 y 좌표에 따라 정렬합니다.
    left_points = sorted(points[:2], key = lambda x: x[1])
    right_points = sorted(points[2:], key = lambda x: x[1])

    # 결과를 top_left, top_right, bottom_right, bottom_left 순으로 반환합니다.
    return [left_points[0], right_points[0], right_points[1], left_points[1]]


def draw_rectangle(image, points):
    if len(points) == 4:
        rect = arrange_rectangle_points(points)
        cv2.polylines(image, [np.array(rect)], True, (255, 255, 255), 2)  # draw rectangle

def find_points(p1, p2, ratio): # rato = l2 / l1
    # 중점 구하기
    mid_point = ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)

    # p1과 p2 사이의 기울기 구하기
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    angle = math.atan2(dy, dx)

    # l1의 절반 구하기
    half_l1 = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) / 2

    # l2에 대한 절반의 거리 계산 (l2 = l1 / ratio)
    half_l2 = half_l1 * ratio

    # p3, p4 구하기
    p3 = (mid_point[0] - half_l2 * math.cos(angle), mid_point[1] - half_l2 * math.sin(angle))
    p4 = (mid_point[0] + half_l2 * math.cos(angle), mid_point[1] + half_l2 * math.sin(angle))
    
    return np.round(p3, 3), np.round(p4, 3)



def interpolate_points(p1, p2, num_string):
    # 간격을 구합니다
    x_step = (p2[0] - p1[0]) / (num_string - 1)
    y_step = (p2[1] - p1[1]) / (num_string - 1)

    points = []
    for i in range(0, num_string):
        points.append((p1[0] + i * x_step, p1[1] + i * y_step))
        
    return points

def check_point_in_polygon(point, polygon):
    path = Path(polygon)
    return path.contains_point(point)

def draw_space(image, points, color=(0, 0, 255)):
    if len(points) == 4:  # 사각형 모양의 주차 공간을 그립니다.
        cv2.polylines(image, [points], True, color, 2)

def sort_polygons(roiList):
    # Calculate the center coordinates for each polygon
    center_coordinates = [(np.mean(polygon, axis=0), idx) for idx, polygon in enumerate(roiList)]

    # Find the leftmost points
    x_small = min(center_coordinates, key=lambda x: x[0][0])[0][0]
    leftmost_points = [center for center in center_coordinates if (x_small - 10) <= center[0][0] <= (x_small + 10)]

    # From the leftmost points, find the topmost point
    topmost_leftmost_point = min(leftmost_points, key=lambda x: x[0][1])[0]

    # Calculate the absolute horizontal and vertical distances from each point to the leftmost point
    distances = [(idx, np.abs(center[0] - topmost_leftmost_point[0]), np.abs(center[1] - topmost_leftmost_point[1])) for center, idx in center_coordinates]
    # print('D:',distances)
    # Sort the polygons first by vertical distance, then by horizontal distance
    sorted_indices = [idx for idx, dx, dy in sorted(distances, key=lambda x: (x[1], x[2]))]

    # Reorder the polygons according to the sorted indices
    sorted_roiList = [roiList[idx] for idx in sorted_indices]

    return sorted_roiList


def find_chord(idx_pressed):
    
        # Convert input list to set for efficient membership checks
    arr_set = set(idx_pressed)

    # Check for each condition and return appropriate value
    
    if {19, 26, 34}.issubset(arr_set): #C chord
        return 0
    elif {18, 25, 23}.issubset(arr_set): # G chord
        return 1
    elif {26, 27}.issubset(arr_set): # Am chord
        return 2
    elif {30, 19, 20}.issubset(arr_set): # F chord
        return 3
    elif {25, 26}.issubset(arr_set):    # Em chord
        return 4
    else:
        return -1
    
    
    

def processor(frame, points, finger_points):
    if len(points) == 4:
        rect_points = arrange_rectangle_points(points)

        stiker_fretEdge_ratio = 0.75
        left_stiker_points = find_points(rect_points[0], rect_points[3], stiker_fretEdge_ratio)
        right_stiker_points = find_points(rect_points[1], rect_points[2], stiker_fretEdge_ratio)

        for point in left_stiker_points:
            cv2.circle(frame, (int(point[0]), int(point[1])), radius=5, color=(255, 255, 0), thickness=-1)

        for point in right_stiker_points:
            cv2.circle(frame, (int(point[0]), int(point[1])), radius=5, color=(255, 255, 0), thickness=-1)

        fretEdge_string_ratio = 0.8
        left_string_points = find_points(left_stiker_points[0], left_stiker_points[1], fretEdge_string_ratio)
        right_string_points = find_points(right_stiker_points[0], right_stiker_points[1], fretEdge_string_ratio)

        fret_rect_points = left_stiker_points + right_stiker_points
        fret_rect_points = [(int(round(point[0])), int(round(point[1]))) for point in fret_rect_points]      

        test_tl = left_string_points[0]
        test_bl = left_string_points[1]
        test_tr = right_string_points[0]
        test_br = right_string_points[1]
        points_tl_bl = interpolate_points(test_tl, test_bl, 6)
        points_tr_br = interpolate_points(test_tr, test_br, 6)
        for point in points_tl_bl:
            cv2.circle(frame, (int(point[0]), int(point[1])), radius=5, color=(0, 255, 0), thickness=-1)

        for point in points_tr_br:
            cv2.circle(frame, (int(point[0]), int(point[1])), radius=5, color=(0, 255, 0), thickness=-1)

        for i in range(len(points_tl_bl)):
            point_left = points_tl_bl[i]
            point_right = points_tr_br[i]
            cv2.line(frame, (int(point_left[0]), int(point_left[1])), (int(point_right[0]), int(point_right[1])), (0, 255, 255), thickness=2)

        divided_points = []
        ratios = np.array([0, 1.5, 1.7, 1.85, 2.1, 2.25, 2.7]) / 12.05  # 퍼센트로 변환

        for i in range(6):
            # 각 좌, 우 point간의 벡터를 계산
            vector = np.subtract(points_tr_br[i], points_tl_bl[i])
            previous_point = points_tl_bl[i]

            for ratio in ratios:
                # 각 등분점을 계산하여 저장
                divided_point = np.add(previous_point, vector * ratio)
                divided_points.append(tuple(divided_point))
                previous_point = divided_point
        # print(divided_points)
#-----------------------------------------------------------------------------#
        rect_divided_points_6 = []
        vector = np.subtract(rect_points[1], rect_points[0])
        previous_point = rect_points[0]

        for ratio in ratios:
                # 각 등분점을 계산하여 저장
                rect_divided_point_6 = np.add(previous_point, vector * ratio)
                rect_divided_points_6.append(tuple(rect_divided_point_6))
                previous_point = rect_divided_point_6

        rect_divided_points_6 = [tuple(map(int, point)) for point in rect_divided_points_6]
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
        rect_divided_points_1 = []
        vector = np.subtract(rect_points[2], rect_points[3])
        previous_point = rect_points[3]

        for ratio in ratios:
                # 각 등분점을 계산하여 저장
                rect_divided_point_1 = np.add(previous_point, vector * ratio)
                rect_divided_points_1.append(tuple(rect_divided_point_1))
                previous_point = rect_divided_point_1

        rect_divided_points_1 = [tuple(map(int, point)) for point in rect_divided_points_1]
#-----------------------------------------------------------------------------#
        

        divided_points = [tuple(map(int, point)) for point in divided_points]
        # 각각의 점들을 표시합니다
        for point in divided_points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)  # 점을 그립니다. 
        # 인덱스 1부터 6까지 추출 / 6번 줄(두꺼운거)
        divided_points_6 = divided_points[:num_width]
        # 인덱스 30부터 36까지 추출 / 1번 줄(얇은거)
        divided_points_1 = divided_points[(num_width-1)*num_height:num_width*num_height]
        
        middle_points_vertical = []
        # 리스트를 6개의 점씩 나눕니다.
        divided_points_split = [divided_points[i::num_width] for i in range(num_width)]

        for points in divided_points_split:
            for i in range(num_height-1):  # 각 세로줄에 대해
                # 시작점과 끝점의 중간점을 계산합니다.
                middle_point = ((points[i][0] + points[i+1][0]) / 2, (points[i][1] + points[i+1][1]) / 2)
                middle_points_vertical.append(middle_point)

        # middle_points_vertical 리스트에는 각 세로줄의 중간점이 저장되어 있습니다.

        # 각각의 점들을 표시합니다
        for point in middle_points_vertical:
            pt = tuple(map(int, point))
            cv2.circle(frame, pt, 5, (0, 100, 100), -1)  # 점을 그립니다.
        
        
        middle_points_vertical = [tuple(map(int, point)) for point in middle_points_vertical]


#-------------------------------------------------- 점 배열 방법 결정-------------------------------------------#
        # 가장 왼쪽에 있는 점을 찾습니다
        leftmost_point = min(middle_points_vertical, key=lambda x: x[0])

        # 각 점과 가장 왼쪽에 있는 점과의 수평 거리와 수직 거리를 계산합니다
        distances = [(point, np.abs(point[0]-leftmost_point[0]), np.abs(point[1]-leftmost_point[1])) for point in middle_points_vertical]

        # 먼저 수직 거리에 따라 점들을 정렬하고, 그다음 수평 거리에 따라 점들을 정렬합니다
        points_sorted = [point for point, dx, dy in sorted(distances, key=lambda x: (x[2], x[1]))]
#-----------------------------------------------------------------------------------------------------------------#

        #ROI 설정
        roiList = []

        # 정렬된 points에 대해 반복, 4개 점마다 하나의 사각형을 만듭니다
        ROI_height = num_height -1
        for i in range(0, len(middle_points_vertical)-(ROI_height+1), 1): 

            if i % ROI_height != ROI_height-1:
                # 사각형의 꼭짓점을 추출
                p1 = middle_points_vertical[i]
                p2 = middle_points_vertical[i+1]
                p3 = middle_points_vertical[i+ROI_height]
                p4 = middle_points_vertical[i+ROI_height+1]
                
                # 점들의 좌표를 int로 변환 (OpenCV는 float를 인식하지 않음)
                p1 = tuple(map(int, p1))
                p2 = tuple(map(int, p2))
                p3 = tuple(map(int, p3))
                p4 = tuple(map(int, p4))

                # 사각형을 형성하는 점들
                rect_points = [p1, p2, p3, p4]
                rect_points = arrange_rectangle_points(rect_points)

                roiList.extend([np.array(rect_points)])

                # 사각형 그리기
                cv2.polylines(frame, [np.array(rect_points, dtype=np.int32)], True, (0,255,0), 1)

        for i in range(0, num_width-1, 1):
            p1 = rect_divided_points_6[i]
            p2 = rect_divided_points_6[i+1]
            p3 = middle_points_vertical[i*ROI_height]
            p4 = middle_points_vertical[(i+1)*ROI_height]

            # 점들의 좌표를 int로 변환 (OpenCV는 float를 인식하지 않음)
            p1 = tuple(map(int, p1))
            p2 = tuple(map(int, p2))
            p3 = tuple(map(int, p3))
            p4 = tuple(map(int, p4))

            # 사각형을 형성하는 점들
            rect_points = [p1, p2, p3, p4]
            rect_points = arrange_rectangle_points(rect_points)
            roiList.extend([np.array(rect_points)])

            # 사각형 그리기
            cv2.polylines(frame, [np.array(rect_points, dtype=np.int32)], True, (255,255,255), 1)

        for i in range(0, num_width-1, 1):
            p1 = rect_divided_points_1[i]
            p2 = rect_divided_points_1[i+1]
            p3 = middle_points_vertical[(i+1)*ROI_height-1]
            p4 = middle_points_vertical[(i+2)*ROI_height-1]

            # 점들의 좌표를 int로 변환 (OpenCV는 float를 인식하지 않음)
            p1 = tuple(map(int, p1))
            p2 = tuple(map(int, p2))
            p3 = tuple(map(int, p3))
            p4 = tuple(map(int, p4))

            # 사각형을 형성하는 점들
            rect_points = [p1, p2, p3, p4]
            rect_points = arrange_rectangle_points(rect_points)
            roiList.extend([np.array(rect_points)])

            # 사각형 그리기
            cv2.polylines(frame, [np.array(rect_points, dtype=np.int32)], True, (255,255,255), 1)
 
        sorted_roiList = sort_polygons(roiList)

        for idx, polygon in enumerate(sorted_roiList):
            # Calculate the moments of the polygon
            M = cv2.moments(polygon)
            if M["m00"] != 0:  # avoid division by zero
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            # Place index as text on the polygon
            cv2.putText(frame, str(idx), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        idx_pressed = []
        
        if finger_points:
            for finger_point in finger_points:
                for idx, polygon in enumerate(sorted_roiList):
                    if check_point_in_polygon(finger_point, polygon):
                        idx_pressed.append(idx)
                        draw_space(frame, polygon, color=(0, 0, 255))

                        # Calculate the moments of the polygon
                        M = cv2.moments(polygon)
                        if M["m00"] != 0:  # avoid division by zero
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                        else:
                            cX, cY = 0, 0
                        # Place index as text on the polygon
                        cv2.putText(frame, str(idx), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        
    
        
    global last_chord
    update_chord(idx_pressed)

    if last_chord is not None:
        cv2.putText(frame, str(last_chord), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 4, cv2.LINE_AA)
        # Generate frames and write them to the video file
        x_position = 300
        for i, (chord, duration) in enumerate(Chord_Time):
            hyphens = round(duration*0.7)  # 1초마다 하이픈 추가. 역수 곱해라
            repr = chord_repr[chord]
            for _ in range(hyphens):
                for j, string in enumerate(repr):
                    cv2.putText(frame, string, (x_position, 50 + j * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                x_position += 50
    return idx_pressed




