'''
업데이트 내역

- 손 관절 인식 가능해짐
- Opencv tracker 사용해서 네 ROI 지정 후 사각형 그려줌

'''


import cv2
import numpy as np
import mediapipe
import myFunc

drawingModule = mediapipe.solutions.drawing_utils
handsModule   = mediapipe.solutions.hands

# Tracker creators
trackers = [cv2.TrackerBoosting_create,
            cv2.TrackerMIL_create,
            cv2.TrackerKCF_create,
            cv2.TrackerTLD_create,
            cv2.TrackerMedianFlow_create,
            cv2.TrackerGOTURN_create,
            cv2.TrackerCSRT_create,
            cv2.TrackerMOSSE_create]

trackerIdx = 6 
trackers_instances = [None] * 4
isFirst = [True] * 4
bbox_center = [None] * 4

video_src = "test_0614.mp4"
# video_src = 0

win_name = 'Tracking APIs'

cap = cv2.VideoCapture(video_src)
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000/fps)


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('fret_detect.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))


with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None: break
        # frame  = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # 전체 손가락 그려주는거
        #res
        
        # Finger 1: 검지
        # Finger 2: 중지
        # Finger 3: 약지
        # Finger 4: 소지
            
    
        
        
            
        # If finger detected..
        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS) 
                

                finger_1 = [handLandmarks.landmark[handsModule.HandLandmark(8).value].x, handLandmarks.landmark[handsModule.HandLandmark(8).value].y]
                finger_2 = [handLandmarks.landmark[handsModule.HandLandmark(12).value].x, handLandmarks.landmark[handsModule.HandLandmark(12).value].y]
                finger_3 = [handLandmarks.landmark[handsModule.HandLandmark(16).value].x, handLandmarks.landmark[handsModule.HandLandmark(16).value].y]
                finger_4 = [handLandmarks.landmark[handsModule.HandLandmark(20).value].x, handLandmarks.landmark[handsModule.HandLandmark(20).value].y]
        
        else:
            finger_1 = [0, 0]
            finger_2 = [0, 0]
            finger_3 = [0, 0]
            finger_4 = [0, 0]
        #results = cv2.circle(frame, (finger_1[0], finger_1[1]), 1, (255, 0, 0))
                
            #print(f'{handsModule.HandLandmark(8).name}:') 
            #print(f'x: {handLandmarks.landmark[handsModule.HandLandmark(8).value].x}')
            #print(f'y: {handLandmarks.landmark[handsModule.HandLandmark(8).value].y}')
            
 
        if not ret:
            print('Cannot read video file')
            break

        img_draw = frame.copy()

        # Draw instructions
        if not all([t for t in trackers_instances]):
            cv2.putText(img_draw, "Press the Space to set ROI!!", \
                        (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2, cv2.LINE_AA)

        for i in range(4):
            if trackers_instances[i]:
                ok, bbox = trackers_instances[i].update(frame)
                (x, y, w, h) = bbox
                if ok:
                    cv2.rectangle(img_draw, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2, 1)
                    # Save bbox center
                    bbox_center[i] = (int(x + w / 2), int(y + h / 2))
                else:
                    cv2.putText(img_draw, f"Tracking {i+1} fail.", (100, 80 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        if all(bbox_center):
            # Create a square
            pts = np.array(bbox_center, np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(img_draw, [pts], True, (0,255,255))
            
            new_list = [
                (int(finger_1[0]*frame_width), int(finger_1[1]*frame_height)), 
                (int(finger_2[0]*frame_width), int(finger_2[1]*frame_height)), 
                (int(finger_3[0]*frame_width), int(finger_3[1]*frame_height)), 
                (int(finger_4[0]*frame_width), int(finger_4[1]*frame_height))
                ]
            # print(new_list)

            myFunc.processor(frame, bbox_center, new_list)
            
            

        cv2.imshow(win_name, img_draw)
        cv2.imshow('test_new', frame)
        out.write(frame)
        key = cv2.waitKey(delay) & 0xff

        if key == ord(' ') or (video_src != 0 and any(isFirst)):
            for i in range(4):
                if not trackers_instances[i]:
                    isFirst[i] = False
                    roi = cv2.selectROI(win_name, frame, False)
                    if roi[2] and roi[3]:
                        trackers_instances[i] = trackers[trackerIdx]()
                        isInit = trackers_instances[i].init(frame, roi)
                    break
        elif key in range(48, 56):
            trackerIdx = key-48
            for i in range(4):
                if trackers_instances[i] is not None:
                    trackers_instances[i] = trackers[trackerIdx]()
                    isInit = trackers_instances[i].init(frame, bbox)
        elif key == 27 : 
            break
    else:
        print( "Could not open video")
        
    

cap.release()
cv2.destroyAllWindows()
