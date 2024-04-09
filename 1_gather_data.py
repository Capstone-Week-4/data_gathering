import cv2
import mediapipe as mp
import numpy as np
import time, os
import shutil

# 실행경로를 설정한다.
# 현재 .py파일의 상위 폴더를 cwd로 설정한다.
CURDIR = os.path.realpath(__file__)
os.chdir(os.path.dirname(CURDIR))

# 학습하려는 단어의 개수를 입력
num = int(input("INSERT THE NUMBER OF SIGNS : "))
actions = []
for i in range(num):
    actions.append(input(f"[{i+1}]INSERT THE VOCABULARY EACH : "))

# 각 동작을 30초씩 프레임 단위로 기록한다.
secs_for_action = 30

# Mediapipe 인식 model을 정의한다.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2, # 한 손만을 인식할 수 있도록 한다.
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 손 동작 학습을 위한 웹캠을 정의한다.
cap = cv2.VideoCapture(0)

# 새롭게 생성할 데이터셋의 생성시점을 정의하고
# 데이터셋을 정의할 폴더를 생성한다.
created_time = int(time.time())

# dataset 폴더가 존재한다면, 기존 dataset 폴더 내의 데이터들을 *모두삭제* 한다.
# dataset 폴더가 존재하지 않는다면 새롭게 폴더를 생성한다.
if os.path.isdir('dataset'):
    shutil.rmtree('dataset')
os.makedirs('dataset', exist_ok=True)

while cap.isOpened():
    # A-Z 액션의 데이터를 수집하도록 한다.
    for idx, action in enumerate(actions):

        # 예) data에는 'A' 액션의 정보가 저장된다.
        data = []

        # 현재 웹캠의 이미지를 찍는다
        ret, img = cap.read()

        # 이미지 반전
        img = cv2.flip(img, 1)
        
        # 이미지를 화면에 표시
        cv2.putText(img, f"take '{action}' please", 
                    org=(10, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, 
                    fontScale=1, color=(0, 255, 0), 
                    thickness=2)
        cv2.imshow('img', img)

        # 3초간 대기했다가 본격적으로 사진을 촬영할 수 있도록 한다.
        cv2.waitKey(2000)

        start_time = time.time()
        
        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()
            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 촬영한 이미지로부터 mediapipe - 관절 할당하기
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # ** 손이 두 쪽 다 인식되었을 때를 기준으로 모델을 생성함 **
            if result.multi_hand_landmarks:
                if len(result.multi_handedness) == 2:
                    lr_map = {} 
                    for res, res2 in zip(result.multi_hand_landmarks, result.multi_handedness):
                        # 인식된 손이 어느 쪽인지 which_hand 변수에 할당한다.
                        if res2.classification[0].label == "Left":
                            which_hand = 0
                        elif res2.classification[0].label == "Right":
                            which_hand = 1
                        else:
                            continue
                        
                        # 해당 이미지의 joint 정보를 저장할 벡터 생성
                        joint = np.zeros((21, 4))
                        for j, lm in enumerate(res.landmark):

                            # 21개 joint point의 x/y/z 좌표와 visibility를 저장
                            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                        # 인접한 joint 간의 관계를 계산하여 벡터화
                        # 20개의 마디를 저장한다고 이해할 수 있음
                        v1 = joint[[0,1,2,3,0,5,6,7,5,0,9,10,11,9,0,13,14,15,13,0,17,18,19], :3] # Parent joint
                        v2 = joint[[1,2,3,4,5,6,7,8,9,9,10,11,12,13,13,14,15,16,17,17,18,19,20], :3] # Child joint
                        v = v2 - v1 # [23, 3]

                        # 20개의 관절 정보를 정규화한다
                        # (손가락 관절의 길이와 상관 없이 인식할 수 있도록)
                        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                        # Get angle using arcos of dot product
                        angle = np.arccos(np.einsum('nt,nt->n',
                            v[[0,1,2,4,5,6,4,8,9,10,11,9,13,14,15,16,14,18,19,20,21],:], 
                            v[[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],:])) # [21,]

                        angle = np.degrees(angle) # Convert radian to degree
                        angle = np.array(angle, dtype=np.float32)

                        if which_hand == 0:
                            lr_map["left"] = angle
                        else:
                            lr_map["right"] = angle

                        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
                    
                    if len(lr_map) > 1:
                        data_row = np.concatenate([lr_map["left"], lr_map["right"]], axis=0)
                    else:
                        continue

                    data_row = np.append(data_row, idx)
                    d = np.concatenate([joint[:,3], v.flatten(), data_row])
                    data.append(d)


            cv2.putText(img, str(round(time.time() - start_time, 2)), 
            org=(10, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, 
            fontScale=1, color=(0, 255, 0), 
            thickness=2)
            cv2.imshow('img', img)

        # 하나의 sign language에 대해 데이터셋 생성
        data = np.array(data)
        print(f"{'='*30}\n[{action}] {data.shape[0]} data are gathered")
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)
        cv2.waitKey(500)

    break