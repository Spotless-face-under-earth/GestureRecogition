from queue import Queue
import time
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from threading import Thread
from collections import deque  # 使用双端队列记录历史位置

def get_hand_gesture(landmarks):
    """
    根据关键点坐标判断手势类型：
    - "thumb_up"（大拇指）
    - "wave"（摇手）
    - "fist"（握拳）
    - "unknown"（未知）
    """
    global wrist_history

    # 逻辑判断
    if is_thumb_up(landmarks):  # 拇指朝上,五指握拳
        return "thumb up"
    elif is_fist(landmarks):  # 拇指和食指接触（握拳）
        return "fist"
    elif is_open_palm(landmarks):  # 先判断是否是手掌
        wrist = landmarks[0]  # 手腕坐标
         # 记录当前手腕位置（归一化坐标）
        wrist_history.append((wrist.x, wrist.y))
        result_queue = Queue()
        wave_thread = Thread(target=is_waving, args=(landmarks,result_queue))  # 创建线程
        wave_thread.start()
        wave_thread.join()      # 等待线程完成
        if not result_queue.empty():
            if result_queue.get():
                return "wave hand"
        else:
            return "palm"
        return "palm"
    else:  # 其他情况
        return "unknown"

# 判断是否点赞
def is_thumb_up(landmarks):
    # 关键点
    thumb_tip = landmarks[4]
    # thumb_base = landmarks[1]
    thumb_ip = landmarks[3]
    index_tip = landmarks[8]
    index_ip = landmarks[7]
    index_base = landmarks[5]
    middle_base = landmarks[9]
    ring_base = landmarks[13]
    
    # 计算手部基准尺度
    palm_width = abs(landmarks[5].x - landmarks[17].x)
    
    # 条件1：拇指竖直分量
    thumb_vector_y = thumb_ip.y - thumb_tip.y  # 指尖的y更小
    vertical_condition = thumb_vector_y > (palm_width * 0.3)
    
    # 条件2：拇指-食指二端分离
    thumb_index_ip_dist = np.linalg.norm([thumb_tip.x - index_ip.x, thumb_tip.y - index_ip.y])
    separation_condition = thumb_index_ip_dist > (palm_width * 0.5)

    # 条件3：食指尖-其他手指根部距离
    thumb_index_tip_dist = np.linalg.norm([index_tip.x - index_base.x, index_tip.y - index_base.y])
    thumb_index_tip_condition = thumb_index_tip_dist < (palm_width * 1.0)
    index_middle_tip_dist = np.linalg.norm([index_tip.x - middle_base.x, index_tip.y - middle_base.y])
    index_middle_condition = index_middle_tip_dist < (palm_width * 1.0)
    index_dist = np.linalg.norm([ 2*(index_tip.x) - (index_base.x + middle_base.x), 2*(index_tip.y) - (index_base.y + middle_base.y) ])
    index_condition = index_dist < (palm_width * 1.5)
    
    # # 条件3：拇指朝前（可选）
    # z_condition = thumb_tip.z < thumb_ip.z - 0.05
    
    return vertical_condition and separation_condition and (thumb_index_tip_condition or index_middle_condition or index_condition) # and z_condition

# OK我感觉这里写的很OK了
def is_fist(landmarks, threshold_scale=0.7):
    # 关键点定义
    fingertips = [4, 8, 12, 16, 20]  # 指尖
    palm_indices = [0, 1, 5, 9, 13, 17]  # 掌心参考点
    
    # 计算掌心（2D）
    palm_points = np.array([(landmarks[i].x, landmarks[i].y) for i in palm_indices])
    palm_center = np.mean(palm_points, axis=0)
    
    # 动态阈值（基于手掌宽度）
    palm_width = abs(landmarks[1].x - landmarks[17].x)
    dynamic_threshold = palm_width * threshold_scale
    
    # 检查所有指尖
    for tip in fingertips:
        tip_pos = np.array([landmarks[tip].x, landmarks[tip].y])
        dist = np.linalg.norm(tip_pos - palm_center)
        if tip == 4:  # 拇指
            if dist > 2*dynamic_threshold:
                return False
        elif dist > dynamic_threshold:
            return False
    return True

# 判断是否成手掌
def is_open_palm(landmarks):
    fingertips = [4, 8, 12, 16, 20]
    palm_center = np.mean([(landmarks[i].x, landmarks[i].y) for i in [0, 5, 9, 13, 17]], axis=0)

    # 动态阈值（基于手掌宽度）
    palm_width = abs(landmarks[1].x - landmarks[17].x)
    dynamic_threshold = palm_width * 1.0

    # 所有指尖远离掌心
    return all(
        np.linalg.norm([landmarks[i].x - palm_center[0], landmarks[i].y - palm_center[1]]) > dynamic_threshold
        for i in fingertips
    )

# 判断是否摇手
def is_waving(landmarks, result_queue):
    """
    通过历史位置判断是否摇手（横向移动超过阈值）
    """
    # 动态阈值（基于手掌宽度）
    palm_width = abs(landmarks[1].x - landmarks[17].x)
    dynamic_threshold = palm_width * 1.0
    if len(wrist_history) < HISTORY_MAXLEN:
        result_queue.put(False)
        return False
    
    total_movement = sum(
        abs(wrist_history[i][0] - wrist_history[i-1][0])
        for i in range(1, len(wrist_history))
    )
    if total_movement > 1.0 * dynamic_threshold:
        result_queue.put(True)
        return True
    else:
        result_queue.put(False)  
        return False


# 打开摄像头
cap = cv2.VideoCapture(0)
last_gesture = "unknow"

# 初始化 MediaPipe 手部检测
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# 初始化语音引擎
engine = pyttsx3.init()

# 全局变量：记录手腕位置历史（最多保存5帧）
HISTORY_MAXLEN = 5
wrist_history = deque(maxlen=HISTORY_MAXLEN)  # 存储(x, y)坐标

# 全局变量控制暂停
last_gesture_time = 0
PAUSE_DURATION = 3  # 暂停3秒

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # 转换为 RGB 并检测手势
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if time.time() - last_gesture_time < PAUSE_DURATION:
        # 显示手势名称
        cv2.putText(
            frame, last_gesture, (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 绘制关键点
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # 获取手势类型
            current_gesture = get_hand_gesture(
                [lm for lm in hand_landmarks.landmark]
            )

            last_gesture_time = time.time()

            # 显示手势名称
            cv2.putText(
                frame, current_gesture, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            # 语音提示（仅当手势变化时）
            if current_gesture != "unknown":
                # 语音提示
                engine.say(f"Detected {current_gesture}")
                engine.runAndWait()
                last_gesture = current_gesture

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()