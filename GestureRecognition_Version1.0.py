# from queue import Queue
import time
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
# from threading import Thread
from collections import deque  # 使用双端队列记录历史位置
# import os
# from datetime import datetime

# # 在函数外，创建一个保存目录
# SAVE_DIR = "snapshots"
# os.makedirs(SAVE_DIR, exist_ok=True)

# 全局记录手腕历史
HISTORY_MAXLEN = 5
wrist_history = deque(maxlen=HISTORY_MAXLEN)

def get_hand_gesture(landmarks):
    global wrist_history

    # —— 1. 每帧都记录手腕横坐标 —— 
    wrist = landmarks[0]
    wrist_history.append(wrist.x)

    # —— 2. 优先检测大拇指、握拳等静态手势 —— 
    if is_thumb_up(landmarks):
        return "thumb up"
    if is_fist(landmarks):
        return "fist"

    # —— 3. 再看是不是“open palm” —— 
    if is_open_palm(landmarks):
        # 只有当我们已经有足够帧数时，才做“摇手”判断
        if len(wrist_history) >= HISTORY_MAXLEN and is_waving(landmarks):
            return "wave hand"
        else:
            return "palm"

    # —— 4. 其他都算 unknown —— 
    return "unknown"


def is_waving(landmarks):
    """
    通过横坐标 extremes & 方向变化次数 来判断左右摇手
    """
    xs = list(wrist_history)
    # 真正的掌宽
    palm_width = abs(landmarks[5].x - landmarks[17].x)
    # 只要求移动超过 palm_width * 0.5 即可
    if max(xs) - min(xs) < 0.8 * palm_width:
        return False

    # 检测方向来回变化次数
    changes = 0
    last_dir = 0
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i-1]
        if abs(dx) < 1e-3:
            continue
        dir = 1 if dx > 0 else -1
        if last_dir != 0 and dir != last_dir:
            changes += 1
        last_dir = dir

    # 至少来回一次（changes>=1）
    return changes >= 1

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

# # 判断是否摇手
# def is_waving(landmarks, result_queue):
#     """
#     判断是否摇手，通过左右摇摆超过阈值+方向变换次数判断
#     """
#     if len(wrist_history) < HISTORY_MAXLEN:
#         result_queue.put(False)
#         return False

#     # 提取历史横坐标
#     x_positions = [pos[0] for pos in wrist_history]

#     # 动态阈值
#     palm_width = abs(landmarks[1].x - landmarks[17].x)
#     min_total_movement = palm_width * 1.5
#     min_direction_changes = 2  # 至少来回摆动1次（左右方向变化2次）

#     # 计算总位移
#     total_movement = abs(x_positions[-1] - x_positions[0])

#     # 检测方向变化次数
#     direction_changes = 0
#     last_direction = 0  # -1 表示左移，1 表示右移

#     for i in range(1, len(x_positions)):
#         dx = x_positions[i] - x_positions[i - 1]
#         if abs(dx) < 1e-4:
#             continue  # 忽略微小波动

#         current_direction = 1 if dx > 0 else -1
#         if current_direction != last_direction and last_direction != 0:
#             direction_changes += 1
#         last_direction = current_direction

#     # 判断是否满足摇手条件
#     if total_movement > min_total_movement and direction_changes >= min_direction_changes:
#         result_queue.put(True)
#         return True
#     else:
#         result_queue.put(False)
#         return False


# 直接调用这个函数，得到手势的识别后的string类型变量
def get_hand_gesture_by_vedio():

    # # 打开摄像头
    # cap = cv2.VideoCapture(0)
    last_gesture = "unknown"

    # 初始化 MediaPipe 手部检测
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils

    """
    语音提示初始化，已有的话去掉这段
    """
    # 初始化语音引擎
    engine = pyttsx3.init()

    # 全局变量控制暂停
    last_gesture_time = 0
    PAUSE_DURATION = 1  # 暂停3秒
    max_plam_count = 5  # 检测到手掌超过五次且不摇手则判断为手掌
    temp_palm_count = 0

    while cap.isOpened():
        current_gesture = "unknown"
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
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # continue

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

        # cv2.imshow("Gesture Recognition", frame)
         
        # 这里需要把摇手和其他情况区分开；其他情况直接结束；检测到手掌时判断是否继续摇手

        if current_gesture != "unknown" and current_gesture != "palm":
            break
        elif current_gesture == "palm":
            # 用当前时间做文件名，避免重名
            # fn = datetime.now().strftime("palm_%Y%m%d_%H%M%S.jpg")
            # path = os.path.join(SAVE_DIR, fn)
            # cv2.imwrite(path, frame)
            temp_palm_count += 1
            if temp_palm_count > max_plam_count:
                break
        else:
            continue

        # 将if cap.isOpened()；中的if改为while后，取消注释可以实现循环识别。
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # cap.release()
    # cv2.destroyAllWindows()

    return current_gesture

if __name__ == "__main__":
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    # 直接调用这个函数，得到手势的识别后的string类型变量
    gesture = get_hand_gesture_by_vedio()
    print(gesture)
    cap.release()
    cv2.destroyAllWindows()

