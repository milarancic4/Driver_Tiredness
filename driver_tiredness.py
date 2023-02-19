import cv2
import dlib
from math import hypot
import time
import os
from playsound import playsound
import easygui
import pymsgbox
import simpleaudio as sa

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


blinking_threshold = 4.1
start_blinking_threshold = 3.85
font = cv2.FONT_HERSHEY_PLAIN
last_blinking_time = 0
frequently_blinking_rate_time_threshold = 2
blinking_frequency = frequently_blinking_rate_time_threshold  
frequently_blinking_frame_count = 0 

frequently_blinking_alert_points = 0
blinking_duration_alert_points = 0
blinking_duration_frame_count = 0
total_alert_points = 0
eyes_were_closed = False
blinking_duration_frame_count_threshold = 13
wave_obj = sa.WaveObject.from_wave_file('mixkit-alarm-tone-996.wav')
total_alert_points = 0
alarm_occurred = False


def get_frame_rate():
    for i in range(0, 120):
        ret, frame = cap.read()
    end = time.time()

    seconds = end - start
    print("Time taken : {0} seconds".format(seconds))
    # Calculate frames per second
    fps = 120 / seconds
    print("Estimated frames per second : {0}".format(fps))


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def calculate_ear(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length: float = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_length / ver_line_length

    return ratio


def show_alert_message(alert_level):
    return pymsgbox.confirm(text='You need to rest! Take a break!', title='ALERT', buttons=['OK', 'Cancel'])


def check_tiredness():
    global alarm_occurred
    global total_alert_points
    if total_alert_points > 10000:
        alarm_occurred = True
        play_obj = wave_obj.play()
        response = show_alert_message(total_alert_points / 1000)
        if response == 'OK':
            alarm_occurred = False
            total_alert_points = 0
            if play_obj.is_playing():
                play_obj.stop()


def show_warning_message():
    pymsgbox.alert('WARNING', 'WARNING', button='OK')

def facePoints2(image, faceLandmarks, color=(0, 255, 0), radius=2):
  for p in faceLandmarks.parts():
    cv2.circle(image, (p.x, p.y), radius, color, -1)

while True:
    if not alarm_occurred:
        check_tiredness()
    start = time.time()
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:

        landmarks = predictor(gray, face)
        # facePoints2(frame,landmarks)

        left_ear = calculate_ear([36, 37, 38, 39, 40, 41], landmarks)
        right_ear = calculate_ear([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_ear + right_ear) / 2

        if blinking_ratio > start_blinking_threshold:  # 1st STATE - BLINKING HAS STARTED
            blinking_duration_frame_count += 1

            if blinking_ratio > blinking_threshold:  # 2nd STATE - BLINKED
                eyes_were_closed = True
                if blinking_duration_frame_count > blinking_duration_frame_count_threshold:
                    # print(blinking_frame_count)
                    blinking_duration_alert_points += (blinking_duration_frame_count * blinking_duration_frame_count)
                    total_alert_points += blinking_duration_alert_points
        else:  # 3rd STATE BEGINS - EYES ARE OPENNED AGAIN
            if blinking_duration_frame_count > blinking_duration_frame_count_threshold:
                # print(blinking_frame_count)
                blinking_duration_alert_points += (blinking_duration_frame_count * blinking_duration_frame_count)
                # print("blinking_duration_alert_points", blinking_duration_alert_points)
            total_alert_points += blinking_duration_alert_points
            blinking_duration_alert_points = 0
            blinking_duration_frame_count = 0

            if eyes_were_closed:
                eyes_were_closed = False
                # Eyes were closed which means that blinking has happened,
                # so we need to calculate blinking frequency
                current_blinking_time = time.time()
                if last_blinking_time != 0:  # to handle 0th case
                    blinking_frequency = time.time() - last_blinking_time

                last_blinking_time = current_blinking_time
                if blinking_frequency < frequently_blinking_rate_time_threshold:
                    frequently_blinking_critical_ratio = frequently_blinking_rate_time_threshold / blinking_frequency
                    frequently_blinking_alert_points += frequently_blinking_critical_ratio
                    frequently_blinking_frame_count += 1
                else:
                    # Blink rate is in the 'normal' range again, reset everything
                    if frequently_blinking_frame_count > 5:
                        frequently_blinking_alert_points *= frequently_blinking_frame_count
                        print("frequently_blinking_alert_points: ", frequently_blinking_alert_points)
                        total_alert_points += (frequently_blinking_alert_points * 2)
                        frequently_blinking_frame_count = 0
                        frequently_blinking_alert_points = 0


    cv2.imshow("Video", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
