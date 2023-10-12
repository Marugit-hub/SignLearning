import cv2
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from flask import Flask, render_template, Response, redirect

app = Flask(__name__)
cap = None
detector = HandDetector(maxHands=1)
classifier = Classifier("Models/keras_model.h5", "Models/labels.txt")
offset = 20
labels = ["A", "B", "C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
current_label_index = 0
course_completed = False
button_pressed = False
webcam_started = False
aspect_ratio = 0
detected_label = None
img_output=None

def generate_frames():
    global current_label_index, course_completed, button_pressed, webcam_started, aspect_ratio, detected_label

    while not course_completed:
        try:
            success, img = cap.read()
            if not success:
                break

            img_output = img.copy()
            hands, img = detector.findHands(img)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                img_white = np.ones((300, 300, 3), np.uint8) * 255
                img_crop = img[y-offset:y+h+offset, x-offset:x+w+offset]

                aspect_ratio = h / w

                if aspect_ratio > 1:
                    k = 300 / h
                    w_cal = math.ceil(k * w)
                    img_resize = cv2.resize(img_crop, (w_cal, 300))
                    w_gap = math.ceil((300 - w_cal) / 2)
                    img_white[:, w_gap:w_cal + w_gap] = img_resize
                    prediction, index = classifier.getPrediction(img_white, draw=False)
                else:
                    k = 300 / w
                    h_cal = math.ceil(k * h)
                    img_resize = cv2.resize(img_crop, (300, h_cal))
                    h_gap = math.ceil((300 - h_cal) / 2)
                    img_white[h_gap:h_cal+h_gap, :] = img_resize
                    prediction, index = classifier.getPrediction(img_white, draw=False)

                detected_label = labels[index]

                if current_label_index == index:
                    if detected_label == "A":
                        if not button_pressed:
                            cv2.putText(img_output, detected_label, (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
                        else:
                            button_pressed = False  
                else:
                    detected_label = "unknown"

                cv2.putText(img_output, detected_label, (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', img_output)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print("An error occurred:", str(e))
            cv2.putText(img_output, "error", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', img_output)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/start_webcam')
def start_webcam():
    global cap, webcam_started
    cap = cv2.VideoCapture(0)
    webcam_started = True
    return redirect('/index')

@app.route('/')
def home():
    return render_template("Final_Home.html")

@app.route('/Explore')
def Explore():
    return render_template("Final_Explore.html")

@app.route('/History')
def History():
    return render_template("Final_History.html")

@app.route('/finish')
def finish():
    return render_template("finish.html")

@app.route('/index')
def index():
    global course_completed, button_pressed, current_label_index, webcam_started, aspect_ratio, detected_label

    if course_completed:
        cap.release()
        return render_template("finish.html")

    return render_template('index.html', current_label=labels[current_label_index], button_pressed=button_pressed, webcam_started=webcam_started, aspect_ratio=aspect_ratio)


@app.route('/detect_next_label', methods=['GET'])
def detect_next_label():
    global button_pressed, current_label_index, course_completed, cap, aspect_ratio, detected_label

    if not webcam_started and button_pressed == True and not detected_label:
        return "failure"

    if detected_label is None or detected_label == "unknown" or detected_label != labels[current_label_index]:
        return "failure"

    button_pressed = True

    if current_label_index >= len(labels) - 1:
        course_completed = True
        return redirect('/index')

    current_label_index += 1
    button_pressed = False

    aspect_ratio = 0

    return "success"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
