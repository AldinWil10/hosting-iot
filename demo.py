from flask import Flask, render_template, Response
import cv2
import datetime
import time

app = Flask(__name__)

# Load the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the database (the .xml file)
recognizer.read('latihwajah/training.xml')

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the database of known faces (ID to Name mapping)
id_to_name = {1: "Aldin", 2: "Wildan", 3: 'Ramelan', 4: 'Alex', 5:'Maul'}

# Start the video capture
cap = cv2.VideoCapture(0)

minWidth = 0.1*cap.get(3)
minHeight = 0.1*cap.get(4)
prev_time = 0
curr_time = 0
start_time = time.time()

# Loop through the frames of the video
def generate_frames():
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        curr_time = cv2.getTickCount()
        time_elapsed = (curr_time - prev_time) / cv2.getTickFrequency()
        fps = 1 / time_elapsed
        today = datetime.datetime.now()
        date_str = today.strftime("%d-%m-%Y")

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Loop through each face in the frame
        for (x, y, w, h) in faces:
            # Extract the face region of interest (ROI)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Resize the face ROI to a fixed size
            roi_gray = cv2.resize(roi_gray, (200, 200), interpolation=cv2.INTER_LINEAR)

            # Recognize the face using the LBPH algorithm
            id_, confidence = recognizer.predict(roi_gray)
            
            # Check if the confidence level is below a certain threshold (i.e. the face is unknown)
            if confidence > 100:
                # Draw a rectangle around the face and display the label "Unknown"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, 'Unknown', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, date_str, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Draw a rectangle around the face and display the name of the recognized person
                name = id_to_name[id_]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, date_str, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

prev_time = curr_time
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

# # Release the video capture and close all windows
# cap.release()
# cv2.destroyAllWindows()
