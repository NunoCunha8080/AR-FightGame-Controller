import mediapipe as mp

from Switch import *
from directkeys import PressKey, ReleaseKey


# Function to draw bounding box
def draw_bbox(image, bbox):
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


# Function to calculate angle between three points
def calculate_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle if angle <= 180 else 360 - angle


def press_and_release_key(key, cont=False):
    PressKey(key)
    if cont:
        pass  # no delay
    else:
        time.sleep(0.07)  # min gap btw press and release
        ReleaseKey(key)

    # 0.35 for perfect cont, 0.01 will some cont in many attempts
    time.sleep(0.01)  # min gap time for continuous press


mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture(0)
pTime = 0

button = Buttons(training=False)
start_position = None

while cap.isOpened():
    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:

        # Display the person box
        landmarks = results.pose_landmarks.landmark
        image_height, image_width, _ = img.shape
        x_min, x_max = image_width, 0
        y_min, y_max = image_height, 0
        for landmark in landmarks:
            x, y = int(landmark.x * image_width), int(landmark.y * image_height)
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        if start_position is None:
            start_position = int(landmarks[0].x * image_width), int(landmarks[0].y * image_height)

        current_position = int(landmarks[0].x * image_width), int(landmarks[0].y * image_height)
        cv2.circle(img, start_position, 10, (0, 255, 0), -1)
        cv2.circle(img, current_position, 10, (255, 0, 0), -1)

        diff_from_initial_point = start_position[0] - current_position[0]
        if abs(diff_from_initial_point) > 70:
            if diff_from_initial_point > 0:
                cv2.putText(img, "Moving left", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
                press_and_release_key(D, True)
            else:
                cv2.putText(img, "Moving right", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
                press_and_release_key(A, True)
        else:
            ReleaseKey(A)
            ReleaseKey(D)

        diff_from_initial_point = start_position[1] - current_position[1]
        if abs(diff_from_initial_point) > 30:
            if diff_from_initial_point > 0:
                cv2.putText(img, "Moving Up", (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
                press_and_release_key(W)
            else:
                cv2.putText(img, "Moving Down", (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
                press_and_release_key(S, True)
        else:
            # ReleaseKey(W)
            ReleaseKey(S)

        # Extract keypoint coordinates
        keypoint_coords = [(lm.x, lm.y) for lm in landmarks]

        # Check kicking motion
        left_knee = keypoint_coords[mpPose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = keypoint_coords[mpPose.PoseLandmark.LEFT_ANKLE.value]
        left_hip = keypoint_coords[mpPose.PoseLandmark.LEFT_HIP.value]
        left_shoulder = keypoint_coords[mpPose.PoseLandmark.LEFT_SHOULDER.value]
        left_wrist = keypoint_coords[mpPose.PoseLandmark.LEFT_WRIST.value]

        right_knee = keypoint_coords[mpPose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = keypoint_coords[mpPose.PoseLandmark.RIGHT_ANKLE.value]
        right_hip = keypoint_coords[mpPose.PoseLandmark.RIGHT_HIP.value]
        right_shoulder = keypoint_coords[mpPose.PoseLandmark.RIGHT_SHOULDER.value]
        right_wrist = keypoint_coords[mpPose.PoseLandmark.RIGHT_WRIST.value]

        left_kick_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
        right_kick_angle = calculate_angle(right_shoulder, right_hip, right_ankle)
        left_punch_angle = calculate_angle(left_hip, left_shoulder, left_wrist)
        right_punch_angle = calculate_angle(right_hip, right_shoulder, right_wrist)

        if 120 > right_punch_angle > 70 or 120 > left_punch_angle > 70:
            cv2.putText(img, "punch detected", (10, 120), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            press_and_release_key(J)

        if right_kick_angle < 130 or left_kick_angle < 130:
            cv2.putText(img, "kick detected", (10, 140), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            press_and_release_key(K)

        draw_bbox(img, bbox)
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        start_position = None

cap.release()
cv2.destroyAllWindows()
