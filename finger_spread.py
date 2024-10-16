import cv2
import mediapipe as mp
import numpy as np
import math
import subprocess 

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# Get screen dimensions.
cap = cv2.VideoCapture(0)
screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize variables for color cycling.
color_index = 0
color_palette = [
    (255, 0, 0),   # R
    (255, 127, 0), # O
    (255, 255, 0), # Y
    (0, 255, 0),   # G
    (0, 0, 255),   # B
    (75, 0, 130),  # I
    (148, 0, 211)  # V
]

# FPS variables
prev_frame_time = 0
new_frame_time = 0

def get_current_volume():
    # Get the current system volume using osascript
    output = subprocess.check_output(["osascript", "-e", "output volume of (get volume settings)"])
    return int(output.strip())

def set_volume(volume):
    # Set the system volume using osascript
    volume = max(0, min(100, int(volume * 100)))  # Ensure volume is between 0 and 100
    subprocess.call(["osascript", "-e", f"set volume output volume {volume}"])

while cap.isOpened():
    # Read a frame from the webcam.
    success, image = cap.read()
    if not success:
        continue

    # Flip the image horizontally.
    image = cv2.flip(image, 1)

    # Convert the image to RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands.
    results = hands.process(image_rgb)

    # Get color for current frame.
    color = color_palette[color_index]
    color_index = (color_index + 1) % len(color_palette)
    
    # Draw hand landmarks and calculate volume.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
            )
            
            # Calculate distance between thumb and index finger.
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = math.dist((thumb_tip.x, thumb_tip.y), (index_tip.x, index_tip.y))

            # Map distance to volume range (0 to 1).
            volume = np.interp(distance, [0, 0.2], [0, 1])  # Adjust [0, 0.2] based on your setup
            volume = np.clip(volume, 0, 1)
            
            # Set the system volume.
            set_volume(volume)

            # Draw line between thumb and index finger.
            cv2.line(image, (int(thumb_tip.x * screen_width), int(thumb_tip.y * screen_height)), 
                     (int(index_tip.x * screen_width), int(index_tip.y * screen_height)), color, 2)

    # Calculate and display FPS.
    new_frame_time = cv2.getTickCount()
    fps = 1/(new_frame_time - prev_frame_time) * cv2.getTickFrequency()
    prev_frame_time = new_frame_time
    cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Display current volume.
    current_volume = get_current_volume()
    cv2.putText(image, f"Volume: {int(current_volume)}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Display the image.
    cv2.imshow('Hand Tracker', image)
    
    # Exit on 'q' key press.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Cleanup.
cap.release()
cv2.destroyAllWindows()
                 
