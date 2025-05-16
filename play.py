
import cv2
import time
import numpy as np 
import random
import sys
import os
import pygame

# Import TensorFlow and Keras for loading ML model
from tensorflow.keras.models import load_model

# Import custom utility functions
from utils import (
    initialize_pose_model,
    process_frame,
    blink_screen,
    draw_bold_text,
    extract_landmarks,
    save_data,
    predict_pose_v2,
    display_gameover_message,
)

## A dictionary mapping pose labels to integer codes as used during model training (Dict[str, int]); see training_model.ipynb.
# If model is retrained, please also make sure to adjust function 'predict_pose' accordingly.
LABEL_MAP = {"X": 0, "Hide": 1, "Pose": 2, "Squat": 3, "Stand": 4}

# A list of pose labels used to randomly select the next pose.
POSES = [pose for pose in LABEL_MAP.keys()]

# Loading Pose Classificatin model
MODEL_PATH = "./model/model_strike_a_pose.h5"
MODEL = load_model(MODEL_PATH)

# Flag to save recorded gameplay data (can be further analyzed via the procedure described in training_model.ipynb).
SAVE_DATA = False
SAVE_PATH = "play_data"

# Settings for the "postion rectangle" (outlining the 'region of interest' (ROI) used for collecting data).
RECT_TOP_LEFT = (950, 20)
RECT_BOTTOM_RIGHT = (350, 700)
RECT_COLOR = (0, 255, 0)  # green
RECT_THICKNESS = 4

## Handling command-line argument passing (number of rounds, seconds between poses)
# Print warning and exit when needed arguments are not provided.
if len(sys.argv) != 3:
    print("EXITING GAME")
    print("Please provide the number of rounds and the time between poses as integers.")
    print(
        "Usage: python play.py <ROUNDS> <TIME>; e.g., 10 rounds with 5s between poses: python play.py 10 5"
    )
    sys.exit(1)

# Convert arguments to integers and handle conversion errors
try:
    ROUNDS = int(sys.argv[1])
    COUNTDOWN = int(sys.argv[2])
except ValueError:
    print("Invalid arguments. Please use integers.")
    print(
        "Usage: python play.py <ROUNDS> <TIME>; e.g., 10 rounds with 5s between poses: python play.py 10 5"
    )
    sys.exit(1)

def main():
    # Initializes countdown/game variables.
    next_pose = random.choice(POSES)
    current_pose = None

    countdown = False
    count = COUNTDOWN
    rounds_left = ROUNDS

    points = 0
    correct_pose = False
    incorrect_pose = False

    capture = cv2.VideoCapture(0)  # Opens the camera.
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # Initializes MediaPipe's pose landmark detection.
    pose = initialize_pose_model()

    # Initialize pygame for audio processing.
    pygame.init()
    pygame.mixer.init()

    # Initializes audio variables.
    audio_played = False
    pose_sound_played = False
    game_over_sound_played = False

    while True:
        success, frame = capture.read()

        if success:
            # Draws ROI rectangle into the videostream.
            cv2.rectangle(
                frame,
                RECT_TOP_LEFT,
                RECT_BOTTOM_RIGHT,
                RECT_COLOR,
                thickness=RECT_THICKNESS,
            )

            # Processes videostream: Detects pose landmarks and draws them onto frame.
            results, frame = process_frame(frame, pose)

            # Mirrors the videostream, it looks uncanny otherwise ;)
            frame = cv2.flip(frame, 1)

            # Initiates countdown/game when 'space' is pressed.
            if countdown:
                if rounds_left > 0:  # Checks if game is ongoing.
                    # Visual feedback for a correct pose: Screen blinks green.
                    if correct_pose:
                        frame, correct_pose = blink_screen(
                            frame,
                            color_channel=1,  # BGR -> green
                            record_time=record_time,
                            blink_flag=correct_pose,
                        )
                    # Visual feedback for a correct pose: Screen blinks red.
                    elif incorrect_pose:
                        frame, incorrect_pose = blink_screen(
                            frame,
                            color_channel=2,  # BGR -> red
                            record_time=record_time,
                            blink_flag=incorrect_pose,
                        )

                    # Text to be displayed while playing (list of dictionaries).
                    text_playing = [
                        {"text": f"Left: {rounds_left}", "position": (50, 100)},
                        {
                            "text": f"{next_pose.upper()}: {count} s",
                            "position": (430, 100),
                        },
                    ]

                    # Iterate over list to display text.
                    for data in text_playing:
                        draw_bold_text(
                            frame, data["text"], data["position"], color=(0, 0, 255)
                        )

                    # Updates countdown variable after one second has elapsed.
                    if time.time() - last_time_check >= 1:
                        count -= 1
                        last_time_check = time.time()

                    # Resets countdown, selects new pose, preditcs pose, and saves data (optional) when countdown reached zero.
                    if count == 0:
                        count = COUNTDOWN  # Resets countdown variable.
                        record_time = time.time()  # Resets the recording timestamp.
                        current_pose = next_pose  # Updates the current pose variable to the next pose.
                        rounds_left -= 1  # Decreases number of rounds left.
                        pose_sound_played = False  # Resets pose_sound_played flag to announce the next pose via audio cue.

                        # Makes sure that consecutive poses are not repeated.
                        while next_pose == current_pose:
                            next_pose = random.choice(POSES)

                        # Extracts and processes landmark coordinates from results of frame processing.
                        landmark_coordinates = extract_landmarks(results)

                        # Predicts a pose label based on given landmark coordinates and a trained model.
                        predicted_pose = predict_pose_v2(
                            landmark_coordinates, LABEL_MAP, MODEL
                        )

                        # Prints results of the current recording.
                        print(f"")
                        print(f"Round {ROUNDS - rounds_left}")
                        print(f"Pose to pose: {current_pose}")
                        print(f"Recorded Pose: {predicted_pose}")

                        if predicted_pose == current_pose:
                            correct_pose = True  # Set a flag to indicate a correct pose (used for screen green blinking).

                            points += 1

                            print("***")
                            print("Good job!")
                            print(f"Points:{points}/{ROUNDS}")

                        else:
                            incorrect_pose = True  # Set a flag to indicate an incorrect pose (used for screen red blinking).

                            print("***")
                            print("Better luck next time!")
                            print(f"Points:{points}/{ROUNDS}")

                        # Saves frames and landmark coordinates if the SAVE_DATA flag is set accordingly.
                        if SAVE_DATA:
                            save_data(
                                SAVE_PATH, current_pose, frame, results, predicted_pose
                            )
                            print(
                                f"Data for Round {ROUNDS - rounds_left} saved in folder '{SAVE_PATH}'."
                            )

                if rounds_left == 0:  # Checks if game is over.
                    if not game_over_sound_played:
                        # Prints 'Game Over' notification and the total score.
                        print("")
                        print("Game Over!")
                        print(f"Points:{points}/{ROUNDS}")

                    #  Adds a translucent white background to frame.
                    white_rect = np.zeros_like(frame) + 255
                    alpha = 0.5
                    frame = cv2.addWeighted(frame, 1 - alpha, white_rect, alpha, 0)

                    # Display 'Game Over' message (total score, restart + exit instructions).
                    display_gameover_message(frame, points, ROUNDS)

                    key = cv2.waitKey(5)  # Records a key press every 5 ms.

                    # Checks if spacebar is pressed and if so, restarts game by resetting game variables.
                    if key == ord(" "):

                        while pygame.mixer.get_busy():
                            pass

                        countdown = True
                        rounds_left = ROUNDS
                        points = 0
                        last_time_check = time.time()
                        game_over_sound_played = False

                        # Randomly select a new pose to start the game
                        next_pose = random.choice(POSES)
                        current_pose = None

            # If countdown = False (initial setting -> start screen).
            else:
                #  Adds a translucent white background to frame.
                white_rect = np.zeros_like(frame) + 255
                alpha = 0.5
                frame = cv2.addWeighted(frame, 1 - alpha, white_rect, alpha, 0)

                ## Displays text on start screen.
                # Defines a list of text and their properties
                start_screen_text = [
                    {
                        "text": "Press 'SPACE' to start",
                        "position": (270, 650),
                        "scale": 2,
                        "color": (255, 0, 0),  # BGR: blue
                        "thickness": 2,
                        "offset": 1,
                    },
                    {
                        "text": "STRIKE",
                        "position": (470, 110),
                        "scale": 3,
                        "color": (0, 0, 255),  # BGR: red
                        "thickness": 4,
                        "offset": 3,
                    },
                    {
                        "text": "A POSE!",
                        "position": (440, 210),
                        "scale": 3,
                        "color": (0, 0, 255),
                        "thickness": 4,
                        "offset": 3,
                    },
                    
                ]

                # Iterates over the list of text elements and displays each one
                for element in start_screen_text:
                    if (  # Checks if the element is meant to blink.
                        element["text"] == "Press 'SPACE' to start"
                        # Use 'int(time.time() * 2) % 2' to alternate between 0 and 1 every half-second, creating a blinking effect.
                        and int(time.time() * 2) % 2 != 0
                    ):
                        continue  # Skips displaying the text element if both conditions are met.

                    draw_bold_text(
                        frame,
                        element["text"],
                        element["position"],
                        font_scale=element["scale"],
                        color=element["color"],
                        thickness=element["thickness"],
                        offset=element["offset"],
                    )

            # Displays videostream in window.
            cv2.imshow("STRIKE A POSE!", frame)

            key = cv2.waitKey(5)  # Records a key press every 5 ms.

            # Checks if 'q' is pressed and exit the execution loop to terminate the application.
            if key == ord("q"):
                while pygame.mixer.get_busy():
                    pass
                break

            # Checks if the spacebar is pressed and restart the game by resetting game variables.
            if key == ord(" "):
                while pygame.mixer.get_busy():
                    pass

                countdown = True
                last_time_check = time.time()

            # Checks if 'r' is pressed and calls the 'main()' function to restart the execution.
            if key == ord("r"):
                capture.release()
                cv2.destroyAllWindows()
                main()

        else:
            print("Closing camera.")
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
