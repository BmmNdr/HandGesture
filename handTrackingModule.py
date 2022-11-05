import cv2
import mediapipe as mp


class handTracker:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        self.results = None
        # perform the hand recognition algorithm
        self.mpHands = mp.solutions.hands
        # configuration of the model
        self.hands = self.mpHands.Hands()
        # draw the detected point
        self.mpDraw = mp.solutions.drawing_utils
        # coordinates
        self.landmarks = []

    def handsFinder(self, image):
        # image to rgb (from bgr)
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Processes an RGB image and returns the hand landmarks prediction, handedness and hand landmarks in 3D
        self.results = self.hands.process(imageRGB)

        # if a hand is detected
        if self.results.multi_hand_landmarks:
            # for each hand
            for handLms in self.results.multi_hand_landmarks:
                # Draw landmarks and connections on parameter image
                # mpHands.HAND_CONNECTIONS --> how landmarks should be connected
                self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)

                # store the coordinates
                for lm in handLms.landmark:
                    lmx = lm.x
                    lmy = lm.y
                    self.landmarks.append([lmx, lmy])

        return image
