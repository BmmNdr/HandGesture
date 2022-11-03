import cv2
import handTrackingModule


def main():
    cap = cv2.VideoCapture(0)
    tracker = handTrackingModule.handTracker()

    while True:
        success, image = cap.read()
        image = tracker.handsFinder(image)
        lmList = tracker.positionFinder(image)

        cv2.imshow("Video", cv2.flip(image, 1))
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
