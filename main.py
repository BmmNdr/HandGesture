import cv2
import handTrackingModule


def main():
    # video capture
    cap = cv2.VideoCapture(0)
    # hand tracking module
    tracker = handTrackingModule.handTracker()

    while True:
        success, image = cap.read()
        image = tracker.handsFinder(image)

        cv2.imshow("Video", cv2.flip(image, 1))

        # Close pressing 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    print(tracker.landmarks)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
