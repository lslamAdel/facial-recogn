import cv2
from deepface import DeepFace

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    frame_count = 0
    skip_frames = 5
    last_results = []  # ← cache last detections

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        frame_count += 1

        if frame_count % skip_frames == 0:
            try:
                results = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                last_results = results  # ← update cache
            except Exception as e:
                print(f"Analysis error: {e}")
                last_results = []  # clear on failure

        # ← draw cached results on EVERY frame
        for result in last_results:
            region = result['region']
            dominant_emotion = result['dominant_emotion']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, dominant_emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Face Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()