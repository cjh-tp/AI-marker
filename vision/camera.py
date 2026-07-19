import cv2
import time
from pathlib import Path

SAVE_DIR = Path(__file__).parent.parent / "images"
SAVE_DIR.mkdir(exist_ok=True)

SAVE_IMAGE = SAVE_DIR / "current_page.png"

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise Exception("Cannot open webcam.")

previous_gray = None
stable_start = None

STABILITY_THRESHOLD = 5000      # smaller = more sensitive
STABILITY_TIME = 2.0            # seconds

print("Starting live camera...")

while True:

    ret, frame = camera.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    status = "Scanning..."

    if previous_gray is not None:

        difference = cv2.absdiff(previous_gray, gray)

        score = difference.sum()

        if score < STABILITY_THRESHOLD:

            if stable_start is None:
                stable_start = time.time()

            elapsed = time.time() - stable_start

            status = f"Page Stable ({elapsed:.1f}s)"

            if elapsed >= STABILITY_TIME:

                cv2.imwrite(str(SAVE_IMAGE), frame)

                print(f"\nCaptured stable page:\n{SAVE_IMAGE}")

                status = "Captured"

                stable_start = None

                # TODO
                # Next lesson:
                # call gemini_vision.py here

                time.sleep(1)

        else:

            stable_start = None

            status = "Move paper into position"

    previous_gray = gray

    cv2.putText(
        frame,
        status,
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("AI Marker Live Camera", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()