import cv2

camera = cv2.VideoCapture(0)

print("Press SPACE to capture")
print("Press ESC to quit")

while True:
    ret, frame = camera.read()

    if not ret:
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)

    if key == 27:      # ESC
        break

    if key == 32:      # SPACE
        cv2.imwrite("images/test_photo.png", frame)
        print("Image saved!")
        break

camera.release()
cv2.destroyAllWindows()