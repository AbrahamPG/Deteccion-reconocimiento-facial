import cv2

# Cambia esta URL por la que te da DroidCam
cap = cv2.VideoCapture("http://192.168.1.38:4747/video")

if not cap.isOpened():
    print("❌ No se pudo abrir DroidCam.")
    exit()

print("✅ DroidCam abierta correctamente. Presiona ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error al capturar frame.")
        break

    cv2.imshow("DroidCam OpenCV", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
