import cv2
from ultralytics import YOLO   #Añadimos yolo para poder leer modelos preeentrenados


#si es la camara de la laptop es (0) sino ip del celular
#cap = cv2.VideoCapture(0)  
cap = cv2.VideoCapture("http://192.168.1.38:4747/video")


if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    exit()

print("✅ Cámara abierta correctamente. Presiona ESC para salir.")

# ---------------------------
# 2. Cargar modelo YOLO
# ---------------------------
model = YOLO("yolov8n-face.pt")  # debe estar en la misma carpeta

print("🤖 Modelo YOLO cargado correctamente.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error al capturar frame.")
        break
    # ---------------------------
    # 4. Procesar frame con YOLO
    # ---------------------------
    results = model(frame, verbose=False)

    # ---------------------------
    # 5. Dibujar cajas en el frame
    # ---------------------------
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # Dibujar caja
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Mostrar confianza
        cv2.putText(frame,
                    f"{conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

    # ---------------------------
    # 6. Mostrar ventana
    # ---------------------------
    cv2.imshow("Detección de Rostros - YOLOv8", frame)


    if cv2.waitKey(1) & 0xFF == 27:  # Presiona ESC
        break

cap.release()
cv2.destroyAllWindows()
