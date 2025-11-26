import cv2
from ultralytics import YOLO   #Añadimos yolo para poder leer modelos preeentrenados
import os
import face_recognition
import numpy as np

#si es la camara de la laptop es (0) sino ip del celular
#cap = cv2.VideoCapture(0)  
cap = cv2.VideoCapture("http://192.168.1.38:4747/video")


# Carpeta para guardar fotos de rostros detectados
if not os.path.exists("faces_data"):
    os.makedirs("faces_data")
face_id = 0

# Carpeta con rostros conocidos para reconocimiento
known_face_encodings = []
known_face_names = []

for filename in os.listdir("known_faces"):
    image = face_recognition.load_image_file(f"known_faces/{filename}")
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(os.path.splitext(filename)[0])


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
    # 4. Procesar frame con YOLO, detecta
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
        # Guardar rostro detectado
        # ---------------------------
        face_img = frame[y1:y2, x1:x2]
        cv2.imwrite(f"faces_data/face_{face_id}.jpg", face_img)
        face_id += 1

        # ---------------------------
        # Reconocimiento facial
        # ---------------------------
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(face_img_rgb)

        name = "Desconocido"
        if len(face_encodings) > 0:
            matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
            face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Mostrar nombre sobre el rostro
        cv2.putText(frame, name, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ---------------------------
    # 6. Mostrar ventana
    # ---------------------------
    cv2.imshow("Detección de Rostros - YOLOv8", frame)


    if cv2.waitKey(1) & 0xFF == 27:  # Presiona ESC
        break

cap.release()
cv2.destroyAllWindows()
