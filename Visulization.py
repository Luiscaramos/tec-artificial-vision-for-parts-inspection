from ultralytics import YOLO
import cv2

version = r'HYBRID_2-1\runs\detect\train\weights\best.pt'

# Cargar el modelo entrenado
model = YOLO(version)

# Iniciar la cámara (0 = cámara integrada principal)

cap = cv2.VideoCapture(1)

# Verificar si se abrió la cámara correctamente
if not cap.isOpened():
    print("❌ No se pudo acceder a la cámara.")
    exit()

print("✅ Cámara encendida. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo leer el frame de la cámara.")
        break
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gris = cv2.merge([gris, gris, gris])
    
    # Hacer inferencia con YOLO
    results = model(gris)

    # Dibujar predicciones sobre la imagen
    annotated_frame = results[0].plot()

    # Mostrar la imagen en ventana
    cv2.imshow("Detección en vivo - YOLOv8", annotated_frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()