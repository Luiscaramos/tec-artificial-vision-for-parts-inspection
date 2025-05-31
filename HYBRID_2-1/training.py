from ultralytics import YOLO

# Cargar un modelo preentrenado o uno nuevo
model = YOLO('yolov8m.pt')  # Cambia a yolov8s.pt, yolov8m.pt, etc., según tu necesidad

# Entrenar
if __name__ == "__main__":
    model.train(data='C:\Users\Luis Carlos Ramos\Personal\tec-artificial-vision-for-parts-inspection\HYBRID_2-1\data.yaml', epochs=50, imgsz=640, device=0)  # Usa device=0 para CUDA