from ultralytics import YOLO

# Cargar un modelo preentrenado o uno nuevo
model = YOLO('yolov8l.pt')  # Cambia a yolov8s.pt, yolov8m.pt, etc., según tu necesidad

# Entrenar
if __name__ == "__main__":
    model.train(data='Data_sets\Data_set_2_valreal\data.yaml', epochs=50, imgsz=640, device=0)  # Usa device=0 para CUDA
