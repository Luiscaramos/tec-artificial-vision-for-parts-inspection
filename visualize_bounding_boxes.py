import cv2


dir_imagen = r"Data_sets\Data_set_2imagen_001.png"
dir_text = r"Data_sets\Data_set_2imagen_001.txt"

imagen = cv2.imread(dir_imagen)

txt = open(dir_text)
lineas = txt.readlines()

height, width, _ = imagen.shape  # Obtener las dimensiones de la imagen

for i in range(len(lineas)):
    
    coordanates = lineas[i].split()
    # Coordenadas normalizadas de YOLO (ejemplo)
    x_center = float(coordanates[1])
    y_center = float(coordanates[2])
    obj_width = float(coordanates[3])
    obj_height = float(coordanates[4])

    # Convertir las coordenadas normalizadas a coordenadas absolutas
    x_min = int((x_center - obj_width / 2) * width)
    y_min = int((y_center - obj_height / 2) * height)
    x_max = int((x_center + obj_width / 2) * width)
    y_max = int((y_center + obj_height / 2) * height)

    # Dibujar el rectángulo en la imagen (color en BGR, grosor del rectángulo)
    cv2.rectangle(imagen, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    print(coordanates)



cv2.imshow("Bounding_box",imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()

