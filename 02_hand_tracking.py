import cv2
import mediapipe as mp

# inicializar modulos de MediaPipe
mp_hand_mesh = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# configuración de la cámara
cap = cv2.VideoCapture(0)

with mp_hand_mesh.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    print("Sistema corriendo. Presiona ESC para salir.")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue # Si no se captura la imagen, continuar al siguiente ciclo

        # 1. Pre-procesamiento: Convertir la imagen de RGB a BGR para que MediaPipe la procese
        image.flags.writeable = False # para optimizar velocidad
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Inferencia: Procesar la imagen para detectar la malla facial
        results = hands.process(image)

        # 3. Post-procesamiento: Volver a BRG para mostrarlo en la pantalla
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            
        # 4. Visualización: Dibujar los resultados si se detecta una mano
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # dibujar la malla (red de líneas)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hand_mesh.HAND_CONNECTIONS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
        # 5. Mostrar la imagen final
        cv2.imshow('Vision Core Lab v1.0', image)

        # Romper el bucle si presionas la tecla ESC
        if cv2.waitKey(5) & 0xFF == 27:
            break

# salir y liberar recursos
cap.release()
cv2.destroyAllWindows()

      