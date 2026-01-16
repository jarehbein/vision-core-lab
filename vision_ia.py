import cv2
import mediapipe as mp

# inicializar modulos de MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# configuración de la cámara
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    print("Sistema corriendo. Presiona ESC para salir.")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue # Si no se captura la imagen, continuar al siguiente ciclo

        # 1. Pre-procesamiento: Convertir la imagen de RGB a BGR para que MediaPipe la procese
        image.flags.writeable = False # para optimizar velocidad
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Inferencia: Procesar la imagen para detectar la malla facial
        results = face_mesh.process(image)

        # 3. Post-procesamiento: Volver a BRG para mostrarlo en la pantalla
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            
        # 4. Visualización: Dibujar los resultados si se detecta una cara
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # dibujar la malla (red de líneas)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        # 5. Mostrar la imagen final
        cv2.imshow('Vision Core Lab v1.0', image)

        # Romper el bucle si presionas la tecla ESC
        if cv2.waitKey(5) & 0xFF == 27:
            break

# salir y liberar recursos
cap.release()
cv2.destroyAllWindows()

      