from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np

vehicle_model = YOLO(".\\models\\vehicle_best.pt")
light_model = YOLO(".\\models\\lights_best.pt")
sign_model = YOLO(".\\models\\signs_best.pt")

tracker = DeepSort(max_age=150, n_init=10, nms_max_overlap=0.6, max_cosine_distance=0.7, nn_budget=100, max_iou_distance=0.7)

cap = cv2.VideoCapture(".\\vids\\rojo2.mp4")

# Configurar video de salida
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(".\\vids\\output\\violations_output.mp4", fourcc, fps, (frame_width, frame_height))

# Estados de los vehículos por track_id
vehicle_states = {}

# Parámetros de detención
FRAMES_STOP_ALTO = 5  # Reducido de 10 a 5 frames
FRAMES_STOP_RED = 5   # Reducido de 10 a 5 frames
MIN_MOVEMENT = 5  # Píxeles mínimos para considerar movimiento

# Parámetros U-Turn
UTURN_ANGLE_THRESHOLD = 55  # Grados - ajustado para compensar ángulo de cámara (30 + 25)
UTURN_HISTORY_FRAMES = 60  # Frames para analizar el historial de movimiento (aumentado para capturar giro completo)
UTURN_DETECTION_RADIUS = 400  # Píxeles - radio aumentado para detectar vehículos grandes

# Tamaño de fuente
FONT_SIZE_ID = 1.2
FONT_SIZE_VIOL = 1.5

# Definición de clases
# Signs model: 0=Alto/Stop, 1=No U-turn
# Lights model: 0=Yellow, 1=Red, 2=Green

def calculate_angle_change(positions):
    """
    Calcula el cambio total de ángulo en una secuencia de posiciones.
    Retorna el ángulo acumulado en grados.
    """
    if len(positions) < 3:
        return 0
    
    total_angle = 0
    for i in range(len(positions) - 2):
        p1, p2, p3 = positions[i], positions[i+1], positions[i+2]
        
        # Vectores
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        # Magnitudes
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < MIN_MOVEMENT or norm2 < MIN_MOVEMENT:
            continue
        
        # Normalizar
        v1 = v1 / norm1
        v2 = v2 / norm2
        
        # Calcular ángulo usando producto punto y producto cruz
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        
        angle = np.arccos(dot) * 180 / np.pi
        
        # Determinar dirección del giro
        if cross < 0:
            angle = -angle
        
        total_angle += angle
    
    return abs(total_angle)

def is_approaching_from_behind(prev_pos, curr_pos, sign_pos):
    """
    Detecta si un vehículo está aproximándose desde atrás de la señal.
    """
    if prev_pos is None:
        return False
    
    # Vector de movimiento
    vehicle_direction = np.array([curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1]])
    
    # Vector desde la señal hacia el vehículo
    from_sign = np.array([curr_pos[0] - sign_pos[0], curr_pos[1] - sign_pos[1]])
    
    vehicle_norm = np.linalg.norm(vehicle_direction)
    sign_norm = np.linalg.norm(from_sign)
    
    if vehicle_norm < MIN_MOVEMENT or sign_norm < 1:
        return False
    
    vehicle_direction = vehicle_direction / vehicle_norm
    from_sign = from_sign / sign_norm
    
    # Si el vehículo se mueve en la misma dirección que apunta desde la señal,
    # está detrás de la señal moviéndose en dirección opuesta
    dot_product = np.dot(vehicle_direction, from_sign)
    
    return dot_product > 0.5  # Moviéndose alejándose de la señal desde atrás

def is_approaching_sign(prev_pos, curr_pos, sign_pos):
    """
    Determina si un vehículo se está acercando a una señal desde el frente.
    Verifica tanto la distancia como el ángulo de aproximación.
    """
    if prev_pos is None:
        return False
    
    prev_dist = np.sqrt((prev_pos[0] - sign_pos[0])**2 + (prev_pos[1] - sign_pos[1])**2)
    curr_dist = np.sqrt((curr_pos[0] - sign_pos[0])**2 + (curr_pos[1] - sign_pos[1])**2)
    
    # Calcular vector de movimiento del vehículo
    vehicle_direction = np.array([curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1]])
    
    # Calcular vector hacia la señal
    to_sign = np.array([sign_pos[0] - curr_pos[0], sign_pos[1] - curr_pos[1]])
    
    # Normalizar vectores
    vehicle_norm = np.linalg.norm(vehicle_direction)
    sign_norm = np.linalg.norm(to_sign)
    
    if vehicle_norm < MIN_MOVEMENT or sign_norm < 1:
        return False
    
    vehicle_direction = vehicle_direction / vehicle_norm
    to_sign = to_sign / sign_norm
    
    # Calcular ángulo usando producto punto
    dot_product = np.dot(vehicle_direction, to_sign)
    
    # Solo considerar "approaching" si:
    # 1. La distancia está disminuyendo
    # 2. El ángulo es menor a 45 grados (dot_product > 0.7 ≈ cos(45°))
    is_getting_closer = curr_dist < prev_dist - MIN_MOVEMENT
    is_aligned = dot_product > 0.7
    
    return is_getting_closer and is_aligned

while True:
    ret, frame = cap.read()
    if not ret:
        break

    v_detections = []
    sign_detections = []
    light_detections = []
    
    # Diccionario persistente para etiquetas de señales (fuera del loop si no existe)
    if 'sign_labels' not in globals():
        global sign_labels
        sign_labels = {}  # {(x, y, w, h): label}

    # 1. Detectar vehículos con umbral más bajo para capturar vehículos en giros
    results = vehicle_model(frame, imgsz=640, conf=0.3, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            v_detections.append([[x1, y1, x2 - x1, y2 - y1], conf, 0])

    # 2. Detectar señales (Signs: 0=Alto, 1=No U-turn)
    results = sign_model.predict(frame, imgsz=640, conf=0.5, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Crear clave de posición para tracking persistente
            sign_key = (int(x1 // 20) * 20, int(y1 // 20) * 20)
            
            # Si ya existe esta señal y fue marcada como No U-Turn, mantener esa clase
            if sign_key in sign_labels and sign_labels[sign_key] == "No U-Turn":
                cls = 1  # Forzar a No U-Turn
            elif cls == 1:  # Si se detecta como No U-Turn, registrarlo permanentemente
                sign_labels[sign_key] = "No U-Turn"
            
            sign_detections.append([x1, y1, x2, y2, conf, cls])

    # 3. Detectar semáforos (Lights: 0=Yellow, 1=Red, 2=Green)
    results = light_model.predict(frame, imgsz=640, conf=0.5, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            light_detections.append([x1, y1, x2, y2, conf, cls])

    # 4. Actualizar tracker
    tracks = tracker.update_tracks(v_detections, frame=frame)

    # 5. Procesar cada vehículo
    for track in tracks:
        # Solo procesar tracks confirmados Y con detección reciente (máximo 5 frames sin detección para U-turns)
        if not track.is_confirmed() or track.time_since_update > 5:
            continue
        x1, y1, x2, y2 = track.to_ltrb()
        track_id = track.track_id
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Inicializar estado
        if track_id not in vehicle_states:
            vehicle_states[track_id] = {
                'frames_stop_alto': 0,
                'frames_stop_red': 0,
                'violation_alto': False,
                'violation_red': False,
                'violation_uturn': False,
                'prev_pos': None,
                'position_history': [],  # Historial para U-turn
                'approaching_alto': {},  # Dict de sign_id: bool
                'approaching_red': {},
                'exempt_alto': {},  # Vehículos que NO deben ser penalizados
                'exempt_red': {},
                'near_no_uturn': False  # Flag para detectar si está cerca de señal No U-turn
            }

        state = vehicle_states[track_id]
        curr_pos = (cx, cy)
        
        # Actualizar historial de posiciones para U-turn
        state['position_history'].append(curr_pos)
        if len(state['position_history']) > UTURN_HISTORY_FRAMES:
            state['position_history'].pop(0)
        
        # Resetear flag de No U-turn
        state['near_no_uturn'] = False
        
        # Procesar señales de tráfico (Signs)
        for idx, (x1_s, y1_s, x2_s, y2_s, conf, cls) in enumerate(sign_detections):
            sign_center = ((x1_s + x2_s) / 2, (y1_s + y2_s) / 2)
            
            # La clase ya viene corregida desde la detección
            if cls == 0:  # Alto/Stop sign
                inflation = 30
                stop_zone_x1 = x1_s - 20
                stop_zone_x2 = x2_s + 20
                stop_zone_y1 = y1_s
                stop_zone_y2 = y2_s + inflation

                in_stop_zone = stop_zone_x1 < cx < stop_zone_x2 and stop_zone_y1 < cy < stop_zone_y2

                # Verificar si el vehículo se está acercando a esta señal
                is_facing = is_approaching_sign(state['prev_pos'], curr_pos, sign_center)
                is_behind = is_approaching_from_behind(state['prev_pos'], curr_pos, sign_center)

                # Marcar como exento si viene de atrás
                if is_behind:
                    state['exempt_alto'][idx] = True
                
                state['approaching_alto'][idx] = is_facing
                
                if in_stop_zone and is_facing:
                    state['frames_stop_alto'] += 1
                    # Si detuvo correctamente, marcar como exento permanentemente
                    if state['frames_stop_alto'] >= FRAMES_STOP_ALTO:
                        state['exempt_alto'][idx] = True
                        state['violation_alto'] = False  # Limpiar violación si detuvo
                elif is_facing and state['approaching_alto'].get(idx, False):
                    # El vehículo salió de la zona de detención
                    # Solo marcar violación si NO está exento y NO detuvo lo suficiente
                    if (state['frames_stop_alto'] < FRAMES_STOP_ALTO and 
                        not state['exempt_alto'].get(idx, False)):
                        state['violation_alto'] = True
                    # Resetear contador solo después de evaluar
                    if not in_stop_zone:
                        state['frames_stop_alto'] = 0
            
            elif cls == 1:  # No U-turn sign
                # Definir zona de influencia para señal No U-turn
                distance = np.sqrt((cx - sign_center[0])**2 + (cy - sign_center[1])**2)
                if distance < UTURN_DETECTION_RADIUS:  # Radio de influencia en píxeles
                    state['near_no_uturn'] = True
                    # Debug: dibujar línea entre vehículo y señal
                    cv2.line(frame, (int(cx), int(cy)), (int(sign_center[0]), int(sign_center[1])), (255, 255, 0), 1)
                    cv2.putText(frame, f"Dist: {distance:.0f}px", (int(cx), int(cy) + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Detectar U-turn si hay suficiente historial
        # Una vez detectado, la violación es permanente (no se puede sobreescribir)
        if not state['violation_uturn'] and len(state['position_history']) >= UTURN_HISTORY_FRAMES:
            angle_change = calculate_angle_change(state['position_history'])
            
            # Debug: mostrar el ángulo acumulado y estado
            debug_color = (0, 255, 255) if state['near_no_uturn'] else (128, 128, 128)
            cv2.putText(frame, f"Angle: {angle_change:.1f} {'NEAR!' if state['near_no_uturn'] else ''}", 
                       (int(x1), int(y2) + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_color, 2)
            
            # Marcar violación si detecta un giro significativo cerca de señal No U-turn
            if angle_change >= UTURN_ANGLE_THRESHOLD and state['near_no_uturn']:
                state['violation_uturn'] = True  # Esta violación es permanente
                print(f"U-TURN VIOLATION DETECTED! Vehicle {track_id}, Angle: {angle_change:.1f}°")

        # Procesar semáforos (Lights)
        for idx, (x1_l, y1_l, x2_l, y2_l, conf, cls) in enumerate(light_detections):
            light_center = ((x1_l + x2_l) / 2, (y1_l + y2_l) / 2)
            
            inflation = 30
            stop_zone_x1 = x1_l - 20
            stop_zone_x2 = x2_l + 20
            stop_zone_y1 = y1_l
            stop_zone_y2 = y2_l + inflation

            in_stop_zone = stop_zone_x1 < cx < stop_zone_x2 and stop_zone_y1 < cy < stop_zone_y2

            # Verificar si el vehículo se está acercando a este semáforo
            is_facing = is_approaching_sign(state['prev_pos'], curr_pos, light_center)
            is_behind = is_approaching_from_behind(state['prev_pos'], curr_pos, light_center)

            if cls == 1:  # Red light
                # Marcar como exento si viene de atrás
                if is_behind:
                    state['exempt_red'][idx] = True
                
                state['approaching_red'][idx] = is_facing
                
                # Solo procesar si el vehículo está enfrentando el semáforo
                if in_stop_zone and is_facing:
                    state['frames_stop_red'] += 1
                    # Si detuvo correctamente, marcar como exento permanentemente
                    if state['frames_stop_red'] >= FRAMES_STOP_RED:
                        state['exempt_red'][idx] = True
                        state['violation_red'] = False  # Limpiar violación si detuvo
                elif is_facing and state['approaching_red'].get(idx, False):
                    # El vehículo salió de la zona de detención
                    # Solo marcar violación si NO está exento y NO detuvo lo suficiente
                    if (state['frames_stop_red'] < FRAMES_STOP_RED and 
                        not state['exempt_red'].get(idx, False)):
                        state['violation_red'] = True
                    # Resetear contador solo después de evaluar
                    if not in_stop_zone:
                        state['frames_stop_red'] = 0

            elif cls == 2:  # Green light
                if is_facing and not state['exempt_red'].get(idx, False):
                    state['violation_red'] = False
                    state['frames_stop_red'] = FRAMES_STOP_RED

        # Actualizar posición previa
        state['prev_pos'] = curr_pos

        # Dibujar vehículo y posibles infracciones
        color = (0, 255, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"Vehicle {track_id}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE_ID, color, 2)

        # Mostrar violaciones (ajustar posición vertical)
        y_offset = 25
        if state['violation_alto']:
            cv2.putText(frame, "ALTO VIOLATION", (int(x1), int(y1) - y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE_VIOL, (0, 0, 255), 3)
            y_offset += 30
        if state['violation_red']:
            cv2.putText(frame, "RED LIGHT VIOLATION", (int(x1), int(y1) - y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE_VIOL, (0, 0, 255), 3)
            y_offset += 30
        if state['violation_uturn']:
            cv2.putText(frame, "U-TURN VIOLATION", (int(x1), int(y1) - y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE_VIOL, (0, 0, 255), 3)

    # Dibujar señales de tráfico (AZUL)
    for x1_s, y1_s, x2_s, y2_s, conf, cls in sign_detections:
        color = (255, 0, 0)  # Azul (BGR)
        
        # La clase ya viene con el tracking persistente aplicado
        if cls == 0:
            label = "ALTO Sign"
        elif cls == 1:
            label = "No U-Turn"
        else:
            label = f"Sign {cls}"
        
        cv2.rectangle(frame, (int(x1_s), int(y1_s)), (int(x2_s), int(y2_s)), color, 2)
        cv2.putText(frame, label, (int(x1_s), int(y1_s) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Dibujar semáforos (ROJO)
    for x1_l, y1_l, x2_l, y2_l, conf, cls in light_detections:
        color = (0, 0, 255)  # Rojo (BGR)
        if cls == 0:
            label = "Yellow Light"
        elif cls == 1:
            label = "Red Light"
        elif cls == 2:
            label = "Green Light"
        else:
            label = f"Light {cls}"
        
        cv2.rectangle(frame, (int(x1_l), int(y1_l)), (int(x2_l), int(y2_l)), color, 2)
        cv2.putText(frame, label, (int(x1_l), int(y1_l) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    cv2.imshow("Traffic Violations", frame)
    out.write(frame)  # Guardar frame al video
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()  # Cerrar video de salida
cv2.destroyAllWindows()