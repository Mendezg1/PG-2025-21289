# Sistema de Detección de Infracciones de Tránsito

Sistema inteligente de visión por computadora para la detección automática de infracciones de tránsito en tiempo real, incluyendo violaciones a señales de alto, semáforos en rojo y giros en U prohibidos.

## Descripción del Proyecto

Este proyecto implementa un sistema de detección automática de infracciones de tránsito utilizando técnicas avanzadas de visión por computadora y aprendizaje profundo. El sistema es capaz de:

- **Detectar y rastrear vehículos** en tiempo real mediante algoritmos de tracking
- **Identificar señales de tránsito** (Alto/Stop, No U-Turn)
- **Reconocer estados de semáforos** (Rojo, Amarillo, Verde)
- **Detectar violaciones** de tránsito:
  - Violación de señal de Alto
  - Violación de semáforo en rojo
  - Giros en U prohibidos
- **Generar videos anotados** con las infracciones detectadas

El sistema utiliza modelos YOLO personalizados para la detección de objetos y DeepSORT para el seguimiento de vehículos, garantizando precisión y robustez en la detección de infracciones.

## Tecnologías Utilizadas

- **Python 3.8+**
- **YOLOv8** (Ultralytics) - Detección de objetos en tiempo real
- **DeepSORT** - Algoritmo de tracking multi-objeto
- **OpenCV (cv2)** - Procesamiento de video e imágenes
- **NumPy** - Cálculos numéricos y procesamiento de arrays
- **PyTorch** - Backend para modelos de deep learning

## Requisitos Previos

Antes de instalar el proyecto, asegúrate de tener instalado:

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git (opcional, para clonar el repositorio)
- Al menos 4GB de RAM disponible
- GPU compatible con CUDA (opcional, pero recomendado para mejor rendimiento)

## 🔧 Instalación

Sigue estos pasos para configurar el proyecto en tu máquina local:

### 1. Clonar el repositorio

```bash
git clone https://github.com/Mendezg1/PG-2025-21289
cd .\src
```

### 2. Crear un entorno virtual (recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install ultralytics
pip install deep-sort-realtime
pip install opencv-python
pip install numpy
```

**O instalar desde requirements.txt (si está disponible):**

```bash
pip install -r requirements.txt
```

### 4. Estructura de directorios

Asegúrate de que tu proyecto tenga la siguiente estructura:

```
proyecto/
│
├── src/
│   ├── models/
│   │   ├── vehicle_best.pt
│   │   ├── lights_best.pt
│   │   └── signs_best.pt
│   │
│   ├── vids/
│   │   ├── rojo2.mp4
│   │   └── output/
│   │       └── (aquí se guardarán los videos procesados)
│   │
│   └── main.py
│
├── demo/
│   └── demo.mp4
│
├── docs/
│   └── informe_final.pdf
│
├── requirements.txt
├── README.md
└── .gitignore
```

### 5. Descargar modelos entrenados

Coloca los siguientes modelos entrenados en la carpeta `models/`:
- `vehicle_best.pt` - Modelo para detección de vehículos
- `lights_best.pt` - Modelo para detección de semáforos
- `signs_best.pt` - Modelo para detección de señales de tránsito

> **Nota:** Los modelos deben ser entrenados previamente con YOLOv8 o descargados desde la fuente especificada por el proyecto.

## Ejecución

### Ejecutar el sistema de detección

```bash
python main.py
```

El script procesará el video de entrada (`vids/rojo2.mp4`) y generará:
- Visualización en tiempo real con las detecciones
- Video de salida con anotaciones en `vids/output/violations_output.mp4`

### Detener la ejecución

- Presiona la tecla **ESC** durante la reproducción para detener el procesamiento
- O espera a que el video termine de procesarse completamente

### Procesar un video diferente

Para procesar un video diferente, modifica la línea en `main.py`:

```python
cap = cv2.VideoCapture(".\\vids\\<Algún nombre de prueba>.mp4")
```

## Parámetros Configurables

El script incluye varios parámetros ajustables en la sección de configuración:

```python
# Parámetros de detención
FRAMES_STOP_ALTO = 5      # Frames requeridos de detención en señal de alto
FRAMES_STOP_RED = 5       # Frames requeridos de detención en luz roja
MIN_MOVEMENT = 5          # Píxeles mínimos para considerar movimiento

# Parámetros U-Turn
UTURN_ANGLE_THRESHOLD = 55        # Grados para detectar giro en U
UTURN_HISTORY_FRAMES = 60         # Frames de historial para análisis
UTURN_DETECTION_RADIUS = 400      # Radio de influencia de señal (píxeles)

# Configuración de tracker DeepSORT
tracker = DeepSort(
    max_age=150,              # Frames máximos sin detección antes de eliminar track
    n_init=10,                # Frames necesarios para confirmar un track
    nms_max_overlap=0.6,      # Overlap máximo para non-maximum suppression
    max_cosine_distance=0.7,  # Distancia máxima para matching
    nn_budget=100,            # Budget para nearest neighbor
    max_iou_distance=0.7      # Distancia IoU máxima
)
```

## Características del Sistema

### Detección de Violaciones

1. **Violación de Señal de Alto (ALTO VIOLATION)**
   - Detecta vehículos que no se detienen completamente ante señales de alto
   - Requiere detención de al menos 5 frames consecutivos
   - Identifica la dirección de aproximación del vehículo

2. **Violación de Semáforo en Rojo (RED LIGHT VIOLATION)**
   - Detecta vehículos que cruzan con luz roja
   - Valida que el vehículo esté aproximándose de frente al semáforo
   - Excluye vehículos que ya están detenidos correctamente

3. **Violación de Giro en U Prohibido (U-TURN VIOLATION)**
   - Detecta giros en U cerca de señales de "No U-Turn"
   - Calcula el ángulo de giro acumulado en trayectoria
   - Radio de detección de 400 píxeles alrededor de la señal

### Visualización

- **Vehículos:** Cuadros delimitadores verdes con ID de tracking
- **Señales de tránsito:** Cuadros azules con etiquetas
- **Semáforos:** Cuadros rojos con estado de la luz
- **Violaciones:** Texto rojo sobre los vehículos infractores
- **Debug info:** Ángulos de giro y distancias a señales

## Demostración

Para ver una demostración del sistema en funcionamiento, consulta el video de demostración ubicado en:

**[Ver demo](demo/demo.mp4)**

El video muestra ejemplos reales de detección de infracciones en diferentes escenarios de tráfico.

## Documentación

Para información detallada sobre el desarrollo, metodología, resultados y análisis del proyecto, consulta el informe final:

**[Informe Final](docs/informe_final.pdf)**

El informe incluye:
- Marco teórico y fundamentos
- Arquitectura del sistema
- Metodología de entrenamiento de modelos
- Resultados y métricas de desempeño
- Análisis de casos de prueba
- Conclusiones y trabajo futuro

## Autor

**José Ricardo Méndez González**  
Carnet 21289
Universidad del Valle de Guatemala
Facultad de Ingeniería
Trabajo de Graduación 
Segundo Semestre 2025  
Noviembre 2025

## Notas Adicionales

### Consideraciones Técnicas

- El sistema funciona mejor con videos grabados desde una posición elevada y estable
- La precisión de detección depende de la calidad del video de entrada y de los modelos entrenados
- Se recomienda usar videos con resolución mínima de 640x480 píxeles
- El tiempo de procesamiento varía según las especificaciones del hardware (CPU vs GPU)

### Limitaciones

- El sistema requiere que las señales y semáforos sean visibles en el frame
- La detección puede verse afectada por condiciones de iluminación extremas
- Se asume una perspectiva de cámara relativamente fija
- Los modelos están entrenados específicamente para las clases definidas

### Rendimiento

- **FPS esperado:** 15-30 FPS en CPU moderna, 30-60 FPS con GPU
- **Precisión de detección:** Depende de la calidad del entrenamiento de los modelos
- **Uso de memoria:** ~2-4 GB de RAM durante ejecución

## Solución de Problemas

### Error: "No module named 'ultralytics'"
```bash
pip install ultralytics
```

### Error: "No module named 'deep_sort_realtime'"
```bash
pip install deep-sort-realtime
```

### Error: "Cannot open video file"
Verifica que:
- El archivo de video existe en la ruta especificada
- La ruta en el código coincide con la ubicación del archivo
- El formato del video es compatible (MP4, AVI, MOV)
- Tienes permisos de lectura en el directorio

### Error: "Model file not found"
Asegúrate de que:
- Los archivos .pt están en la carpeta `models/`
- Los nombres de los archivos coinciden exactamente
- Los modelos fueron entrenados con YOLOv8

### Bajo rendimiento / Video lento
- Considera reducir la resolución del video de entrada
- Ajusta el parámetro `imgsz` en las predicciones YOLO (ej: de 640 a 416)
- Verifica que estás usando una GPU si está disponible
- Reduce el número de frames de historial para U-turn detection
- Aumenta el umbral de confianza en las detecciones

### El video no se guarda correctamente
Verifica que:
- La carpeta `vids/output/` existe
- Tienes permisos de escritura en el directorio
- Hay suficiente espacio en disco

## Mejoras Futuras

- Implementar detección de exceso de velocidad
- Agregar reconocimiento de placas vehiculares
- Integrar base de datos para registro de infracciones
- Mejorar la precisión con modelos más grandes
- Implementar procesamiento multi-cámara
- Añadir interfaz gráfica de usuario
- Exportar reportes en formato JSON/CSV

## Referencias

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [DeepSORT Paper](https://arxiv.org/abs/1703.07402)
- [OpenCV Documentation](https://docs.opencv.org/)

## Contacto

Para preguntas, sugerencias o reportar problemas:
- Email: rmendezg324@gmail.com
- GitHub: [Mendezg1](https://github.com/Mendezg1)
- LinkedIn: [jr-mendez](https://www.linkedin.com/in/jr-mendez/)

## Licencia

Este proyecto es para fines académicos y educativos. 

---

**Desarrollado con 💻 y ☕ en Guatemala**

**© 2025 - Sistema de Detección de Infracciones de Tránsito**