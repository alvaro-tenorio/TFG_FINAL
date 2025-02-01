# Desarrollo de un sistema de adquisición de registros y generación de alarmas guiado por Inteligencia Artificial

## Descripción
Este repositorio contiene el código fuente y documentación del Trabajo de Fin de Grado titulado **"Desarrollo de un sistema de adquisición de registros y generación de alarmas guiado por Inteligencia Artificial"**.

El proyecto se centra en la adquisición y análisis de datos de audio y vibración de maquinaria industrial para la detección de anomalías mediante técnicas de Inteligencia Artificial. Está desarrollado en **Python** e implementado en una **Raspberry Pi**, utilizando sensores como un **acelerómetro** y un **micrófono**, además de un **acelerador Coral USB** para la inferencia de modelos de Machine Learning.

## Características principales
- Adquisición de datos de audio mediante un micrófono.
- Adquisición de datos de vibración mediante un sensor acelerómetro.
- Procesado de señales con **Espectrograma Log-Mel**.
- Implementación de modelos de **detección de anomalías**:
  - Gaussian Mixture Model (GMM)
  - Autoencoders
- Generación de alarmas en tiempo real.
- Optimización del modelo para el **Coral USB Accelerator**.

## Estructura del repositorio
```
📂 Proyecto-TFG               
├── 📂V1_Audio          # Version 1 para audio
├── 📂V1_acc            # Version 1 para vibracion
├── 📂V2_Audio          # Version 2 para audio
├── 📂V2_acc          # Version 2 para vibracion
├── 📂V3_Audio          # Version 3 para audio
├── 📂V3_acc          # Version 3 para vibracion
├── 📂V4_Audio         # Version 4 para audio
├── 📂V4_acc         # Version 4 para vibracion
├── 📂V5_Audio         # Version 5 para audio
├── 📂V5_acc         # Version 5 para vibracion
│── 📂 Almacenamiento       # Modulo de almacenamiento
    │── 📂 audio            # medidas de audio
    │── 📂 medidas          # medidas de vibracion
    │── 📂 models           # modelos entrenados guardados
│── README.md              # Introducción y guía de uso
│── module_versions.txt       # librerias necesarias junto con sus versiones
```

## Instalación y configuración
### Requisitos previos
Antes de ejecutar el sistema, asegúrate de tener instalados:
- **Python 3.11**
- **Raspberry Pi OS**
- **Coral USB Accelerator** (opcional, pero recomendado)
- Las librerias listadas en `module_versions.txt`

### Instalación

# Clonar el repositorio
git clone https://github.com/alvaro-tenorio/TFG_FINAL.git
cd TFG_FINAL


## Contacto
Autor: **Álvaro Tenorio Pérez**  
Email: [alvaro.tenorio@alumnos.upm.es](mailto:alvaro.tenorio@alumnos.upm.es)  
Universidad Politécnica de Madrid - ETSIT

Si encuentras algún problema o tienes sugerencias, no dudes en abrir un **issue** en este repositorio.
