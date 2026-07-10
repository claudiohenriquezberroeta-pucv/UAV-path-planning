# UAV-path-planning
Proyecto del curso Deep Neural Networks
UAV Autonomous Navigation & Obstacle Avoidance via Deep Reinforcement Learning (DRL) and Curriculum Learning
Este repositorio contiene la implementación completa del sistema de Path Planning autónomo y evasión de obstáculos en entornos interiores para el dron DJI Mavic 2 PRO en el simulador físico Webots.

El proyecto hereda y adapta la formulación de Curriculum Learning propuesta por Park, Jang & Shin. Para mitigar la brecha de control físico en Webots (que requiere comandos individuales de velocidad angular para las 4 hélices en lugar de comandos de velocidad abstractos), diseñamos un low-level flight controller híbrido que ejecuta un lazo de control de actitud PID acoplado directamente con las decisiones de navegación tomadas por las policies de alto nivel de Deep Reinforcement Learning (DRL).

# Arquitectura del Sistema
El flujo de control y datos se organiza de manera jerárquica:

High-Level Decision-Maker (Agente de DRL): Desarrollado sobre la especificación de OpenAI Gymnasium y entrenado mediante la librería Stable-Baselines3. Soporta dos algoritmos comparativos:
PPO: Agente on-policy con un discrete Action Space compuesto por 15 combinaciones de forward linear velocity y yaw rate.
TD3: Agente off-policy con un continuous Action Space en $[-1.0, 1.0]^2$ acoplado a un Replay Buffer y sintonizado con ruido gaussiano de exploración.
Environment Wrapper (Gymnasium): El script drone_env.py define el State Space ($\mathbb{R}^{38}$ con lecturas de LiDAR de 36 canales, distancia al target y error de heading) y calcula la Reward Function en cada step.
Low-Level Flight Controller (PID): El script flight_controller.py actúa como puente de control, recibiendo los comandos de velocidad lineal y angular del agente de DRL y traduciéndolos a comandos de hélice en rad/s, forzando de manera absoluta el ángulo de roll a cero para estabilizar el cono de percepción del LiDAR.

# Requisitos de Instalación
Requisitos del Sistema
Ubuntu 20.04 / 22.04 LTS o Windows 10/11
Webots R2023a / R2023b o posterior
Python 3.10 / 3.11 / 3.12
Configuración del Entorno Virtual (venv)
Se recomienda encarecidamente utilizar un entorno virtual aislado para evitar conflictos de dependencias:

# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/proyecto-uav-mavic.git
cd proyecto-uav-mavic

# 2. Crear el entorno virtual
python3 -m venv venv

# 3. Activar el venv
# En Linux/macOS:
source venv/bin/activate
# En Windows:
venv\Scripts\activate

# 4. Instalar dependencias requeridas
pip install --upgrade pip

pip install -r requirements.txt
Contenido de requirements.txt
gymnasium==0.28.1
stable-baselines3[extra]==2.1.0
numpy==1.24.3
matplotlib==3.7.1
pandas==2.0.3
protobuf==4.23.3
# Estructura del Código
uav_rl_controller.py: Punto de entrada (entry point) del controlador de Webots. Configura los flujos de log en tiempo real y lee los argumentos de ejecución desde el campo controllerArgs del archivo de mundo .wbt.
train.py: Administra el bucle de entrenamiento, la vectorización del entorno, el callback adaptativo del curriculum y la rutina de evaluación final.
drone_env.py: Envuelve la API del supervisor de Webots bajo el estándar de Gymnasium. Administra el cálculo del State Space, la lógica de colisión y las recompensas térmicas.
flight_controller.py: Implementa los PID de velocidad de avance a cabeceo (pitch), estabilidad de altura y velocidad de guiñada (yaw rate).
config.py: Archivo centralizado que contiene los hyperparameters de los optimizadores, geometrías del mundo, configuraciones físicas de motores y constantes del curriculum.
make_figures.py / plot_learning_curve.py: Herramientas de visualización para procesar los logs en CSV y generar gráficos de curvas de aprendizaje y Goal Rate.
# Instrucciones de Ejecución
El comportamiento del controlador se define mediante argumentos de terminal configurados en Webots o ejecutados directamente:

Uso: python uav_rl_controller.py [--mode {train,eval}] [--reward-model {1,2}] [--stage {1,2}] [--algo {ppo,td3}] [--episodes N]
1. Entrenamiento (Training Mode)
Stage 1 (Navegación libre, sin obstáculos): El dron aprende los lazos básicos de control de orientación y avance en worlds/indoor_no_obstacles.wbt escalando su dificultad desde $\mathcal{D}=0.0$ a $1.0$.

# Entrenamiento con PPO discreto y Reward Model 2 (Recomendado)
python uav_rl_controller.py --mode train --algo ppo --reward-model 2 --stage 1

# Entrenamiento con TD3 continuo
python uav_rl_controller.py --mode train --algo td3 --reward-model 2 --stage 1
Stage 2 (Evasión de obstáculos / Fine-Tuning): El dron se inicializa cargando los mejores pesos del Stage 1 en worlds/indoor_with_obstacles.wbt para aprender a esquivar 6 cilindros fijos utilizando el sensor LiDAR.

# Continuar entrenamiento con PPO cargando pesos del Stage 1
python uav_rl_controller.py --mode train --algo ppo --reward-model 2 --stage 2
2. Evaluación (Evaluation Mode)
Para evaluar el desempeño y registrar la métrica final de Goal Rate sobre la tarea completa (forzando dificultad $\mathcal{D}=1.0$) a lo largo de un número fijo de episodes:

# Evaluar el mejor modelo guardado de PPO en Stage 2
python uav_rl_controller.py --mode eval --algo ppo --stage 2 --episodes 100
# Hiperparámetros de los Algoritmos (config.py)
Hyperparameter	PPO (Discrete)	TD3 (Continuous)
Learning Rate	$1.5 \times 10^{-4}$ (bajada de $3.0\times 10^{-4}$)	$3.0 \times 10^{-4}$
Batch Size	$125$	$256$
Discount Factor ($\gamma$)	$0.99$	$0.99$
Replay Buffer Size	N/A	$200,000$ steps
Target KL Limit	$0.20$	N/A
Policy Delay	N/A	2 gradient updates
Exploration Noise	N/A	$\sigma = 0.10$ (Gaussiano)
Steps per Iteration	$10,000$ steps	$10,000$ steps
# Resumen de Resultados Experimentales
El modelo fue evaluado empíricamente a lo largo de 50 iteraciones de lote ($500,000$ timesteps totales de simulación por configuración):

Algoritmo	Escenario (Stage)	Tasa de Éxito Final (Última It.)	Media Móvil Final (MA5)	Dificultad Máxima Alcanzada
PPO	Stage 1 (Sin obstáculos)	$61.11%$	$67.58%$	$1.0$ (Máxima)
TD3	Stage 1 (Sin obstáculos)	$83.93%$	$81.53%$	$1.0$ (Máxima)
PPO	Stage 2 (Con obstáculos)	$67.80%$	$55.63%$	$1.0$ (Máxima)
TD3	Stage 2 (Con obstáculos)	$40.74%$	$49.13%$	$0.7$ (Intermedia)
# Análisis de Comportamiento Físico y de Control
La Eficiencia de TD3 en Stage 1: TD3 demostró una tasa de convergencia superior en mapas despejados. Debido a su naturaleza off-policy de alta eficiencia muestral y optimización continua, eliminó el fenómeno de arranque en frío (cold-start) de PPO (el cual comenzó con apenas un $6.82%$ de éxito en su primera iteración). TD3 consolidó trayectorias directas y fluidas, logrando un notable $83.93%$ de Goal Rate final.
El Colapso Aerodinámico de TD3 en Stage 2: Bajo entornos densificados con obstáculos, las políticas continuas deterministas de TD3 sufren un problema de sobreajuste crítico: buscan explotar velocidades máximas en trayectorias demasiado cercanas al radio de colisión de los cilindros. En Webots, la simulación física realista de las fuerzas del motor añade una gran inercia traslacional. Al escalar el curriculum por encima de la dificultad de $0.7$ (lo que aumenta drásticamente la distancia y desalineación de spawneo), el dron inercial de TD3 derrapó en curvas cerradas de evasión, provocando colisiones continuas que congelaron la promoción del curriculum.
La Resiliencia Estocástica de PPO: PPO demostró una estabilidad excepcional ante obstáculos. Aunque sufrió fluctuaciones importantes de rendimiento al inicio (cayendo a un $29.67%$ en la iteración 3 del Stage 2), la restricción del paso de actualización (controlado por el Target KL Limit de $0.20$) impidió cambios destructivos en la política. PPO asimiló con éxito el sensor LiDAR de 36 canales en un espacio discreto de toma de decisiones (15 acciones combinadas), logrando alcanzar la máxima complejidad del mapa ($\mathcal{D}=1.0$) y cerrando con un sólido $67.80%$ de éxito.
# Buenas Prácticas de Simulación en Webots
Line Buffering en Consola: Como Webots redirige el flujo de la salida estándar, el script fuerza la reconfiguración sys.stdout.reconfigure(line_buffering=True) en el punto de entrada para que las métricas de entrenamiento se muestren en tiempo real sin retrasos en la consola de Webots.
Control de Inestabilidad Física (Tilt Limit): Para evitar que fallos de exploración iniciales vuelquen el dron de forma irreversible (provocando divergencias numéricas en el simulador que generen valores NaN en los vectores de estado), el entorno detiene de inmediato el episodio si el dron se inclina más de $1.0\text{ rad}$ ($\sim 57^{\circ}$), declarando colisión y aplicando un coste de $R_{\text{fail}} = -500.0$.
