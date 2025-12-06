import cv2
import serial
import time
import random
import json
import os
import numpy as np
from datetime import datetime

# === CONFIGURACIÓN SERIAL ===
ARDUINO_PORT = 'COM7s'
BAUD_RATE = 9600
arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

# === ARCHIVOS Y DIRECTORIOS ===
DATASET_DIR = "dataset"
TRAINER_FILE = "trainer.yml"
JUGADORES_FILE = "jugadores.json"

os.makedirs(DATASET_DIR, exist_ok=True)

# === CARGAR JUGADORES ===
if os.path.exists(JUGADORES_FILE):
    with open(JUGADORES_FILE, "r") as f:
        jugadores = json.load(f)
else:
    jugadores = {}

# LIMPIAR JSON (solo al cargar)
for j in jugadores.values():
    if "mejor tiempo" in j:
        j["record"] = j.get("mejor tiempo")
        del j["mejor tiempo"]

# === CLASIFICADOR ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

if os.path.exists(TRAINER_FILE):
    recognizer.read(TRAINER_FILE)
    entrenado = True
else:
    entrenado = False

cam = cv2.VideoCapture(2)

# === FUNCIONES ===
def guardar_jugadores():
    with open(JUGADORES_FILE, "w") as f:
        json.dump(jugadores, f, indent=2)

def obtener_siguiente_id():
    if not jugadores:
        return 0
    ids = [datos["id"] for datos in jugadores.values()]
    return max(ids) + 1

def enviar_lcd(texto):
    arduino.write(f"lcd:{texto}\n".encode())
    time.sleep(1)

def capturar_imagenes(nombre):
    jugador_id = jugadores[nombre]["id"]
    carpeta = os.path.join(DATASET_DIR, nombre)
    os.makedirs(carpeta, exist_ok=True)
    
    enviar_lcd("Mira la camara")
    count = 0

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            count += 1
            rostro = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{carpeta}/{count}.jpg", rostro)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow("Capturando Rostro", frame)
        if cv2.waitKey(1) == 27 or count >= 50:
            break

    cv2.destroyAllWindows()
    enviar_lcd("Captura completa/")
    print("✅ Captura completada")

def entrenar_modelo():
    print("\n🧠 Entrenando modelo LBPH...")

    faces, ids = [], []
    for nombre, datos in jugadores.items():
        jugador_id = datos["id"]
        carpeta = os.path.join(DATASET_DIR, nombre)
        if not os.path.exists(carpeta):
            continue
        for imagen in os.listdir(carpeta):
            img_path = os.path.join(carpeta, imagen)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                ids.append(jugador_id)

    if not faces:
        print("⚠️ No hay imágenes para entrenar.")
        enviar_lcd("Sin imagenes/")
        return

    recognizer.train(faces, np.array(ids))
    recognizer.write(TRAINER_FILE)
    print("✅ Modelo entrenado correctamente")

def asegurar_ids_jugadores():
    siguiente_id = 0
    for nombre, datos in jugadores.items():
        if "id" not in datos:
            datos["id"] = siguiente_id
            siguiente_id += 1
        else:
            siguiente_id = max(siguiente_id, datos["id"] + 1)
    guardar_jugadores()

def reconocer_jugador():
    print("\n📷 Reconociendo jugador...")
    enviar_lcd("Mira la camara/")
    asegurar_ids_jugadores()

    intentos_no_reconocido = 0
    id_to_name = {datos["id"]: nombre for nombre, datos in jugadores.items()}

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            rostro = gray[y:y+h, x:x+w]
            if entrenado:
                try:
                    id_, conf = recognizer.predict(rostro)
                    if conf < 65 and id_ in id_to_name:
                        nombre = id_to_name[id_]
                        enviar_lcd(f"Hola/{nombre}")
                        cv2.putText(frame, f"{nombre}", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                        cv2.imshow("Reconocimiento", frame)
                        cv2.waitKey(1000)
                        cv2.destroyAllWindows()
                        return nombre
                    else:
                        intentos_no_reconocido += 1
                except:
                    pass

        cv2.imshow("Reconocimiento", frame)
        if cv2.waitKey(1) == 27:
            break
        if intentos_no_reconocido > 50:
            cv2.destroyAllWindows()
            return None

def mostrar_tabla():
    print("\n🏆 TOP 5 JUGADORES 🏆")
    print("=" * 55)
    print(f"{'Jugador':<20}{'Record Tiempo':<18}{'Aciertos'}")
    print("-" * 55)

    jugadores_ordenados = sorted(
        jugadores.items(),
        key=lambda x: (-x[1].get("aciertos", 0),
                       x[1].get("record", float("inf")) if x[1].get("record") else float("inf"))
    )

    for i, (nombre, datos) in enumerate(jugadores_ordenados[:5], start=1):
        record = datos.get("record", "Sin récord")
        aciertos = datos.get("aciertos", 0)
        print(f"{i}. {nombre:<18}{str(record):<18}{aciertos}")

    print("=" * 55)
    return jugadores_ordenados

def detectar_gesto_cabeza(duracion=4):
    enviar_lcd("Deseas continuar?/Asienta o Nega")
    print("\n🤖 Mueve la cabeza")

    posiciones_y, posiciones_x = [], []
    inicio = time.time()

    while time.time() - inicio < duracion:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            centro_x = x + w // 2
            centro_y = y + h // 2
            posiciones_x.append(centro_x)
            posiciones_y.append(centro_y)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Gesto", frame)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()

    if len(posiciones_y) < 5:
        return None

    mov_vertical = max(posiciones_y) - min(posiciones_y)
    mov_horizontal = max(posiciones_x) - min(posiciones_x)

    if mov_vertical > 40 and mov_vertical > mov_horizontal:
        enviar_lcd("Continuar/")
        return "continuar"

    elif mov_horizontal > 40 and mov_horizontal > mov_vertical:
        enviar_lcd("Salir/")
        return "salir"

    else:
        enviar_lcd("Sin movimiento/")
        return None

def jugar(nombre):
    jugador_data = jugadores.get(nombre, {"record": None, "aciertos": 0})
    record = jugador_data.get("record")
    mejor_racha = jugador_data.get("aciertos", 0)

    if record:
        enviar_lcd(f"{nombre}/Record:{mejor_racha}")
        print(f"\nBienvenido {nombre}, Racha máxima: {mejor_racha}")
        time.sleep(5)
    else:
        enviar_lcd(f"Nuevo jugador/{nombre}")
        print(f"\nNuevo jugador {nombre}!")

    mejor_tiempo = record or None
    nivel = 3
    aciertos = 0
    ultimo_tiempo_ok = None  # 🔥 NUEVO

    while True:
        print(f"\n🎮 Nivel {nivel}")
        enviar_lcd(f"Nivel/{nivel}")

        secuencia = ''.join(str(random.randint(0, 3)) for _ in range(nivel))
        arduino.write(f"start:{secuencia}\n".encode())

        while True:
            if arduino.in_waiting:
                msg = arduino.readline().decode().strip()

                if msg.startswith("ok:"):
                    tiempo = int(msg[3:])
                    print(f"✅ Correcto en {tiempo} ms")

                    aciertos += 1
                    nivel += 1
                    ultimo_tiempo_ok = tiempo  # 🔥 GUARDAMOS EL TIEMPO REAL DEL RECORD

                    if mejor_tiempo is None or tiempo < mejor_tiempo:
                        mejor_tiempo = tiempo

                elif msg.startswith("fail:"):
                    tiempo_fail = int(msg[5:])
                    print(f"❌ Fallaste Racha: {aciertos} | ({tiempo_fail} ms) |")

                    # ======================================================
                    # 🎯 GUARDAR SOLO SI HAY NUEVO RÉCORD Y CON TIEMPO CORRECTO
                    # ======================================================
                    if aciertos > mejor_racha:

                        mejor_racha = aciertos
                        jugadores[nombre]["aciertos"] = mejor_racha

                        # 🔥 AQUÍ VA EL FIX
                        jugadores[nombre]["record"] = ultimo_tiempo_ok

                        jugadores[nombre]["ultimo_record"] = datetime.now().strftime("%Y-%m-%d %H:%M")

                        guardar_jugadores()
                        print(f"🏆 ¡Nuevo récord! Aciertos: {mejor_racha}, Tiempo: {ultimo_tiempo_ok} ms")

                    else:
                        print("ℹ No se superó el récord. No se actualiza la tabla.")

                    tabla = mostrar_tabla()
                    pos = [i+1 for i,(n,_) in enumerate(tabla) if n == nombre][0]
                    enviar_lcd(f"{nombre}/Aciertos:{mejor_racha} Pos:{pos}")
                    time.sleep(5)

                    enviar_lcd("Deseas continuar?/Asienta o Nega")
                    gesto = detectar_gesto_cabeza()

                    if gesto == "continuar":
                        nivel = 3
                        aciertos = 0
                        ultimo_tiempo_ok = None
                        mejor_tiempo = jugadores[nombre].get("record")
                        break

                    elif gesto == "salir":
                        enviar_lcd("Pulsa S + Enter/")
                        print("\n👋 Has salido del juego.")
                        return

                    else:
                        enviar_lcd("Fin del juego/")
                        print("\n⏹ Fin del juego.")
                        return

                elif msg == "next":
                    break

# === BUCLE PRINCIPAL ===
while True:
    mostrar_tabla()
    enviar_lcd("Deseas empezar?/Presiona S+Enter")
    opcion = input("\n¿Deseas empezar? (s/n): ").strip().lower()

    if opcion != "s":
        enviar_lcd("Adios!/Hasta pronto")
        print("👋 Saliendo del programa.")
        break

    jugador = reconocer_jugador()

    if not jugador:
        print("\n🆕 Rostro no reconocido. Registrando nuevo jugador...")
        enviar_lcd("Escribe tu nombre/Y da enter")

        nombre = input("Escribe tu nombre: ").strip()
        nuevo_id = obtener_siguiente_id()

        jugadores[nombre] = {
            "id": nuevo_id,
            "record": None,
            "aciertos": 0,
            "ultimo_record": None
        }

        guardar_jugadores()
        capturar_imagenes(nombre)
        entrenar_modelo()
        jugador = nombre

    jugar(jugador)

arduino.close()
cam.release()
cv2.destroyAllWindows()