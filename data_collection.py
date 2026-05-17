import sounddevice as sd
import numpy as np
import soundfile as sf
import queue
import os
import time

# ======================================
# CONFIG
# ======================================

FS = 48000
BLOCK = 256
RECORD_TIME = 3      # seconds

MIC_IDS = [17,18,19,20]
MIC_NAMES = ["mic1","mic2","mic3","mic4"]



ANGLES = [0,30,60,90,120,150,180,210,240,270,300,330]
SHOTS_PER_ANGLE = 50

SAVE_DIR = "D:\Propeller_Dataset"

os.makedirs(SAVE_DIR, exist_ok=True)

# ======================================
# AUDIO STREAM SETUP
# ======================================

queues = [queue.Queue() for _ in MIC_IDS]

def make_callback(idx):

    def callback(indata, frames, time_info, status):
        queues[idx].put(indata[:,0].copy())

    return callback


streams = []

for i in range(len(MIC_IDS)):

    s = sd.InputStream(
        device=MIC_IDS[i],
        channels=1,
        samplerate=FS,
        blocksize=BLOCK,
        callback=make_callback(i)
    )

    streams.append(s)

for s in streams:
    s.start()

print("\nStreams started\n")

samples_needed = int(RECORD_TIME * FS)

# ======================================
# RECORD FUNCTION
# ======================================

def record_shot():

    buffers = [[] for _ in MIC_IDS]

    while len(buffers[0]) < samples_needed:

        if any(q.empty() for q in queues):
            continue

        for i in range(len(MIC_IDS)):
            buffers[i].extend(queues[i].get())

    audios = []

    for i in range(len(MIC_IDS)):
        audio = np.array(buffers[i][:samples_needed])
        audios.append(audio)

    return audios


# ======================================
# DATASET COLLECTION
# ======================================

try:

    for angle in ANGLES:

        print(f"\n====== ANGLE {angle}° ======")

        for shot in range(1, SHOTS_PER_ANGLE + 1):

            input(f"\nPress ENTER for shot {shot}")

            recordings = record_shot()

            for i in range(len(MIC_IDS)):

                filename = f"{angle}_{shot}_{MIC_NAMES[i]}.wav"
                path = os.path.join(SAVE_DIR, filename)

                sf.write(path, recordings[i], FS)

                print("Saved:", filename)

            time.sleep(0.5)

finally:

    for s in streams:
        s.stop()
        s.close()

print("\nDataset collection finished")