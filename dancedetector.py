
import time
import math
from collections import deque

import cv2
import numpy as np
import pygame

MOTION_HISTORY_LEN = 15        # how many recent frames we look at for the decision
START_DANCE_RATIO = 0.55       # fraction of recent frames that must be "high motion" to START dancing
STOP_DANCE_RATIO = 0.15        # fraction that must be "high motion" to KEEP dancing (else stop)
DIFF_THRESHOLD = 25            # pixel intensity diff considered "changed"
MOTION_PIXEL_FRACTION = 0.03   # fraction of frame pixels that must change to count this frame as "high motion"
CAT_BOX_SIZE = 260             # size (px) of the picture-in-picture cat box
MUSIC_VOLUME = 0.4


def make_dance_loop(sample_rate=44100, bpm=140):
    """Generate a simple 8-bit-style arpeggio loop as a numpy waveform."""
    beat_dur = 60.0 / bpm
    note_dur = beat_dur / 2.0

    # A goofy little arpeggio pattern (frequencies in Hz), repeats to feel "dancy"
    pattern = [261.63, 329.63, 392.00, 523.25,   # C E G C
               392.00, 329.63, 261.63, 196.00]   # G E C G

    samples_per_note = int(sample_rate * note_dur)
    waveform = np.zeros(samples_per_note * len(pattern), dtype=np.float32)

    for i, freq in enumerate(pattern):
        t = np.linspace(0, note_dur, samples_per_note, endpoint=False)
        # square-ish wave for that chiptune feel + quick fade to avoid clicks
        tone = 0.6 * np.sign(np.sin(2 * np.pi * freq * t))
        tone += 0.25 * np.sin(2 * np.pi * (freq * 2) * t)  # octave harmonic
        fade = np.ones_like(tone)
        fade_len = int(0.01 * sample_rate)
        fade[:fade_len] = np.linspace(0, 1, fade_len)
        fade[-fade_len:] = np.linspace(1, 0, fade_len)
        waveform[i * samples_per_note:(i + 1) * samples_per_note] = tone * fade

    waveform = np.clip(waveform, -1.0, 1.0)
    stereo = np.column_stack((waveform, waveform))
    audio = (stereo * 32767).astype(np.int16)
    return audio


def init_audio():
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.mixer.init()
    audio_array = make_dance_loop()
    sound = pygame.sndarray.make_sound(audio_array)
    sound.set_volume(MUSIC_VOLUME)
    return sound



def draw_dancing_cat(size, phase, dancing):
    """
    Returns a BGR image (size x size) with a cartoon cat.
    `phase` drives the sway/bounce animation over time.
    `dancing` = False freezes the cat mid-pose (no motion).
    """
    img = np.full((size, size, 3), (245, 235, 225), dtype=np.uint8)  # light background

    cx, cy = size // 2, size // 2

    if dancing:
        sway = int(18 * math.sin(phase * 4.0))          # left-right hip sway
        bounce = int(10 * abs(math.sin(phase * 4.0)))    # up-down bounce
        ear_wiggle = 8 * math.sin(phase * 8.0)
        tail_swing = 25 * math.sin(phase * 3.0 + 1.0)
    else:
        sway, bounce, ear_wiggle, tail_swing = 0, 0, 0, 0

    body_center = (cx + sway, cy + 20 - bounce)
    head_center = (cx + sway, cy - 40 - bounce)

    # tail
    tail_base = (body_center[0] - 45, body_center[1] + 30)
    tail_tip = (int(tail_base[0] - 40 + tail_swing), tail_base[1] - 40)
    cv2.line(img, tail_base, tail_tip, (60, 60, 60), 10, lineType=cv2.LINE_AA)

    # body
    cv2.ellipse(img, body_center, (55, 65), 0, 0, 360, (70, 70, 70), -1, cv2.LINE_AA)

    # legs (little stubs that "step" with the sway)
    leg_off = int(10 * math.sin(phase * 4.0)) if dancing else 0
    cv2.ellipse(img, (body_center[0] - 30, body_center[1] + 55 + leg_off), (14, 20), 0, 0, 360, (60, 60, 60), -1)
    cv2.ellipse(img, (body_center[0] + 30, body_center[1] + 55 - leg_off), (14, 20), 0, 0, 360, (60, 60, 60), -1)

    # head
    cv2.circle(img, head_center, 55, (80, 80, 80), -1, cv2.LINE_AA)

    # ears (triangles), wiggling
    left_ear = np.array([
        [head_center[0] - 45, head_center[1] - 25],
        [head_center[0] - 15 + int(ear_wiggle), head_center[1] - 65],
        [head_center[0] - 5, head_center[1] - 15],
    ], np.int32)
    right_ear = np.array([
        [head_center[0] + 45, head_center[1] - 25],
        [head_center[0] + 15 - int(ear_wiggle), head_center[1] - 65],
        [head_center[0] + 5, head_center[1] - 15],
    ], np.int32)
    cv2.fillPoly(img, [left_ear], (70, 70, 70), cv2.LINE_AA)
    cv2.fillPoly(img, [right_ear], (70, 70, 70), cv2.LINE_AA)
    cv2.fillPoly(img, [left_ear], (60, 60, 60))
    cv2.fillPoly(img, [right_ear], (60, 60, 60))

    # goggles ("scuba" vibe) - two circles + strap
    g_y = head_center[1] - 5
    cv2.line(img, (head_center[0] - 55, g_y), (head_center[0] + 55, g_y), (40, 40, 200), 6, cv2.LINE_AA)
    for dx in (-25, 25):
        gcenter = (head_center[0] + dx, g_y)
        cv2.circle(img, gcenter, 22, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(img, gcenter, 17, (255, 240, 200), -1, cv2.LINE_AA)
        pupil_shift = int(6 * math.sin(phase * 4.0)) if dancing else 0
        cv2.circle(img, (gcenter[0] + pupil_shift, gcenter[1]), 6, (20, 20, 20), -1, cv2.LINE_AA)

    # nose + mouth
    nose = (head_center[0], head_center[1] + 20)
    cv2.circle(img, nose, 5, (30, 30, 150), -1, cv2.LINE_AA)
    cv2.ellipse(img, (nose[0] - 10, nose[1] + 5), (10, 6), 0, 20, 160, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.ellipse(img, (nose[0] + 10, nose[1] + 5), (10, 6), 0, 20, 160, (30, 30, 30), 2, cv2.LINE_AA)

    # whiskers
    for side in (-1, 1):
        for i in range(3):
            y = nose[1] - 5 + i * 6
            cv2.line(img, (nose[0] + side * 15, y),
                      (nose[0] + side * 60, y - 5 + i * 3), (30, 30, 30), 1, cv2.LINE_AA)

    label = "DANCING!" if dancing else "..."
    cv2.putText(img, label, (10, size - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 120, 0) if dancing else (120, 120, 120), 2, cv2.LINE_AA)

    return img


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera index/permissions.")

    sound = init_audio()
    music_playing = False

    prev_gray = None
    motion_history = deque(maxlen=MOTION_HISTORY_LEN)
    dancing = False
    phase_start = time.time()

    print("Move / dance in front of the camera to wake up the cat. Press 'q' to quit, 'c' to recalibrate.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror, feels natural
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray is None:
            prev_gray = gray
            continue

        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        changed_fraction = np.count_nonzero(thresh) / thresh.size
        prev_gray = gray

        motion_history.append(changed_fraction > MOTION_PIXEL_FRACTION)

        if len(motion_history) == motion_history.maxlen:
            high_ratio = sum(motion_history) / len(motion_history)
            if not dancing and high_ratio >= START_DANCE_RATIO:
                dancing = True
                phase_start = time.time()
            elif dancing and high_ratio <= STOP_DANCE_RATIO:
                dancing = False

        # --- audio control ---
        if dancing and not music_playing:
            sound.play(loops=-1)
            music_playing = True
        elif not dancing and music_playing:
            sound.stop()
            music_playing = False

        # --- draw cat overlay (picture-in-picture, top-right corner) ---
        phase = time.time() - phase_start
        cat_img = draw_dancing_cat(CAT_BOX_SIZE, phase, dancing)

        h, w = frame.shape[:2]
        x_off = w - CAT_BOX_SIZE - 20
        y_off = 20
        if x_off > 0 and y_off + CAT_BOX_SIZE < h:
            frame[y_off:y_off + CAT_BOX_SIZE, x_off:x_off + CAT_BOX_SIZE] = cat_img
            cv2.rectangle(frame, (x_off, y_off), (x_off + CAT_BOX_SIZE, y_off + CAT_BOX_SIZE),
                          (255, 255, 255), 2)

        status_text = "STATUS: DANCING! Keep it up" if dancing else "STATUS: Still (start dancing to wake the cat)"
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 200, 0) if dancing else (0, 0, 200), 2, cv2.LINE_AA)

        cv2.imshow("Scuba Cat Dance-Along (press q to quit, c to recalibrate)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            motion_history.clear()
            print("Recalibrated baseline.")

    if music_playing:
        sound.stop()
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

if __name__ == "__main__":
    main()