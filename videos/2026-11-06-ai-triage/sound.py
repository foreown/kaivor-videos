#!/usr/bin/env python3
"""Original sound for the AI triage video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

music/ holds no licensed tracks yet, so this is the rule 3.2 fallback. A 108 BPM A-minor ward pulse (5/9 s per beat): a two-stroke
heartbeat kick, a monitor beep, a filtered saw bass and dark electric-piano stabs, turning to C major on the end card.
The groove drops out for "Get a second opinion", where only the monitor beeps.
Cuts land on beats: hook 0-4, turn 4-7, action 7-13, report 13-19, triage 19-27, decide 27-32, end 32-38.
"""
import sys, wave
import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import butter, lfilter

SR = 48000
BEAT = 5 / 9
DUR = 1267 / 60
rng = np.random.default_rng(1106)
mix = np.zeros((int(SR * DUR), 2))


def env(n, attack=0.002, decay=0.08):
    t = np.arange(n) / SR
    return np.minimum(1, t / attack) * np.exp(-t / decay)


def add(at, sig, gain=1.0, pan=0.0):
    i = int(at * SR)
    if i >= len(mix) or i < 0:
        return
    sig = sig[: len(mix) - i] * gain
    mix[i:i + len(sig), 0] += sig * (1 - pan) ** 0.5
    mix[i:i + len(sig), 1] += sig * (1 + pan) ** 0.5


def beat(b):
    return b * BEAT


def heart(strong=True):     # one stroke of the heartbeat: low and round
    n = int(SR * 0.3)
    t = np.arange(n) / SR
    f = (120 if strong else 95) * np.exp(-t * 26) + 44
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.006, 0.09 if strong else 0.065)


def beep(freq=988.0, dur=0.1):   # the monitor
    n = int(SR * dur)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * freq * t) + 0.18 * np.sin(2 * np.pi * freq * 2 * t)) * np.minimum(1, t / 0.003) * np.minimum(1, (dur - t) / 0.012)


def hat(decay=0.014):
    n = int(SR * 0.05)
    s = np.clip(np.diff(rng.standard_normal(n + 1)), -2.2, 2.2) / 2.2   # clipped, so no single hit spikes past the mix
    return s * env(n, 0.0008, decay)


def rim():
    n = int(SR * 0.05)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 1750 * t) + 0.6 * np.sin(2 * np.pi * 820 * t)) * env(n, 0.0004, 0.009)


def epiano(freq, dur=0.9):  # FM electric piano
    n = int(SR * dur)
    t = np.arange(n) / SR
    mod = np.sin(2 * np.pi * freq * 3.0 * t) * 1.2 * np.exp(-t / 0.14)
    return np.sin(2 * np.pi * freq * t + mod) * np.minimum(1, t / 0.003) * np.exp(-t / 0.42)


def stab(freqs, dur=0.9):
    return sum(epiano(f, dur) for f in freqs) / len(freqs)


def bass(freq, dur):        # a saw through a low-pass that closes on each note
    n = int(SR * dur)
    t = np.arange(n) / SR
    saw = 2 * ((freq * t) % 1) - 1
    b, a = butter(2, 520 / (SR / 2))
    s = lfilter(b, a, saw)
    return np.tanh(s * 2.2) * np.minimum(1, t / 0.005) * np.minimum(1, (dur - t) / 0.02)


def thud():                 # a report card landing
    n = int(SR * 0.16)
    t = np.arange(n) / SR
    f = 230 * np.exp(-t * 46) + 82
    click = np.convolve(rng.standard_normal(n), np.ones(9) / 9, "same") * env(n, 0.0004, 0.006) * 2
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.001, 0.05) + click


def key():
    n = int(SR * 0.03)
    return np.convolve(rng.standard_normal(n), np.ones(5) / 5, "same") * env(n, 0.0005, 0.006)


def pop(pitch=700):
    n = int(SR * 0.13)
    t = np.arange(n) / SR
    f = pitch * (1 + 0.5 * np.exp(-t * 90))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.001, 0.032)


def click():                # a mouse button
    n = int(SR * 0.025)
    return np.clip(np.diff(rng.standard_normal(n + 1)), -2.2, 2.2) / 2.2 * env(n, 0.0003, 0.003)


def whoosh(dur=0.22):       # the form opening
    n = int(SR * dur)
    s = rng.standard_normal(n)
    b, a = butter(2, [500 / (SR / 2), 3200 / (SR / 2)], "band")
    return lfilter(b, a, s) * np.sin(np.pi * np.linspace(0, 1, n)) ** 1.5


def bell(freq):
    n = int(SR * 1.6)
    t = np.arange(n) / SR
    return sum(a * np.sin(2 * np.pi * freq * k * t) * np.exp(-t / d) for k, a, d in ((1, 1, 0.7), (2.0, 0.4, 0.35), (3.01, 0.25, 0.2))) * np.minimum(1, t / 0.002)


# ---- groove: one chord per bar, four beats ----
A, F, C, G = 55.0, 43.65, 65.41, 49.0
ROOTS = [A, F, A, G, A, F, C, G, C, C]
CHORDS = [(220, 261.63, 329.63), (174.61, 220, 261.63), (220, 261.63, 329.63), (196, 246.94, 293.66),
          (220, 261.63, 329.63), (174.61, 220, 261.63), (196, 261.63, 329.63), (196, 246.94, 293.66),
          (196, 261.63, 329.63), (196, 261.63, 329.63)]
TURN = (4, 7)               # the drop, in beats
for b16 in range(0, 38 * 4):
    bt = b16 / 4
    tb = beat(bt)
    bar = int(bt // 4)
    pos = bt % 4
    if TURN[0] <= bt < TURN[1]:
        continue
    if pos in (0, 2):
        add(tb, heart(True), 0.8)
    if pos in (0.5, 2.5):
        add(tb, heart(False), 0.5)
    if pos in (1, 3) and bt >= 7:
        add(tb, rim(), 0.2, 0.15)
    if pos == int(pos):
        add(tb + BEAT / 2, hat(0.02), 0.13, -0.2)
    if bt >= 7:
        add(tb, hat(), 0.07 if b16 % 2 else 0.1, 0.2)
    if bt >= 19 and pos == 3.75:
        add(tb, rim(), 0.12, -0.2)
    if b16 % 2 == 0 and not (pos in (1.5, 3.5)):
        add(tb, bass(ROOTS[bar] * (2 if pos == 3 else 1), BEAT * 0.42), 0.3)
    if pos in (0, 2.5):
        add(tb, stab(CHORDS[bar], 1.0), 0.2 if pos == 0 else 0.13, 0.12 if pos == 0 else -0.12)

# the monitor, on the backbeat through the groove, on every beat through the drop
for b in range(0, 38):
    if TURN[0] <= b < TURN[1]:
        add(beat(b), beep(), 0.3, 0.1)
        add(beat(b + 0.5), beep(), 0.2, -0.1)
    elif b % 4 == 1 and b < 32:
        add(beat(b), beep(), 0.1, 0.25)


# ---- effects, in beats from the start ----
# hook: two more reports land
add(0, thud(), 0.5)
add(beat(1), thud(), 0.55, -0.1)
add(beat(2), thud(), 0.6, 0.1)

# action: keys, the send, Kaivor, the line lights up
for c in range(10):
    add(beat(7) + 0.12 + c * 0.085, key(), 0.5, rng.uniform(-0.2, 0.2))
add(beat(9), pop(620), 0.34)
add(beat(10), pop(940), 0.36)
add(beat(11.5), beep(1319, 0.07), 0.18)

# report: right click, the menu, left click, the form, keys, submit, the reply
add(beat(13) + 0.42, click(), 0.5)
add(beat(13) + 1.08, click(), 0.5)
add(beat(15), whoosh(), 0.2)
for c in range(16):
    add(beat(15) + 0.16 + c * 0.053, key(), 0.45, rng.uniform(-0.2, 0.2))
add(beat(17), click(), 0.5)
add(beat(17) + 0.1, pop(940), 0.34)

# triage: the card, the suggestion, three quick beeps on "Severity: high"
add(beat(19), thud(), 0.6)
add(beat(21), pop(940), 0.36)
for i in range(3):
    add(beat(23) + i * BEAT / 3, beep(1319, 0.075), 0.3)

# decide: the delete click
add(beat(29), click(), 0.55)
add(beat(29), thud(), 0.4)

# end card: resolves on C
add(beat(32), bell(523.25), 0.16)
add(beat(32), bell(783.99), 0.08, -0.2)
add(beat(33), pop(940), 0.26)
add(beat(34), bell(1046.5), 0.1, 0.2)
add(beat(36), stab((130.81, 196, 261.63, 329.63), 1.2), 0.3)

t = np.arange(len(mix)) / SR
mix *= np.minimum(1, (DUR - t) / 0.5)[:, None] ** 0.5
mix *= np.minimum(1, t / 0.01)[:, None]

# master: gentle lowpass, soft saturation and a look-ahead limiter, so the AAC file stays under -2 dBTP after loudnorm
b, a = butter(2, 13000 / (SR / 2))
mix = lfilter(b, a, mix, axis=0)
mix /= max(1e-9, np.abs(mix).max())
mix = np.tanh(mix * 2.4)
peak = np.abs(mix).max(axis=1)
win = int(0.006 * SR)
ceil = 10 ** (-5 / 20) * peak.max()
gain = np.minimum(1, ceil / np.maximum(1e-9, maximum_filter1d(peak, win * 2 + 1)))
gain = uniform_filter1d(gain, win)
mix *= gain[:, None]
mix /= np.abs(mix).max() / 0.7
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype("<i2").tobytes())
print(sys.argv[1])
