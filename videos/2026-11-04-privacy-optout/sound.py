#!/usr/bin/env python3
"""Original sound for the privacy opt-out video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

music/ holds no licensed tracks yet, so this is the rule 3.2 fallback. A dry 100 BPM groove in D minor
(0.6 s per beat), with marker strokes for the highlights and redaction bars. Cuts land on beats:
hook 0-3.0, turn 3.0-4.2, action 4.2-9.6, result 9.6-13.2, covers 13.2-16.2, fact 16.2-18.6, end 18.6-21.6.
"""
import sys, wave
import numpy as np

SR = 48000
BEAT = 0.6
DUR = 21.6
rng = np.random.default_rng(11)
mix = np.zeros((int(SR * DUR), 2))


def env(n, attack=0.002, decay=0.08):
    t = np.arange(n) / SR
    return np.minimum(1, t / attack) * np.exp(-t / decay)


def add(at, sig, gain=1.0, pan=0.0):
    i = int(at * SR)
    if i >= len(mix):
        return
    sig = sig[: len(mix) - i] * gain
    mix[i:i + len(sig), 0] += sig * (1 - pan) ** 0.5
    mix[i:i + len(sig), 1] += sig * (1 + pan) ** 0.5


def kick():
    n = int(SR * 0.28)
    t = np.arange(n) / SR
    f = 120 * np.exp(-t * 28) + 48
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.001, 0.09)


def hat(decay=0.018):
    n = int(SR * 0.08)
    s = np.diff(rng.standard_normal(n + 1))
    return s * env(n, 0.0005, decay)


def rim():
    n = int(SR * 0.09)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 1700 * t) * 0.5 + np.diff(rng.standard_normal(n + 1)) * 0.5) * env(n, 0.0005, 0.012)


def bass(freq, dur):
    n = int(SR * dur)
    t = np.arange(n) / SR
    s = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(2 * np.pi * freq * 2 * t) + 0.12 * np.sin(2 * np.pi * freq * 3 * t)
    return s * np.minimum(1, t / 0.006) * np.minimum(1, (dur - t) / 0.03)


def pluck(freq, dur=0.5):
    n = int(SR * dur)
    t = np.arange(n) / SR
    s = sum(a * np.sin(2 * np.pi * freq * k * t) for k, a in ((1, 1), (2, 0.4), (3, 0.16)))
    return s * env(n, 0.002, 0.16)


def pop(pitch=760):        # a chat message arriving
    n = int(SR * 0.14)
    t = np.arange(n) / SR
    f = pitch * (1 + 0.5 * np.exp(-t * 90))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.001, 0.035)


def key():                 # a soft key press
    n = int(SR * 0.03)
    s = np.convolve(rng.standard_normal(n), np.ones(5) / 5, "same")
    return s * env(n, 0.0005, 0.006)


def marker(dur=0.17):      # a felt marker dragged across paper
    n = int(SR * dur)
    s = np.convolve(rng.standard_normal(n), np.ones(9) / 9, "same")
    s = np.diff(np.concatenate([[0], s]))
    shape = np.sin(np.pi * np.linspace(0, 1, n)) ** 0.6
    return s * shape * (1 + 0.4 * np.sin(2 * np.pi * 37 * np.arange(n) / SR)) * 2.2


def swish():               # a message deleted
    n = int(SR * 0.16)
    s = rng.standard_normal(n)
    a = np.linspace(0.02, 0.5, n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = y[i - 1] + a[i] * (s[i] - y[i - 1])
    return y * np.sin(np.pi * np.linspace(0, 1, n)) ** 2


def denied():              # the "nothing to snipe" answer
    return pluck(146.83, 0.4) * 0.8 + pluck(138.59, 0.4) * 0.5


def chord(freqs):
    return sum(pluck(f, 0.9) for f in freqs)


# ---- groove ----
ROOTS = [73.42, 58.27, 65.41, 55.0]            # D Bb C A
CHORDS = [(293.66, 349.23, 440.0), (233.08, 293.66, 349.23), (261.63, 329.63, 392.0), (220.0, 277.18, 329.63)]
beats = int(round(DUR / BEAT))
for b in range(beats):
    at = b * BEAT
    bar = (b // 4) % 4
    if b % 4 in (0, 2) or b % 8 == 7:
        add(at, kick(), 0.85)
    if b % 4 in (1, 3):
        add(at, rim(), 0.22, 0.15)
    add(at + BEAT / 2, hat(), 0.14, -0.2)
    if b % 2 == 1:
        add(at + BEAT * 0.75, hat(0.008), 0.08, 0.2)
    add(at, bass(ROOTS[bar], BEAT * 0.7), 0.32)
    if b % 4 == 0:
        add(at, chord(CHORDS[bar]), 0.075, -0.15)
    if b % 4 == 2:
        add(at + BEAT / 2, chord(CHORDS[bar]), 0.05, 0.15)

# ---- effects, global seconds ----
# hook: delete, snipe command, Kaivor's embed, kofi
add(0.55, key(), 0.8)
add(0.6, swish(), 0.5)
add(1.2, pop(620), 0.34)
add(1.8, pop(990), 0.42)
add(2.4, pop(760), 0.34)

# turn: the marker across "opt out."
add(3.0 + 0.3, marker(0.25), 0.5)

# action: click, keys, send, Kaivor, highlights
A = 4.2
add(A + 0.85, key(), 0.9)
for c in range(15):
    add(A + 0.9 + c * (0.75 / 15) + rng.uniform(-0.006, 0.006), key(), 0.6, rng.uniform(-0.2, 0.2))
add(A + 1.8, pop(620), 0.34)
add(A + 2.4, pop(990), 0.42)
for i in range(6):
    add(A + 2.7 + i * 0.3, marker(0.15), 0.3, rng.uniform(-0.2, 0.2))

# result: delete, snipe, nothing
R = 9.6
add(R + 0.55, key(), 0.8)
add(R + 0.6, swish(), 0.5)
add(R + 1.2, pop(620), 0.34)
add(R + 1.8, denied(), 0.4)
add(R + 2.4, pop(760), 0.34)
add(R + 3.0, pop(700), 0.3)

# covers: one bar stroke per word
for at in (0.3, 0.9, 1.5, 1.8, 2.1):
    add(13.2 + at, marker(0.16), 0.46)

# end card: the groove resolves on D
add(18.6 + 0.6, chord((293.66, 440.0, 587.33)), 0.11)

t = np.arange(len(mix)) / SR
mix *= np.minimum(1, (DUR - t) / 0.5)[:, None] ** 0.5

mix /= max(1e-9, np.abs(mix).max()) / 0.7
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype("<i2").tobytes())
print(sys.argv[1])
