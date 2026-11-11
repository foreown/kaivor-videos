#!/usr/bin/env python3
"""Original sound for the ticket ratings video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

A warm 116 BPM Bb-major "night desk" groove: a kalimba (sine tine with a fast metallic overtone) playing a
syncopated pattern over Bbmaj7 / Gm7 / Ebmaj7 / F, a round sub bass, a soft kick, a wooden rim click on 2 and 4
and a brushed shaker. Every shot in index.html is a whole number of beats, so the cuts land on the grid:
hook 0-6, turn 6-8, action 8-18, result 18-25, fact 25-33, end 33-39 (beats).
The idea: the five stars are heard. Each star that fills on screen plays one note of a rising Bb pentatonic
glass chime, so a 5-star rating is a little five-note run (in the action shot when mira clicks 5★, again in the
result shot and on the end card). In the hook only a low-passed kalimba and a soft kick play (nobody knows how
the ticket went), the turn adds a rising swell, and the full groove comes in with the DM. The fact shot goes
half-time with a soft tick for each highlighted stat, and the end card resolves on Bbmaj9.
On top: message pops, button clicks, a soft whoosh for the form, key clicks for the comment.
"""
import sys, wave
import numpy as np
from scipy.signal import lfilter

SR = 48000
FPS = 60
BPM = 116
BEAT = 60 / BPM
SHOT_BEATS = [6, 2, 10, 7, 8, 6]
frames = [round(b * BEAT * FPS) for b in SHOT_BEATS]          # same rounding as tools/capture.mjs
starts = np.concatenate([[0], np.cumsum(frames)]) / FPS        # shot start times in seconds
DUR = starts[-1]
rng = np.random.default_rng(1111)
n_all = int(SR * DUR) + 1
kal = np.zeros((n_all, 2))
drums = np.zeros((n_all, 2))
low = np.zeros((n_all, 2))
fx = np.zeros((n_all, 2))


def B(k):
    return k * BEAT


def add(bus, at, sig, gain=1.0, pan=0.0):
    i = int(round(at * SR))
    if i >= n_all or i < 0:
        return
    sig = sig[: n_all - i] * gain
    bus[i:i + len(sig), 0] += sig * np.sqrt((1 - pan) / 2) * 1.414
    bus[i:i + len(sig), 1] += sig * np.sqrt((1 + pan) / 2) * 1.414


def tt(dur):
    return np.arange(int(dur * SR)) / SR


def lp(x, cutoff):
    a = np.exp(-2 * np.pi * cutoff / SR)
    return lfilter([1 - a], [1, -a], x)


def hp(x, cutoff):
    return x - lp(x, cutoff)


def hz(m):
    return 440 * 2 ** ((m - 69) / 12)


def tine(m, dur=0.9, v=1.0):
    """Kalimba: sine fundamental, a short inharmonic overtone and a tiny click."""
    t = tt(dur)
    f = hz(m)
    body = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.42)
    over = 0.35 * np.sin(2 * np.pi * f * 5.93 * t) * np.exp(-t / 0.025)
    oct2 = 0.12 * np.sin(2 * np.pi * f * 2.01 * t) * np.exp(-t / 0.12)
    click = hp(rng.standard_normal(len(t)), 2500) * np.exp(-t / 0.002) * 0.15
    return (body + over + oct2 + click) * np.minimum(1, t / 0.001) * np.minimum(1, (dur - t) / 0.03) * v


def glass(m, dur=1.2):
    """The star chime: bright and bell-like."""
    t = tt(dur)
    f = hz(m)
    return (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * 2.76 * f * t) * np.exp(-t / 0.15)
            + 0.25 * np.sin(2 * np.pi * 5.4 * f * t) * np.exp(-t / 0.05)) * np.minimum(1, t / 0.002) * np.exp(-t / 0.45)


def kick():
    t = tt(0.3)
    f = 44 + 100 * np.exp(-t / 0.028)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.13)


def rim():
    t = tt(0.09)
    tone = np.sin(2 * np.pi * 1750 * t) * np.exp(-t / 0.012) + 0.6 * np.sin(2 * np.pi * 820 * t) * np.exp(-t / 0.02)
    return tone + hp(rng.standard_normal(len(t)), 3000) * np.exp(-t / 0.004) * 0.4


def shaker(v=1.0):
    t = tt(0.1)
    n = hp(rng.standard_normal(len(t)), 5500)
    return n * np.minimum(1, t / 0.012) * np.exp(-t / 0.03) * v


def sub(m, dur):
    t = tt(dur)
    e = np.minimum(1, t / 0.01) * np.exp(-t / 0.8) * np.minimum(1, (dur - t) / 0.05)
    return (np.sin(2 * np.pi * hz(m) * t) + 0.15 * np.sin(4 * np.pi * hz(m) * t)) * e


def key():
    t = tt(0.035)
    return hp(rng.standard_normal(len(t)), 3000) * np.exp(-t / 0.005) + 0.25 * np.sin(2 * np.pi * 2900 * t) * np.exp(-t / 0.007)


def click():
    t = tt(0.06)
    return np.sin(2 * np.pi * 1200 * t) * np.exp(-t / 0.008) + hp(rng.standard_normal(len(t)), 2000) * np.exp(-t / 0.003) * 0.5


def pop(f):
    t = tt(0.15)
    return np.sin(2 * np.pi * (f + 300 * np.exp(-t / 0.018)) * t) * np.exp(-t / 0.045)


def whoosh(dur=0.32):
    t = tt(dur)
    n = rng.standard_normal(len(t))
    k = t / dur
    env = np.sin(np.pi * k) ** 2
    return lp(hp(n, 400), 3000) * env * 0.5


def swell(dur):
    t = tt(dur)
    n = lp(hp(rng.standard_normal(len(t)), 1500), 6000)
    return n * (t / dur) ** 2.5 * np.minimum(1, (dur - t) / 0.02)


# ---- the band: one chord per bar (4 beats) ----
CHORDS = [  # (bass, kalimba notes)
    (34, [62, 65, 69, 70, 74, 77]),   # Bbmaj7 (D F A Bb D F)
    (31, [58, 62, 65, 67, 70, 74]),   # Gm7
    (39, [58, 63, 67, 70, 74, 75]),   # Ebmaj7
    (41, [60, 65, 69, 72, 74, 77]),   # F(add9)
]
PATTERN = [(0, 0), (0.75, 2), (1.5, 4), (2, 1), (2.75, 3), (3.25, 5), (3.5, 2)]   # (beat in bar, note index)
total_beats = sum(SHOT_BEATS)
s0, s1, s2, s3, s4, s5 = starts[:6]
for bar in range(int(np.ceil(total_beats / 4))):
    root, notes = CHORDS[bar % 4]
    if bar * 4 >= 36:                      # the end card resolves on Bb
        root, notes = CHORDS[0]
    for off, idx in PATTERN:
        beat = bar * 4 + off
        if beat >= total_beats - 2:
            break
        if 25 <= beat < 33 and off not in (0, 2, 1.5):   # fact shot: sparser
            continue
        add(kal, B(beat), tine(notes[idx], 0.9, 1.0 if off in (0, 2) else 0.75), 0.42, -0.4 + 0.16 * idx)
    if bar * 4 >= 8:                       # the bass comes in with the DM
        add(low, B(bar * 4), sub(root, B(2.6)), 0.55)
        add(low, B(bar * 4 + 2.75), sub(root + 7, B(1.1)), 0.3)

for b in range(total_beats):
    at = B(b)
    if b < 8:                              # hook: soft kick on 1 and 3; turn: four-on-floor build
        if b < 6 and b % 2 == 0:
            add(drums, at, kick(), 0.5)
        if 6 <= b < 8:
            add(drums, at, kick(), 0.55)
            add(drums, at + B(0.5), kick(), 0.3)
        continue
    if b >= total_beats - 2:               # last two beats ring out
        continue
    if 25 <= b < 33:                       # fact: half-time
        if b % 4 == 1:
            add(drums, at, kick(), 0.8)
        if b % 4 == 3:
            add(drums, at, rim(), 0.35, 0.15)
        add(drums, at, shaker(0.7), 0.1, 0.35)
        continue
    if b % 2 == 0:
        add(drums, at, kick(), 0.85)
    if b % 4 == 3:
        add(drums, at + B(0.5), kick(), 0.5)
    if b % 2 == 1:
        add(drums, at, rim(), 0.38, 0.15)
    for h in range(4):
        add(drums, at + B(h / 4) + (0.012 if h % 2 else 0), shaker(1.0 if h % 2 == 0 else 0.55), 0.1, 0.35)

# end: a Bbmaj9 on the downbeat of the last bar, rolled
for j, m in enumerate([46, 58, 62, 65, 69, 72, 74]):
    add(kal, s5 + B(3) + j * 0.03, tine(m, 2.0), 0.36, -0.3 + 0.1 * j)
add(low, s5 + B(3), sub(34, B(3)), 0.6)

# ---- the star chimes: one note per star (times match index.html) ----
PENTA = [74, 77, 79, 82, 86]   # D F G Bb D


def star_run(at, gap, gain=0.17):
    for i, m in enumerate(PENTA):
        add(fx, at + i * gap, glass(m), gain, -0.5 + 0.25 * i)


star_run(s2 + B(3), B(0.25))                 # mira clicks 5★
star_run(s3 + B(0.5), B(0.5), 0.15)          # result shot stars fill one by one
star_run(s5 + B(1), B(0.4), 0.15)            # end card stars

# ---- UI sounds ----
def typing(at, n, span):
    for c in range(n):
        add(fx, at + c * span / n + rng.uniform(-0.005, 0.005), key(), 0.32, rng.uniform(-0.2, 0.2))


add(fx, s0 + B(0.75), pop(820), 0.26)        # "all back, thank you!!"
add(fx, s0 + B(2), pop(820), 0.26)           # ,ticket close
add(fx, s0 + B(2.75), pop(1250), 0.28)       # Closing…
add(fx, s1 + B(0.5), swell(B(1.5)), 0.12)    # turn: rising swell into the DM
add(fx, s2 + B(0.4), pop(1250), 0.3)         # the rating DM arrives
add(fx, s2 + B(0.4) + 0.07, glass(86, 0.6), 0.06, 0.3)
add(fx, s2 + B(3), click(), 0.45)            # 5★
add(fx, s2 + B(3.25), whoosh(), 0.35)        # the form opens
typing(s2 + B(4), len('teo had my roles back in five minutes'), B(3))
add(fx, s2 + B(8.5), click(), 0.45)          # Submit
add(fx, s2 + B(8.75), pop(1000), 0.28)       # the DM turns into "Thanks!"
add(fx, s3 + B(3), pop(1250), 0.3)           # the log embed
add(fx, s4 + B(0.75), pop(1250), 0.3)        # ticket stats
for k in (2.5, 3.5, 4.5):                    # each highlighted stat
    add(fx, s4 + B(k), glass(82 if k < 4 else 86, 0.4), 0.07, 0.2)

t = np.arange(n_all) / SR
# the hook is muffled: kalimba low-passed until the turn
thin = np.stack([lp(kal[:, c], 1200) for c in range(2)], axis=1)
k = np.clip((t - s1) / 0.1, 0, 1)[:, None]
kal = thin * (1 - k) * 1.6 + kal * k

mix = kal + drums + low + fx
mix *= (np.minimum(1, t / 0.015) * np.minimum(1, (DUR - t) / 0.5))[:, None]
mix /= max(1e-9, np.abs(mix).max()) / 0.7
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype("<i2").tobytes())
print(sys.argv[1], f"{DUR:.3f}s")
