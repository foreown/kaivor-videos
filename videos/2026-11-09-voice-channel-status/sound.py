#!/usr/bin/env python3
"""Original sound for the voice channel status video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

A laid-back 112 BPM Eb-major "night radio" groove: a tine electric piano (additive partials with a bell attack)
on Ebmaj9 / Cm9 / Abmaj9 / Bb7sus, a sine bass, a soft kick, a rim click, a brushed shaker and a little tape hiss.
Every shot in index.html is a whole number of beats, so the cuts land on the grid:
hook 0-5, turn 5-8, action 8-16, result 16-22, voicemaster 22-29, fact 29-34, end 34-40 (beats).
The idea: in the hook the music is heard "through the wall" (low-passed, quiet) because nobody knows what is
playing in lounge. The turn is a radio dial sweep through static that locks onto the station. When Kaivor posts
"Now playing" the groove opens up to full brightness, and each status line gets a two-note chime. In the fact
shot a short falling swish clears the status and the band drops to just keys for a beat. The end card resolves
on Ebmaj9. On top: key clicks for typing and message pops.
"""
import sys, wave
import numpy as np
from scipy.signal import lfilter, butter, sosfilt

SR = 48000
FPS = 60
BPM = 112
BEAT = 60 / BPM
SHOT_BEATS = [5, 3, 8, 6, 7, 5, 6]
frames = [round(b * BEAT * FPS) for b in SHOT_BEATS]          # same rounding as tools/capture.mjs
starts = np.concatenate([[0], np.cumsum(frames)]) / FPS        # shot start times in seconds
DUR = starts[-1]
rng = np.random.default_rng(1109)
n_all = int(SR * DUR) + 1
band = np.zeros((n_all, 2))     # drums + bass + keys
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


def epiano(m, dur, vel=1.0):
    t = tt(dur)
    f = hz(m)
    body = (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.6)
            + 0.12 * np.sin(2 * np.pi * 3.01 * f * t) * np.exp(-t / 0.25))
    tine = 0.25 * np.sin(2 * np.pi * 7.1 * f * t) * np.exp(-t / 0.05)
    trem = 1 + 0.12 * np.sin(2 * np.pi * 4.6 * t)
    e = np.minimum(1, t / 0.004) * np.exp(-t / 1.1) * np.minimum(1, (dur - t) / 0.08)
    return (body + tine) * e * trem * vel


def kick():
    t = tt(0.35)
    f = 48 + 90 * np.exp(-t / 0.035)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.16)


def rim():
    t = tt(0.08)
    n = hp(rng.standard_normal(len(t)), 1800) * np.exp(-t / 0.012)
    return 0.6 * n + 0.5 * np.sin(2 * np.pi * 1650 * t) * np.exp(-t / 0.018)


def shaker(v=1.0):
    t = tt(0.09)
    return hp(rng.standard_normal(len(t)), 6000) * np.minimum(1, t / 0.01) * np.exp(-t / 0.03) * v


def bass(m, dur):
    t = tt(dur)
    e = np.minimum(1, t / 0.01) * np.exp(-t / 0.7) * np.minimum(1, (dur - t) / 0.05)
    return (np.sin(2 * np.pi * hz(m) * t) + 0.15 * np.sin(4 * np.pi * hz(m) * t)) * e


def key():
    t = tt(0.04)
    return hp(rng.standard_normal(len(t)), 2500) * np.exp(-t / 0.006) + 0.3 * np.sin(2 * np.pi * 2400 * t) * np.exp(-t / 0.008)


def pop(f):
    t = tt(0.16)
    return np.sin(2 * np.pi * (f + 260 * np.exp(-t / 0.02)) * t) * np.exp(-t / 0.05)


def chime(m1, m2):
    out = np.zeros(int(0.9 * SR))
    for k, m in enumerate([m1, m2]):
        t = tt(0.9 - k * 0.09)
        f = hz(m)
        s = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2.76 * f * t) * np.exp(-t / 0.15)) * np.exp(-t / 0.4)
        i = int(k * 0.09 * SR)
        out[i:i + len(s)] += s * np.minimum(1, t / 0.003)
    return out


# ---- the band: 4-chord loop, one chord per bar (4 beats), from beat 0 to the end ----
CHORDS = [  # (bass, voicing)
    (39, [58, 62, 63, 65, 70]),   # Ebmaj9
    (36, [55, 58, 62, 63, 67]),   # Cm9
    (32, [55, 58, 60, 63, 67]),   # Abmaj9
    (34, [56, 60, 63, 65, 68]),   # Bb7sus
]
total_beats = sum(SHOT_BEATS)
keys_bus = np.zeros((n_all, 2))
for bar in range(int(np.ceil(total_beats / 4))):
    root, voicing = CHORDS[bar % 4]
    at = B(bar * 4)
    if bar * 4 >= 38:                       # end card resolves on Ebmaj9
        root, voicing = CHORDS[0]
    for j, m in enumerate(voicing):
        add(keys_bus, at + j * 0.012, epiano(m, B(2.6), 0.18), 1, -0.3 + 0.15 * j)
        add(keys_bus, at + B(2.5) + j * 0.01, epiano(m, B(1.4), 0.10), 1, 0.3 - 0.15 * j)
    add(band, at, bass(root, B(1.8)), 0.55)
    add(band, at + B(2.5), bass(root + 7 if bar % 2 == 0 else root + 12, B(1.3)), 0.4)
for b in range(total_beats):
    at = B(b)
    if b in (29 + 2, 29 + 3):               # drums drop out for a beat after the status clears
        continue
    if b % 4 in (0, 2) or (b % 4 == 3 and b % 8 == 7):
        add(band, at if b % 4 != 3 else at + B(0.5), kick(), 0.8)
    if b % 4 in (1, 3):
        add(band, at, rim(), 0.35, 0.15)
    for h in range(4):                       # sixteenth shaker, lightly swung
        sw = 0.03 if h % 2 else 0
        add(band, at + B(h / 4) + sw, shaker(1.0 if h % 2 == 0 else 0.6), 0.10, -0.25)
band += keys_bus * 1.0

# ---- "through the wall" in the hook, opening at the now-playing embed ----
s0, s1, s2, s3, s4, s5, s6 = starts[:7]
open_at = s2 + B(4.25)
muffled = np.stack([lp(lp(band[:, c], 420), 420) for c in range(2)], axis=1)
t = np.arange(n_all) / SR
mixk = np.clip((t - open_at) / 0.25, 0, 1)
# during the turn the band is replaced by the radio sweep
gate = np.ones(n_all)
gate[(t >= s1) & (t < s2)] = 0.25
gate = lp(gate, 30)
band = (muffled * (1 - mixk)[:, None] * 0.9 + band * mixk[:, None]) * gate[:, None]

# ---- radio dial sweep in the turn ----
d = s2 - s1
tw = tt(d)
static = hp(rng.standard_normal(len(tw)), 900)
static = sosfilt(butter(2, [1200, 5200], btype='band', fs=SR, output='sos'), static)
lock = np.clip((tw - 0.25 - 0.9) / 0.15, 0, 1)
sweep_f = 300 + 2400 * (1 - np.exp(-np.clip(tw - 0.25, 0, None) / 0.35))
whistle = 0.18 * np.sin(2 * np.pi * np.cumsum(sweep_f) / SR) * (1 - lock)
sw = (static * 0.5 * (1 - 0.85 * lock) + whistle) * np.minimum(1, tw / 0.03) * np.minimum(1, (d - tw) / 0.05)
add(fx, s1, sw, 0.5)
add(fx, s1 + 0.25 + 1.05, chime(75, 82), 0.22)            # station locked

# ---- typing, pops, chimes ----
def typing(at, n, span):
    for c in range(n):
        add(fx, at + c * span / n + rng.uniform(-0.006, 0.006), key(), 0.4, rng.uniform(-0.2, 0.2))

add(fx, s0 + B(1.5), pop(760), 0.26)
add(fx, s0 + B(3.25), pop(760), 0.26)
add(fx, s2 + B(1), key(), 0.5)                             # cursor click
typing(s2 + B(1) + 0.08, len(',play night drive'), B(2))
add(fx, s2 + B(3.5), pop(760), 0.28)
add(fx, s2 + B(4.25), pop(1180), 0.3)
add(fx, s2 + B(5.5), chime(79, 86), 0.26)                  # lounge status appears
add(fx, s3 + B(2.5), pop(760), 0.26)
typing(s4 + 0.2, len(',vcstatus Ranked grind and chill'), B(2.8))
add(fx, s4 + B(3.5), pop(760), 0.28)
add(fx, s4 + B(4.25), pop(1180), 0.3)
add(fx, s4 + B(4.75), chime(77, 84), 0.26)                 # mira's status appears
# the status clears: a short falling swish
tc = tt(0.45)
swish = sosfilt(butter(2, 2500, btype='high', fs=SR, output='sos'), rng.standard_normal(len(tc)))
swish *= np.sin(np.pi * np.clip(tc / 0.45, 0, 1)) ** 2
fall = 0.3 * np.sin(2 * np.pi * np.cumsum(1400 * np.exp(-tc / 0.15) + 200) / SR) * np.exp(-tc / 0.2)
add(fx, s5 + B(2), swish * 0.35 + fall, 0.4)
add(fx, s6 + 0.0, chime(75, 87), 0.2)                     # end card

mix = band + fx
mix *= (np.minimum(1, t / 0.02) * np.minimum(1, (DUR - t) / 0.4))[:, None]
mix += 0.004 * lp(rng.standard_normal((n_all, 2)), 5000)  # tape hiss
mix /= max(1e-9, np.abs(mix).max()) / 0.7
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype("<i2").tobytes())
print(sys.argv[1], f"{DUR:.3f}s")
