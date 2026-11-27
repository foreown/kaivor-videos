#!/usr/bin/env python3
"""Original sound for the autoresponders video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

A 128.57 BPM (28 video frames a beat) UK-garage two-step in Bb minor (Bbm9 - Gbmaj7 - Abadd9 - Fm7, one bar each):
a skipping two-step kick, a clap on 2 and 4, swung sixteenth hats, a round sub bass, and short detuned-saw organ
stabs on the off-beats through a low-pass filter. Every shot in index.html is a whole number of beats, so cuts land
on the grid: hook 0-7, turn 7-10, command 10-20, result 20-29, fact 29-34, end 34-42 (beats).
The idea: a front-desk service bell is Kaivor's answer. The hook plays the groove muffled (low-passed, like it is
playing in the next room) under a notification blip for each of the six questions, each one a step higher, and a
flat two-note sigh for nova's "it's pinned. again." The turn drops everything for a reverse swell into one bell
strike. The command brings the groove up with key ticks for every typed character, a send blip, and the bell for
Kaivor's reply. In the result each question gets a blip and each of Kaivor's answers a bell, a fifth apart. The fact
strips back to bass, hats and stabs with a soft pop per line and a thunk for the flag. The end card plays the full
groove and rings out on a rolled Bbm(add9) with a last bell to the final frame.
"""
import sys, wave
import numpy as np
from scipy.signal import lfilter, butter

SR = 48000
FPS = 60
BEAT_FRAMES = 28
BEAT = BEAT_FRAMES / FPS                                      # 0.4667 s
SHOT_BEATS = [7, 3, 10, 9, 5, 8]
frames = [round(b * BEAT_FRAMES) for b in SHOT_BEATS]         # same rounding as tools/capture.mjs
starts_b = np.concatenate([[0], np.cumsum(SHOT_BEATS)])
s_hook, s_turn, s_cmd, s_res, s_fact, s_end, s_total = starts_b
DUR = sum(frames) / FPS
rng = np.random.default_rng(1127)
n_all = int(SR * DUR) + 1
buses = {k: np.zeros((n_all, 2)) for k in ('muffled', 'music', 'drums', 'fx')}


def T(beat):
    return beat * BEAT


def add(bus, at, sig, gain=1.0, pan=0.0):
    i = int(round(at * SR))
    if i >= n_all or i < 0:
        return
    sig = sig[: n_all - i] * gain
    b = buses[bus]
    b[i:i + len(sig), 0] += sig * np.sqrt((1 - pan) / 2) * 1.414
    b[i:i + len(sig), 1] += sig * np.sqrt((1 + pan) / 2) * 1.414


def tt(dur):
    return np.arange(int(dur * SR)) / SR


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def filt(x, lo=None, hi=None, order=2):
    if lo and hi:
        b, a = butter(order, [lo / (SR / 2), hi / (SR / 2)], 'band')
    elif lo:
        b, a = butter(order, lo / (SR / 2), 'high')
    else:
        b, a = butter(order, hi / (SR / 2), 'low')
    return lfilter(b, a, x, axis=0)


def env(n, a=0.005, r=0.1, hold=0.0):
    t = np.arange(n) / SR
    e = np.minimum(1, t / max(a, 1e-4))
    rel = np.where(t > a + hold, np.exp(-(t - a - hold) / max(r, 1e-4)), 1.0)
    return e * rel


# ---------------- instruments ----------------
def kick():
    t = tt(0.35)
    f = 48 + 110 * np.exp(-t / 0.03)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-t / 0.16) + 0.25 * filt(rng.standard_normal(len(t)), lo=2000) * np.exp(-t / 0.004)


def clap():
    t = tt(0.25)
    n = filt(rng.standard_normal(len(t)), 900, 4500)
    e = np.zeros(len(t))
    for d in (0.0, 0.011, 0.022):
        e += np.where(t >= d, np.exp(-(t - d) / 0.012), 0) * 0.5
    e += np.where(t >= 0.03, np.exp(-(t - 0.03) / 0.09), 0)
    return n * e


def hat(open_=False):
    t = tt(0.25 if open_ else 0.06)
    n = filt(rng.standard_normal(len(t)), lo=7000)
    return n * np.exp(-t / (0.07 if open_ else 0.014))


def rim():
    t = tt(0.06)
    return (np.sin(2 * np.pi * 1700 * t) * 0.6 + filt(rng.standard_normal(len(t)), 1500, 5000)) * np.exp(-t / 0.012)


def sub(m, dur):
    t = tt(dur)
    s = np.sin(2 * np.pi * hz(m) * t) + 0.18 * np.sin(4 * np.pi * hz(m) * t)
    return s * env(len(t), 0.006, 0.12, hold=dur * 0.55)


def stab(chord, dur=0.2, cutoff=2200):
    """Detuned saws through a low-pass: a short garage organ stab."""
    t = tt(dur + 0.12)
    s = np.zeros(len(t))
    for m in chord:
        for det in (-0.08, 0.0, 0.08):
            f = hz(m + det)
            s += 2 * ((t * f) % 1.0) - 1
    s = filt(s / (len(chord) * 3), hi=cutoff, order=2)
    return s * env(len(t), 0.004, 0.07, hold=dur * 0.5)


def bell(m, dur=1.8):
    """Service-desk bell: a bright strike with inharmonic partials and a long ring."""
    t = tt(dur)
    f = hz(m)
    parts = [(1.0, 1.0, 1.4), (2.76, 0.45, 0.6), (5.40, 0.25, 0.3), (8.93, 0.12, 0.15), (1.003, 0.5, 1.6)]
    s = sum(a * np.sin(2 * np.pi * f * k * t) * np.exp(-t / d) for k, a, d in parts)
    s += 0.3 * filt(rng.standard_normal(len(t)), lo=4000) * np.exp(-t / 0.003)
    return s * np.minimum(1, t / 0.0008)


def blip(m, dur=0.14):
    t = tt(dur)
    f = hz(m) * (1 + 0.04 * np.exp(-t / 0.01))
    return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * env(len(t), 0.002, 0.045)


def key_tick():
    t = tt(0.03)
    return filt(rng.standard_normal(len(t)), 2500, 7000) * np.exp(-t / 0.004)


def sigh(m1, m2):
    t = tt(0.7)
    f = np.where(t < 0.28, hz(m1), hz(m2))
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sign(np.sin(ph)) * 0.4 + np.sin(ph)
    return filt(s, hi=900) * env(len(t), 0.02, 0.18, hold=0.35)


def swell(dur):
    t = tt(dur)
    n = filt(rng.standard_normal(len(t)), 600, 6000)
    return n * (t / dur) ** 3


def thunk():
    t = tt(0.2)
    f = 140 * np.exp(-t / 0.05) + 70
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.07)


def whoosh():
    t = tt(0.18)
    return filt(rng.standard_normal(len(t)), 1500, 8000) * np.sin(np.pi * t / 0.18) ** 2


# ---------------- the groove ----------------
# Bbm9 - Gbmaj7 - Abadd9 - Fm7 (stab voicings, bass roots)
CHORDS = [([61, 65, 68, 72], 46), ([61, 65, 66, 70], 42), ([60, 63, 68, 70], 44), ([60, 63, 65, 68], 41)]
SWING = 0.13 * BEAT  # how late the off sixteenths land


def groove(b0, b1, bus='music', drums=True, stabs=True, bass=True, kicks=True, level=1.0):
    dbus = 'drums' if bus == 'music' else bus
    for b in range(int(b0), int(b1)):
        bar = b // 4
        chord, root = CHORDS[bar % 4]
        beat_in_bar = b % 4
        if drums:
            if kicks and beat_in_bar == 0:
                add(dbus, T(b), kick(), 0.9 * level)
            if kicks and beat_in_bar == 2:
                add(dbus, T(b) + 0.75 * BEAT, kick(), 0.75 * level)   # the two-step skip
            if beat_in_bar in (1, 3):
                add(dbus, T(b), clap(), 0.45 * level, pan=0.05)
            for k in range(4):
                at = T(b) + k * BEAT / 4 + (SWING if k % 2 else 0)
                add(dbus, at, hat(open_=(k == 2)), (0.16 if k % 2 else 0.1) * level, pan=0.3)
            if beat_in_bar == 3:
                add(dbus, T(b) + 0.5 * BEAT + SWING, rim(), 0.18 * level, pan=-0.3)
        if bass:
            if beat_in_bar == 0:
                add(bus, T(b), sub(root - 12, BEAT * 1.4), 0.55 * level)
            if beat_in_bar == 2:
                add(bus, T(b) + 0.75 * BEAT, sub(root - 12, BEAT * 0.9), 0.45 * level)
            if beat_in_bar == 3:
                add(bus, T(b) + 0.5 * BEAT + SWING, sub(root - 5, BEAT * 0.4), 0.35 * level)
        if stabs:
            add(bus, T(b) + 0.5 * BEAT, stab(chord), 0.22 * level, pan=-0.15)
            if beat_in_bar == 1:
                add(bus, T(b) + 0.75 * BEAT + SWING, stab(chord, 0.12), 0.14 * level, pan=0.15)


# ---------------- the edit ----------------
# hook: the groove in the next room, a blip per question, nova's sigh
groove(s_hook, s_turn, bus='muffled', level=0.9)
H_Q = [0.5, 1.25, 2, 2.75, 3.5, 4.25]
for i, b in enumerate(H_Q):
    add('fx', T(s_hook + b), blip(79 + [0, 2, 3, 5, 7, 8][i]), 0.32, pan=[-0.3, 0.2, -0.1, 0.3, -0.2, 0.1][i])
add('fx', T(s_hook + 5), sigh(58, 56), 0.28)

# turn: reverse swell into one bell strike
add('fx', T(s_turn) - 0.05, whoosh(), 0.25)
add('fx', T(s_turn + 1) - 0.6, swell(0.6), 0.16)
add('fx', T(s_turn + 1), bell(82, 2.2), 0.42)
add('music', T(s_turn + 1), sub(46, BEAT * 2), 0.4)

# command: groove up, key ticks, send, bell for Kaivor's reply
groove(s_cmd, s_res, level=0.85, kicks=True)
CMD = ',autoresponder add get roles, grab them in #roles --not_strict --reply'
c_type, c_send = T(0.5), T(5)
d = c_send - c_type - 0.12


def inout(x):
    x = min(1, max(0, x))
    return 8 * x ** 4 if x < 0.5 else 1 - (-2 * x + 2) ** 4 / 2


last = 0
for f in range(int(c_type * FPS), int(c_send * FPS)):
    tl = f / FPS
    n = int(np.ceil(len(CMD) * inout((tl - c_type) / d)))
    if n > last:
        add('fx', T(s_cmd) + tl, key_tick(), 0.2, pan=float(rng.uniform(-0.25, 0.25)))
        last = n
add('fx', T(s_cmd + 5), whoosh(), 0.2)
add('fx', T(s_cmd + 5), blip(84, 0.1), 0.2)
add('fx', T(s_cmd + 5.75), bell(82), 0.36)

# result: blip per question, bell per answer
groove(s_res, s_fact, level=1.0)
add('fx', T(s_res + 0.5), blip(80), 0.28, pan=-0.2)
add('fx', T(s_res + 1.25), bell(82), 0.36, pan=-0.1)
add('fx', T(s_res + 3), blip(82), 0.28, pan=0.2)
add('fx', T(s_res + 3.75), bell(89), 0.32, pan=0.1)

# fact: stripped back, a pop per line, a thunk for the flag
groove(s_fact, s_end, kicks=False, level=0.8)
for i, b in enumerate([1.5, 2, 2.5]):
    add('fx', T(s_fact + b), blip(77 + 3 * i, 0.1), 0.24)
add('fx', T(s_fact + 3.25), thunk(), 0.4)

# end: full groove for the card, then a rolled chord and a last bell to the final frame
groove(s_end, s_total - 2, level=1.0)
for i, m in enumerate([58, 61, 65, 68, 72]):
    add('music', T(s_total - 2) + i * 0.04, stab([m], dur=1.6, cutoff=3000), 0.2)
add('music', T(s_total - 2), sub(46, T(2)), 0.5)
add('fx', T(s_total - 2), bell(82, 1.6), 0.3)

# ---------------- mix ----------------
mix = buses['music'] + buses['drums'] * 0.9 + buses['fx']
mix += filt(buses['muffled'], hi=650, order=4) * 1.4
# a short room on the fx bus
d_n = int(0.09 * SR)
room = np.zeros_like(buses['fx'])
room[d_n:] = buses['fx'][:-d_n] * 0.18
mix += filt(room, hi=5000)
# fade the last 0.2 s just enough to avoid a click
fade = int(0.2 * SR)
mix[-fade:] *= np.linspace(1, 0.0, fade)[:, None] ** 0.5
mix /= max(1e-9, np.abs(mix).max()) / 0.8
out = (mix * 32767).astype(np.int16)
with wave.open(sys.argv[1], 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(out.tobytes())
print(f'{sys.argv[1]}: {DUR:.3f} s, {len(out)} samples')
