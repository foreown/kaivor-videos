#!/usr/bin/env python3
"""Original sound for the role shop video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

music/ holds no licensed tracks yet, so this is the rule 3.2 fallback. A bright 120 BPM marimba-and-bass groove
in F major (0.5 s per beat), with till sounds on top: coin dings, a cash drawer, receipt printer ticks and a stamp.
The groove drops out for "20 hours later", where only a clock runs fast.
Cuts land on beats: hook 0-3, turn 3-4, action 4-9.5, receipt 9.5-13, staff 13-16, fact 16-18.5, end 18.5-21.5.
"""
import sys, wave
import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d

SR = 48000
BEAT = 0.5
DUR = 21.5
rng = np.random.default_rng(23)
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
    n = int(SR * 0.24)
    t = np.arange(n) / SR
    f = 150 * np.exp(-t * 34) + 52
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.004, 0.075)


def clap():
    n = int(SR * 0.16)
    s = np.diff(rng.standard_normal(n + 1))
    e = sum(env(n, 0.0005, 0.012) * (np.arange(n) >= d * SR / 1000) for d in (0, 11, 22)) / 2 + env(n, 0.001, 0.05) * 0.5
    return s * e


def shaker(decay=0.02):
    n = int(SR * 0.07)
    return np.diff(rng.standard_normal(n + 1)) * env(n, 0.003, decay)


def marimba(freq, dur=0.32):
    n = int(SR * dur)
    t = np.arange(n) / SR
    s = np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.13) + 0.32 * np.sin(2 * np.pi * freq * 4 * t) * np.exp(-t / 0.03)
    return s * np.minimum(1, t / 0.001)


def bass(freq, dur):
    n = int(SR * dur)
    t = np.arange(n) / SR
    s = np.tanh(1.6 * (np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * freq * 2 * t)))
    return s * np.minimum(1, t / 0.008) * np.minimum(1, (dur - t) / 0.03)


def coin(pitch=1975.5):     # a small coin ding
    n = int(SR * 0.5)
    t = np.arange(n) / SR
    s = sum(a * np.sin(2 * np.pi * pitch * k * t) * np.exp(-t / d) for k, a, d in ((1, 1, 0.16), (1.5, 0.5, 0.1), (2.76, 0.35, 0.05)))
    return s * np.minimum(1, t / 0.001)


def drawer():               # the cash drawer: a ratchet, then a bell
    n = int(SR * 0.3)
    s = np.zeros(n)
    for k in range(7):
        i = int(k * 0.02 * SR)
        b = np.diff(rng.standard_normal(int(SR * 0.012) + 1)) * env(int(SR * 0.012), 0.0003, 0.003)
        s[i:i + len(b)] += b
    return s


def tick(pitch=2100):       # a clock
    n = int(SR * 0.03)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * pitch * t) + 0.5 * rng.standard_normal(n)) * env(n, 0.0003, 0.005)


def printer(dur):           # receipt printer teeth
    n = int(SR * dur)
    t = np.arange(n) / SR
    teeth = (np.sin(2 * np.pi * 310 * t) > 0.4).astype(float)
    s = teeth * (0.6 + 0.4 * rng.standard_normal(n))
    s = np.diff(np.concatenate([[0], s]))
    return s * np.sin(np.pi * np.linspace(0, 1, n)) ** 0.4


def stamp():                # a rubber stamp on a counter
    n = int(SR * 0.26)
    t = np.arange(n) / SR
    f = 190 * np.exp(-t * 40) + 62
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.0005, 0.07)
            + 0.5 * np.convolve(rng.standard_normal(n), np.ones(12) / 12, "same") * 3 * env(n, 0.0005, 0.018))


def pop(pitch=760):         # a chat message arriving
    n = int(SR * 0.13)
    t = np.arange(n) / SR
    f = pitch * (1 + 0.5 * np.exp(-t * 90))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.001, 0.032)


def key():                  # a soft key press
    n = int(SR * 0.03)
    s = np.convolve(rng.standard_normal(n), np.ones(5) / 5, "same")
    return s * env(n, 0.0005, 0.006)


def nope():                 # "You cannot afford that yet."
    return marimba(174.61, 0.3) + marimba(164.81, 0.3) * 0.8


# ---- groove ----
F, D, Bb, C = 87.31, 73.42, 58.27, 65.41
ROOTS = [F, D, Bb, C]
CHORDS = [(349.23, 440.0, 523.25), (293.66, 349.23, 440.0), (293.66, 349.23, 466.16), (329.63, 392.0, 523.25)]
PATTERN = (0, 0.75, 1.5, 2.25, 3.0, 3.5)       # marimba 3-3-2 over a 4-beat bar
for bar in range(int(DUR / (BEAT * 4)) + 1):
    at = bar * BEAT * 4
    prog = bar % 4
    for b in range(4):
        tb = at + b * BEAT
        if 3.0 <= tb < 4.0 or tb >= DUR:      # the drop under "20 hours later"
            continue
        add(tb, kick(), 0.82)
        if b in (1, 3):
            add(tb, clap(), 0.2, 0.1)
        add(tb + BEAT / 2, shaker(), 0.16, -0.2)
        add(tb + BEAT * 0.75, shaker(0.01), 0.08, 0.2)
        add(tb, bass(ROOTS[prog], BEAT * 0.42), 0.28)
        add(tb + BEAT * 0.75, bass(ROOTS[prog] * 2, BEAT * 0.2), 0.14)
    for i, off in enumerate(PATTERN):
        tp = at + off * BEAT
        if 3.0 <= tp < 4.0 or tp >= DUR:
            continue
        note = CHORDS[prog][i % 3] * (2 if i in (2, 5) else 1)
        add(tp, marimba(note), 0.13, -0.25 + 0.1 * i)
        if off in (0, 1.5, 3.0):
            add(tp, marimba(CHORDS[prog][(i + 1) % 3]), 0.08, 0.2)

# ---- effects, global seconds ----
# hook: ,buy 2, Kaivor says no, teo
add(0.5, pop(620), 0.34)
add(1.0, nope(), 0.5)
add(2.0, pop(760), 0.3)

# turn: a clock running fast through the drop
for i in range(16):
    add(3.0 + i * 0.0625, tick(2100 if i % 2 else 1600), 0.3 + 0.012 * i, 0.15 if i % 2 else -0.15)

# action: keys, ,daily, coin, keys, ,buy, the till, the name turns
A = 4.0
for c in range(6):
    add(A + 0.05 + c * 0.06, key(), 0.6, rng.uniform(-0.2, 0.2))
add(A + 0.5, pop(620), 0.34)
add(A + 1.0, coin(), 0.3)
add(A + 1.08, coin(2349.3), 0.22, 0.2)
for c in range(6):
    add(A + 2.05 + c * 0.06, key(), 0.6, rng.uniform(-0.2, 0.2))
add(A + 2.5, pop(620), 0.34)
add(A + 3.0, drawer(), 0.5)
add(A + 3.0, coin(1567.98), 0.26)
add(A + 3.12, coin(2093.0), 0.3, -0.15)
add(A + 3.5, coin(2637.0), 0.34, 0.1)
add(A + 4.5, pop(760), 0.3)

# receipt: one printer burst per row, then the stamp
Rc = 9.5
for i in range(4):
    add(Rc + i * 0.5, printer(0.14), 0.16, 0.1)
add(Rc + 1.86, printer(0.14), 0.16, 0.1)
add(Rc + 2.0, stamp(), 0.55)

# staff: two commands, two answers
S = 13.0
add(S + 0.25, pop(620), 0.32)
add(S + 0.75, pop(990), 0.36)
add(S + 1.25, pop(620), 0.32)
add(S + 1.75, coin(1975.5), 0.24)

# fact: the refusal
add(16.25, pop(620), 0.32)
add(16.75, nope(), 0.5)

# end card: a last coin, resolving on F
add(18.5 + 0.5, coin(2093.0), 0.26)
add(18.5 + 1.0, sum(marimba(f, 1.2) for f in (349.23, 523.25, 698.46)), 0.12)

t = np.arange(len(mix)) / SR
mix *= np.minimum(1, (DUR - t) / 0.5)[:, None] ** 0.5
mix *= np.minimum(1, t / 0.02)[:, None]

# master: soft saturation so kicks, stamp and dings don't spike past the true-peak limit after AAC
# (a second-order lowpass at 13 kHz takes off the edge the noise bursts add)
a = np.exp(-2 * np.pi * 13000 / SR)
for ch in range(2):
    for _ in range(2):
        y, prev = np.empty(len(mix)), 0.0
        for i, v in enumerate(mix[:, ch]):
            prev = (1 - a) * v + a * prev
            y[i] = prev
        mix[:, ch] = y
mix /= max(1e-9, np.abs(mix).max())
mix = np.tanh(mix * 2.6)
# a look-ahead limiter (6 ms window) holds the crest down, so the AAC file stays under -2 dBTP after loudnorm
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
