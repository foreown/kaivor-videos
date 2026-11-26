#!/usr/bin/env python3
"""Original sound for the reminders video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

A 116.13 BPM (31 video frames a beat) bossa nova in A minor (Am9 - D9 - Gmaj7 - Cmaj7, two beats each): a nylon guitar
(Karplus-Strong) comping in the bossa rhythm, a round upright-style bass on root and fifth, a cross-stick clave in the
3-2 pattern, a brushed shaker and a soft kick, and a breathy flute (sine with vibrato and filtered breath noise) for
the tune. Every shot in index.html is a whole number of beats, so cuts land on the grid:
hook 0-7, turn 7-11, command 11-19, dm 19-26, done 26-30, fact 30-35, end 35-41 (beats).
The idea: the hook is a lone wall clock (tick-tock on every beat) under a quiet, muted guitar; the date divider jumps
two days with a quick tape-rewind sweep, mira's ping is a notification blip, "oh no" is a sagging two-note flute and
"He forgot." a low felt thud. On the turn each split-flap card drops with a clack, the four rising a fifth apart. The
groove starts with the command: key ticks for each typed character, a send blip, a two-note flute "got it" for
Kaivor's reply. Two hours later the clock flaps clack again and the DM lands on a three-note bell. The next shot plays
the flute tune over the groove; the fact strips back to bass, clave and guitar; the end card plays one more bar and
rings out on a rolled A6/9 to the last frame.
"""
import sys, wave
import numpy as np
from scipy.signal import lfilter, butter

SR = 48000
FPS = 60
BEAT_FRAMES = 31
BEAT = BEAT_FRAMES / FPS                                      # 0.5167 s
SHOT_BEATS = [7, 4, 8, 7, 4, 5, 6]
frames = [round(b * BEAT_FRAMES) for b in SHOT_BEATS]         # same rounding as tools/capture.mjs
starts_b = np.concatenate([[0], np.cumsum(SHOT_BEATS)])
s_hook, s_turn, s_cmd, s_dm, s_done, s_fact, s_end, s_total = starts_b
DUR = sum(frames) / FPS
rng = np.random.default_rng(1126)
n_all = int(SR * DUR) + 1
buses = {k: np.zeros((n_all, 2)) for k in ('quiet', 'music', 'drums', 'fx')}


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
def nylon(m, dur=1.4, bright=0.5, damp=0.996):
    """Karplus-Strong pluck, darkened for a nylon string."""
    f = hz(m)
    N = int(SR / f)
    n = int(dur * SR)
    buf = filt(rng.uniform(-1, 1, N), hi=1500 + 4000 * bright)
    out = np.zeros(n)
    # vectorised by periods: each period is the smoothed previous one
    prev = buf.copy()
    for k in range(0, n, N):
        seg = prev[: min(N, n - k)]
        out[k:k + len(seg)] = seg
        nxt = damp * 0.5 * (prev + np.roll(prev, -1))
        prev = nxt
    out = filt(out, lo=70)
    return out * env(n, 0.002, dur * 0.5, hold=dur * 0.2)


def strum(at, notes, dur, gain=0.11, roll=0.014, bright=0.45, bus='music', down=True):
    order = notes if down else list(reversed(notes))
    for k, m in enumerate(order):
        add(bus, at + k * roll, nylon(m, dur, bright), gain, pan=(k - len(notes) / 2) * 0.1)


def bass(m, dur):
    t = tt(dur)
    s = np.sin(2 * np.pi * hz(m) * t) + 0.3 * np.sin(2 * np.pi * 2 * hz(m) * t) * np.exp(-t / 0.15)
    return np.tanh(1.3 * s) * env(len(t), 0.008, dur * 0.5, hold=dur * 0.25)


def flute(m, dur, vel=1.0, bend=0.0):
    t = tt(dur)
    vib = 1 + 0.006 * np.sin(2 * np.pi * 5.2 * t) * np.minimum(1, t / 0.25)
    f = hz(m) * vib * (1 + bend * np.minimum(1, t / dur))
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) + 0.18 * np.sin(2 * ph) + 0.05 * np.sin(3 * ph)
    breath = filt(rng.standard_normal(len(t)), hz(m) * 1.5, min(hz(m) * 5, 16000)) * 0.12
    e = np.minimum(1, t / 0.05) * np.minimum(1, np.maximum(0, (dur - t)) / 0.12)
    return (s + breath) * e * vel


def bell(m, dur=1.6, vel=1.0):
    t = tt(dur)
    f = hz(m)
    mod = np.sin(2 * np.pi * f * 3.5 * t) * 2.2 * np.exp(-t / 0.25)
    return np.sin(2 * np.pi * f * t + mod) * env(len(t), 0.002, 0.7) * vel


def kick(vel=1.0):
    t = tt(0.35)
    f = 55 * (1 + 1.8 * np.exp(-t / 0.025))
    s = np.sin(2 * np.pi * np.cumsum(f) / SR)
    return np.tanh(1.6 * s) * env(len(t), 0.002, 0.12) * vel


def xstick(vel=1.0):
    t = tt(0.12)
    tone = np.sin(2 * np.pi * 1850 * t) * env(len(t), 0.0004, 0.012) + 0.6 * np.sin(2 * np.pi * 820 * t) * env(len(t), 0.0004, 0.025)
    n = filt(rng.standard_normal(len(t)), 1500, 6000) * env(len(t), 0.0004, 0.01)
    return (tone + 0.5 * n) * vel


def shaker(vel=1.0):
    t = tt(0.1)
    n = filt(rng.standard_normal(len(t)), 5000, 13000)
    e = np.minimum(1, t / 0.012) * np.exp(-t / 0.03)
    return n * e * vel


def tick(hi=True):
    t = tt(0.05)
    f = 3200 if hi else 2400
    return (np.sin(2 * np.pi * f * t) * 0.5 + filt(rng.standard_normal(len(t)), 2000, 8000)) * env(len(t), 0.0004, 0.006)


def keytick():
    t = tt(0.04)
    return filt(rng.standard_normal(len(t)), 2500, 9000) * env(len(t), 0.0005, 0.006)


def blip(m=86):
    t = tt(0.14)
    return np.sin(2 * np.pi * hz(m) * t * (1 + 0.25 * t)) * env(len(t), 0.002, 0.045)


def flap(m=60):
    """A split-flap card dropping: a plastic clack with a short pitched knock."""
    t = tt(0.16)
    n = filt(rng.standard_normal(len(t)), 900, 5000) * env(len(t), 0.0004, 0.012)
    knock = np.sin(2 * np.pi * hz(m) * t) * env(len(t), 0.0005, 0.035)
    rattle = np.zeros(len(t))
    j = int(0.028 * SR)
    rattle[j:] = n[: len(t) - j] * 0.35
    return n + 0.8 * knock + rattle


def rewind(dur=0.45):
    t = tt(dur)
    f = 300 * np.exp(np.log(12) * t / dur)
    s = np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)) * 0.3
    s = filt(s, hi=4000) + filt(rng.standard_normal(len(t)), 1000, 8000) * 0.3
    return s * np.sin(np.pi * t / dur) ** 1.5


def thud():
    t = tt(0.7)
    f = 46 * (1 + 1.4 * np.exp(-t / 0.04))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(len(t), 0.003, 0.24)


# ---------------- arrangement ----------------
AM9, D9, GM7, CM7 = (57, 60, 64, 67, 71), (54, 60, 64, 69), (55, 59, 62, 66), (55, 60, 64, 71)
PROG = [(45, AM9), (38, D9), (43, GM7), (36, CM7)]
# bossa comping inside two beats (in beats): a stab on 1, a pushed stab on the "and" of 1, and one on the "and" of 2
COMP = [(0.0, 1.0), (0.5, 0.75), (1.5, 0.85)]

# hook: lone wall clock, muted guitar, rewind on the date jump, blips, sagging flute, thud
for b in range(0, int(s_turn)):
    add('fx', T(b), tick(b % 2 == 0), 0.32, pan=0.15)
for b, (root, notes) in zip((0, 2, 4), ((45, AM9), (38, D9), (45, AM9))):
    strum(T(b), notes, T(2) + 0.3, gain=0.07, bright=0.2, bus='quiet')
    add('quiet', T(b), bass(root - 12, T(2)), 0.22)
add('fx', T(2) - 0.42, rewind(0.45), 0.22)
add('fx', T(2.5), blip(88), 0.28)
add('fx', T(2.5) + 0.07, blip(93), 0.22)
add('fx', T(3.75), blip(84), 0.22)
add('fx', T(4.0), flute(69, 0.42, 0.8), 0.20)
add('fx', T(4.0) + 0.42, flute(68, 0.6, 0.8, bend=-0.03), 0.20)     # "oh no"
add('fx', T(5), thud(), 0.75)
add('quiet', T(5), bass(33, T(2)), 0.25)

# turn: four flap clacks rising a fifth apart, a pad of guitar, pickup into the groove
for k, b in enumerate((1, 1.5, 2, 2.5)):
    add('fx', T(s_turn + b), flap(55 + 7 * k), 0.55, pan=(-0.3 + 0.2 * k))
strum(T(s_turn), CM7, T(4), gain=0.08, bright=0.35)
add('music', T(s_turn), bass(36, T(3)), 0.25)
add('drums', T(s_turn + 3), xstick(0.7), 0.30)
add('drums', T(s_turn + 3.5), xstick(0.8), 0.32)
add('music', T(s_turn + 3.5), flute(76, 0.25), 0.12)

# groove: command, dm, done (beats 11-30); stripped in the fact (30-35); last bar in the end card (35-41)
CLAVE = [0, 0.75, 1.5, 2.5, 3.0]                 # 3-2 clave over four beats (positions in beats)
for b in range(int(s_cmd), int(s_end)):
    rel = b - int(s_cmd)
    in_fact = s_fact <= b < s_end
    if rel % 2 == 0:
        root, notes = PROG[(rel // 2) % 4]
        for off, v in COMP:
            strum(T(b + off), notes, T(0.6) + 0.15, gain=0.10 * v, bright=0.45, down=(off == 0))
        add('music', T(b), bass(root - 12, T(1.4)), 0.36)
        add('music', T(b + 1), bass(root - 12 + 7, T(0.9)), 0.26)
        add('music', T(b + 1.5), bass(root - 12, T(0.45)), 0.18)
    bar_pos = rel % 4
    for c in CLAVE:
        if bar_pos <= c < bar_pos + 1:
            add('drums', T(b + (c - bar_pos)), xstick(), 0.24, pan=-0.2)
    if not in_fact:
        add('drums', T(b), kick(0.9 if rel % 2 == 0 else 0.6), 0.5)
        add('drums', T(b + 0.75), kick(0.45), 0.35)
        for h in (0, 0.25, 0.5, 0.75):
            add('drums', T(b + h), shaker(1.0 if h in (0, 0.5) else 0.55), 0.18, pan=0.3)
    else:
        for h in (0, 0.5):
            add('drums', T(b + h), shaker(0.6), 0.12, pan=0.3)

# command: key ticks per typed character, send blip, a two-note flute for the reply
CMD_LEN = len(",remind 2h check mira's map")
C_TYPE, C_SEND, C_REPLY = 0.5, 3.5, 4.25
d = C_SEND - C_TYPE - 0.12 / BEAT


def inout(x):
    return 8 * x ** 4 if x < 0.5 else 1 - (-2 * x + 2) ** 4 / 2


prev = 0
for step in range(1, 601):
    x = step / 600
    n_chars = int(np.ceil(CMD_LEN * inout(x)))
    if n_chars > prev:
        add('fx', T(s_cmd + C_TYPE + x * d), keytick(), 0.30, pan=rng.uniform(-0.3, 0.3))
        prev = n_chars
add('fx', T(s_cmd + C_SEND), blip(86), 0.25)
add('fx', T(s_cmd + C_REPLY), flute(76, 0.22), 0.20)
add('fx', T(s_cmd + C_REPLY) + 0.2, flute(81, 0.5), 0.20)

# dm: flap clacks at 0.5 (two cards) and 1.5 beats (one card), the DM on a three-note bell
add('fx', T(s_dm + 0.5), flap(62), 0.5, pan=-0.2)
add('fx', T(s_dm + 0.5) + 0.06, flap(64), 0.45, pan=-0.05)
add('fx', T(s_dm + 1.5) + 0.0, flap(67), 0.5, pan=-0.1)
for k, m in enumerate((81, 85, 88)):
    add('fx', T(s_dm + 2.5) + k * 0.09, bell(m, 1.5), 0.17)
# the tune, from the DM to the end of the done shot
TUNE = [(2.5, 76, 0.5), (3.0, 79, 0.5), (3.5, 83, 1.0), (4.5, 81, 0.5), (5.0, 79, 0.5), (5.5, 76, 1.5),
        (s_done - s_dm + 0.0, 74, 0.5), (s_done - s_dm + 0.5, 76, 0.5), (s_done - s_dm + 1.0, 79, 1.0),
        (s_done - s_dm + 2.0, 78, 0.5), (s_done - s_dm + 2.5, 76, 1.5)]
for b, m, dur in TUNE:
    add('music', T(s_dm + b), flute(m, T(dur) * 0.95, 0.8), 0.10, pan=0.1)

# done: teo's message blip, mira's reply blip
add('fx', T(s_done + 0.5), blip(88), 0.20)
add('fx', T(s_done + 1.75), blip(91), 0.20)

# fact: "Once." one flute note, "Or every day." a little turn, the pill on a bell
add('fx', T(s_fact), flute(76, T(0.9)), 0.12)
for k, m in enumerate((79, 81, 83)):
    add('fx', T(s_fact + 1) + k * T(0.25), flute(m, T(0.24)), 0.11)
add('fx', T(s_fact + 2.5), bell(88, 1.2), 0.12)

# end: one more bar, then a rolled A6/9 to the last frame
end_len = DUR - T(s_end) + 0.05
add('music', T(s_end), bass(33, end_len), 0.38)
strum(T(s_end), (45, 52, 57, 61, 66, 71, 76), end_len, gain=0.10, roll=0.05, bright=0.5)
add('drums', T(s_end), kick(), 0.5)
add('drums', T(s_end), xstick(), 0.22, pan=-0.2)
for b in (1, 2, 3):
    add('drums', T(s_end + b), shaker(0.5), 0.12, pan=0.3)
add('fx', T(s_end + 2), bell(88, 1.8), 0.12)       # kaivor.space
add('music', T(s_end + 1), flute(76, T(3), 0.7), 0.08)

# ---------------- mix ----------------
quiet = filt(buses['quiet'], hi=1400)
music = filt(buses['music'], hi=10000)
drums = filt(buses['drums'], hi=11000)
mix = quiet + music + drums * 0.9 + buses['fx']
mix = np.tanh(mix * 1.2) / 1.2
fade_in = np.minimum(1, np.arange(n_all) / (0.02 * SR))
fade_out = np.minimum(1, (n_all - np.arange(n_all)) / (0.06 * SR))
mix *= (fade_in * np.maximum(fade_out, 0.0) ** 0.5)[:, None]
mix /= np.max(np.abs(mix)) * 1.12

out = sys.argv[1] if len(sys.argv) > 1 else 'sound.wav'
with wave.open(out, 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype('<i2').tobytes())
print(f'{out}: {DUR:.3f} s')
