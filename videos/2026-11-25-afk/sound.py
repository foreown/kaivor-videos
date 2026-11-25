#!/usr/bin/env python3
"""Original sound for the AFK video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

An 85.71 BPM (42 video frames a beat) lo-fi lullaby in F minor (Fm9 - Dbmaj9 - Abmaj7 - Eb6, two beats each): a dusty
swung kit (soft kick, rim-and-noise snare on 2 and 4, brushed eighth-note hats), an FM electric piano with tremolo, a
round sine bass, a celesta for Kaivor, and vinyl crackle under everything (clicks drawn at random, never a loop).
Every shot in index.html is a whole number of beats, so cuts land on the grid:
hook 0-5, turn 5-8, command 8-14, result 14-20, back 20-24, fact 24-27, end 27-31 (beats).
The idea: the hook is a low, muffled F-minor pad, "through the pillow", with six notification pings that pile up, each a
semitone higher, then a felt thud on "He was asleep." On the turn the door hanger swings in on a soft swish and lands
with a wooden clack; AFK is a three-note celesta and the reason a small bell. The groove starts on the command: key
ticks for each typed character, a send blip, and a two-note celesta lullaby for Kaivor's reply. In the result each ping
is answered by Kaivor's descending "shh" chime. Next morning the piano opens up (the low-pass lifts) with a rising
celesta for teo's message and a brighter chime for the welcome back. The fact drops to kick, hats and pad; the end card
plays one more bar and rings out on a rolled Abmaj9 to the last frame.
"""
import sys, wave
import numpy as np
from scipy.signal import lfilter, butter

SR = 48000
FPS = 60
BEAT_FRAMES = 42
BEAT = BEAT_FRAMES / FPS                                      # 0.7 s
SHOT_BEATS = [5, 3, 6, 6, 4, 3, 4]
frames = [round(b * BEAT_FRAMES) for b in SHOT_BEATS]         # same rounding as tools/capture.mjs
starts_b = np.concatenate([[0], np.cumsum(SHOT_BEATS)])
s_hook, s_turn, s_cmd, s_res, s_back, s_fact, s_end, s_total = starts_b
DUR = sum(frames) / FPS
rng = np.random.default_rng(1125)
n_all = int(SR * DUR) + 1
buses = {k: np.zeros((n_all, 2)) for k in ('pillow', 'music', 'drums', 'fx')}


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
def epiano(m, dur=1.6, vel=1.0):
    t = tt(dur)
    f = hz(m)
    idx = 1.6 * vel * np.exp(-t / 0.35)
    mod = np.sin(2 * np.pi * f * t) * idx
    s = np.sin(2 * np.pi * f * t + mod) + 0.25 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.2)
    trem = 1 + 0.12 * np.sin(2 * np.pi * 4.8 * t)
    return s * trem * env(len(t), 0.004, 0.9) * vel


def chord(at, notes, dur, bus='music', gain=0.16, roll=0.0, vel=1.0):
    for k, m in enumerate(notes):
        add(bus, at + k * roll, epiano(m, dur, vel), gain, pan=(k - len(notes) / 2) * 0.12)


def bass(m, dur):
    t = tt(dur)
    s = np.sin(2 * np.pi * hz(m) * t) + 0.18 * np.sin(2 * np.pi * 2 * hz(m) * t)
    return np.tanh(1.4 * s) * env(len(t), 0.02, dur * 0.6, hold=dur * 0.3)


def pad(notes, dur, att=0.6):
    t = tt(dur)
    s = np.zeros(len(t))
    for m in notes:
        for d in (-0.08, 0.08):
            s += np.sin(2 * np.pi * hz(m + d) * t + rng.uniform(0, 6))
    e = np.minimum(1, t / att) * np.minimum(1, (dur - t) / 0.4)
    return s * e / len(notes)


def celesta(m, dur=1.4, vel=1.0):
    t = tt(dur)
    f = hz(m)
    s = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 4 * f * t) * np.exp(-t / 0.08) + 0.15 * np.sin(2 * np.pi * 3 * f * t)
    return s * env(len(t), 0.002, 0.55) * vel


def kick(vel=1.0):
    t = tt(0.4)
    f = 52 * (1 + 2.2 * np.exp(-t / 0.03))
    s = np.sin(2 * np.pi * np.cumsum(f) / SR)
    return np.tanh(2 * s) * env(len(t), 0.002, 0.16) * vel


def snare(vel=1.0):
    t = tt(0.3)
    n = filt(rng.standard_normal(len(t)), 1500, 7000) * env(len(t), 0.001, 0.07)
    body = np.sin(2 * np.pi * 210 * t) * env(len(t), 0.001, 0.04)
    rim = np.sin(2 * np.pi * 1700 * t) * env(len(t), 0.0005, 0.012)
    return (0.8 * n + 0.5 * body + 0.4 * rim) * vel


def hat(vel=1.0, open_=False):
    t = tt(0.25 if open_ else 0.09)
    n = filt(rng.standard_normal(len(t)), 6000, 14000)
    return n * env(len(t), 0.001, 0.08 if open_ else 0.022) * vel


def ping(m, vel=1.0):
    a = celesta(m, 0.5, 1.0) * 0.6
    b = celesta(m + 7, 0.6, 1.0) * 0.6
    out = np.zeros(int(0.7 * SR))
    out[:len(a)] += a
    j = int(0.07 * SR)
    out[j:j + len(b)] += b[: len(out) - j]
    return out * vel


def swish(dur=0.45, lo=400, hi=3000):
    t = tt(dur)
    n = rng.standard_normal(len(t))
    out = np.zeros(len(t))
    seg = 1024
    for k in range(0, len(t), seg):
        c = lo + (hi - lo) * (k / len(t))
        out[k:k + seg] = filt(n[k:k + seg], c * 0.7, min(c * 1.4, 20000))
    return out * np.sin(np.pi * t / dur) ** 2


def clack():
    t = tt(0.2)
    n = filt(rng.standard_normal(len(t)), 500, 2500) * env(len(t), 0.0005, 0.02)
    tone = np.sin(2 * np.pi * 620 * t) * env(len(t), 0.0005, 0.05) + 0.5 * np.sin(2 * np.pi * 1130 * t) * env(len(t), 0.0005, 0.03)
    return n + 0.7 * tone


def thud():
    t = tt(0.7)
    f = 44 * (1 + 1.5 * np.exp(-t / 0.04))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(len(t), 0.003, 0.25)


def tick():
    t = tt(0.04)
    return filt(rng.standard_normal(len(t)), 2500, 9000) * env(len(t), 0.0005, 0.006)


def blip(m=84):
    t = tt(0.12)
    return np.sin(2 * np.pi * hz(m) * t * (1 + 0.3 * t)) * env(len(t), 0.002, 0.04)


# ---------------- arrangement ----------------
FM9, DBM9, ABM7, EB6 = (56, 60, 63, 67), (53, 56, 60, 63), (55, 60, 63, 67), (55, 58, 60, 63)
PROG = [(41, FM9), (37, DBM9), (44, ABM7), (39, EB6)]

# hook: muffled pad, six piling pings, a thud
add('pillow', T(0), pad((41, 53, 56, 60, 63), T(5) + 0.3, att=0.3), 0.30)
add('pillow', T(0), bass(29, T(3.5)), 0.20)
for i in range(6):
    add('fx', T(0.25 + 0.5 * i), ping(81 + i, 0.6 + 0.07 * i), 0.5, pan=(-0.3 if i % 2 else 0.3))
add('fx', T(3.5), thud(), 0.8)
add('pillow', T(3.5), pad((41, 48, 56), T(1.5) + 0.4, att=0.05), 0.25)

# turn: swish, clack, celesta AFK, bell for the reason
add('fx', T(s_turn + 0.5) - 0.1, swish(0.5), 0.35)
add('fx', T(s_turn + 0.5) + 0.42, clack(), 0.55)
for k, m in enumerate((77, 80, 84)):
    add('fx', T(s_turn + 1.25) + k * 0.07, celesta(m, 1.2), 0.30)
add('fx', T(s_turn + 1.75), celesta(87, 1.4), 0.25)
add('music', T(s_turn), pad((41, 56, 60, 63, 67), T(3) + 0.3, att=0.4), 0.22)
add('music', T(s_turn + 2), epiano(53, 1.0, 0.6), 0.10)   # pickup into the groove
add('drums', T(s_turn + 2.5), snare(0.5), 0.35)
add('drums', T(s_turn + 2.75), snare(0.7), 0.4)

# groove: command, result, back (beats 8-24); lighter in the fact (24-27), last bar in the end card (27-31)
SWING = 0.58
for b in range(int(s_cmd), int(s_end)):
    pi = ((b - int(s_cmd)) // 2) % 4
    root, notes = PROG[pi]
    if (b - int(s_cmd)) % 2 == 0:
        bright = b >= s_back
        chord(T(b), [n + (12 if bright and k == len(notes) - 1 else 0) for k, n in enumerate(notes)], T(2) + 0.4,
              gain=0.15, roll=0.012, vel=1.0 if bright else 0.85)
        add('music', T(b), bass(root - 12, T(2)), 0.34)
        add('music', T(b + 1.5 - (0.5 - SWING) * 0), bass(root - 12 + 7, T(0.5)), 0.18)
    in_fact = s_fact <= b < s_end
    pos = (b - int(s_cmd)) % 4
    if pos in (0,) or (pos == 2 and not in_fact):
        add('drums', T(b), kick(), 0.65)
    if pos == 2 and in_fact:
        add('drums', T(b), kick(0.8), 0.55)
    if pos in (1, 3) and not in_fact:
        add('drums', T(b), snare(), 0.42)
    if pos == 3 and not in_fact:
        add('drums', T(b + SWING), kick(0.6), 0.45)
    for h in (0, SWING):
        add('drums', T(b + h), hat(0.9 if h == 0 else 0.6), 0.22, pan=0.25)

# command: key ticks per typed character, send blip, reply chime
CMD_LEN = len(',afk sleeping, back at 9')
C_TYPE, C_SEND, C_REPLY = 0.5, 2.5, 3.25
d = C_SEND - C_TYPE - 0.12 / BEAT


def inout(x):
    return 8 * x ** 4 if x < 0.5 else 1 - (-2 * x + 2) ** 4 / 2


prev = 0
for step in range(1, 601):
    x = step / 600
    n_chars = int(np.ceil(CMD_LEN * inout(x)))
    if n_chars > prev:
        add('fx', T(s_cmd + C_TYPE + x * d), tick(), 0.30, pan=rng.uniform(-0.3, 0.3))
        prev = n_chars
add('fx', T(s_cmd + C_SEND), blip(86), 0.25)
add('fx', T(s_cmd + C_REPLY), celesta(84, 1.2), 0.26)
add('fx', T(s_cmd + C_REPLY) + 0.14, celesta(80, 1.4), 0.24)

# result: two pings, each answered by Kaivor's descending chime
for ping_b, reply_b in ((0.5, 1.25), (2.25, 3.0)):
    add('fx', T(s_res + ping_b), ping(84, 0.7), 0.42)
    add('fx', T(s_res + reply_b), celesta(87, 1.2), 0.24)
    add('fx', T(s_res + reply_b) + 0.13, celesta(84, 1.2), 0.22)
    add('fx', T(s_res + reply_b) + 0.26, celesta(80, 1.5), 0.20)

# back: rising celesta for teo's message, brighter chime for the welcome back
for k, m in enumerate((72, 75, 79, 84)):
    add('fx', T(s_back + 0.5) + k * 0.06, celesta(m, 1.0), 0.20)
add('fx', T(s_back + 1.25), celesta(87, 1.2), 0.25)
add('fx', T(s_back + 1.25) + 0.12, celesta(91, 1.5), 0.22)

# fact and end: pad, then the last bar and a rolled Abmaj9 to the end
add('music', T(s_fact), pad((44, 56, 60, 63, 67), T(3) + 0.3, att=0.3), 0.18)
end_len = DUR - T(s_end) + 0.05
add('music', T(s_end), bass(32, end_len), 0.38)
chord(T(s_end), (56, 60, 63, 67, 70, 75), end_len, gain=0.15, roll=0.05)
add('music', T(s_end), pad((44, 56, 63, 67, 70), end_len, att=0.2), 0.18)
add('drums', T(s_end), kick(), 0.65)
for b in range(1, 4):
    add('drums', T(s_end + b), hat(0.6), 0.16, pan=0.25)
add('fx', T(s_end + 1.75), celesta(87, 1.6), 0.18)   # kaivor.space

# vinyl crackle under everything: random clicks and a soft hiss, never a loop
crackle = filt(rng.standard_normal(n_all), 1500, 9000) * 0.012
pops = rng.random(n_all) < 7 / SR
crackle[pops] += rng.uniform(-0.25, 0.25, pops.sum())
crackle = filt(crackle, 800, 10000)
add('fx', 0, np.tile(crackle[:, None], 1)[:, 0], 0.5)

# ---------------- mix ----------------
pillow = filt(buses['pillow'], hi=700)
music = buses['music'].copy()
cut = int(T(s_back) * SR)
music[:cut] = filt(music[:cut], hi=3200)
music[cut:] = filt(music[cut:], hi=9000)
drums = filt(buses['drums'], hi=8000)
mix = pillow + music + drums * 0.9 + buses['fx']
mix = np.tanh(mix * 1.2) / 1.2
fade_in = np.minimum(1, np.arange(n_all) / (0.02 * SR))
mix *= fade_in[:, None]
mix /= np.max(np.abs(mix)) * 1.12

out = sys.argv[1] if len(sys.argv) > 1 else 'sound.wav'
with wave.open(out, 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype('<i2').tobytes())
print(f'{out}: {DUR:.3f} s')
