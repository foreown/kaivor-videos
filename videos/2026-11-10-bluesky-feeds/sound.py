#!/usr/bin/env python3
"""Original sound for the Bluesky feeds video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

A bright 104 BPM A-major "open air" groove: a plucked nylon-ish guitar (Karplus-Strong) picking eighth-note
arpeggios over A / F#m / D / E, a round sub bass, a soft kick, a finger-snap clap on 2 and 4 and a light
tambourine. Every shot in index.html is a whole number of beats, so the cuts land on the grid:
hook 0-5, turn 5-8, action 8-16, result 16-23, fact 23-28, end 28-34 (beats).
The idea: the "wire" motif is heard as a short gliding tone that pans from left (the profile) to right (the
channel) whenever the dot travels on screen, and it lands with a soft two-note ping. In the hook only the
guitar (a little muffled) and a soft kick play (nobody in the server knows what's going on), the turn adds the wire, and the full groove comes in
with the command. Kaivor's feed post arrives with a notification ding. The fact shot goes half-time with two
wire glides, and the end card resolves on Amaj9. On top: key clicks for typing and message pops.
"""
import sys, wave
import numpy as np
from scipy.signal import lfilter

SR = 48000
FPS = 60
BPM = 104
BEAT = 60 / BPM
SHOT_BEATS = [5, 3, 8, 7, 5, 6]
frames = [round(b * BEAT * FPS) for b in SHOT_BEATS]          # same rounding as tools/capture.mjs
starts = np.concatenate([[0], np.cumsum(frames)]) / FPS        # shot start times in seconds
DUR = starts[-1]
rng = np.random.default_rng(1110)
n_all = int(SR * DUR) + 1
gtr = np.zeros((n_all, 2))
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


def pluck(m, dur, bright=0.5):
    """Karplus-Strong string."""
    n = int(dur * SR)
    period = SR / hz(m)
    p = int(round(period))
    hist = np.zeros(n + p + 1)
    hist[:p + 1] = lp(rng.uniform(-1, 1, p + 1), 2500 + 6000 * bright)
    decay = 0.996
    h = hist.tolist()
    for j in range(p + 1, n + p + 1):
        h[j] = decay * 0.5 * (h[j - p] + h[j - p - 1])
    out = np.array(h[p + 1:])
    env = np.minimum(1, (dur - tt(dur)) / 0.04)
    return out * env


_pluck_cache = {}


def pl(m, dur, bright):
    key = (m, round(dur, 3), bright)
    if key not in _pluck_cache:
        _pluck_cache[key] = pluck(m, dur, bright)
    return _pluck_cache[key]


def kick():
    t = tt(0.32)
    f = 46 + 110 * np.exp(-t / 0.03)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.14)


def snap():
    t = tt(0.16)
    n = rng.standard_normal(len(t))
    burst = np.zeros(len(t))
    for d in (0, 0.009, 0.018):
        burst += np.exp(-np.clip(t - d, 0, None) / 0.006) * (t >= d)
    body = lfilter([1], [1, -1.6 * np.cos(2 * np.pi * 1400 / SR), 0.81], n) * 0.08
    return hp(n, 900) * burst * 0.7 + body * np.exp(-t / 0.05)


def tamb(v=1.0):
    t = tt(0.12)
    n = hp(rng.standard_normal(len(t)), 7000)
    jingle = sum(np.sin(2 * np.pi * f * t) for f in (5200, 6900, 8300)) * 0.15
    return (n + jingle) * np.minimum(1, t / 0.004) * np.exp(-t / 0.04) * v


def sub(m, dur):
    t = tt(dur)
    e = np.minimum(1, t / 0.008) * np.exp(-t / 0.9) * np.minimum(1, (dur - t) / 0.05)
    return (np.sin(2 * np.pi * hz(m) * t) + 0.2 * np.sin(4 * np.pi * hz(m) * t)) * e


def key():
    t = tt(0.035)
    return hp(rng.standard_normal(len(t)), 3000) * np.exp(-t / 0.005) + 0.25 * np.sin(2 * np.pi * 2900 * t) * np.exp(-t / 0.007)


def pop(f):
    t = tt(0.15)
    return np.sin(2 * np.pi * (f + 300 * np.exp(-t / 0.018)) * t) * np.exp(-t / 0.045)


def bell(m, dur=1.0):
    t = tt(dur)
    f = hz(m)
    return (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * 2.0 * f * t) * np.exp(-t / 0.2)
            + 0.2 * np.sin(2 * np.pi * 3.0 * f * t) * np.exp(-t / 0.08)) * np.minimum(1, t / 0.002) * np.exp(-t / 0.35)


def ping(m1, m2):
    out = np.zeros(int(1.1 * SR))
    a = bell(m1, 1.0)
    b = bell(m2, 1.0)
    out[:len(a)] += a
    i = int(0.075 * SR)
    out[i:i + len(b)] += b
    return out


def glide(at, dur, f0, f1, gain=0.16):
    """The wire: a soft gliding tone that pans left to right with the dot (same inOut easing as the page)."""
    t = tt(dur)
    k = np.clip(t / dur, 0, 1)
    e = np.where(k < .5, 8 * k ** 4, 1 - (-2 * k + 2) ** 4 / 2)
    f = f0 * (f1 / f0) ** e
    ph = 2 * np.pi * np.cumsum(f) / SR
    tone = (np.sin(ph) + 0.25 * np.sin(2 * ph)) * np.minimum(1, t / 0.05) * np.minimum(1, (dur - t) / 0.06) * gain
    pan = -0.85 + 1.7 * e
    i = int(round(at * SR))
    m = min(len(tone), n_all - i)
    fx[i:i + m, 0] += tone[:m] * np.sqrt((1 - pan[:m]) / 2) * 1.414
    fx[i:i + m, 1] += tone[:m] * np.sqrt((1 + pan[:m]) / 2) * 1.414


# ---- the band: one chord per bar (4 beats) ----
CHORDS = [  # (bass, arpeggio notes)
    (33, [57, 64, 69, 71, 73, 76]),   # Aadd9
    (30, [54, 61, 66, 68, 69, 73]),   # F#m9
    (38, [50, 57, 62, 64, 66, 69]),   # Dadd9
    (40, [52, 59, 64, 66, 68, 71]),   # Eadd9
]
ORDER = [0, 1, 3, 2, 4, 5, 3, 1]       # picking pattern over the 6 notes, eighth notes
total_beats = sum(SHOT_BEATS)
s0, s1, s2, s3, s4, s5 = starts[:6]
for bar in range(int(np.ceil(total_beats / 4))):
    root, notes = CHORDS[bar % 4]
    if bar * 4 >= 28:                     # the end card resolves on A
        root, notes = CHORDS[0]
    for e8 in range(8):
        beat = bar * 4 + e8 / 2
        if beat >= total_beats:
            break
        if 23 <= beat < 28 and e8 % 2:    # fact shot: half-time picking
            continue
        m = notes[ORDER[e8]]
        sw = 0.018 if e8 % 2 else 0.0
        add(gtr, B(beat) + sw, pl(m, B(1.6), 0.55 if e8 % 2 == 0 else 0.35), 0.5 if e8 % 2 == 0 else 0.36,
            -0.35 + 0.14 * ORDER[e8])
    if bar * 4 >= 8:                      # the bass comes in with the command
        add(low, B(bar * 4), sub(root, B(2.4)), 0.55)
        add(low, B(bar * 4 + 2.5), sub(root + 12 if bar % 2 else root + 7, B(1.3)), 0.32)

for b in range(total_beats):
    at = B(b)
    if b < 8:                              # hook: guitar and a soft kick on the downbeats; turn: a heartbeat kick
        if b < 5 and b % 2 == 0:
            add(drums, at, kick(), 0.5)
        if 5 <= b < 8:
            add(drums, at, kick(), 0.4)
        continue
    if 23 <= b < 28:                       # fact: half-time
        if b % 4 == 0:
            add(drums, at, kick(), 0.8)
        if b % 4 == 2:
            add(drums, at, snap(), 0.5, 0.1)
        add(drums, at, tamb(0.6), 0.07, 0.3)
        continue
    if b % 2 == 0 or (b % 4 == 3):
        add(drums, at if b % 4 != 3 else at + B(0.5), kick(), 0.85)
    if b % 2 == 1:
        add(drums, at, snap(), 0.5, 0.1)
    for h in range(2):
        add(drums, at + B(h / 2) + (0.02 if h else 0), tamb(1.0 if h else 0.6), 0.09, 0.3)

# the last bar rings out: one strummed Amaj9 on the final downbeat stretch
for j, m in enumerate([45, 57, 64, 68, 71, 73, 76]):
    add(gtr, s5 + B(4) + j * 0.022, pl(m, B(2.2), 0.6), 0.4, -0.3 + 0.1 * j)

# ---- the wire glides and pings (times match index.html) ----
glide(s1 + 0.2, B(2), 520, 1040)
add(fx, s1 + 0.2 + B(2), ping(81, 88), 0.2, 0.7)
glide(s3 + B(0.5), B(1.5), 520, 1040)
add(fx, s3 + B(2), ping(88, 93), 0.24, 0.2)            # Kaivor posts the feed item: notification ding
for k in (2, 4):                                        # fact shot: the feed keeps coming
    glide(s4 + B(k), B(2), 440, 880, 0.12)
add(fx, s4 + B(4), ping(81, 85), 0.14, 0.7)
glide(s5 + 0.8, B(1.5), 520, 1040)
add(fx, s5 + 0.8 + B(1.5), ping(81, 88), 0.2, 0.7)


# ---- typing, pops, clicks ----
def typing(at, n, span):
    for c in range(n):
        add(fx, at + c * span / n + rng.uniform(-0.005, 0.005), key(), 0.36, rng.uniform(-0.2, 0.2))


add(fx, s0 + B(1.25), pop(820), 0.26)
add(fx, s0 + B(3), pop(820), 0.26)
add(fx, s2 + B(1), key(), 0.5)                              # cursor click
typing(s2 + B(1) + 0.08, len(',notify add bluesky @lumenpark.bsky.social #news'), B(2.5))
add(fx, s2 + B(4), pop(820), 0.28)
add(fx, s2 + B(4.75), pop(1250), 0.3)
add(fx, s2 + B(6), bell(85, 0.8), 0.08, 0.3)
add(fx, s3 + B(4.5), pop(820), 0.24)

t = np.arange(n_all) / SR
# the hook is a little thinner: guitar low-passed until the turn
thin = np.stack([lp(gtr[:, c], 1800) for c in range(2)], axis=1)
k = np.clip((t - s1) / 0.1, 0, 1)[:, None]
gtr = thin * (1 - k) * 1.9 + gtr * k

mix = gtr * 0.9 + drums + low + fx
mix *= (np.minimum(1, t / 0.015) * np.minimum(1, (DUR - t) / 0.45))[:, None]
mix /= max(1e-9, np.abs(mix).max()) / 0.7
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype("<i2").tobytes())
print(sys.argv[1], f"{DUR:.3f}s")
