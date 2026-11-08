#!/usr/bin/env python3
"""Original sound for the soundboard volume video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

A swung 92 BPM C-minor boom-bap loop: FM electric piano on Cm9 / Abmaj7 / Fm9 / G7, a round sub bass, a dusty
kick and snare, swung hats. Every shot in index.html is a whole number of beats, so the cuts land on the grid:
hook 0-5, turn 5-7, action 7-15, result 15-21, fact 21-25, end 25-31 (beats).
The airhorn is the joke: it blasts twice at full volume in the hook, the music ducks to a filtered hush on
"Turn it down", a fader slide drops in pitch when Kaivor sets 40%, and the same horn plays again in the result
at 0.4 of its amplitude (40% volume). On top: key clicks, message pops, the log pop.
"""
import sys, wave
import numpy as np
from scipy.signal import lfilter

SR = 48000
FPS = 60
BPM = 92
BEAT = 60 / BPM
SHOT_BEATS = [5, 2, 8, 6, 4, 6]
frames = [round(b * BEAT * FPS) for b in SHOT_BEATS]          # same rounding as tools/capture.mjs
starts = np.concatenate([[0], np.cumsum(frames)]) / FPS        # shot start times in seconds
DUR = starts[-1]
rng = np.random.default_rng(1108)
music = np.zeros((int(SR * DUR) + 1, 2))
fx = np.zeros_like(music)
n_all = len(music)


def B(k):
    return k * BEAT


def add(at, sig, gain=1.0, pan=0.0, bus=None):
    bus = music if bus is None else bus
    i = int(round(at * SR))
    if i >= n_all or i < 0:
        return
    sig = sig[: n_all - i] * gain
    bus[i:i + len(sig), 0] += sig * np.sqrt((1 - pan) / 2) * 1.414
    bus[i:i + len(sig), 1] += sig * np.sqrt((1 + pan) / 2) * 1.414


def env(n, attack=0.002, decay=0.1):
    t = np.arange(n) / SR
    return np.minimum(1, t / attack) * np.exp(-t / decay)


def lp(x, cutoff):
    a = np.exp(-2 * np.pi * cutoff / SR)
    return lfilter([1 - a], [1, -a], x)


def hp(x, cutoff):
    return x - lp(x, cutoff)


def midi(m):
    return 440 * 2 ** ((m - 69) / 12)


def saw(freq, n):
    """Band-limited-ish saw; freq may be an array."""
    freq = np.broadcast_to(freq, (n,)).astype(float)
    phase = np.cumsum(freq) / SR
    out = np.zeros(n)
    for k in range(1, 16):
        out += ((-1) ** (k + 1)) * np.sin(2 * np.pi * k * phase) / k * (freq * k < SR / 2.3)
    return out


# ---------- instruments ----------
def epiano(m, dur=0.9, vel=1.0):
    n = int(SR * dur)
    t = np.arange(n) / SR
    f = midi(m)
    index = 1.6 * vel * np.exp(-t / 0.25)
    s = np.sin(2 * np.pi * f * t + index * np.sin(2 * np.pi * f * t))
    s += 0.18 * np.sin(2 * np.pi * f * 14 * t) * np.exp(-t / 0.02) * vel
    return s * np.minimum(1, t / 0.003) * np.exp(-t / 0.7) * np.minimum(1, (dur - t) / 0.05)


def sub(m, dur=0.5):
    n = int(SR * dur)
    t = np.arange(n) / SR
    f = midi(m)
    s = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * 2 * f * t)
    return s * np.minimum(1, t / 0.006) * np.exp(-t / 0.35) * np.minimum(1, (dur - t) / 0.04)


def kick():
    n = int(SR * 0.4)
    t = np.arange(n) / SR
    f = 44 + 95 * np.exp(-t * 28)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.001, 0.16)
    knock = lp(rng.standard_normal(n), 1800) * env(n, 0.0005, 0.006)
    return np.tanh(1.6 * (body + 0.5 * knock)) * 0.9


def snare():
    n = int(SR * 0.3)
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * 190 * t) * env(n, 0.001, 0.05)
    noise = hp(lp(rng.standard_normal(n), 7000), 900) * env(n, 0.001, 0.11)
    return (0.6 * tone + noise) * 0.7


def hat(open_=False):
    n = int(SR * (0.2 if open_ else 0.06))
    return hp(rng.standard_normal(n), 7000) * env(n, 0.0005, 0.07 if open_ else 0.015)


def horn_note(dur, gain=1.0):
    """One air-horn note: three detuned saws on a B-flat with a short pitch scoop and a rough edge."""
    n = int(SR * dur)
    t = np.arange(n) / SR
    base = midi(70) * (1 - 0.06 * np.exp(-t / 0.025)) * (1 + 0.004 * np.sin(2 * np.pi * 6 * t))
    s = saw(base * 0.995, n) + saw(base * 1.006, n) + 0.7 * saw(base * 1.498, n)
    s = lp(s, 3800)
    s = np.tanh(2.2 * s) * 0.6
    return s * np.minimum(1, t / 0.008) * np.minimum(1, (dur - t) / 0.03) * gain


S16 = BEAT / 4


def blast(at, pattern, gain):
    """pattern: list of (start in sixteenths, length in sixteenths)."""
    for s, length in pattern:
        add(at + s * S16, horn_note(length * S16 * 0.94), gain, 0.0, fx)


def pop(pitch=900):
    n = int(SR * 0.16)
    t = np.arange(n) / SR
    f = pitch * (1 + 0.5 * np.exp(-t * 60))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.001, 0.035)


def key():
    n = int(SR * 0.025)
    noise = np.convolve(rng.standard_normal(n), np.ones(5) / 5, "same")
    return noise * env(n, 0.0004, 0.005)


def click():
    n = int(SR * 0.04)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 2400 * t) * 0.5 + rng.standard_normal(n) * 0.3) * env(n, 0.0003, 0.006)


def slide(dur=0.6):
    """A fader sliding down: filtered noise falling in pitch plus a soft thunk at the end."""
    n = int(SR * dur)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    out = np.zeros(n)
    seg = int(SR * 0.01)
    for i in range(0, n, seg):                  # piecewise band, falling 2400 Hz -> 300 Hz
        c = 2400 * (300 / 2400) ** (i / n)
        chunk = noise[max(0, i - 400):i + seg]
        band = hp(lp(chunk, c * 1.4), c * 0.7)[-len(noise[i:i + seg]):]
        out[i:i + seg] = band
    shape = np.minimum(1, t / 0.03) * np.minimum(1, (dur - t) / 0.08)
    thunk = np.zeros(n)
    m = int(SR * 0.08)
    thunk[-m:] = np.sin(2 * np.pi * 120 * np.arange(m) / SR) * env(m, 0.001, 0.03)
    return out * shape * 1.8 + thunk * 0.8


# ---------- the groove ----------
CHORDS = [[60, 63, 67, 70, 74], [56, 60, 63, 67], [53, 56, 60, 63, 67], [55, 59, 62, 65, 68]]
ROOTS = [36, 32, 29, 31]
TOTAL = sum(SHOT_BEATS)
SWING = 0.58  # the off-beat eighth lands at 58% of the beat

for beat in range(TOTAL):
    at = B(beat)
    bar = beat // 4
    chord, root = CHORDS[bar % 4], ROOTS[bar % 4]
    in_hook = beat < 5
    in_turn = 5 <= beat < 7
    in_fact = 21 <= beat < 25
    in_end = beat >= 25
    full = not (in_turn or in_fact)
    late_end = beat >= 29

    if in_turn:
        continue
    # drums: kick on 1 and the swung "and" of 2, snare on 2 and 4
    if full and not late_end:
        if beat % 4 == 0:
            add(at, kick(), 0.9)
        if beat % 4 == 1:
            add(at + BEAT * SWING, kick(), 0.65)
        if beat % 4 == 2 and not in_hook:
            add(at, kick(), 0.7)
        if beat % 2 == 1:
            add(at, snare(), 0.55, 0.05)
    if in_fact and beat % 2 == 0:
        add(at, kick(), 0.5)
    if not late_end:
        add(at, hat(), 0.16 if full else 0.09, 0.3)
        add(at + BEAT * SWING, hat(open_=full and beat % 4 == 3), 0.12 if full else 0.07, 0.3)
    # e-piano comping: on beat 1 and the swung "and" of 2
    if beat % 4 == 0 and not late_end:
        vel = 0.5 if in_fact else 1.0
        for i, m in enumerate(chord):
            add(at + i * 0.008, epiano(m, BEAT * 1.4, vel), 0.09, (-0.4, -0.2, 0, 0.2, 0.4)[i])
    if beat % 4 == 1 and not late_end and not in_fact:
        for i, m in enumerate(chord):
            add(at + BEAT * SWING + i * 0.006, epiano(m, BEAT * 1.1, 0.7), 0.065, (-0.4, -0.2, 0, 0.2, 0.4)[i])
    # sub bass
    if full and not late_end:
        if beat % 4 == 0:
            add(at, sub(root, BEAT * 1.4), 0.5)
        if beat % 4 == 2:
            add(at + BEAT * SWING, sub(root + 7 if root < 34 else root - 5, BEAT * 0.9), 0.38)

# turn: everything drops to one low, dark chord ("turn it down")
turn_len = starts[2] - starts[1]
for i, m in enumerate([48, 55, 58, 63]):
    s = lp(epiano(m, turn_len, 0.4), 700)
    add(starts[1], s, 0.12, (-0.3, -0.1, 0.1, 0.3)[i])
add(starts[1], sub(36, turn_len), 0.35)

# fact: a soft held Fm9 pad under the half-time beat, so the shot never goes quiet
fact_len = starts[5] - starts[4]
for i, m in enumerate([53, 56, 60, 63, 67]):
    n = int(SR * fact_len)
    tt = np.arange(n) / SR
    pad = lp(saw(midi(m) * 0.998, n) + saw(midi(m) * 1.003, n), 650)
    pad *= np.minimum(1, tt / 0.15) * np.minimum(1, (fact_len - tt) / 0.05)
    add(starts[4], pad, 0.05, (-0.4, -0.2, 0, 0.2, 0.4)[i])

# end card: a last Cm9 that rings out
ring_at = starts[5] + B(4)
for i, m in enumerate([48, 60, 63, 67, 70, 74]):
    n = int(SR * (DUR - ring_at))
    s = epiano(m, DUR - ring_at, 0.8)
    s *= np.minimum(1, ((DUR - ring_at) - np.arange(n) / SR) / 0.5)
    add(ring_at, s, 0.09, (-0.4, -0.25, -0.1, 0.1, 0.25, 0.4)[i])
add(ring_at, kick(), 0.8)
add(ring_at, sub(36, DUR - ring_at), 0.4)

# ---------- effects, timed to index.html ----------
s0, s1, s2, s3, s4, s5 = starts[:6]
# hook: two full-volume airhorn blasts (B(0.5) for B(1.5), B(3) for B(1.25)), two messages
blast(s0 + B(0.5), [(0, 1.4), (2, 0.6), (3, 0.6), (4, 2.0)], 1.0)
blast(s0 + B(3), [(0, 0.6), (1, 0.6), (2, 3.0)], 1.0)
add(s0 + B(2), pop(980), 0.22, -0.2, fx)
add(s0 + B(3.75), pop(1060), 0.22, 0.2, fx)
# action: click, typing, send, reply, the fader slide
add(s2 + B(1), click(), 0.5, 0, fx)
cmd = ',soundboard volume airhorn 40'
for c in range(len(cmd)):
    add(s2 + B(1) + 0.1 + c * (B(2.5) / len(cmd)) + rng.uniform(-0.006, 0.006), key(), 0.45, rng.uniform(-0.2, 0.2), fx)
add(s2 + B(4), pop(760), 0.28, 0, fx)
add(s2 + B(5), pop(1180), 0.3, 0, fx)
add(s2 + B(5) + 0.15, slide(0.6), 0.3, 0, fx)
# result: typing ",soundboard", send, the list, then the same horn at 40%
for c in range(len(',soundboard')):
    add(s3 + 0.15 + c * (B(1) / 11) + rng.uniform(-0.006, 0.006), key(), 0.45, rng.uniform(-0.2, 0.2), fx)
add(s3 + B(1.5), pop(760), 0.28, 0, fx)
add(s3 + B(2), pop(1180), 0.3, 0, fx)
blast(s3 + B(3.5), [(0, 1.4), (2, 0.6), (3, 0.6), (4, 2.0)], 0.4)
# fact: the log entry arriving
add(s4 + 0.3, pop(880), 0.3, 0, fx)

# duck the music under the full-volume blasts so the horn reads as loud
duck = np.ones(n_all)
for a, d in [(s0 + B(0.5), B(1.5)), (s0 + B(3), B(1.25))]:
    i, j = int(a * SR), int((a + d) * SR)
    duck[i:j] = 0.55
duck = lp(duck, 12)
music *= duck[:, None]

mix = music + fx
t = np.arange(n_all) / SR
mix *= (np.minimum(1, t / 0.02) * np.minimum(1, (DUR - t) / 0.3))[:, None]
mix /= max(1e-9, np.abs(mix).max()) / 0.7
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype("<i2").tobytes())
print(sys.argv[1], f"{DUR:.3f}s")
