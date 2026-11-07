#!/usr/bin/env python3
"""Original sound for the gradient role colours video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

A bright 124 BPM G-major house groove: detuned saw plucks on Gmaj7 / Em7 / Cmaj7 / D, a short bass, soft kick,
claps on 2 and 4, off-beat hats. Every shot in index.html is a whole number of beats, so the cuts land on the grid:
hook 0-6, turn 6-8, action 8-17, result 17-24, holographic 24-32, fact 32-37, end 37-44 (beats).
On top: message pops, key clicks, a colour sweep when the gradient lands, a bell shimmer for holographic.
"""
import sys, wave
import numpy as np

SR = 48000
FPS = 60
BPM = 124
BEAT = 60 / BPM
SHOT_BEATS = [6, 2, 9, 7, 8, 5, 7]
frames = [round(b * BEAT * FPS) for b in SHOT_BEATS]          # same rounding as tools/capture.mjs
starts = np.concatenate([[0], np.cumsum(frames)]) / FPS        # shot start times in seconds
DUR = starts[-1]
rng = np.random.default_rng(1107)
mix = np.zeros((int(SR * DUR) + 1, 2))
n_all = len(mix)


def B(k):
    return k * BEAT


def add(at, sig, gain=1.0, pan=0.0):
    i = int(round(at * SR))
    if i >= n_all or i < 0:
        return
    sig = sig[: n_all - i] * gain
    if sig.ndim == 1:
        mix[i:i + len(sig), 0] += sig * np.sqrt((1 - pan) / 2) * 1.414
        mix[i:i + len(sig), 1] += sig * np.sqrt((1 + pan) / 2) * 1.414
    else:
        mix[i:i + len(sig)] += sig


def env(n, attack=0.002, decay=0.1):
    t = np.arange(n) / SR
    return np.minimum(1, t / attack) * np.exp(-t / decay)


def lowpass(x, cutoff):
    """One-pole low-pass; cutoff may be an array (Hz per sample)."""
    cutoff = np.broadcast_to(cutoff, x.shape)
    a = np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = (1 - a[i]) * x[i] + a[i] * acc
        y[i] = acc
    return y


def saw(freq, n, harmonics=14):
    t = np.arange(n) / SR
    return sum(((-1) ** (k + 1)) * np.sin(2 * np.pi * freq * k * t) / k
               for k in range(1, harmonics + 1) if freq * k < SR / 2.2)


def midi(m):
    return 440 * 2 ** ((m - 69) / 12)


# ---------- instruments ----------
def pluck(m, dur=0.32, bright=1.0):
    n = int(SR * dur)
    f = midi(m)
    s = saw(f * 0.997, n, 10) + saw(f * 1.003, n, 10)
    t = np.arange(n) / SR
    cut = 600 + 3800 * bright * np.exp(-t / 0.07)
    return lowpass(s, cut) * env(n, 0.002, 0.16) * 0.5


def bass(m, dur=0.22):
    n = int(SR * dur)
    t = np.arange(n) / SR
    f = midi(m)
    s = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t) + 0.12 * np.sin(2 * np.pi * 3 * f * t)
    return s * np.minimum(1, t / 0.004) * np.exp(-t / 0.14)


def kick():
    n = int(SR * 0.32)
    t = np.arange(n) / SR
    f = 48 + 110 * np.exp(-t * 32)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.001, 0.11)


def clap():
    n = int(SR * 0.2)
    noise = rng.standard_normal(n)
    noise = noise - lowpass(noise, 900)
    e = np.zeros(n)
    for off in (0, 0.009, 0.019):
        i = int(off * SR)
        e[i:] += env(n - i, 0.0005, 0.012 if off < 0.019 else 0.07)
    return noise * e * 0.5


def hat(open_=False):
    n = int(SR * (0.16 if open_ else 0.05))
    noise = rng.standard_normal(n)
    noise = noise - lowpass(noise, 6000)
    return noise * env(n, 0.0005, 0.05 if open_ else 0.012)


def pop(pitch=900):            # a chat message arriving
    n = int(SR * 0.16)
    t = np.arange(n) / SR
    f = pitch * (1 + 0.5 * np.exp(-t * 60))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.001, 0.035)


def key():                     # a soft key press
    n = int(SR * 0.025)
    noise = np.convolve(rng.standard_normal(n), np.ones(5) / 5, "same")
    return noise * env(n, 0.0004, 0.005)


def click():                   # mouse press
    n = int(SR * 0.04)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 2400 * t) * 0.5 + rng.standard_normal(n) * 0.3) * env(n, 0.0003, 0.006)


def sweep(dur=0.6, f0=300, f1=3200):   # the colour washing over the names
    n = int(SR * dur)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    cut = f0 * (f1 / f0) ** (t / dur)
    band = lowpass(noise, cut) - lowpass(noise, cut * 0.5)
    shape = np.sin(np.pi * np.minimum(1, t / dur)) ** 2
    return band * shape * 2.2


def bell(m, dur=1.2):
    n = int(SR * dur)
    t = np.arange(n) / SR
    f = midi(m)
    s = np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t / 0.15) + 0.2 * np.sin(2 * np.pi * f * 5.4 * t) * np.exp(-t / 0.06)
    return s * env(n, 0.002, 0.45)


def riser(dur):
    n = int(SR * dur)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    cut = 400 * (9000 / 400) ** (t / dur)
    return lowpass(noise, cut) * (t / dur) ** 2


# ---------- the groove ----------
# one chord per bar (4 beats): Gmaj7, Em7, Cmaj7, D
CHORDS = [[55, 59, 62, 66], [52, 55, 59, 62], [48, 52, 55, 59], [50, 54, 57, 62]]
ROOTS = [43, 40, 36, 38]
TOTAL_BEATS = sum(SHOT_BEATS)
# pluck rhythm in sixteenths within a bar: a syncopated stab pattern
STABS = [0, 3, 6, 8, 11, 14]

for beat in range(TOTAL_BEATS):
    bar = beat // 4
    chord = CHORDS[bar % 4]
    root = ROOTS[bar % 4]
    at = B(beat)
    in_hook = beat < 6
    in_turn = 6 <= beat < 8
    in_fact = 32 <= beat < 37
    in_end = beat >= 37
    full = not (in_hook or in_turn or in_fact)

    # kick on every beat once the turn has dropped; on the end card only for its first bar
    if full and not (in_end and beat >= 41):
        add(at, kick(), 0.9)
    if in_hook and beat % 2 == 0:
        add(at, kick(), 0.45)
    # claps on 2 and 4
    if full and beat % 2 == 1:
        add(at, clap(), 0.45, 0.05)
    # off-beat hats
    if not in_turn and not (in_end and beat >= 42):
        add(at + BEAT / 2, hat(open_=full and beat % 4 == 3), 0.22 if full else 0.14, 0.3)
        if full:
            add(at + BEAT / 4, hat(), 0.08, -0.3)
            add(at + 3 * BEAT / 4, hat(), 0.08, -0.3)
    # plucks: on the beat's sixteenths that match the stab pattern
    if not in_turn and beat < 41:
        for s16 in STABS:
            if s16 // 4 == beat % 4:
                bright = 0.5 if in_hook else (0.35 if in_fact else 1.0)
                t16 = at + (s16 % 4) * BEAT / 4
                for i, m in enumerate(chord):
                    add(t16, pluck(m + 12, 0.3, bright), 0.11, (-0.35, -0.1, 0.1, 0.35)[i])
    # bass: root on the off-beat eighth, an octave hop on beat 4
    if full and beat < 41:
        add(at + BEAT / 2, bass(root), 0.42)
        if beat % 4 == 3:
            add(at + 3 * BEAT / 4, bass(root + 12, 0.12), 0.25)

# turn: a riser into the action drop
add(starts[1], riser(starts[2] - starts[1]), 0.22)

# fact: a held, filtered chord instead of stabs
for i, m in enumerate(CHORDS[0]):
    n = int(SR * (starts[6] - starts[5]))
    t = np.arange(n) / SR
    pad = lowpass(saw(midi(m), n, 8) + saw(midi(m) * 1.004, n, 8), 900) * np.minimum(1, t / 0.08) * np.exp(-t / 1.6)
    add(starts[5], pad, 0.07, (-0.4, -0.15, 0.15, 0.4)[i])

# end card: a last chord that rings out to the end
end_len = DUR - starts[6]
for i, m in enumerate([55, 59, 62, 66, 71]):
    n = int(SR * end_len)
    t = np.arange(n) / SR
    s = lowpass(saw(midi(m), n, 10) + saw(midi(m) * 1.003, n, 10), 1400 + 1800 * np.exp(-t / 0.4))
    s *= np.minimum(1, t / 0.005) * np.exp(-t / 2.2) * np.minimum(1, (end_len - t) / 0.6)
    add(starts[6], s, 0.07, (-0.4, -0.2, 0, 0.2, 0.4)[i])
add(starts[6], kick(), 0.8)
add(starts[6] + 0.18, bell(79), 0.12, 0.2)
add(starts[6] + 0.53, bell(86), 0.08, -0.2)

# ---------- sound effects, timed to index.html ----------
s0, s1, s2, s3, s4, s5, s6 = starts[:7]
# hook: messages arriving
for k, at in enumerate([0.15, B(1.5), B(2.5), B(3.5), B(4.5)]):
    add(s0 + at, pop([880, 990, 940, 1040, 1100][k]), 0.25, rng.uniform(-0.3, 0.3))
# turn: five paint chips landing
for i in range(5):
    add(s1 + 0.06 + i * 0.05 + 0.12, click(), 0.25, -0.4 + 0.2 * i)
# action: cursor click, typing, send, Kaivor's reply, the colour sweep
add(s2 + B(2), click(), 0.6)
cmd_len = len(',role color VIP #e8475f #f5a524')
for c in range(cmd_len):
    add(s2 + B(2) + 0.1 + c * (B(2.6) / cmd_len) + rng.uniform(-0.006, 0.006), key(), 0.5, rng.uniform(-0.2, 0.2))
add(s2 + B(5), pop(760), 0.3)
add(s2 + B(6), pop(1180), 0.32)
add(s2 + B(6) + 0.2, sweep(0.7), 0.35)
# result: member rows landing
for i in range(7):
    add(s3 + 0.2 + i * 0.06 + 0.05, click(), 0.14, -0.3 + 0.1 * i)
# holographic: typing, send, reply, a bell shimmer, two replies
holo_len = len(',role color VIP holographic')
for c in range(holo_len):
    add(s4 + 0.25 + c * (B(2.2) / holo_len) + rng.uniform(-0.006, 0.006), key(), 0.5, rng.uniform(-0.2, 0.2))
add(s4 + B(3), pop(760), 0.3)
add(s4 + B(4), pop(1180), 0.32)
for i, m in enumerate([79, 83, 86, 90, 91, 95]):
    add(s4 + B(4) + 0.22 + i * BEAT / 4, bell(m, 0.9), 0.07, (-0.5, 0.5, -0.25, 0.25, -0.1, 0.1)[i])
add(s4 + B(5.5), pop(940), 0.22, -0.2)
add(s4 + B(6.75), pop(1040), 0.22, 0.2)

# fade in over the first 20 ms, out over the last 0.3 s
t = np.arange(n_all) / SR
mix *= (np.minimum(1, t / 0.02) * np.minimum(1, (DUR - t) / 0.3))[:, None]
mix /= max(1e-9, np.abs(mix).max()) / 0.7
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype("<i2").tobytes())
print(sys.argv[1], f"{DUR:.3f}s")
