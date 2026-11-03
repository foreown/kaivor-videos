#!/usr/bin/env python3
"""Original sound for the sticky messages video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

Timings follow the shot list in index.html (global seconds):
hook 0-2.6, turn 2.6-3.6, action 3.6-8.0, result 8.0-12.6, fact 12.6-15.0, end 15.0-18.2.
"""
import sys, wave
import numpy as np

SR = 48000
DUR = 18.2
rng = np.random.default_rng(7)
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


def tone(freq, dur, decay=0.15, harmonics=(1, 0.3, 0.1)):
    n = int(SR * dur)
    t = np.arange(n) / SR
    s = sum(a * np.sin(2 * np.pi * freq * (k + 1) * t) for k, a in enumerate(harmonics))
    return s * env(n, 0.003, decay)


def pop(pitch=880):        # a chat message arriving
    return tone(pitch, 0.18, 0.05, (1, 0.25)) + 0.4 * tone(pitch * 1.5, 0.18, 0.03, (1,))


def key():                 # a soft key press
    n = int(SR * 0.03)
    noise = rng.standard_normal(n)
    noise = np.convolve(noise, np.ones(6) / 6, "same")
    return noise * env(n, 0.0005, 0.006)


def thump():               # hard cut
    n = int(SR * 0.35)
    t = np.arange(n) / SR
    f = 90 * np.exp(-t * 9) + 45
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.002, 0.12)


def chime(base):           # sticky repost / end card
    return sum(tone(base * r, 1.2, 0.5, (1, 0.15)) * g for r, g in ((1, 1), (1.5, 0.5), (2, 0.3)))


# low pad bed under everything, quiet, slow swell
t = np.arange(len(mix)) / SR
pad = sum(np.sin(2 * np.pi * f * t + p) for f, p in ((110, 0), (164.8, 1), (220.5, 2), (277.2, 3)))
pad *= 0.035 * (0.6 + 0.4 * np.sin(2 * np.pi * t / 6.5)) * np.minimum(1, t / 0.4) * np.minimum(1, (DUR - t) / 1.5)
mix += np.stack([pad, np.roll(pad, 240)], axis=1)

# hook: the flood of messages, slightly varied pitch and pan
for k in range(11):
    add(0.15 + 0.2 * k, pop(rng.choice([740, 830, 880, 990])), 0.22, rng.uniform(-0.4, 0.4))

for cut in (2.6, 3.6, 8.0, 12.6, 15.0):
    add(cut, thump(), 0.6)

# action: cursor click, typing, send, Kaivor's two messages
add(3.6 + 0.95, key(), 0.9)
for c in range(28):
    add(3.6 + 1.1 + c * (1.15 / 28) + rng.uniform(-0.008, 0.008), key(), 0.55, rng.uniform(-0.2, 0.2))
add(3.6 + 2.45, pop(660), 0.3)
add(3.6 + 2.95, chime(523.25), 0.22)
add(3.6 + 3.25, pop(988), 0.3)

# result: chat continues, sticky reposts land with a chime
for at in (0.35, 0.7, 1.0, 3.0):
    add(8.0 + at, pop(rng.choice([740, 830, 880])), 0.22, rng.uniform(-0.3, 0.3))
for at in (2.35, 3.85):
    add(8.0 + at, chime(523.25), 0.22)

# end card
add(15.15, chime(392.0), 0.3)
add(15.6, chime(587.33), 0.18)

mix /= max(1e-9, np.abs(mix).max()) / 0.7
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype("<i2").tobytes())
print(sys.argv[1])
