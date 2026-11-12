#!/usr/bin/env python3
"""Original sound for the contest reminder (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

A punchy 110 BPM F#-minor "ticket stub" beat: a detuned-saw synth stab through a low-pass that opens shot by shot,
F#m / D / A / E, a clap on 2 and 4, a tight kick, a muted bass, and a paper "ticket punch" (click + tear) on every
cut. Shots are whole beats: hook 0-6, prize 6-11, enter 11-22, wins 22-28, end 28-35. In the hook the beat is
filtered and the stabs sit on the downbeat only (a count-in feel for "Ends Sunday"), the prize shot opens the
filter, the three entry steps each get a rising pluck (1, 2, 3), "Most interaction wins." gets a three-hit stab,
and the end card resolves on F#m(add9) and rings out.
"""
import sys, wave
import numpy as np
from scipy.signal import lfilter

SR = 48000
FPS = 60
BPM = 110
BEAT = 60 / BPM
SHOT_BEATS = [6, 5, 11, 6, 7]
frames = [round(b * BEAT * FPS) for b in SHOT_BEATS]
starts = np.concatenate([[0], np.cumsum(frames)]) / FPS
DUR = starts[-1]
rng = np.random.default_rng(1113)
n_all = int(SR * DUR) + 1
mix = np.zeros((n_all, 2))


def B(k):
    return k * BEAT


def add(at, sig, gain=1.0, pan=0.0):
    i = int(round(at * SR))
    if i >= n_all or i < 0:
        return
    sig = sig[: n_all - i] * gain
    mix[i:i + len(sig), 0] += sig * np.sqrt((1 - pan) / 2) * 1.414
    mix[i:i + len(sig), 1] += sig * np.sqrt((1 + pan) / 2) * 1.414


def tt(d):
    return np.arange(int(d * SR)) / SR


def lp(x, c):
    a = np.exp(-2 * np.pi * c / SR)
    return lfilter([1 - a], [1, -a], x)


def hp(x, c):
    return x - lp(x, c)


def hz(m):
    return 440 * 2 ** ((m - 69) / 12)


def env(t, a, d, dur):
    return np.minimum(1, t / a) * np.exp(-t / d) * np.minimum(1, np.maximum(0, (dur - t) / 0.02))


def stab(notes, dur, cutoff):
    t = tt(dur)
    s = np.zeros(len(t))
    for m in notes:
        for det in (-0.006, 0.006):
            f = hz(m) * (1 + det)
            s += 2 * ((f * t + rng.uniform()) % 1) - 1
    return lp(lp(s / (2 * len(notes)), cutoff), cutoff * 1.5) * env(t, 0.004, 0.2, dur)


def pluck(m, dur=0.5):
    t = tt(dur)
    f = hz(m)
    return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * env(t, 0.002, 0.15, dur)


def kick():
    t = tt(0.3)
    return np.sin(2 * np.pi * np.cumsum(48 + 120 * np.exp(-t / 0.025)) / SR) * np.exp(-t / 0.12)


def clap():
    t = tt(0.2)
    n = lp(hp(rng.standard_normal(len(t)), 900), 5000)
    e = np.exp(-t / 0.05) + 0.6 * np.exp(-np.maximum(0, t - 0.011) / 0.004) * (t > 0.011) + 0.5 * np.exp(-np.maximum(0, t - 0.022) / 0.004) * (t > 0.022)
    return n * e


def bass(m, dur):
    t = tt(dur)
    return np.sin(2 * np.pi * hz(m) * t) * env(t, 0.004, 0.25, dur) + 0.3 * np.sin(4 * np.pi * hz(m) * t) * env(t, 0.004, 0.1, dur)


def punch():
    t = tt(0.3)
    click = np.sin(2 * np.pi * 2100 * t) * np.exp(-t / 0.004)
    tear = lp(hp(rng.standard_normal(len(t)), 1500), 6000) * (0.5 + 0.5 * np.sin(2 * np.pi * 60 * t)) * np.exp(-np.maximum(0, t - 0.02) / 0.06) * (t > 0.02)
    return click + tear * 0.6


PROG = [(42, [66, 69, 73]), (38, [66, 69, 74]), (45, [64, 69, 73]), (40, [64, 68, 71])]   # F#m D A E
total = sum(SHOT_BEATS)
shot_of = lambda k: int(np.searchsorted(np.cumsum(SHOT_BEATS), k, side='right'))
for k in range(total):
    t0 = B(k)
    sh = shot_of(k)
    b, chord = PROG[(k // 4) % 4]
    if sh == 4:
        b, chord = 42, [66, 68, 69, 73]          # F#m(add9)
    cutoff = [900, 2200, 2600, 3000, 2400][sh]
    if sh == 0:
        if k % 2 == 0:
            add(t0, stab(chord, B(0.9), cutoff), 0.35)
    elif sh == 4 and k >= 30:
        pass
    else:
        add(t0, stab(chord, B(0.4), cutoff), 0.30)
        add(t0 + B(0.5), stab(chord, B(0.3), cutoff), 0.22)
    if not (sh == 4 and k >= 31):
        add(t0, kick(), 0.6 if sh else 0.35)
        add(t0, bass(b, B(0.8)), 0.35)
    if k % 2 == 1 and sh != 0 and not (sh == 4 and k >= 31):
        add(t0, clap(), 0.35, pan=0.1)
    if sh in (1, 2, 3):
        add(t0 + B(0.5), lp(hp(rng.standard_normal(int(0.05 * SR)), 7000), 12000) * np.exp(-tt(0.05) / 0.012), 0.12, pan=-0.3)
# end chord rings out
add(B(30), stab([54, 66, 68, 69, 73], B(5), 2000), 0.38)
# punches on every cut and the first frame
for s in starts[:-1]:
    add(s, punch(), 0.5)
# entry steps 1, 2, 3
for i, at in enumerate((B(1), B(4), B(7))):
    add(starts[2] + at, pluck(73 + [0, 4, 7][i]), 0.35)
    add(starts[2] + at, pluck(85 + [0, 4, 7][i], 0.3), 0.12)
# "Most interaction wins." three hits
for i, at in enumerate((0.0, 0.13, 0.26)):
    add(starts[3] + at, stab([66, 69, 73, 78], 0.25, 3500), 0.3)

fade = np.ones(n_all)
fade[: int(0.02 * SR)] = np.linspace(0, 1, int(0.02 * SR))
fade[-int(0.5 * SR):] *= np.linspace(1, 0, int(0.5 * SR))
mix *= fade[:, None]
mix /= np.abs(mix).max() / 0.8
out = sys.argv[1] if len(sys.argv) > 1 else 'sound.wav'
with wave.open(out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype('<i2').tobytes())
print(f'wrote {out}: {DUR:.3f} s, {BPM} BPM')
