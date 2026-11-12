#!/usr/bin/env python3
"""Original sound for the repeating events video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

A bright 126 BPM E-major "wall planner" groove: a wall clock (woodblock tick-tock) that keeps time through the whole
video, a drawbar-organ chord stab on the off-beats over E / C#m7 / Amaj7 / B(sus), a plucked bass, a soft four-on-the-
floor kick and an open-hat "ts" on the and. Every shot in index.html is a whole number of beats, so cuts land on the
grid: hook 0-7, turn 7-9, action 9-21, result 21-31, fact 31-38, end 38-44 (beats).
The idea: the calendar is heard. Every Saturday that gets stamped on screen is a rubber-stamp thump plus one note of a
rising E-major bell (14, 21, 28, then 5 December), so the repeat is a little climbing figure. In the hook only the
clock and a slightly sour, falling music-box line play (game night was missed), with a pen scratch when last
Saturday is crossed out; the turn is a reversed swell into a stamp; the band comes in when the command is sent.
The result shot gets two notification dings (a low one for "Coming up", a brighter chord for "Starting now"),
the fact shot drops the kick for the three stamps, and the end card resolves on Emaj9 with the clock ticking out.
On top: key clicks while the commands are typed, message pops.
"""
import sys, wave
import numpy as np
from scipy.signal import lfilter

SR = 48000
FPS = 60
BPM = 126
BEAT = 60 / BPM
SHOT_BEATS = [7, 2, 12, 10, 7, 6]
frames = [round(b * BEAT * FPS) for b in SHOT_BEATS]          # same rounding as tools/capture.mjs
starts = np.concatenate([[0], np.cumsum(frames)]) / FPS        # shot start times in seconds
DUR = starts[-1]
rng = np.random.default_rng(1112)
n_all = int(SR * DUR) + 1
buses = {k: np.zeros((n_all, 2)) for k in ('band', 'drums', 'clock', 'fx')}


def B(k):
    return k * BEAT


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


def lp(x, cutoff):
    a = np.exp(-2 * np.pi * cutoff / SR)
    return lfilter([1 - a], [1, -a], x)


def hp(x, cutoff):
    return x - lp(x, cutoff)


def hz(m):
    return 440 * 2 ** ((m - 69) / 12)


def env(t, a, d, dur):
    return np.minimum(1, t / a) * np.exp(-t / d) * np.minimum(1, np.maximum(0, (dur - t) / 0.02))


def woodblock(f, v=1.0):
    t = tt(0.12)
    return (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * f * 2.7 * t)) * env(t, 0.0008, 0.018, 0.12) * v


def organ(notes, dur, bright=1.0):
    t = tt(dur)
    s = np.zeros(len(t))
    for m in notes:
        f = hz(m)
        s += np.sin(2 * np.pi * f * t) + 0.5 * np.sin(4 * np.pi * f * t) * bright + 0.25 * np.sin(6 * np.pi * f * t) * bright
    click = hp(rng.standard_normal(len(t)), 3000) * np.exp(-t / 0.003) * 0.3
    return (s / len(notes) + click) * env(t, 0.004, 0.16, dur)


def pluck_bass(m, dur):
    t = tt(dur)
    f = hz(m)
    saw = 2 * ((f * t) % 1) - 1
    return lp(saw, 300 + 900 * 1) * env(t, 0.003, 0.22, dur) * 0.9 + np.sin(2 * np.pi * f * t) * env(t, 0.003, 0.3, dur)


def kick():
    t = tt(0.32)
    f = 46 + 110 * np.exp(-t / 0.03)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.14)


def hat(open_=False):
    d = 0.22 if open_ else 0.06
    t = tt(d)
    return hp(rng.standard_normal(len(t)), 7000) * np.minimum(1, t / 0.002) * np.exp(-t / (0.07 if open_ else 0.015))


def music_box(m, dur=0.9, detune=0.0):
    t = tt(dur)
    f = hz(m) * (1 + detune)
    return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 4.1 * t) * np.exp(-t / 0.06)) * env(t, 0.001, 0.35, dur)


def bell(m, dur=1.4):
    t = tt(dur)
    f = hz(m)
    return (np.sin(2 * np.pi * f * t) + 0.45 * np.sin(2 * np.pi * 2.0 * f * t) * np.exp(-t / 0.3)
            + 0.3 * np.sin(2 * np.pi * 3.01 * f * t) * np.exp(-t / 0.12)) * env(t, 0.002, 0.5, dur)


def stamp():
    """Rubber stamp: a dull low thump plus a short papery slap."""
    t = tt(0.25)
    thump = np.sin(2 * np.pi * np.cumsum(90 + 60 * np.exp(-t / 0.01)) / SR) * np.exp(-t / 0.05)
    slap = lp(hp(rng.standard_normal(len(t)), 600), 4000) * np.exp(-t / 0.012) * 0.7
    return thump + slap


def scratch(dur=0.24):
    t = tt(dur)
    n = hp(rng.standard_normal(len(t)), 2500)
    am = 0.6 + 0.4 * np.sin(2 * np.pi * 38 * t)
    return lp(n, 7000) * am * np.sin(np.pi * t / dur) * 0.5


def key():
    t = tt(0.035)
    return hp(rng.standard_normal(len(t)), 3000) * np.exp(-t / 0.005) + 0.25 * np.sin(2 * np.pi * 2600 * t) * np.exp(-t / 0.007)


def pop(f):
    t = tt(0.15)
    return np.sin(2 * np.pi * (f + 280 * np.exp(-t / 0.018)) * t) * np.exp(-t / 0.045)


def ding(notes, dur=0.9):
    t = tt(dur)
    s = sum(np.sin(2 * np.pi * hz(m) * t) + 0.2 * np.sin(2 * np.pi * hz(m) * 3 * t) * np.exp(-t / 0.08) for m in notes)
    return s / len(notes) * env(t, 0.002, 0.3, dur)


def swell(dur):
    t = tt(dur)
    n = lp(hp(rng.standard_normal(len(t)), 800), 5000)
    tone = sum(np.sin(2 * np.pi * hz(m) * t) for m in (64, 68, 71)) / 3
    return (n * 0.6 + tone * 0.5) * (t / dur) ** 3


g = [starts[i] for i in range(6)]   # shot starts
s0, s1, s2, s3, s4, s5 = g
total = sum(SHOT_BEATS)
beat_at = lambda k: B(k)            # global beat -> seconds (shots are whole beats, frame rounding < 8 ms)

# ---- the wall clock: tick-tock on every beat, all the way through ----
for k in range(total):
    add('clock', beat_at(k), woodblock(1850 if k % 2 == 0 else 1250), 0.32, pan=-0.35 if k % 2 == 0 else 0.35)

# ---- hook: falling, slightly sour music box (beats 0-7) ----
for k, m in enumerate([76, 75, 73, 71, 68, 66, 64]):
    add('band', beat_at(k) + 0.0, music_box(m, 1.0, detune=-0.012 if k % 2 else 0.0), 0.32)
for k in (0, 4):
    add('band', beat_at(k), organ([52, 59, 64], B(3.8), bright=0.2), 0.09)
add('fx', s0 + B(0.5), pop(700), 0.30)
add('fx', s0 + B(2), pop(620), 0.30)
add('fx', s0 + B(3.5), pop(560), 0.30)
add('fx', s0 + B(4.5), scratch(), 0.75)

# ---- turn: reversed swell into a stamp on the downbeat of the action ----
add('fx', s1, swell(s2 - s1), 0.45)
add('fx', s2, stamp(), 0.55)

# ---- the band from the action shot to the end (beats 9-44) ----
PROG = [  # (bass, organ chord) per bar of 4 beats
    (40, [64, 68, 71, 76]),     # E
    (37, [61, 64, 68, 71]),     # C#m7
    (45, [61, 64, 68, 69]),     # Amaj7
    (47, [63, 66, 71, 76]),     # B(sus feel)
]
for k in range(9, total):
    t0 = beat_at(k)
    bar = (k - 9) // 4
    bass, chord = PROG[bar % 4]
    in_fact = 31 <= k < 38
    in_end = k >= 38
    if in_end:
        bass, chord = 40, [63, 66, 68, 71, 78]   # Emaj9 on the end card
    # organ stab on the "and"
    add('band', t0 + B(0.5), organ(chord, B(0.45)), 0.16 if not in_fact else 0.11, pan=0.15)
    # bass: root on the beat, octave on the and of 2 and 4
    add('band', t0, pluck_bass(bass, B(0.45)), 0.30)
    if k % 2 == 1:
        add('band', t0 + B(0.5), pluck_bass(bass + 12, B(0.3)), 0.16)
    # drums
    if not in_fact and not (in_end and k >= 42):
        add('drums', t0, kick(), 0.55)
    add('drums', t0 + B(0.5), hat(open_=True), 0.10 if not in_fact else 0.06, pan=0.25)
    add('drums', t0 + B(0.25), hat(), 0.05, pan=-0.2)
    add('drums', t0 + B(0.75), hat(), 0.05, pan=-0.2)
# final chord held on the end card
add('band', s5 + B(4), organ([52, 63, 66, 68, 71, 78], B(2), bright=0.6), 0.16)

# ---- action: typing, sends, replies, and the four Saturday tiles ----
CMD1, CMD2 = 60, 22   # characters typed (see index.html)
for i in range(CMD1):
    add('fx', s2 + B(0.25) + B(2.6) * i / CMD1 + rng.uniform(-0.01, 0.01), key(), 0.16, pan=rng.uniform(-0.3, 0.3))
for i in range(CMD2):
    add('fx', s2 + B(5.5) + B(1.3) * i / CMD2 + rng.uniform(-0.01, 0.01), key(), 0.16, pan=rng.uniform(-0.3, 0.3))
for at, f in ((B(3.25), 520), (B(4), 760), (B(7.25), 520), (B(8), 760)):
    add('fx', s2 + at, pop(f), 0.34)
for i, (at, m) in enumerate(((4.5, 76), (5, 80), (5.5, 83), (6, 88))):
    add('fx', s2 + B(at), stamp(), 0.5)
    add('fx', s2 + B(at), bell(m), 0.20, pan=-0.3 + 0.2 * i)

# ---- result: Coming up, the DM, rin, Starting now ----
add('fx', s3 + B(1), ding([71, 76]), 0.32)
add('fx', s3 + B(2), pop(820), 0.30)
add('fx', s3 + B(3.5), pop(600), 0.26)
add('fx', s3 + B(6), ding([76, 80, 83, 88], 1.2), 0.36)

# ---- fact: three stamps, rising bell ----
for i, (at, m) in enumerate(((1.5, 76), (2.5, 80), (3.5, 83))):
    add('fx', s4 + B(at), stamp(), 0.55)
    add('fx', s4 + B(at), bell(m), 0.24, pan=-0.2 + 0.2 * i)

# ---- end card: one last stamp with the logo ----
add('fx', s5, stamp(), 0.45)
add('fx', s5, bell(88, 2.0), 0.18)

# mix
mix = buses['band'] * 1.0 + buses['drums'] * 0.9 + buses['clock'] * 0.8 + buses['fx'] * 1.0
# fade in over 30 ms, out over the last 0.6 s
fade = np.ones(n_all)
fade[: int(0.03 * SR)] = np.linspace(0, 1, int(0.03 * SR))
fade[-int(0.6 * SR):] *= np.linspace(1, 0, int(0.6 * SR))
mix *= fade[:, None]
mix /= max(1e-9, np.abs(mix).max()) / 0.8

out = sys.argv[1] if len(sys.argv) > 1 else 'sound.wav'
with wave.open(out, 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype('<i2').tobytes())
print(f'wrote {out}: {DUR:.3f} s, {BPM} BPM, shots at ' + ', '.join(f'{s:.2f}' for s in starts[:-1]))
