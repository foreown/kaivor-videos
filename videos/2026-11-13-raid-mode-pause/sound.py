#!/usr/bin/env python3
"""Original sound for the raid mode video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

A tense 138 BPM D-minor "night shift" track. Every shot in index.html is a whole number of beats, so cuts land on the
grid: hook 0-7, turn 7-10, action 10-21, result 21-32, fact 32-39, end 39-47 (beats).
The idea: the raid is heard as a door chime that climbs (D, F, A, C, D) with each of the five joins, over a low sub
drone and a quiet ticking 16th hat; the fifth join tips into a short two-tone alarm. The turn ("Kaivor isn't.") cuts
everything to one low hit and a reversed swell. The band comes in on the action shot: a pulsing 8th-note
sawtooth bass and dark stabs over Dm / Bb / F / C, four-on-the-floor kick, claps on 2 and 4. In the result the alert
lands with a falling siren blip, and the two shutters are metal roller rattles that end in a clang. The fact shot
lifts to D major / Bb with half-time drums while the shutters rattle back up, and the end card resolves on Dmaj9.
On top: key clicks while the command is typed, message pops and a two-note success chime.
"""
import sys, wave
import numpy as np
from scipy.signal import lfilter

SR = 48000
FPS = 60
BPM = 138
BEAT = 60 / BPM
SHOT_BEATS = [7, 3, 11, 11, 7, 8]
frames = [round(b * BEAT * FPS) for b in SHOT_BEATS]          # same rounding as tools/capture.mjs
starts = np.concatenate([[0], np.cumsum(frames)]) / FPS        # shot start times in seconds
DUR = starts[-1]
rng = np.random.default_rng(1113)
n_all = int(SR * DUR) + 1
buses = {k: np.zeros((n_all, 2)) for k in ('band', 'drums', 'fx')}


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


def saw(f, t):
    return 2 * ((f * t) % 1) - 1


def bass(m, dur, cut=700):
    t = tt(dur)
    f = hz(m)
    s = 0.5 * (saw(f, t) + saw(f * 1.004, t))
    return lp(lp(s, cut), cut) * env(t, 0.003, 0.12, dur) + 0.6 * np.sin(2 * np.pi * f * t) * env(t, 0.003, 0.18, dur)


def stab(notes, dur, cut=2200):
    t = tt(dur)
    s = sum(saw(hz(m), t) + saw(hz(m) * 1.006, t) for m in notes) / (2 * len(notes))
    return lp(s, cut) * env(t, 0.004, 0.11, dur)


def pad(notes, dur, cut=1600):
    t = tt(dur)
    s = sum(saw(hz(m), t) + saw(hz(m) * 1.005, t) + saw(hz(m) * 0.995, t) for m in notes) / (3 * len(notes))
    a = np.minimum(1, t / 0.25) * np.minimum(1, np.maximum(0, (dur - t) / 0.6))
    return lp(lp(s, cut), cut) * a


def drone(dur, m=26):
    t = tt(dur)
    f = hz(m)
    s = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)
    return s * np.minimum(1, t / 0.05) * np.minimum(1, np.maximum(0, (dur - t) / 0.08)) * (0.75 + 0.25 * np.sin(2 * np.pi * (BPM / 60) * 2 * t))


def kick():
    t = tt(0.3)
    f = 44 + 120 * np.exp(-t / 0.028)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.13)


def clap():
    t = tt(0.22)
    n = lp(hp(rng.standard_normal(len(t)), 900), 6000)
    e = np.zeros(len(t))
    for d in (0, 0.009, 0.018):
        e += (t >= d) * np.exp(-np.maximum(0, t - d) / (0.006 if d < 0.018 else 0.07))
    return n * e * 0.6


def hat(open_=False):
    d = 0.2 if open_ else 0.05
    t = tt(d)
    return hp(rng.standard_normal(len(t)), 8000) * np.minimum(1, t / 0.001) * np.exp(-t / (0.06 if open_ else 0.012))


def chime(m, dur=0.6):
    t = tt(dur)
    f = hz(m)
    return (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2.76 * f * t) * np.exp(-t / 0.05)) * env(t, 0.001, 0.2, dur)


def alarm(dur=0.5):
    """Two-tone beep: D6 then G#5, square-ish, short."""
    t = tt(dur)
    f = np.where(t < dur / 2, hz(86), hz(80))
    s = np.tanh(3 * np.sin(2 * np.pi * np.cumsum(f) / SR))
    return lp(s, 5000) * np.minimum(1, t / 0.004) * np.minimum(1, np.maximum(0, (dur - t) / 0.03)) * 0.6


def siren_drop(dur=0.55):
    t = tt(dur)
    f = hz(88) * np.exp(-t * 2.2)
    s = np.tanh(2 * np.sin(2 * np.pi * np.cumsum(f) / SR))
    return lp(s, 4500) * env(t, 0.003, 0.25, dur) * 0.7


def rattle(dur, up=False):
    """Roller shutter: metallic noise clicked at a slat rate that slows down (or speeds up going up)."""
    t = tt(dur)
    rate = 34 - 14 * (t / dur) if not up else 20 + 14 * (t / dur)
    phase = np.cumsum(rate) / SR
    clicks = np.exp(-((phase % 1) * 18))
    n = hp(rng.standard_normal(len(t)), 1500)
    ring = np.sin(2 * np.pi * 2300 * t) * 0.25 + np.sin(2 * np.pi * 3170 * t) * 0.15
    body = lp(hp(rng.standard_normal(len(t)), 120), 900) * 0.5
    a = np.minimum(1, t / 0.04) * np.minimum(1, np.maximum(0, (dur - t) / 0.05))
    return ((n * 0.5 + ring) * clicks + body) * a


def clang():
    t = tt(0.9)
    s = sum(np.sin(2 * np.pi * f * t) * np.exp(-t / d) for f, d in ((180, 0.25), (437, 0.18), (1210, 0.09), (2650, 0.05)))
    thud = np.sin(2 * np.pi * np.cumsum(70 + 80 * np.exp(-t / 0.01)) / SR) * np.exp(-t / 0.08)
    return s * 0.5 + thud


def low_hit():
    t = tt(1.2)
    s = np.sin(2 * np.pi * np.cumsum(38 + 60 * np.exp(-t / 0.04)) / SR) * np.exp(-t / 0.5)
    n = lp(rng.standard_normal(len(t)), 300) * np.exp(-t / 0.08)
    return s + n * 0.4


def swell(dur):
    t = tt(dur)
    n = lp(hp(rng.standard_normal(len(t)), 600), 6000)
    tone = sum(np.sin(2 * np.pi * hz(m) * t) for m in (62, 65, 69)) / 3
    return (n * 0.6 + tone * 0.4) * (t / dur) ** 3


def key():
    t = tt(0.035)
    return hp(rng.standard_normal(len(t)), 3000) * np.exp(-t / 0.005) + 0.25 * np.sin(2 * np.pi * 2500 * t) * np.exp(-t / 0.007)


def pop(f):
    t = tt(0.15)
    return np.sin(2 * np.pi * (f + 280 * np.exp(-t / 0.018)) * t) * np.exp(-t / 0.045)


def ding(notes, dur=0.9):
    t = tt(dur)
    s = sum(np.sin(2 * np.pi * hz(m) * t) + 0.2 * np.sin(2 * np.pi * hz(m) * 3 * t) * np.exp(-t / 0.08) for m in notes)
    return s / len(notes) * env(t, 0.002, 0.3, dur)


s0, s1, s2, s3, s4, s5 = [starts[i] for i in range(6)]
total = sum(SHOT_BEATS)

# ---- hook (beats 0-7): sub drone, ticking hats, five climbing door chimes, alarm on the fifth ----
add('band', 0, drone(s1), 0.32)
for k in range(28):
    add('drums', B(k * 0.25), hat(), 0.07 if k % 4 else 0.10, pan=0.2 if k % 2 else -0.2)
for i, (at, m) in enumerate(zip((0.6, 1.3, 2.0, 2.7, 3.4), (74, 77, 81, 84, 86))):
    add('fx', s0 + B(at), chime(m), 0.30, pan=-0.3 + 0.15 * i)
    add('fx', s0 + B(at), pop(500 + 60 * i), 0.18)
add('fx', s0 + B(3.6), alarm(), 0.22)
add('fx', s0 + B(4.1), alarm(), 0.18)
add('fx', s0 + B(4.6), pop(640), 0.28)
for k in (4, 5, 6):
    add('drums', B(k), kick(), 0.35)

# ---- turn (beats 7-10): one low hit, then a swell into the action ----
add('fx', s1, low_hit(), 0.65)
add('fx', s1 + B(0.5), swell(s2 - s1 - B(0.5)), 0.40)

# ---- the band (beats 10-47) ----
PROG_MIN = [(38, [62, 65, 69]), (34, [62, 65, 70]), (41, [60, 65, 69]), (36, [60, 64, 67])]   # Dm Bb F C
PROG_MAJ = [(38, [62, 66, 69]), (34, [62, 65, 70])]                                             # D  Bb
for k in range(10, total):
    t0 = B(k)
    in_fact = 32 <= k < 39
    in_end = k >= 39
    if in_fact:
        root, chord = PROG_MAJ[((k - 32) // 4) % 2]
    elif in_end:
        root, chord = 38, [62, 66, 69, 73, 76]
    else:
        root, chord = PROG_MIN[((k - 10) // 4) % 4]
    # bass: 8ths, octave on the off-beat
    if not in_end or k < 45:
        add('band', t0, bass(root, B(0.45), 600 if not in_fact else 900), 0.30)
        add('band', t0 + B(0.5), bass(root + 12, B(0.4), 600 if not in_fact else 900), 0.18)
    # stab on the "and" of 2 and 4 (in the end card only on the downbeats of the last bars)
    if not in_end and k % 2 == 1:
        add('band', t0 + B(0.5), stab([c + 12 for c in chord], B(0.4)), 0.14, pan=0.2)
    # drums
    if in_fact:
        if (k - 32) % 2 == 0:
            add('drums', t0, kick(), 0.45)
        if (k - 32) % 4 == 2:
            add('drums', t0, clap(), 0.35)
    elif not (in_end and k >= 45):
        add('drums', t0, kick(), 0.55)
        if (k - 10) % 2 == 1:
            add('drums', t0, clap(), 0.40)
    add('drums', t0 + B(0.5), hat(open_=True), 0.10 if not in_fact else 0.06, pan=0.25)
    add('drums', t0 + B(0.25), hat(), 0.05, pan=-0.2)
    add('drums', t0 + B(0.75), hat(), 0.05, pan=-0.2)
# pads: low under the action/result, lifting in the fact, resolving on the end card
add('band', s2, pad([50, 57, 62], s3 - s2), 0.10)
add('band', s3, pad([50, 57, 62], s4 - s3, 1200), 0.10)
add('band', s4, pad([50, 54, 57, 62], s5 - s4, 2000), 0.12)
add('band', s5, pad([50, 57, 61, 64, 66], DUR - s5 + 0.2, 2400), 0.16)

# ---- action: typing, send, reply ----
CMD = 48   # characters typed (see index.html)
# the page types with an ease-in-out (inout in index.html), so each character appears where inout(x) passes i/CMD
xs = np.linspace(0, 1, 4001)
ys = np.where(xs < 0.5, 8 * xs ** 4, 1 - (-2 * xs + 2) ** 4 / 2)
for i in range(CMD):
    x = xs[np.searchsorted(ys, i / CMD, side='right')] if i > 0 else 0.0
    add('fx', s2 + B(0.3) + B(4.0) * x + rng.uniform(-0.006, 0.006), key(), 0.16, pan=rng.uniform(-0.3, 0.3))
add('fx', s2 + B(4.6), pop(520), 0.34)
add('fx', s2 + B(5.4), pop(760), 0.30)
add('fx', s2 + B(5.4), ding([74, 81]), 0.26)

# ---- result: alert, two shutters down with clangs ----
add('fx', s3 + B(1), siren_drop(), 0.30)
add('fx', s3 + B(1), pop(700), 0.25)
add('fx', s3 + B(3), rattle(B(2)), 0.42, pan=-0.35)
add('fx', s3 + B(5), clang(), 0.42, pan=-0.35)
add('fx', s3 + B(4), rattle(B(2)), 0.42, pan=0.35)
add('fx', s3 + B(6), clang(), 0.42, pan=0.35)

# ---- fact: the "ended" alert, shutters back up ----
add('fx', s4 + B(0.8), ding([69, 74, 78]), 0.30)
add('fx', s4 + B(2), rattle(B(1.6), up=True), 0.36, pan=-0.35)
add('fx', s4 + B(2.5), rattle(B(1.6), up=True), 0.36, pan=0.35)

# ---- end card ----
add('fx', s5, low_hit(), 0.35)
add('fx', s5, ding([74, 78, 81, 86], 1.6), 0.28)

mix = buses['band'] * 1.0 + buses['drums'] * 0.9 + buses['fx'] * 1.0
fade = np.ones(n_all)
fade[: int(0.02 * SR)] = np.linspace(0, 1, int(0.02 * SR))
fade[-int(0.7 * SR):] *= np.linspace(1, 0, int(0.7 * SR))
mix *= fade[:, None]
mix /= max(1e-9, np.abs(mix).max()) / 0.8

out = sys.argv[1] if len(sys.argv) > 1 else 'sound.wav'
with wave.open(out, 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype('<i2').tobytes())
print(f'wrote {out}: {DUR:.3f} s, {BPM} BPM, shots at ' + ', '.join(f'{s:.2f}' for s in starts[:-1]))
