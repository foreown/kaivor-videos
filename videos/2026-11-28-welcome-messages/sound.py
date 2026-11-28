#!/usr/bin/env python3
"""Original sound for the welcome-messages video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

A 109.09 BPM (33 video frames a beat) front-porch strum in F# major (F#maj7 - D#m7 - Bmaj7 - C#7, one bar each):
a strummed four-string ukulele (Karplus-Strong strings, each strum spread across the strings, down on the beat and
up on the off-beat), finger snaps on 2 and 4, a shaker, a round plucked bass, and a whistled melody for the end
card. Every shot in index.html is a whole number of beats, so cuts land on the grid: hook 0-7, turn 7-11,
command 11-20, result 20-28, fact 28-33, end 33-39 (beats).
The idea: a two-tone doorbell is someone joining. In the hook the strum plays in D# minor behind a closed door
(low-passed), and each of the three joins rings a doorbell that nobody answers: muffled, a little further away
each time, with a soft knock after the last. The turn rings the doorbell clean, then a latch click and a door swish
as the door opens, and the strum steps into F# major. The command has a key tick per typed character, a send blip
and a two-note chime for Kaivor's reply. In the result each join rings the doorbell and each of Kaivor's welcomes
answers with a bright uke flourish, juniper's "oh hi! thanks" a little blip. The fact strips back to bass, snaps
and shaker with a pop per row. The end card plays the whole groove with the whistle and rings out on a rolled
F#maj9 and a last doorbell to the final frame.
"""
import sys, wave
import numpy as np
from scipy.signal import lfilter, butter

SR = 48000
FPS = 60
BEAT_FRAMES = 33
BEAT = BEAT_FRAMES / FPS                                      # 0.55 s
SHOT_BEATS = [7, 4, 9, 8, 5, 6]
frames = [round(b * BEAT_FRAMES) for b in SHOT_BEATS]         # same rounding as tools/capture.mjs
starts_b = np.concatenate([[0], np.cumsum(SHOT_BEATS)])
s_hook, s_turn, s_cmd, s_res, s_fact, s_end, s_total = starts_b
DUR = sum(frames) / FPS
rng = np.random.default_rng(1128)
n_all = int(SR * DUR) + 1
buses = {k: np.zeros((n_all, 2)) for k in ('muffled', 'distant', 'music', 'drums', 'fx')}


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
def ks(m, dur=0.6, bright=0.6, decay=0.993):
    """Karplus-Strong string: a filtered noise burst through a damped comb (lfilter, so it is fast)."""
    n = int(dur * SR)
    N = max(2, int(round(SR / hz(m))))
    burst = rng.uniform(-1, 1, N)
    burst = filt(burst, hi=1500 + 6000 * bright)
    x = np.zeros(n)
    x[:N] = burst
    a = np.zeros(N + 2)
    a[0] = 1.0
    a[N] = -decay / 2
    a[N + 1] = -decay / 2
    y = lfilter([1.0], a, x)
    return y * env(n, 0.001, dur * 0.45, hold=dur * 0.2)


def strum(chord, up=False, dur=0.5, spread=0.012, bright=0.6):
    strings = list(reversed(chord)) if up else list(chord)
    out = np.zeros(int((dur + spread * 4) * SR))
    for k, m in enumerate(strings):
        s = ks(m, dur, bright=bright)
        i = int(k * spread * SR)
        out[i:i + len(s)] += s * (0.8 if up else 1.0)
    return out / len(strings)


def snap():
    t = tt(0.09)
    n = filt(rng.standard_normal(len(t)), 1800, 6000)
    return (n + 0.4 * np.sin(2 * np.pi * 2200 * t)) * np.exp(-t / 0.011)


def shaker(accent=False):
    t = tt(0.07)
    n = filt(rng.standard_normal(len(t)), lo=6000)
    e = np.minimum(1, t / 0.012) * np.exp(-t / 0.02)
    return n * e * (1.0 if accent else 0.6)


def bass(m, dur):
    t = tt(dur)
    f = hz(m)
    s = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t)
    return s * env(len(t), 0.004, dur * 0.4, hold=dur * 0.25)


def chime(m, dur=2.4):
    """One bar of a two-tone doorbell: a soft mallet on a metal bar, a few partials and a long ring."""
    t = tt(dur)
    f = hz(m)
    parts = [(1.0, 1.0, 1.3), (2.0, 0.18, 0.7), (3.01, 0.22, 0.45), (4.18, 0.08, 0.25), (1.002, 0.4, 1.5)]
    s = sum(a * np.sin(2 * np.pi * f * k * t) * np.exp(-t / d) for k, a, d in parts)
    s += 0.12 * filt(rng.standard_normal(len(t)), 1500, 5000) * np.exp(-t / 0.004)
    return s * np.minimum(1, t / 0.002)


def doorbell(dur=2.6):
    """Ding (A#5), dong (F#5) three-quarters of a beat later."""
    gap = 0.75 * BEAT
    out = np.zeros(int((gap + dur) * SR))
    a = chime(82, dur)
    b = chime(78, dur)
    out[:len(a)] += a
    i = int(gap * SR)
    out[i:i + len(b)] += b
    return out


def knock():
    t = tt(0.14)
    f = 110 + 60 * np.exp(-t / 0.02)
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.5 * filt(rng.standard_normal(len(t)), 300, 1500)) * np.exp(-t / 0.035)


def latch():
    t = tt(0.08)
    return (filt(rng.standard_normal(len(t)), 2000, 7000) + 0.6 * np.sin(2 * np.pi * 3100 * t)) * np.exp(-t / 0.008)


def swish(dur=0.45):
    t = tt(dur)
    n = filt(rng.standard_normal(len(t)), 300, 2500)
    return n * np.sin(np.pi * t / dur) ** 2


def blip(m, dur=0.13):
    t = tt(dur)
    f = hz(m) * (1 + 0.03 * np.exp(-t / 0.01))
    return (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)) * env(len(t), 0.002, 0.04)


def key_tick():
    t = tt(0.03)
    return filt(rng.standard_normal(len(t)), 2500, 7000) * np.exp(-t / 0.004)


def whistle(m, dur):
    t = tt(dur)
    vib = 1 + 0.006 * np.sin(2 * np.pi * 5.5 * t) * np.minimum(1, t / 0.15)
    ph = 2 * np.pi * np.cumsum(hz(m) * vib) / SR
    breath = 0.05 * filt(rng.standard_normal(len(t)), 1500, 4000)
    return (np.sin(ph) + 0.04 * np.sin(2 * ph) + breath) * env(len(t), 0.03, 0.12, hold=dur * 0.6)


def flourish(chord):
    """A quick upward arpeggio on the uke, then the top note again: Kaivor saying hi."""
    out = np.zeros(int(1.2 * SR))
    for k, m in enumerate(chord + [chord[-1] + 5]):
        s = ks(m + 12, 0.8, bright=0.9)
        i = int(k * 0.055 * SR)
        out[i:i + len(s)] += s
    return out / 3


# ---------------- the groove ----------------
# uke voicings and bass roots, one chord a bar
MAJOR = [([65, 66, 70, 73], 42), ([63, 66, 70, 73], 39), ([63, 66, 70, 71], 35), ([61, 65, 68, 71], 37)]  # F#maj7 D#m7 Bmaj7 C#7
MINOR = [([63, 66, 70, 73], 39), ([63, 66, 70, 71], 35), ([61, 66, 70, 73], 42), ([61, 65, 68, 73], 37)]  # D#m7 Bmaj7 F# C#
# strum pattern per beat (down on the beat, up on the "and"): which halves get a strum
PATTERN = [(True, False), (True, True), (False, True), (True, True)]


def groove(b0, b1, bus='music', chords=MAJOR, uke=True, perc=True, low=True, level=1.0, bar0=None):
    dbus = 'drums' if bus == 'music' else bus
    bar0 = b0 if bar0 is None else bar0
    for b in range(int(b0), int(b1)):
        rel = b - int(bar0)
        chord, root = chords[(rel // 4) % 4]
        bib = rel % 4
        if uke:
            down, up = PATTERN[bib]
            if down:
                add(bus, T(b), strum(chord, bright=0.65), 0.34 * level, pan=-0.12)
            if up:
                add(bus, T(b + 0.5), strum(chord, up=True, dur=0.35, bright=0.55), 0.24 * level, pan=-0.12)
        if perc:
            if bib in (1, 3):
                add(dbus, T(b), snap(), 0.42 * level, pan=0.2)
            for k in range(4):
                add(dbus, T(b + k / 4), shaker(accent=(k == 2)), 0.14 * level, pan=0.35)
        if low:
            if bib == 0:
                add(bus, T(b), bass(root, BEAT * 1.4), 0.5 * level)
            if bib == 2:
                add(bus, T(b), bass(root + 7, BEAT * 0.8), 0.36 * level)
            if bib == 3:
                add(bus, T(b + 0.5), bass(root + 12, BEAT * 0.45), 0.28 * level)


# ---------------- the edit ----------------
# hook: the strum behind a closed door in D# minor; three doorbells nobody answers, further away each time
groove(s_hook, s_turn, bus='muffled', chords=MINOR, level=0.95)
H_J, H_M = [0.5, 2, 3.5], [1.25, 2.75, 4.25]
for i, b in enumerate(H_J):
    add('distant', T(s_hook + b), doorbell(), [0.5, 0.4, 0.32][i], pan=[-0.2, 0.15, -0.05][i])
for i, b in enumerate(H_M):
    add('fx', T(s_hook + b), blip(73 + 2 * i), 0.16, pan=[0.25, -0.2, 0.1][i])
add('fx', T(s_hook + 5.25), knock(), 0.3)
add('fx', T(s_hook + 5.25) + 0.16, knock(), 0.24)

# turn: the doorbell, clean; latch and swish as the door opens; the strum steps into F# major
add('fx', T(s_turn + 0.25), doorbell(2.0), 0.3)
add('fx', T(s_turn + 1), latch(), 0.35)
add('fx', T(s_turn + 1) + 0.04, swish(), 0.25, pan=-0.2)
add('music', T(s_turn + 2), strum(MAJOR[0][0], dur=1.0, spread=0.03, bright=0.8), 0.4, pan=-0.12)
add('music', T(s_turn + 2), bass(42, BEAT * 2), 0.45)

# command: groove up, a key tick per character, send blip, a two-note chime for Kaivor's reply
groove(s_cmd, s_res, level=0.85)
CMD = ",welcome add #general Welcome {user.mention} to {guild.name}! You're member #{guild.member_count}."
c_type, c_send = T(0.25), T(5)
d = c_send - c_type - 0.12


def inout(x):
    x = min(1, max(0, x))
    return 8 * x ** 4 if x < 0.5 else 1 - (-2 * x + 2) ** 4 / 2


last = 0
for f in range(int(c_type * FPS), int(c_send * FPS)):
    tl = f / FPS
    n = int(np.ceil(len(CMD) * inout((tl - c_type) / d)))
    if n > last:
        add('fx', T(s_cmd) + tl, key_tick(), 0.18, pan=float(rng.uniform(-0.25, 0.25)))
        last = n
add('fx', T(s_cmd + 5), blip(85, 0.09), 0.2)
add('fx', T(s_cmd + 5.75), chime(85, 1.4), 0.22)
add('fx', T(s_cmd + 6.0), chime(90, 1.4), 0.2)

# result: a doorbell per join, a uke flourish per welcome, a blip for juniper's reply
groove(s_res, s_fact, level=1.0)
add('fx', T(s_res + 0.5), doorbell(1.8), 0.32, pan=-0.1)
add('fx', T(s_res + 1.25), flourish([66, 70, 73]), 0.5, pan=0.1)
add('fx', T(s_res + 2.5), blip(78), 0.18, pan=0.2)
add('fx', T(s_res + 4), doorbell(1.8), 0.3, pan=0.1)
add('fx', T(s_res + 4.75), flourish([68, 73, 77]), 0.5, pan=-0.1)

# fact: stripped back, a pop per row
groove(s_fact, s_end, uke=False, level=0.85, bar0=s_res)
for i, b in enumerate([1.75, 2.5]):
    add('fx', T(s_fact + b), blip(78 + 4 * i, 0.1), 0.24)
add('music', T(s_fact + 3), strum(MAJOR[2][0], dur=0.8, bright=0.5), 0.3)

# end: the whole groove with a whistle, then a rolled F#maj9 and a last doorbell to the final frame
groove(s_end, s_total - 2, level=1.0)
for m, b, dd in [(78, 0, 0.5), (80, 0.5, 0.5), (82, 1, 1.0), (85, 2, 0.5), (82, 2.5, 1.2)]:
    add('music', T(s_end + b), whistle(m, BEAT * dd + 0.1), 0.2, pan=0.15)
for i, m in enumerate([54, 61, 65, 68, 70, 73]):
    add('music', T(s_total - 2) + i * 0.045, ks(m + 12, 1.8, bright=0.7, decay=0.997), 0.32, pan=-0.1)
add('music', T(s_total - 2), bass(42, T(2)), 0.5)
add('fx', T(s_total - 2) + 0.1, doorbell(1.4), 0.24)

# ---------------- mix ----------------
mix = buses['music'] + buses['drums'] * 2.4 + buses['fx']
mix += filt(buses['muffled'], hi=700, order=4) * 0.6
mix += filt(buses['distant'], hi=1800, order=2) * 0.8
# a small porch: a short room on the fx bus
d_n = int(0.07 * SR)
room = np.zeros_like(buses['fx'])
room[d_n:] = buses['fx'][:-d_n] * 0.2
mix += filt(room, hi=5000)
# fade the last 0.2 s just enough to avoid a click
fade = int(0.2 * SR)
mix[-fade:] *= np.linspace(1, 0.0, fade)[:, None] ** 0.5
mix /= max(1e-9, np.abs(mix).max()) / 0.8
out = (mix * 32767).astype(np.int16)
with wave.open(sys.argv[1], 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(out.tobytes())
print(f'{sys.argv[1]}: {DUR:.3f} s, {len(out)} samples')
