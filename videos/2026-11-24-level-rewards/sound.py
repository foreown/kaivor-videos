#!/usr/bin/env python3
"""Original sound for the level rewards video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

A 105.88 BPM (34 video frames a beat) amapiano-style groove in B major (B - G#m - E - F#): a pitched log-drum bass
with a wooden knock, soft four-on-the-floor kick, sixteenth-note shaker, off-beat rims and a clap on 2 and 4, airy
bell-piano chords and a reedy melodica lead with vibrato. Every shot in index.html is a whole number of beats, so cuts
land on the grid: hook 0-5, problem 5-8, turn 8-12, commands 12-19, result 19-27, fact 27-32, end 32-37 (beats).
The idea: the hook is a sparse G#-minor melodica line over a low pad, with a paper flick for each profile card and a
soft two-note sigh for "One of them joined today." "Nothing shows who stuck around" lands on a felt thud and a hollow
held chord. On the turn, each of the ten pencil marks is a pencil scratch with a climbing B-major pentatonic note; the
tenth rings as a bright chord and a log-drum fill opens the groove. Typing is soft key ticks, each send a blip, each of
Kaivor's replies a two-note bell. In the result teo's message pops, the level-up is a rising four-note run into a bell,
and the role appearing is a sparkle. The fact drops to log drum, shaker and pad; the end card plays one more bar and
rings out on a rolled Bmaj9 to the last frame.
"""
import sys, wave
import numpy as np
from scipy.signal import lfilter, butter

SR = 48000
FPS = 60
BEAT_FRAMES = 34
BEAT = BEAT_FRAMES / FPS                                      # 0.5667 s
SHOT_BEATS = [5, 3, 4, 7, 8, 5, 5]
frames = [round(b * BEAT_FRAMES) for b in SHOT_BEATS]         # same rounding as tools/capture.mjs
starts_b = np.concatenate([[0], np.cumsum(SHOT_BEATS)])
s_hook, s_prob, s_turn, s_cmd, s_res, s_fact, s_end, s_total = starts_b
DUR = sum(frames) / FPS
rng = np.random.default_rng(1124)
n_all = int(SR * DUR) + 1
buses = {k: np.zeros((n_all, 2)) for k in ('music', 'drums', 'fx', 'hollow')}


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
def log_drum(m, dur=0.42, gain=1.0):
    t = tt(dur)
    f = hz(m) * (1 + 0.9 * np.exp(-t / 0.012))             # quick pitch drop: the "log" thwack
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph)
    body = np.tanh(1.8 * body) * np.exp(-t / 0.22)
    knock = filt(rng.standard_normal(len(t)), 700, 2400) * np.exp(-t / 0.006) * 0.5
    return (body * 0.8 + knock) * gain


def kick(dur=0.3):
    t = tt(dur)
    f = 48 + 90 * np.exp(-t / 0.03)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.12)


def shaker(dur=0.07, accent=1.0):
    t = tt(dur)
    n = filt(rng.standard_normal(len(t)), 6000, None)
    return n * np.minimum(1, t / 0.006) * np.exp(-t / 0.022) * 0.35 * accent


def rim():
    t = tt(0.06)
    return (np.sin(2 * np.pi * 1650 * t) * 0.6 + filt(rng.standard_normal(len(t)), 2000, 6000) * 0.5) * np.exp(-t / 0.012)


def clap():
    out = np.zeros(int(0.25 * SR))
    for k, d in enumerate([0, 0.011, 0.022]):
        t = tt(0.2 if k == 2 else 0.02)
        burst = filt(rng.standard_normal(len(t)), 900, 3200) * np.exp(-t / (0.06 if k == 2 else 0.006))
        i = int(d * SR)
        out[i:i + len(burst)] += burst
    return out * 0.6


def bell_piano(m, dur=1.0):
    t = tt(dur)
    f = hz(m)
    s = np.zeros_like(t)
    for k, (amp, dec) in enumerate([(1, 0.9), (0.45, 0.45), (0.22, 0.25), (0.12, 0.15)], 1):
        s += amp * np.sin(2 * np.pi * f * k * t * (1 + 0.0007 * k)) * np.exp(-t / dec)
    return s * np.minimum(1, t / 0.004)


def chord(ms, dur, gain=0.18, roll=0.0, pan=0.0):
    out = np.zeros(int((dur + roll * len(ms)) * SR) + 1)
    for k, m in enumerate(ms):
        s = bell_piano(m, dur)
        i = int(k * roll * SR)
        out[i:i + len(s)] += s
    return out * gain


def melodica(m, dur, gain=0.12):
    t = tt(dur)
    f = hz(m) * (1 + 0.006 * np.sin(2 * np.pi * 5.2 * t) * np.minimum(1, t / 0.25))
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = sum((1 / k) * np.sin(k * ph) for k in range(1, 9)) * 0.6 + 0.2 * np.sin(ph)
    s = filt(s, None, 2600)
    return s * env(len(t), 0.035, 0.12, max(0, dur - 0.2)) * gain


def pad(ms, dur, gain=0.05):
    t = tt(dur)
    s = np.zeros_like(t)
    for m in ms:
        for d in (-0.08, 0.08):
            f = hz(m) * 2 ** (d / 12)
            s += np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)
    s = filt(s, None, 1400)
    a = np.minimum(1, t / 0.4) * np.minimum(1, (dur - t) / 0.4)
    return s * np.clip(a, 0, 1) * gain


def blip(m, dur=0.09, gain=0.2):
    t = tt(dur)
    return np.sin(2 * np.pi * hz(m) * t) * np.exp(-t / 0.03) * gain


def tick():
    t = tt(0.03)
    return filt(rng.standard_normal(len(t)), 2500, 7000) * np.exp(-t / 0.005) * 0.25


def flick():
    t = tt(0.16)
    n = filt(rng.standard_normal(len(t)), 1200, 6000)
    return n * np.sin(np.pi * np.minimum(1, t / 0.16)) ** 2 * 0.3


def pencil(dur=0.11):
    t = tt(dur)
    n = filt(rng.standard_normal(len(t)), 2500, 9000)
    grain = 0.6 + 0.4 * np.sign(np.sin(2 * np.pi * 70 * t))
    return n * grain * np.sin(np.pi * t / dur) * 0.35


def thud():
    t = tt(0.5)
    f = 42 + 50 * np.exp(-t / 0.04)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.18)
    return s + filt(rng.standard_normal(len(t)), None, 400) * np.exp(-t / 0.03) * 0.4


def sparkle(base=83):
    out = np.zeros(int(0.7 * SR))
    for k, m in enumerate([base, base + 4, base + 7, base + 12, base + 16]):
        s = bell_piano(m, 0.45) * 0.08
        i = int(k * 0.04 * SR)
        out[i:i + len(s)] += s
    return out


# ---------------- harmony (MIDI) ----------------
# B major: B - G#m - E - F#, one chord per bar
CHORDS = [[59, 63, 66, 70], [56, 59, 63, 66], [52, 56, 59, 63], [54, 58, 61, 64]]
ROOTS = [35, 32, 40, 42]
PENTA = [71, 73, 75, 78, 80, 83, 85, 87, 90, 92]       # B major pentatonic, climbing

# ---------------- hook (0-5): sparse G#m melodica, two paper flicks, a sigh ----------------
add('music', T(0), pad([44, 51, 56, 59], T(5) + 0.1, 0.06))
for b, m, d in [(0, 68, 0.9), (1, 71, 0.5), (1.5, 70, 0.5), (2, 68, 1.0), (3.5, 63, 0.5), (4, 66, 0.9)]:
    add('music', T(b), melodica(m, d, 0.11), pan=-0.1)
add('drums', T(0), log_drum(32, gain=0.5))
add('drums', T(2.5), log_drum(32, gain=0.4))
add('fx', T(1), flick(), pan=-0.2)
add('fx', T(1.5), flick(), pan=0.2)
add('fx', T(3), blip(75, 0.2, 0.08))
add('fx', T(3.25), blip(71, 0.3, 0.07))

# ---------------- problem (5-8): felt thud, hollow chord ----------------
add('fx', T(5), thud(), 0.8)
add('hollow', T(5), chord([44, 51, 56, 59], 1.6, 0.2))
add('hollow', T(5), pad([44, 51, 56], T(3), 0.07))
add('music', T(6.5), melodica(63, 0.8, 0.07))

# ---------------- turn (8-12): ten pencil marks climbing, a bright chord, log-drum fill ----------------
add('music', T(8), pad([47, 54, 59, 63], T(4), 0.05))
for k in range(10):
    at = T(8.5 + k * 0.25)
    add('fx', at, pencil(), pan=-0.3 + 0.06 * k)
    add('fx', at + 0.01, blip(PENTA[k], 0.18, 0.09 + 0.006 * k), pan=-0.2 + 0.04 * k)
add('music', T(10.75), chord([71, 75, 78, 83], 1.4, 0.14, roll=0.02))
add('fx', T(10.75), sparkle(87))
for b, m in [(11.5, 47), (11.75, 47), (11.875, 49)]:
    add('drums', T(b), log_drum(m - 12, 0.3, 0.7))

# ---------------- groove (12-36) ----------------
LOG_PATTERN = [(0, 0), (0.75, 0), (1.5, 7), (2.5, 0), (2.75, 12), (3.5, 7)]   # (beat in bar, semitones over root)
for bar_start in range(12, 36, 4):
    bi = (bar_start - 12) // 4
    ci = bi % 4
    fact = s_fact <= bar_start < s_end
    for beat in range(4):
        b = bar_start + beat
        if not (s_fact <= b < s_end):
            add('drums', T(b), kick(), 0.55)
            if beat in (1, 3):
                add('drums', T(b), clap(), 0.45, pan=0.05)
            add('drums', T(b + 0.5), rim(), 0.22, pan=0.3)
        for s16 in range(4):
            add('drums', T(b + s16 * 0.25), shaker(accent=1.0 if s16 == 2 else 0.55), pan=-0.35)
    for off, semi in LOG_PATTERN:
        add('drums', T(bar_start + off), log_drum(ROOTS[ci] + semi, gain=0.8))
    # bell-piano chord stabs: on 1 and the "and" of 2
    add('music', T(bar_start), chord(CHORDS[ci], 0.9, 0.13, roll=0.008), pan=0.1)
    add('music', T(bar_start + 1.5), chord(CHORDS[ci], 0.6, 0.09, roll=0.008), pan=0.1)
    add('music', T(bar_start), pad(CHORDS[ci], T(4), 0.035 if not fact else 0.05))
    # melodica answer phrase in the result and end bars (not under the typing)
    if s_res <= bar_start < s_fact or bar_start >= s_end:
        top = CHORDS[ci][-1] + 12
        for b, m, d in [(2, top, 0.3), (2.5, top - 1 if ci != 2 else top - 2, 0.3), (3, top - 4, 0.5)]:
            add('music', T(bar_start + b), melodica(m, d, 0.07), pan=-0.15)

# ---------------- commands (12-19): sends, key ticks, reply bells ----------------
c = s_cmd
add('fx', T(c + 0.5), blip(83, 0.08, 0.16))
add('fx', T(c + 1.25), blip(87, 0.25, 0.1)); add('fx', T(c + 1.25) + 0.09, blip(94, 0.35, 0.09))
n_keys = len(',levels add 10 @Regulars')
for k in range(n_keys):
    u = k / n_keys
    add('fx', T(c + 1.75) + (T(2) - 0.12) * u + rng.uniform(-0.01, 0.01), tick(), pan=rng.uniform(-0.2, 0.2))
add('fx', T(c + 3.75), blip(83, 0.08, 0.16))
add('fx', T(c + 4.5), blip(87, 0.25, 0.1)); add('fx', T(c + 4.5) + 0.09, blip(94, 0.35, 0.09))

# ---------------- result (19-27): teo's message, level-up run, role sparkle ----------------
r = s_res
add('fx', T(r + 1), blip(80, 0.1, 0.14))
for k, m in enumerate([71, 75, 78, 83]):
    add('fx', T(r + 1.5 + k * 0.125), blip(m, 0.15, 0.12))
add('fx', T(r + 2), chord([83, 87, 90, 95], 1.2, 0.1, roll=0.015))
add('fx', T(r + 2.5), sparkle(90))

# ---------------- end (32-37): one more bar, then a rolled Bmaj9 to the last frame ----------------
final_len = DUR - T(36) + 0.05
add('music', T(36), chord([47, 59, 63, 66, 70, 73], final_len, 0.16, roll=0.03))
add('music', T(36), pad([47, 54, 63, 70], final_len, 0.06))
add('drums', T(36), log_drum(35, gain=0.8))
add('drums', T(36), kick(), 0.5)

# ---------------- mix ----------------
hollow = filt(buses['hollow'], None, 900)
mix = buses['music'] * 0.9 + buses['drums'] * 0.85 + buses['fx'] * 1.0 + hollow * 0.9
# a short room
d = int(0.083 * SR)
room = np.zeros_like(mix)
room[d:] += mix[:-d] * 0.18
room[2 * d:] += mix[:-2 * d] * 0.08
mix = mix + filt(room, None, 3000)
fade = int(0.02 * SR)
mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
mix[:int(0.005 * SR)] *= np.linspace(0, 1, int(0.005 * SR))[:, None]
mix = mix / np.max(np.abs(mix)) * 0.89

out = sys.argv[1] if len(sys.argv) > 1 else 'sound.wav'
with wave.open(out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype('<i2').tobytes())
print(f'{out}: {DUR:.3f} s, {n_all} samples')
