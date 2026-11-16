#!/usr/bin/env python3
"""Original sound for the automod-reads-forwards video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

A 133.33 BPM (27 video frames a beat) vibraphone groove in C# minor that opens up to E major on the end card.
Every shot in index.html is a whole number of beats, so cuts land on the grid:
hook 0-9, turn 9-12, action 12-20, result 20-33, fact 33-38, end 38-45 (beats).
The idea: a soft motor-tremolo vibraphone and a round plucked bass play C#m7 - Amaj7 - E - B over a brushed shaker
and a dry clap. A forwarded message arrives as a short panned paper whoosh with a pop. The turn ("Kaivor reads
forwards now.") drops everything for one bright vibraphone stab over a sub. In the action, the typing is soft key
ticks, the send a blip and Kaivor's reply a three-note vibe chime (E, G#, B). In the result the forward whooshes in
again, the scan line is a thin rising sweep that moves with it, the hit is a low two-note buzz, the delete a
reversed paper swish into a thud, Kaivor's notice a pop and the log card a double tick with a vibe note. The fact
shot strips back to bass and pad; the end card lands on Emaj9 with a slow vibe arpeggio that reaches the last frame.
"""
import sys, wave
import numpy as np
from scipy.signal import lfilter, butter

SR = 48000
FPS = 60
BEAT_FRAMES = 27
BEAT = BEAT_FRAMES / FPS                                      # 0.45 s
SHOT_BEATS = [9, 3, 8, 13, 5, 7]
frames = [round(b * BEAT * FPS) for b in SHOT_BEATS]          # same rounding as tools/capture.mjs
starts_b = np.concatenate([[0], np.cumsum(SHOT_BEATS)])
DUR = sum(frames) / FPS
rng = np.random.default_rng(1116)
n_all = int(SR * DUR) + 1
buses = {k: np.zeros((n_all, 2)) for k in ('band', 'drums', 'fx')}


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


def add_panned(bus, at, sig, pans, gain=1.0):
    """Like add(), but the pan moves from pans[0] to pans[1] over the sound."""
    i = int(round(at * SR))
    if i >= n_all:
        return
    sig = sig[: n_all - i] * gain
    pan = np.linspace(pans[0], pans[1], len(sig))
    b = buses[bus]
    b[i:i + len(sig), 0] += sig * np.sqrt((1 - pan) / 2) * 1.414
    b[i:i + len(sig), 1] += sig * np.sqrt((1 + pan) / 2) * 1.414


def tt(dur):
    return np.arange(int(dur * SR)) / SR


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def filt(x, lo=None, hi=None, order=2):
    if lo and hi:
        b, a = butter(order, [lo / (SR / 2), hi / (SR / 2)], btype='band')
    elif hi:
        b, a = butter(order, hi / (SR / 2), btype='low')
    else:
        b, a = butter(order, lo / (SR / 2), btype='high')
    return lfilter(b, a, x)


# ---------- instruments ----------
def vibe(m, dur=1.2, trem=5.2, bright=1.0):
    """Vibraphone: a sine with a quiet 4th-harmonic partial, a strike transient and motor tremolo."""
    t = tt(dur)
    f = hz(m)
    x = np.sin(2 * np.pi * f * t) + 0.22 * bright * np.sin(2 * np.pi * f * 4.0 * t) * np.exp(-t / 0.08)
    x += 0.08 * bright * np.sin(2 * np.pi * f * 10.0 * t) * np.exp(-t / 0.015)
    tremolo = 1 - 0.28 * (0.5 + 0.5 * np.sin(2 * np.pi * trem * t))
    e = np.exp(-t / (dur * 0.45)) * np.minimum(1, t / 0.002)
    r = int(0.03 * SR)
    e[-r:] *= np.linspace(1, 0, r)
    return x * tremolo * e


def chord(notes, dur, gain=0.12, bright=0.8):
    x = sum(vibe(m, dur, bright=bright) for m in notes)
    return x * gain


def pluck_bass(m, dur):
    t = tt(dur)
    f = hz(m)
    x = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.06)
    e = np.exp(-t / 0.22) * np.minimum(1, t / 0.004)
    r = int(0.02 * SR)
    e[-r:] *= np.linspace(1, 0, r)
    return filt(x, hi=900) * e


def pad(notes, dur, gain=0.05, attack=0.2):
    t = tt(dur)
    x = np.zeros(len(t))
    for m in notes:
        for det in (-0.05, 0.05):
            x += np.sin(2 * np.pi * hz(m + det) * t + rng.random() * 6)
            x += 0.3 * np.sin(2 * np.pi * hz(m + det) * 2 * t)
    e = np.minimum(1, t / attack)
    r = min(int(0.3 * SR), len(t) // 3)
    e[-r:] *= np.linspace(1, 0, r)
    return x / len(notes) * e * gain


def kick(dur=0.3):
    t = tt(dur)
    f = 48 + 100 * np.exp(-t / 0.03)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.12)


def clap(dur=0.2):
    t = tt(dur)
    n = filt(rng.standard_normal(len(t)), lo=900, hi=5000)
    e = np.zeros(len(t))
    for k, d in enumerate((0.0, 0.009, 0.018)):
        i = int(d * SR)
        e[i:] += np.exp(-(t[: len(t) - i]) / (0.008 if k < 2 else 0.05))
    return n * e * 0.6


def shaker(dur=0.07, accent=False):
    t = tt(dur)
    n = filt(rng.standard_normal(len(t)), lo=5000, hi=12000)
    return n * np.sin(np.pi * np.minimum(1, t / dur)) ** (1.5 if accent else 3)


def whoosh(dur=0.32, lo=500, hi=4000):
    """Paper-ish air: band-passed noise swept upward."""
    t = tt(dur)
    n = rng.standard_normal(len(t))
    out = np.zeros(len(t))
    chunks = 16
    for k in range(chunks):
        a, b = k * len(t) // chunks, (k + 1) * len(t) // chunks
        c = lo * (hi / lo) ** (k / chunks)
        out[a:b] = filt(n[a:b], lo=c * 0.6, hi=min(c * 1.8, 20000))
    return out * np.sin(np.pi * t / dur) ** 2


def pop(m=84, dur=0.12):
    t = tt(dur)
    f = hz(m) * (1 + 0.6 * np.exp(-t / 0.01))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.03)


def tick(dur=0.018):
    t = tt(dur)
    return filt(rng.standard_normal(len(t)), lo=2500, hi=9000) * np.exp(-t / 0.003)


def scan_sweep(dur):
    """A thin rising tone for the scan line, a narrow pulse through a band-pass."""
    t = tt(dur)
    f = 600 * (2400 / 600) ** (t / dur)
    ph = np.cumsum(f) / SR % 1.0
    x = np.where(ph < 0.18, 1.0, -0.2)
    x = filt(x - x.mean(), lo=500, hi=5000)
    return x * np.minimum(1, t / 0.03) * np.minimum(1, (dur - t) / 0.04)


def buzz(m, dur=0.22):
    t = tt(dur)
    ph = hz(m) * t % 1.0
    x = 2 * ph - 1
    return filt(x, hi=1400) * np.minimum(1, t / 0.004) * np.exp(-t / (dur * 0.6))


def thud(dur=0.3):
    t = tt(dur)
    return np.sin(2 * np.pi * np.cumsum(55 + 70 * np.exp(-t / 0.02)) / SR) * np.exp(-t / 0.08)


def sub(dur=0.9, m=37):
    t = tt(dur)
    return np.sin(2 * np.pi * hz(m) * t) * np.exp(-t / 0.5) * np.minimum(1, t / 0.005)


# ---------- arrangement ----------
s_hook, s_turn, s_act, s_res, s_fact, s_end, s_out = starts_b

# C#m7 - Amaj7 - E - B
prog = [(37, [61, 64, 68, 71]), (33, [61, 64, 68, 69]), (40, [59, 64, 68]), (35, [59, 63, 66])]
BASS_RHYTHM = [(0, 0), (1.5, 0), (2.5, 7), (3.5, 12)]   # (beat in bar, interval)


def groove(b0, b1, drums=True, busy=False, start_bar=0):
    b = b0
    bar = start_bar
    while b < b1 - 1e-9:
        root, notes = prog[bar % 4]
        length = min(4, b1 - b)
        add('band', T(b), chord(notes, T(min(length, 3.6)), gain=0.1))
        add('band', T(b + 2), chord(notes[-2:], T(min(1.6, max(0.1, length - 2))), gain=0.06) if length > 2 else np.zeros(1))
        for at, iv in BASS_RHYTHM:
            if at < length:
                add('band', T(b + at), pluck_bass(root + iv, T(min(1.0, length - at))), 0.5)
        if drums:
            add('drums', T(b), kick(), 0.55)
            if length > 2.5 and busy:
                add('drums', T(b + 2.75), kick(), 0.3)
            for c in (1, 3):
                if c < length:
                    add('drums', T(b + c), clap(), 0.32)
            steps = int(length * 4)
            for s in range(steps):
                add('drums', T(b + s * 0.25), shaker(accent=s % 2 == 1), 0.05 if s % 2 == 0 else 0.08, pan=0.35)
        b += 4
        bar += 1


# hook: groove in from the first frame; the forward lands at beat 1.5
add('fx', T(0), sub(0.8), 0.35)
groove(s_hook, s_turn, drums=True)
add_panned('fx', T(1.5) - 0.12, whoosh(0.34), (-0.6, 0.4), 0.42)
add('fx', T(1.5) + 0.02, pop(86), 0.2)

# turn: everything drops; one bright vibe stab on Amaj9 over a sub, a soft riser into the action
add('band', T(s_turn), chord([57, 61, 64, 68, 71], T(2.8), gain=0.14, bright=1.3))
add('fx', T(s_turn), sub(1.2, 33), 0.55)
add('drums', T(s_turn), kick(), 0.5)
add('fx', T(s_turn + 2.1), whoosh(T(0.9), 300, 6000), 0.22)

# action + result groove
groove(s_act, s_res, start_bar=0)
groove(s_res, s_fact, busy=True, start_bar=2)

# action: typing (16 characters), send, Kaivor's reply chime
for k in range(16):
    at = T(s_act + 0.4) + T(2.8) * (k / 16) ** 0.95 + rng.uniform(-0.008, 0.008)
    add('fx', at, tick(), 0.26, pan=rng.uniform(-0.2, 0.2))
add('fx', T(s_act + 3.6), pop(90, 0.09), 0.18)
for j, m in enumerate([76, 80, 83]):
    add('fx', T(s_act + 4.4 + j * 0.25), vibe(m, 0.9, bright=1.2), 0.17, pan=-0.2 + 0.2 * j)

# result: forward in, scan, hit, delete, notice, log card
add_panned('fx', T(s_res + 1) - 0.12, whoosh(0.34), (-0.6, 0.4), 0.42)
add('fx', T(s_res + 1) + 0.02, pop(86), 0.2)
add('fx', T(s_res + 2), scan_sweep(T(1)), 0.14)
add('fx', T(s_res + 3), buzz(40, 0.2), 0.3)
add('fx', T(s_res + 3.5), buzz(39, 0.26), 0.3)
add_panned('fx', T(s_res + 4) - 0.2, whoosh(0.22, 4000, 600)[::-1], (0.3, -0.3), 0.3)
add('fx', T(s_res + 4), thud(), 0.45)
add('fx', T(s_res + 4.5), pop(88), 0.2)
add('fx', T(s_res + 6.5), tick(), 0.3, pan=-0.2)
add('fx', T(s_res + 6.5) + 0.07, tick(), 0.24, pan=0.2)
add('fx', T(s_res + 6.6), vibe(71, 1.2, bright=1.1), 0.14)

# fact: strip back to bass and pad, a soft swipe for the rule, two vibe notes for the two lines
add('band', T(s_fact), pad([49, 56, 61, 64], T(5), gain=0.06, attack=0.05))
add('band', T(s_fact), pluck_bass(37, T(1.2)), 0.55)
add('band', T(s_fact + 2.5), pluck_bass(37, T(1.2)), 0.42)
add('drums', T(s_fact), kick(), 0.5)
add('drums', T(s_fact + 3), clap(), 0.22)
add('fx', T(s_fact + 1.3), whoosh(0.3, 1500, 7000), 0.16)
add('fx', T(s_fact + 1.7), vibe(73, 1.0), 0.13)
add('fx', T(s_fact + 2.6), vibe(68, 1.0), 0.11)

# end: Emaj9 with a slow vibe arpeggio to the last frame
end_len = DUR - T(s_end)
add('band', T(s_end), pad([40, 52, 56, 59, 63, 66], end_len, gain=0.07, attack=0.06))
add('band', T(s_end), chord([56, 59, 63, 66], T(3), gain=0.1, bright=1.1))
add('band', T(s_end), pluck_bass(40, T(1.8)), 0.55)
add('band', T(s_end + 4), pluck_bass(40, T(1.6)), 0.4)
arp = [64, 68, 71, 75, 78, 75, 71, 68]
for k in range(13):
    add('fx', T(s_end + 0.5 + k * 0.5), vibe(arp[k % len(arp)], 0.9), 0.1, pan=(-0.3, 0.3)[k % 2])
add('drums', T(s_end), kick(), 0.5)
for b in range(7):
    add('drums', T(s_end + b + 0.5), shaker(accent=True), 0.06, pan=0.35)

# ---------- mix ----------
mix = buses['band'] * 0.9 + buses['drums'] * 0.85 + buses['fx'] * 0.95
fade = int(0.3 * SR)
mix[-fade:] *= np.linspace(1, 0.45, fade)[:, None]
mix /= max(1e-9, np.max(np.abs(mix))) / 0.8

out = sys.argv[1] if len(sys.argv) > 1 else 'sound.wav'
with wave.open(out, 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype('<i2').tobytes())
print(f'wrote {out}: {DUR:.3f}s, shots start at {np.round(starts_b * BEAT, 3).tolist()}')
