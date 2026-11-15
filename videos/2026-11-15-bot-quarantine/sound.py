#!/usr/bin/env python3
"""Original sound for the compromised-bot-quarantine video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

A dark 144 BPM half-time "containment" track in G minor that settles on a G suspended chord at the end.
Every shot in index.html is a whole number of beats (one beat = 25 frames), so cuts land on the grid:
hook 0-8, turn 8-11, action 11-20, result 20-32, fact 32-37, end 37-48 (beats).
The idea: the hijacked bot is heard as a bit-crushed digital glitch, a stutter of sample-and-hold noise, each time it
posts, over a filtered square-wave bass pulse in eighths and ticking 16th hats. The turn ("It still has all its
roles.") cuts to a single sub drop and a rising noise riser. The band comes in with the action shot: a half-time
kick and snare, a pulse-wave bass walking G - Eb - C - D, and a low detuned saw pad. Typing is dry key clicks, the
send is a blip and Kaivor's reply is a three-note pulse chime. In the result each of the three posts is a glitch
plus a counting note (G, Bb, D); the third trips the quarantine: a heavy low thump, a metallic clank made of
inharmonic partials and a pressure hiss, like a hatch sealing. The two roles that come off are falling plucks, the
emptied managed role is a muted thunk, and the alert is a low two-tone. The fact shot strips back to the sub and
the pad; the end card lands on Ebmaj7 and then a G sus2 pad with a slow pulse arpeggio that reaches the last frame.
"""
import sys, wave
import numpy as np
from scipy.signal import lfilter, butter

SR = 48000
FPS = 60
BPM = 144
BEAT = 60 / BPM
SHOT_BEATS = [8, 3, 9, 12, 5, 11]
frames = [round(b * BEAT * FPS) for b in SHOT_BEATS]          # same rounding as tools/capture.mjs
starts_b = np.concatenate([[0], np.cumsum(SHOT_BEATS)])        # shot starts in beats
DUR = sum(frames) / FPS
rng = np.random.default_rng(1115)
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


def env(n, a=0.004, d=0.2, sustain=0.0, rel=0.03):
    t = np.arange(n) / SR
    e = np.minimum(1, t / max(a, 1e-4)) * (sustain + (1 - sustain) * np.exp(-t / max(d, 1e-4)))
    r = int(rel * SR)
    if r and n > r:
        e[-r:] *= np.linspace(1, 0, r)
    return e


# ---------- instruments ----------
def square_bass(m, dur, cutoff=700, gain=1.0):
    t = tt(dur)
    f = hz(m)
    x = np.sign(np.sin(2 * np.pi * f * t)) * 0.5 + np.sin(2 * np.pi * f / 2 * t) * 0.6
    return filt(x, hi=cutoff) * env(len(t), 0.003, dur * 0.6, 0.3, 0.02) * gain


def pulse(m, dur, width=0.3, decay=0.25, cutoff=4000):
    t = tt(dur)
    ph = (hz(m) * t) % 1.0
    x = np.where(ph < width, 1.0, -1.0) * 0.5
    return filt(x, hi=cutoff) * env(len(t), 0.002, decay, 0.0, 0.02)


def saw_pad(notes, dur, cutoff=1200, gain=0.1, attack=0.25):
    t = tt(dur)
    x = np.zeros(len(t))
    for m in notes:
        for det in (-0.07, 0.0, 0.08):
            f = hz(m + det)
            x += 2 * ((f * t + rng.random()) % 1.0) - 1
    x = filt(x / (3 * len(notes)), hi=cutoff)
    return x * env(len(t), attack, 99, 1.0, min(0.4, dur * 0.3)) * gain


def kick(dur=0.32):
    t = tt(dur)
    f = 45 + 120 * np.exp(-t / 0.035)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.13)


def snare(dur=0.22):
    t = tt(dur)
    n = filt(rng.standard_normal(len(t)), lo=1200, hi=7000) * np.exp(-t / 0.06)
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.04)
    return n * 0.7 + tone * 0.5


def hat(dur=0.05, open_=False):
    t = tt(dur if not open_ else 0.18)
    return filt(rng.standard_normal(len(t)), lo=7000) * np.exp(-t / (0.012 if not open_ else 0.06))


def glitch(dur=0.24):
    """The hijacked bot: a bit-crushed, sample-and-hold stutter."""
    n = int(dur * SR)
    out = np.zeros(n)
    i = 0
    while i < n:
        seg = int(SR * rng.uniform(0.012, 0.03))
        hold = int(SR / rng.uniform(600, 3200))
        vals = np.repeat(rng.uniform(-1, 1, seg // max(1, hold) + 1), hold)[:seg]
        gate = 1.0 if rng.random() > 0.25 else 0.0
        out[i:i + seg] = vals[: n - i] * gate
        i += seg
    out = np.round(out * 6) / 6                                  # crush to a few levels
    return filt(out, lo=300, hi=6000) * env(n, 0.002, dur * 0.5, 0.3, 0.02)


def ping(m, dur=0.35):
    t = tt(dur)
    return (np.sin(2 * np.pi * hz(m) * t) + 0.3 * np.sin(2 * np.pi * hz(m + 12) * t)) * np.exp(-t / 0.09)


def click(dur=0.02):
    t = tt(dur)
    return filt(rng.standard_normal(len(t)), lo=2000, hi=9000) * np.exp(-t / 0.004)


def clank(dur=1.2):
    """A heavy hatch sealing: inharmonic partials over a low thump."""
    t = tt(dur)
    x = np.zeros(len(t))
    for f, a, d in ((211, 1.0, 0.35), (563, 0.6, 0.22), (877, 0.45, 0.16), (1291, 0.3, 0.12), (1934, 0.2, 0.08)):
        x += a * np.sin(2 * np.pi * f * t + rng.random() * 6) * np.exp(-t / d)
    thump = np.sin(2 * np.pi * np.cumsum(38 + 90 * np.exp(-t / 0.05)) / SR) * np.exp(-t / 0.3)
    return x * 0.35 + thump * 0.9


def hiss(dur=0.8):
    t = tt(dur)
    n = filt(rng.standard_normal(len(t)), lo=2500, hi=9000)
    return n * np.exp(-t / 0.25) * np.minimum(1, t / 0.01)


def riser(dur):
    t = tt(dur)
    n = rng.standard_normal(len(t))
    out = np.zeros(len(t))
    chunks = 24
    for k in range(chunks):
        a, b = k * len(t) // chunks, (k + 1) * len(t) // chunks
        out[a:b] = filt(n[a:b], lo=300 + 5000 * (k / chunks) ** 2, hi=900 + 9000 * (k / chunks) ** 2)
    return out * (t / dur) ** 2


def sub_drop(dur=0.9):
    t = tt(dur)
    f = 80 * np.exp(-t / 0.4) + 30
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.5)


# ---------- arrangement ----------
s_hook, s_turn, s_act, s_res, s_fact, s_end, s_out = starts_b

# hook: filtered square bass in eighths on G, ticking hats, a glitch + mention ping on each of the three posts
add('fx', T(0), sub_drop(0.6), 0.5)
for e in range(16):
    b = e * 0.5
    add('band', T(b), square_bass(31, T(0.45), cutoff=380 + 60 * e), 0.42)
for s in range(32):
    add('drums', T(s * 0.25), hat(), 0.08 if s % 2 else 0.12, pan=0.25)
for k in (0, 4):
    add('drums', T(k), kick(), 0.45)
for b in (1.0, 2.25, 3.5):
    add('fx', T(b), glitch(0.26), 0.5, pan=rng.uniform(-0.3, 0.3))
    add('fx', T(b + 0.12), ping(84), 0.16)
    add('fx', T(b + 0.2), ping(79), 0.12)
add('drums', T(6), snare(), 0.3)
add('drums', T(7.5), snare(), 0.18)

# turn: near silence, one sub drop, a riser into the action
add('fx', T(s_turn), sub_drop(1.0), 0.75)
add('band', T(s_turn), saw_pad([43, 50], T(3), cutoff=500, gain=0.08, attack=0.05))
add('fx', T(s_turn + 0.5), riser(T(2.5)), 0.22)

# action + result groove: half-time kick/snare, pulse bass walking G - Eb - C - D, low pad
prog = [(43, [55, 58, 62]), (39, [55, 58, 63]), (36, [55, 60, 63]), (38, [54, 57, 62])]


def groove(b0, b1, busy=False):
    b = b0
    bar = 0
    while b < b1:
        root, chord = prog[bar % 4]
        length = min(4, b1 - b)
        add('band', T(b), saw_pad(chord, T(length), cutoff=1100 if not busy else 1500, gain=0.075))
        for e in range(int(length * 2)):
            m = root + (12 if e % 4 == 3 else 0)
            add('band', T(b + e * 0.5), square_bass(m, T(0.42), cutoff=900), 0.36)
        add('drums', T(b), kick(), 0.6)
        if length > 2.5:
            add('drums', T(b + 2.5), kick(), 0.4)
        if length > 2:
            add('drums', T(b + 2), snare(), 0.42)
        for s in range(int(length * (4 if busy else 2))):
            step = 0.25 if busy else 0.5
            add('drums', T(b + s * step), hat(), 0.07 if s % 2 else 0.1, pan=0.3)
        b += 4
        bar += 1


groove(s_act, s_res)
groove(s_res, s_fact, busy=True)

# action: typing, send, Kaivor's reply chime
for k in range(26):
    at = T(s_act + 0.6) + T(3.6) * (k / 26) ** 0.9 + rng.uniform(-0.01, 0.01)
    add('fx', at, click(), 0.22, pan=rng.uniform(-0.2, 0.2))
add('fx', T(s_act + 4.6), ping(88, 0.18), 0.14)
for j, m in enumerate([74, 79, 83]):
    add('fx', T(s_act + 5.5 + j * 0.25), pulse(m, 0.5, width=0.25, decay=0.18), 0.16, pan=-0.2 + 0.2 * j)

# result: three posts (glitch + counting note), the seal on the third, roles dropping, the alert
for k, (b, m) in enumerate(zip((0.5, 1.5, 2.5), (67, 70, 74))):
    add('fx', T(s_res + b), glitch(0.2), 0.42, pan=(-0.4, 0.0, 0.4)[k])
    add('fx', T(s_res + b + 0.04), pulse(m, 0.4, width=0.5, decay=0.15), 0.22)
add('fx', T(s_res + 2.5), clank(1.4), 0.55)
add('fx', T(s_res + 2.55), hiss(0.9), 0.14)
add('fx', T(s_res + 3.25), pulse(79, 0.4, width=0.2, decay=0.12) * np.linspace(1, 0.4, int(0.4 * SR)), 0.18, pan=-0.3)
add('fx', T(s_res + 3.75), pulse(74, 0.4, width=0.2, decay=0.12), 0.16, pan=0.3)
add('fx', T(s_res + 4.25), kick(0.2) * 0.6 + filt(rng.standard_normal(int(0.2 * SR)), hi=600) * np.exp(-tt(0.2) / 0.03), 0.3)
add('fx', T(s_res + 5.5), pulse(62, 0.5, width=0.4, decay=0.2), 0.2)
add('fx', T(s_res + 5.75), pulse(58, 0.6, width=0.4, decay=0.25), 0.2)

# fact: strip back to sub and pad, a swipe for the rule, a DM blip
add('band', T(s_fact), saw_pad([43, 50, 58], T(5), cutoff=800, gain=0.09, attack=0.02))
add('band', T(s_fact), square_bass(31, T(2.4), cutoff=300), 0.5)
add('band', T(s_fact + 2.5), square_bass(31, T(2.4), cutoff=300), 0.4)
add('drums', T(s_fact), kick(), 0.55)
add('drums', T(s_fact + 2.5), kick(), 0.35)
add('drums', T(s_fact + 2), snare(), 0.25)
add('fx', T(s_fact + 1.1), riser(T(0.4))[::-1] * 0.6, 0.25)
add('fx', T(s_fact + 1.6), ping(86, 0.25), 0.13)

# end: Ebmaj7, then a G sus2 pad with a slow pulse arpeggio and a soft tail to the last frame
end_len = DUR - T(s_end)
add('fx', T(s_end), riser(T(0.5))[::-1], 0.18)
add('band', T(s_end), saw_pad([51, 55, 58, 62], T(4), cutoff=1600, gain=0.1, attack=0.03))
add('band', T(s_end), square_bass(39, T(3.8), cutoff=500), 0.4)
add('band', T(s_end + 4), saw_pad([43, 50, 55, 57, 62], end_len - T(4), cutoff=1400, gain=0.11, attack=0.08))
add('band', T(s_end + 4), square_bass(31, end_len - T(4), cutoff=400), 0.38)
arp = [67, 69, 74, 79, 74, 69]
for k in range(14):
    add('fx', T(s_end + 0.5 + k * 0.75), pulse(arp[k % len(arp)], 0.5, width=0.3, decay=0.2), 0.12, pan=(-0.3, 0.3)[k % 2])
for b in range(8):
    if b % 4 == 0:
        add('drums', T(s_end + b), kick(), 0.45 if b == 0 else 0.3)
    add('drums', T(s_end + b + 0.5), hat(), 0.07, pan=0.3)

# ---------- mix ----------
mix = buses['band'] * 0.9 + buses['drums'] * 0.85 + buses['fx'] * 0.95
fade = int(0.3 * SR)
mix[-fade:] *= np.linspace(1, 0.4, fade)[:, None]
mix /= max(1e-9, np.max(np.abs(mix))) / 0.8

out = sys.argv[1] if len(sys.argv) > 1 else 'sound.wav'
with wave.open(out, 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype('<i2').tobytes())
print(f'wrote {out}: {DUR:.3f}s, shots start at {np.round(starts_b * BEAT, 3).tolist()}')
