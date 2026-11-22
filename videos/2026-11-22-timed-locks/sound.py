#!/usr/bin/env python3
"""Original sound for the timed locks video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

A 100 BPM (36 video frames a beat) concert-harp groove in G major (G - Em - C - D): harp arpeggios in eighths, a warm
sine bass, a soft kick on one and three, a rim knock on two and four and a ticking hi-hat that plays the part of a
kitchen timer. Every shot in index.html is a whole number of beats, so cuts land on the grid: hook 0-4, forgot 4-7,
turn 7-9, command 9-16, later 16-20, open 20-24, fact 24-28, end 28-34 (beats).
The idea: the hook is a tense, sparse E-minor harp over a clock tick-tock; sol's message is a blip and the lock is a
heavy padlock clack, after which the harp is muffled (the channel is shut). "3 days." lands on a low felt thud and the
#off-topic question is a notification pop. The turn is a kitchen timer being wound (a run of ratchet clicks) and the
groove opens in G. Typing is soft key ticks, the send a blip, and Kaivor's reply a two-note harp chime with a small
padlock click. "Ten minutes later" is the timer running: sixteenth-note ticks over a held D chord whose filter opens as
the clock climbs, a tock on every minute, then a timer bell and a shackle spring at 21:24 that resolve to G. The open
shot runs the full groove with a pop for every message. The fact shot drops the drums. The end card rings out on a
rolled Gmaj9 to the last frame.
"""
import sys, wave
import numpy as np
from scipy.signal import lfilter, butter

SR = 48000
FPS = 60
BEAT_FRAMES = 36
BEAT = BEAT_FRAMES / FPS                                      # 0.6 s
SHOT_BEATS = [4, 3, 2, 7, 4, 4, 4, 6]
frames = [round(b * BEAT_FRAMES) for b in SHOT_BEATS]         # same rounding as tools/capture.mjs
starts_b = np.concatenate([[0], np.cumsum(SHOT_BEATS)])
s_hook, s_forgot, s_turn, s_cmd, s_later, s_open, s_fact, s_end, s_total = starts_b
DUR = sum(frames) / FPS
rng = np.random.default_rng(1122)
n_all = int(SR * DUR) + 1
buses = {k: np.zeros((n_all, 2)) for k in ('band', 'muff', 'drums', 'fx')}


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
    return lfilter(b, a, x)


def env(n, attack=0.004, release=0.03):
    e = np.ones(n)
    a = max(1, int(attack * SR)); r = max(1, int(release * SR))
    e[:a] = np.linspace(0, 1, a)
    e[-r:] *= np.linspace(1, 0, r)
    return e


# ---------- instruments ----------
def harp(m, dur=1.6, vel=1.0):
    """Concert harp: a plucked gut string. Odd-leaning partials that die faster the higher they are, a soft finger
    'thup' on the attack and a slight body resonance."""
    t = tt(dur)
    f = hz(m)
    out = np.zeros(len(t))
    for k, a in enumerate([1.0, 0.42, 0.30, 0.12, 0.09, 0.04, 0.03], start=1):
        out += a * np.sin(2 * np.pi * f * k * t * (1 + 0.0004 * k)) * np.exp(-t * (1.6 + 1.1 * k))
    thup = filt(rng.standard_normal(len(t)), lo=200, hi=1600) * np.exp(-t * 180) * 0.12
    body = filt(out, lo=180, hi=900) * 0.25
    return (out * 0.45 + body + thup) * env(len(t), 0.0015, 0.08) * vel


def bass(m, dur):
    t = tt(dur)
    f = hz(m)
    x = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)
    return filt(x * np.exp(-t * 2.2), hi=500) * env(len(t), 0.006, 0.05)


def kick():
    t = tt(0.32)
    f = 46 + 80 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 11)


def rim():
    t = tt(0.09)
    tone = np.sin(2 * np.pi * 1650 * t) * np.exp(-t * 70)
    n = filt(rng.standard_normal(len(t)), lo=1200, hi=5000) * np.exp(-t * 90)
    return (tone * 0.5 + n * 0.5) * env(len(t), 0.0005, 0.01)


def tick(vel=1.0, bright=6500):
    """The timer tick: a short, dry high click."""
    t = tt(0.035)
    return filt(rng.standard_normal(len(t)), lo=bright, hi=min(bright * 2.2, 22000)) * np.exp(-t * 220) * vel


def tock():
    t = tt(0.08)
    return (np.sin(2 * np.pi * 900 * t) * np.exp(-t * 60) * 0.6
            + filt(rng.standard_normal(len(t)), lo=600, hi=2500) * np.exp(-t * 120) * 0.4)


def padlock_clack():
    """A heavy padlock shackle snapping shut: a metallic click plus a low clunk."""
    t = tt(0.35)
    click = filt(rng.standard_normal(len(t)), lo=2500, hi=9000) * np.exp(-t * 160)
    ring = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t * d) for f, a, d in ((2210, 0.3, 30), (3470, 0.2, 40), (5120, 0.12, 55)))
    clunk = np.sin(2 * np.pi * (90 + 60 * np.exp(-t * 40)) * t) * np.exp(-t * 22)
    return (click * 0.6 + ring * 0.5 + clunk * 0.8) * env(len(t), 0.0005, 0.04)


def small_click():
    t = tt(0.06)
    return filt(rng.standard_normal(len(t)), lo=3000, hi=9000) * np.exp(-t * 150) * env(len(t), 0.0005, 0.01)


def spring():
    """The shackle springing open: a click and a short metallic boing sweeping up."""
    t = tt(0.45)
    f = 380 + 260 * (1 - np.exp(-t * 14))
    boing = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9) * (1 + 0.3 * np.sin(2 * np.pi * 28 * t))
    return (boing * 0.45 + np.concatenate([small_click(), np.zeros(len(t) - len(small_click()))]) * 0.8) * env(len(t), 0.001, 0.05)


def bell():
    """A kitchen-timer bell: a struck bell with inharmonic partials and a quick clapper rattle."""
    t = tt(1.8)
    out = np.zeros(len(t))
    for f, a, d in ((1318, 1.0, 2.2), (3170, 0.45, 3.5), (4420, 0.25, 5), (2050, 0.3, 2.8), (6050, 0.12, 7)):
        out += a * np.sin(2 * np.pi * f * t) * np.exp(-t * d)
    rattle = np.zeros(len(t))
    for k in range(6):
        i = int(k * 0.028 * SR)
        rattle[i:] += np.sin(2 * np.pi * 1318 * t[: len(t) - i]) * np.exp(-t[: len(t) - i] * 14) * (0.5 ** k)
    return (out * 0.35 + rattle * 0.25) * env(len(t), 0.0008, 0.1)


def ratchet(dur, n):
    """Winding the timer: n dry ratchet clicks, slowing slightly."""
    out = np.zeros(int(dur * SR) + int(0.05 * SR))
    for k in range(n):
        at = dur * (k / n) ** 1.15
        c = tick(0.9 - 0.3 * k / n, bright=3200)
        i = int(at * SR)
        out[i:i + len(c)] += c
    return out


def blip(m=86):
    t = tt(0.12)
    return np.sin(2 * np.pi * hz(m) * t) * np.exp(-t * 40) * env(len(t), 0.002, 0.02)


def pop(m=81):
    t = tt(0.16)
    f = hz(m) * (1 + 0.35 * np.exp(-t * 60))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 28) * env(len(t), 0.001, 0.02)


def thud():
    t = tt(0.6)
    f = 52 + 40 * np.exp(-t * 18)
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7)
            + filt(rng.standard_normal(len(t)), hi=400) * np.exp(-t * 30) * 0.4)


def shaker(vel=1.0):
    t = tt(0.07)
    return filt(rng.standard_normal(len(t)), lo=5000, hi=12000) * np.sin(np.pi * t / 0.07) * vel


def pad(ms, dur, cutoff_from, cutoff_to):
    """A soft held chord (detuned triangle waves) whose low-pass opens over its length."""
    t = tt(dur)
    x = np.zeros(len(t))
    for m in ms:
        for det in (-0.06, 0.06):
            ph = 2 * np.pi * hz(m + det) * t
            x += (2 / np.pi) * np.arcsin(np.sin(ph))
    x /= len(ms) * 2
    # block-wise moving filter
    out = np.zeros(len(t))
    block = 2048
    zi = None
    for s in range(0, len(t), block):
        c = cutoff_from * (cutoff_to / cutoff_from) ** (s / len(t))
        b, a = butter(2, c / (SR / 2), 'low')
        seg = x[s:s + block]
        if zi is None:
            zi = np.zeros(max(len(a), len(b)) - 1)
        y, zi = lfilter(b, a, seg, zi=zi)
        out[s:s + block] = y
    return out * env(len(t), 0.25, 0.08)


# ---------- notes ----------
E2, G2, A2, B2, C3, D3, E3 = 40, 43, 45, 47, 48, 50, 52
G3, A3, B3, C4, D4, E4, Fs4, G4 = 55, 57, 59, 60, 62, 64, 66, 67
CHORDS = [  # (bass, arpeggio notes) one bar of 4 beats each
    (G2, [G3, B3, D4, G4, B3 + 12, D4, B3, D4]),
    (E2, [E3, G3, B3, E4, G4, E4, B3, G3]),
    (C3, [C3 + 12, E3, G3, C4, E4, G4, E4, C4]),
    (D3, [D3, Fs4 - 12, A3, D4, Fs4, A3 + 12, Fs4, D4]),
]


def groove(b0, b1, drums=True, bus='band', harp_vel=0.55):
    """The G-major groove from beat b0 to b1 (bar position counted from the start of the groove)."""
    b = b0
    while b < b1 - 1e-6:
        bar = int((b - s_turn) // 4) % 4
        bass_m, arp = CHORDS[bar]
        pos = (b - s_turn) % 4
        # harp eighths
        for e in range(2):
            idx = int(pos * 2 + e) % 8
            if b + e * 0.5 < b1 - 1e-6:
                add(bus, T(b + e * 0.5) + rng.uniform(-0.004, 0.004), harp(arp[idx] + 12, 1.4, harp_vel * (1.0 if e == 0 else 0.8)),
                    pan=-0.35 + 0.1 * idx)
        if pos % 2 == 0:
            add(bus, T(b), bass(bass_m, BEAT * 1.9), 0.75)
        if drums:
            if pos % 2 == 0:
                add('drums', T(b), kick(), 0.85)
            else:
                add('drums', T(b), rim(), 0.35, pan=0.15)
            add('drums', T(b + 0.5), tick(0.55), pan=0.35)
            add('drums', T(b + 0.75), shaker(0.25), pan=-0.4)
        b += 1


# ---------- score ----------
# hook: tense E minor, sparse harp over a clock tick-tock; after the lock the harp is muffled
hook_notes = [(0, E4 + 12), (0.5, B3 + 12), (1.0, G4), (1.5, E4), (2.0, Fs4), (2.5, B3), (3.0, E4), (3.5, G3 + 12)]
for b, m in hook_notes:
    bus = 'band' if b < 2.5 else 'muff'
    add(bus, T(b), harp(m, 1.5, 0.55), pan=rng.uniform(-0.3, 0.3))
for b in range(0, 4):
    add('band' if b < 2.5 else 'muff', T(b), bass(E2, BEAT * 0.95), 0.55)
    add('drums', T(b), tick(0.7), pan=-0.25)
    add('drums', T(b + 0.5), tock(), 0.35, pan=0.25)
add('fx', T(1.5), blip(84), 0.4)                                     # sol's message
add('fx', T(2.5), padlock_clack(), 0.95)                              # the lock

# forgot: still muffled, slower; "3 days." thud; notification pop
for b, m in [(4, E3 + 12), (5, C4), (6, B3)]:
    add('muff', T(b), harp(m, 1.8, 0.5))
add('muff', T(4), bass(C3, BEAT * 2.8), 0.5)
for b in (4, 5, 6):
    add('drums', T(b), tick(0.45), pan=-0.25)
    add('drums', T(b + 0.5), tock(), 0.22, pan=0.25)
add('fx', T(s_forgot + 1), thud(), 0.9)
add('fx', T(s_forgot + 1.75), pop(83), 0.5)

# turn: winding the timer, then a harp glissando up into G
add('fx', T(s_turn + 0.25), ratchet(0.48, 9), 0.6, pan=0.1)
for k, m in enumerate([G3, A3, B3, C4, D4, E4, Fs4, G4, G4 + 12]):
    add('band', T(s_turn + 1.0) + k * 0.035, harp(m + 12 if k < 8 else m, 1.2, 0.45), pan=-0.4 + 0.1 * k)
add('band', T(s_turn), bass(D3, BEAT * 1.9), 0.6)
add('drums', T(s_turn + 1), kick(), 0.7)

# command: full groove; typing ticks, send blip, reply chime + padlock click
groove(s_cmd, s_later)
C_T, C_S, C_R = 0.5, 3.0, 3.75
a, bnd = T(s_cmd + C_T), T(s_cmd + C_S) - 0.1
tb = a
while tb < bnd:
    add('fx', tb + rng.uniform(-0.01, 0.01), tick(0.35, bright=4500), pan=rng.uniform(-0.2, 0.2))
    tb += 0.085 + rng.uniform(-0.015, 0.025)
add('fx', T(s_cmd + C_S), blip(88), 0.4)
add('fx', T(s_cmd + C_R), harp(D4 + 24, 1.0, 0.6), pan=0.2)
add('fx', T(s_cmd + C_R) + 0.1, harp(G4 + 24, 1.2, 0.55), pan=0.2)
add('fx', T(s_cmd + C_R) + 0.02, padlock_clack(), 0.4)

# later: the timer runs. sixteenth ticks over a held D chord opening up, a tock on every minute, bell + spring at 21:24
L_RUN, L_END, L_OPEN = 0.5, 2.75, 3.0
add('band', T(s_later), pad([D3, A3, D4, Fs4], T(L_OPEN) + 0.05, 500, 4200), 1.0)
add('band', T(s_later), bass(D3, T(L_OPEN)), 0.55)
for k in range(int(L_OPEN * 4)):
    add('drums', T(s_later + k * 0.25), tick(0.5 + 0.4 * k / (L_OPEN * 4)), pan=0.3 if k % 2 else -0.3)


def inout(x):
    x = min(1, max(0, x))
    return 8 * x ** 4 if x < 0.5 else 1 - (-2 * x + 2) ** 4 / 2


# a tock where the clock's minute changes (same easing as index.html)
prev = 14
for fr in range(int(T(L_OPEN) * FPS) + 1):
    t = fr / FPS
    k = inout((t - T(L_RUN)) / (T(L_END) - T(L_RUN)))
    mnt = 14 + min(10, int(np.floor(k * 10 + 1e-6)))
    if mnt != prev:
        add('fx', T(s_later) + t, tock(), 0.4, pan=0.0)
        prev = mnt
add('fx', T(s_later + L_OPEN), bell(), 0.75)
add('fx', T(s_later + L_OPEN), spring(), 0.6)
for j, m in enumerate([G3, B3, D4, G4]):
    add('band', T(s_later + L_OPEN) + j * 0.025, harp(m + 12, 1.2, 0.5), pan=-0.2 + 0.13 * j)
add('drums', T(s_later + L_OPEN), kick(), 0.8)

# open: groove; unlock click, a pop per message
groove(s_open, s_fact)
add('fx', T(s_open + 0.5), small_click(), 0.6)
for b, m in ((1.5, 81), (2.5, 84), (3.25, 88)):
    add('fx', T(s_open + b), pop(m), 0.45)

# fact: drums out, harp and bass only, one tick per beat (the timer, quieter)
groove(s_fact, s_end, drums=False, harp_vel=0.5)
for b in range(int(s_fact), int(s_end)):
    add('drums', T(b), tick(0.3), pan=0.3)

# end: rolled Gmaj9 to the last frame
end_dur = DUR - T(s_end)
for j, m in enumerate([G3, D4, Fs4, A3 + 12, B3 + 12, D4 + 12]):
    add('band', T(s_end) + j * 0.045, harp(m, min(3.5, end_dur), 0.5), pan=-0.4 + 0.16 * j)
add('band', T(s_end), bass(G2, end_dur), 0.7)
add('band', T(s_end), pad([G3, B3, D4, Fs4, A3 + 12], end_dur - 0.05, 900, 2200), 0.6)
add('band', T(s_end + 2), harp(D4 + 12, 1.6, 0.35), pan=-0.2)
add('drums', T(s_end), kick(), 0.8)
add('fx', T(s_end + 2), harp(G4 + 24, 2.5, 0.35), pan=0.2)
fade_n = int(0.35 * SR)
for bus in buses.values():
    bus[-fade_n:] *= np.linspace(1, 0.25, fade_n)[:, None]

# ---------- mix ----------
muff = np.stack([filt(buses['muff'][:, c], hi=700, order=2) for c in range(2)], axis=1)
mix = buses['band'] * 0.9 + muff * 1.1 + buses['drums'] * 0.85 + buses['fx'] * 0.95
for d, g in ((0.021, 0.18), (0.043, 0.12), (0.067, 0.07)):   # a short room
    k = int(d * SR)
    mix[k:, 0] += mix[:-k, 1] * g
    mix[k:, 1] += mix[:-k, 0] * g
mix[: int(0.004 * SR)] *= np.linspace(0, 1, int(0.004 * SR))[:, None]
mix /= np.max(np.abs(mix)) + 1e-9
mix *= 0.89

out = sys.argv[1] if len(sys.argv) > 1 else 'sound.wav'
with wave.open(out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.clip(mix, -1, 1) * 32767).astype('<i2').tobytes())
print(out, f'{DUR:.3f}s')
