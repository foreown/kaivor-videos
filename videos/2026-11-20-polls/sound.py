#!/usr/bin/env python3
"""Original sound for the polls video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

A 102.86 BPM (35 video frames a beat) honky-tonk upright-piano shuffle in Db major (Db - Bbm - Gb - Ab), with a
plucked bass, a soft kick, a rim-shot backbeat and a swung shaker. Every shot in index.html is a whole number of
beats, so cuts land on the grid: hook 0-5, turn 5-7, command 7-13, voting 13-18, results 18-24, fact 24-28,
end 28-34 (beats).
The idea: the hook is a film projector clattering over a lonely Bb-minor piano line, and every chat message is a
notification blip, each a little higher and more crowded, so the argument piles up. The turn cuts to the marquee:
a bulb tick on every quarter beat as the lights chase, and a two-hand piano roll up into Db. From the command shot
the full shuffle runs; typing is soft key ticks, the send a blip, the poll arrives on a two-note chime. In the voting
shot the Dune pick and the Vote press are mouse clicks, and each of the 12 votes is a pop climbing a Db pentatonic
scale. The results shot strips to a snare roll under "two days later", the poll closing on a bell, then Kaivor's
results land on a brass "ta-da" chord with a cymbal. The fact shot is bass and piano only; the end card rings out on
Dbmaj9 with a piano arpeggio to the last frame.
"""
import sys, wave
import numpy as np
from scipy.signal import lfilter, butter

SR = 48000
FPS = 60
BEAT_FRAMES = 35
BEAT = BEAT_FRAMES / FPS                                      # 0.5833 s
SHOT_BEATS = [5, 2, 6, 5, 6, 4, 6]
frames = [round(b * BEAT_FRAMES) for b in SHOT_BEATS]         # same rounding as tools/capture.mjs
starts_b = np.concatenate([[0], np.cumsum(SHOT_BEATS)])
s_hook, s_turn, s_cmd, s_vote, s_res, s_fact, s_end, s_total = starts_b
DUR = sum(frames) / FPS
rng = np.random.default_rng(1120)
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
def piano(m, dur=1.2, vel=1.0):
    """Honky-tonk upright: additive partials, two strings detuned against each other, hammer thump."""
    t = tt(dur)
    f = hz(m)
    out = np.zeros_like(t)
    for k, amp in enumerate((1.0, 0.55, 0.32, 0.2, 0.12, 0.07), start=1):
        dec = np.exp(-t * (2.2 + k * 1.3) * (f / 260) ** 0.3)
        for cents in (-7, 7):
            out += amp * dec * np.sin(2 * np.pi * f * k * 2 ** (cents / 1200) * t * (1 + 0.0004 * k * k))
    thump = filt(rng.standard_normal(len(t)), hi=1800) * np.exp(-t * 90) * 0.25
    return (out / 4 + thump) * env(len(t), 0.002, 0.08) * vel


def chord(ms, dur, vel=1.0, spread=0.012):
    n = int(dur * SR) + int(spread * len(ms) * SR) + 1
    out = np.zeros(n)
    for k, m in enumerate(ms):
        s = piano(m, dur, vel)
        i = int(k * spread * SR)
        out[i:i + len(s)] += s
    return out / max(1, len(ms) ** 0.6)


def bass(m, dur):
    t = tt(dur)
    f = hz(m)
    x = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t)
    return filt(x * np.exp(-t * 3.2), hi=900) * env(len(t), 0.003, 0.05)


def kick():
    t = tt(0.32)
    f = 46 + 90 * np.exp(-t * 32)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 11)


def rim():
    t = tt(0.12)
    tone = np.sin(2 * np.pi * 1650 * t) * np.exp(-t * 70)
    noise = filt(rng.standard_normal(len(t)), lo=1800, hi=7000) * np.exp(-t * 55)
    return tone * 0.6 + noise * 0.7


def snare_hit(v=1.0):
    t = tt(0.16)
    return (filt(rng.standard_normal(len(t)), lo=1200, hi=8000) * np.exp(-t * 30) + 0.4 * np.sin(2 * np.pi * 190 * t) * np.exp(-t * 40)) * v


def shaker(v=1.0):
    t = tt(0.07)
    return filt(rng.standard_normal(len(t)), lo=5000) * np.sin(np.pi * np.linspace(0, 1, len(t))) ** 2 * v


def cymbal(dur=1.6):
    t = tt(dur)
    x = filt(rng.standard_normal(len(t)), lo=4500)
    return x * np.exp(-t * 2.6) * env(len(t), 0.001, 0.2)


def blip(m, dur=0.12):
    t = tt(dur)
    return np.sin(2 * np.pi * hz(m) * t) * np.exp(-t * 30) * env(len(t), 0.001, 0.02)


def chime(m, dur=0.9):
    t = tt(dur)
    f = hz(m)
    return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t * 6)) * np.exp(-t * 4.5) * env(len(t), 0.002, 0.05)


def pop(m):
    t = tt(0.14)
    f = hz(m) * (1 + 0.6 * np.exp(-t * 60))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 26) * env(len(t), 0.001, 0.02)


def key_tick():
    t = tt(0.03)
    return filt(rng.standard_normal(len(t)), lo=2500, hi=9000) * np.exp(-t * 160)


def click():
    t = tt(0.05)
    return filt(rng.standard_normal(len(t)), lo=1500, hi=6000) * np.exp(-t * 120) + 0.3 * np.sin(2 * np.pi * 2400 * t) * np.exp(-t * 200)


def brass(ms, dur):
    t = tt(dur)
    out = np.zeros_like(t)
    for m in ms:
        for det in (-0.004, 0.0, 0.004):
            ph = 2 * np.pi * hz(m) * (1 + det) * t
            out += sum(np.sin(ph * k) / k for k in range(1, 9))
    a = np.minimum(1, t / 0.035)
    out = filt(out * a, hi=2800) * np.exp(-t * 1.6)
    return out / (len(ms) * 3) * env(len(t), 0.01, 0.25)


def harmonium(ms, dur, attack=0.12):
    """Soft reed pad (a picture-house harmonium): odd-heavy partials, slow attack, gentle bellows tremolo."""
    t = tt(dur)
    out = np.zeros_like(t)
    for m in ms:
        f = hz(m)
        for k, amp in ((1, 1.0), (2, 0.35), (3, 0.5), (5, 0.18)):
            out += amp * np.sin(2 * np.pi * f * k * t + rng.uniform(0, 6.28))
    out = filt(out, hi=2200) * (1 + 0.12 * np.sin(2 * np.pi * 5.2 * t))
    return out / (len(ms) * 2) * env(len(t), attack, 0.4)


def projector(dur):
    """Film projector: a ~18 Hz shutter clatter over a low motor hum."""
    t = tt(dur)
    shutter = (np.sin(2 * np.pi * 18 * t) > 0.6).astype(float)
    shutter = filt(shutter, hi=400)
    clatter = filt(rng.standard_normal(len(t)), lo=900, hi=4000) * (0.4 + shutter)
    hum = 0.35 * np.sin(2 * np.pi * 50 * t) + 0.15 * np.sin(2 * np.pi * 100 * t)
    return (clatter * 0.5 + hum * 0.6) * env(len(t), 0.05, 0.05)


# chord voicings (midi): Db major family
DB, BBM, GB, AB = [61, 65, 68], [58, 61, 65], [58, 61, 66], [60, 63, 68]
ROOT = {'Db': 37, 'Bbm': 34, 'Gb': 42, 'Ab': 44}
PROG = [('Db', DB), ('Bbm', BBM), ('Gb', GB), ('Ab', AB)]


def groove(b0, b1, drums=True, comp=True, walk=True, vel=1.0):
    """Shuffle from beat b0 to b1: bar = 4 beats, one chord per 2 beats."""
    b = b0
    while b < b1 - 1e-9:
        bar_i = int((b - s_cmd) // 2) % 4
        name, ch = PROG[bar_i]
        r = ROOT[name]
        # bass: root on the beat, fifth on the next
        if walk:
            add('band', T(b), bass(r, T(0.9)), 0.55 * vel)
            if b + 1 < b1:
                add('band', T(b + 1), bass(r + 7 if r < 40 else r - 5, T(0.9)), 0.45 * vel)
        if comp:
            # swung offbeat stabs
            for off in (0.66, 1.66):
                if b + off < b1:
                    add('band', T(b + off), chord(ch, 0.35, 0.7), 0.32 * vel, pan=0.15)
        if drums:
            add('drums', T(b), kick(), 0.6 * vel)
            if b + 1 < b1:
                add('drums', T(b + 1), rim(), 0.22 * vel, pan=0.1)
            for k in range(2):
                for off in (0.0, 0.66):
                    if b + k + off < b1:
                        add('drums', T(b + k + off), shaker(), (0.07 if off else 0.05) * vel, pan=-0.3)
        b += 2


# ---------- hook: projector + lonely Bb-minor piano, message blips pile up ----------
add('fx', T(s_hook), projector(T(s_turn)), 0.11)
for at, m in ((0.0, 70), (0.66, 73), (1.33, 72), (2.0, 70), (2.66, 68), (3.33, 65), (4.0, 66)):
    add('band', T(s_hook + at), piano(m, 1.0, 0.8), 0.28, pan=-0.1)
add('band', T(s_hook), bass(34, T(2)), 0.35)
add('band', T(s_hook + 2.66), bass(30, T(2)), 0.32)
H_MSG = [x * 2 for x in (0.15, 0.45, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)]
for k, at in enumerate(H_MSG):
    m = 79 + k
    add('fx', T(at), blip(m), 0.16 + 0.015 * k, pan=(-0.4, 0.4)[k % 2])
    add('fx', T(at) + 0.05, blip(m + 5), 0.1 + 0.01 * k, pan=(0.4, -0.4)[k % 2])

# ---------- turn: marquee bulbs chase, piano roll up into Db ----------
for q in range(8):
    add('fx', T(s_turn + q * 0.25), key_tick(), 0.22, pan=(-0.5, 0.5)[q % 2])
add('band', T(s_turn), chord([49, 56, 61, 65], 1.5, 1.0), 0.6)
roll = [61, 65, 68, 73, 77, 80, 85]
for k, m in enumerate(roll):
    add('band', T(s_turn + 0.25 + k * 0.2), piano(m, 0.9, 0.8), 0.25, pan=-0.2 + k * 0.07)
add('drums', T(s_turn), kick(), 0.5)
add('drums', T(s_turn + 1.75), snare_hit(0.5), 0.3)

# ---------- command: full shuffle, typing, send, poll chime ----------
groove(s_cmd, s_res)
C_T, C_SEND, C_POLL = 0.25, 2.5, 3.0
n_keys = 26
for k in range(n_keys):
    at = C_T + (C_SEND - C_T - 0.2) * (k / n_keys) ** 0.9
    add('fx', T(s_cmd + at) + rng.uniform(-0.01, 0.01), key_tick(), 0.18 * rng.uniform(0.7, 1.0), pan=rng.uniform(-0.3, 0.3))
add('fx', T(s_cmd + C_SEND), blip(84, 0.1), 0.25)
add('fx', T(s_cmd + C_POLL), chime(85), 0.22)
add('fx', T(s_cmd + C_POLL) + 0.11, chime(89), 0.2)
for at, m in ((0.0, 77), (1.0, 80), (2.0, 82), (3.33, 85), (4.0, 84), (5.0, 80)):
    add('band', T(s_cmd + at), piano(m, 0.8, 0.8), 0.18, pan=-0.25)

# ---------- voting: two clicks, 12 climbing pops ----------
add('fx', T(s_vote + 1.0), click(), 0.4)
add('fx', T(s_vote + 1.5), click(), 0.4)
penta = [73, 75, 77, 80, 82, 85, 87, 89, 92, 94, 97, 99]   # Db pentatonic climbing
for k in range(12):
    add('fx', T(s_vote + 1.5 + k * 0.25), pop(penta[k]), 0.2, pan=(-0.35, 0.35)[k % 2])

# ---------- results: snare roll, poll ends on a bell, brass ta-da with cymbal ----------
add('band', T(s_res), bass(44, T(1.75)), 0.4)
add('band', T(s_res), chord(AB, 1.6, 0.7), 0.3)
roll_n = 28
for k in range(roll_n):
    at = s_res + 0.25 + k * (1.5 / roll_n)
    add('drums', T(at), snare_hit(0.35 + 0.65 * k / roll_n), 0.22)
add('fx', T(s_res + 0.75), chime(80, 1.2), 0.25)
add('band', T(s_res + 1.75), brass([49, 61, 65, 68, 73], T(3)), 0.85)
add('drums', T(s_res + 1.75), cymbal(2.2), 0.3)
add('drums', T(s_res + 1.75), kick(), 0.7)
add('fx', T(s_res + 1.75) + 0.08, chime(85), 0.2)
add('fx', T(s_res + 1.75) + 0.2, chime(89), 0.18)
add('fx', T(s_res + 1.75) + 0.32, chime(92), 0.16)
groove(s_res + 2, s_fact, drums=True, comp=True)

# ---------- fact: bass and piano only ----------
for k, (name, ch) in enumerate(PROG[:2]):
    add('band', T(s_fact + k * 2), bass(ROOT[name], T(1.8)), 0.45)
    add('band', T(s_fact + k * 2), chord(ch, 1.8, 0.7), 0.3)
for at, m in ((0.5, 80), (1.5, 77), (2.5, 82), (3.25, 80)):
    add('band', T(s_fact + at), piano(m, 0.9, 0.7), 0.2)
add('band', T(s_fact), harmonium([61, 65, 68], T(2) + 0.1), 0.22)
add('band', T(s_fact + 2), harmonium([58, 61, 65], T(2) + 0.1), 0.22)
for k in range(8):
    add('drums', T(s_fact + k * 0.5 + (0.16 if k % 2 else 0)), shaker(), 0.06, pan=-0.3)

# ---------- end: Dbmaj9 rings, arpeggio to the last frame ----------
end_len = DUR - T(s_end)
add('band', T(s_end), chord([37, 49, 56, 60, 63, 65], end_len, 1.0, spread=0.02), 0.75)
add('band', T(s_end), bass(37, T(3)), 0.5)
add('drums', T(s_end), kick(), 0.6)
add('drums', T(s_end), cymbal(2.5), 0.15)
arp = [73, 77, 80, 84, 87, 84, 80, 77, 73, 80]
for k in range(10):
    add('band', T(s_end + 0.5 + k * 0.5), piano(arp[k], 1.0, 0.7), 0.22, pan=(-0.3, 0.3)[k % 2])
add('band', T(s_end + 3), chord([49, 56, 60, 63, 68], end_len - T(3), 0.6, spread=0.03), 0.35)
add('band', T(s_end), harmonium([49, 56, 60, 63, 65], end_len, attack=0.05), 0.3)
add('band', T(s_end), harmonium([37], end_len, attack=0.05), 0.25)
for k in range(12):
    add('drums', T(s_end + k * 0.5 + (0.16 if k % 2 else 0)), shaker(), 0.05, pan=-0.3)

# ---------- mix ----------
mix = buses['band'] * 0.9 + buses['drums'] * 0.8 + buses['fx'] * 0.95
fade = int(0.3 * SR)
mix[-fade:] *= np.linspace(1, 0.6, fade)[:, None]
mix /= max(1e-9, np.max(np.abs(mix))) / 0.8

out = sys.argv[1] if len(sys.argv) > 1 else 'sound.wav'
with wave.open(out, 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype('<i2').tobytes())
print(f'wrote {out}: {DUR:.3f}s, shots start at {np.round(starts_b * BEAT, 3).tolist()}')
