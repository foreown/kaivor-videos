#!/usr/bin/env python3
"""Original sound for the Server Tag Rewards video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

A 120 BPM (30 video frames a beat) steel-pan calypso in C major (C - Am - F - G), with a round plucked bass, a soft
kick on every beat, a woodblock on the "and" of two and four, a cowbell clave and a shaker. Every shot in index.html is
a whole number of beats, so cuts land on the grid: hook 0-6, turn 6-8, commands 8-16, sync 16-22, live 22-28,
fact 28-32, end 32-38 (beats).
The idea: the hook is a lonely muted pan line in A minor over a shaker, and every NOVA tag that pops onto the member
list is a small enamel "ting", each a little higher; "Nobody noticed." lands on a dull low thud. The turn cuts to the
pin: a metal clink and a pan glissando up into C. From the commands shot the full calypso runs; typing is soft key
ticks, each send a blip, each Kaivor reply a two-note pan chime. In the sync shot the six names turning pink are six
pings climbing a C major pentatonic scale. In the live shot the switch is a click, the tag a ting, and Kaivor's
thank-you lands on a bright pan chord with a cowbell. The fact shot drops the drums ("Tag off, role off." is two
falling pan notes). The end card rings out on a rolled Cmaj9 to the last frame.
"""
import sys, wave
import numpy as np
from scipy.signal import lfilter, butter

SR = 48000
FPS = 60
BEAT_FRAMES = 30
BEAT = BEAT_FRAMES / FPS                                      # 0.5 s
SHOT_BEATS = [6, 2, 8, 6, 6, 4, 6]
frames = [round(b * BEAT_FRAMES) for b in SHOT_BEATS]         # same rounding as tools/capture.mjs
starts_b = np.concatenate([[0], np.cumsum(SHOT_BEATS)])
s_hook, s_turn, s_cmd, s_sync, s_live, s_fact, s_end, s_total = starts_b
DUR = sum(frames) / FPS
rng = np.random.default_rng(1121)
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
def pan(m, dur=0.9, vel=1.0, muted=False):
    """Steel pan: a fundamental with octave and twelfth partials tuned into it, a tiny pitch scoop on the strike and a
    slow beating on the octave partial (the 'woo' of a pan note)."""
    t = tt(dur)
    f = hz(m) * (1 + 0.012 * np.exp(-t * 60))
    ph = 2 * np.pi * np.cumsum(f) / SR
    dec = 3.2 if not muted else 9.0
    out = (np.sin(ph) * np.exp(-t * dec)
           + 0.55 * np.sin(2 * ph + 0.3) * np.exp(-t * dec * 1.5) * (1 + 0.35 * np.sin(2 * np.pi * 5.5 * t))
           + 0.22 * np.sin(3.01 * ph) * np.exp(-t * dec * 2.6)
           + 0.08 * np.sin(4.02 * ph) * np.exp(-t * dec * 4))
    strike = filt(rng.standard_normal(len(t)), lo=2500, hi=9000) * np.exp(-t * 260) * 0.25
    return (out * 0.5 + strike) * env(len(t), 0.002, 0.05) * vel


def roll(ms, dur, vel=0.6, rate=8):
    """A pan player's roll: the chord notes struck again and again at `rate` per beat."""
    n = int(dur * SR) + int(0.9 * SR)
    out = np.zeros(n)
    step = BEAT / rate
    k = 0
    while k * step < dur:
        for j, m in enumerate(ms):
            s = pan(m, 0.6, vel * (0.75 + 0.25 * rng.random()) / len(ms) ** 0.5)
            i = int((k * step + j * 0.008) * SR)
            out[i:i + len(s)] += s[: n - i]
        k += 1
    fade = np.ones(n)
    fade[int(dur * SR):] = np.exp(-np.arange(n - int(dur * SR)) / SR * 6)
    return out * fade


def bass(m, dur):
    t = tt(dur)
    f = hz(m)
    x = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t) + 0.1 * np.sin(6 * np.pi * f * t)
    return filt(x * np.exp(-t * 4.0), hi=800) * env(len(t), 0.003, 0.04)


def kick():
    t = tt(0.3)
    f = 48 + 85 * np.exp(-t * 34)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 12)


def woodblock():
    t = tt(0.09)
    return (np.sin(2 * np.pi * 980 * t) * 0.8 + np.sin(2 * np.pi * 1510 * t) * 0.3) * np.exp(-t * 60)


def cowbell():
    t = tt(0.3)
    x = np.sign(np.sin(2 * np.pi * 560 * t)) * 0.5 + np.sign(np.sin(2 * np.pi * 845 * t)) * 0.5
    return filt(x, lo=500, hi=4000) * np.exp(-t * 14) * 0.5


def shaker(vel=1.0):
    t = tt(0.07)
    return filt(rng.standard_normal(len(t)), lo=5000, hi=11000) * np.exp(-t * 55) * vel


def tick():
    t = tt(0.025)
    return filt(rng.standard_normal(len(t)), lo=2500, hi=8000) * np.exp(-t * 260) * 0.5


def blip(m=84):
    t = tt(0.1)
    return np.sin(2 * np.pi * hz(m) * t) * np.exp(-t * 40) * env(len(t), 0.002, 0.02)


def ting(m):
    """Small enamel ting: a bell-ish pair of inharmonic partials."""
    t = tt(0.5)
    f = hz(m)
    return (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t * 8)) * np.exp(-t * 7) * 0.5


def click():
    t = tt(0.03)
    return (filt(rng.standard_normal(len(t)), lo=1500, hi=6000) * 0.6 + np.sin(2 * np.pi * 2200 * t) * 0.4) * np.exp(-t * 220)


def thud():
    t = tt(0.5)
    f = 70 + 40 * np.exp(-t * 20)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7)


def clink():
    t = tt(0.6)
    x = sum(np.sin(2 * np.pi * f * t) * np.exp(-t * d) for f, d in ((1870, 7), (2930, 10), (4410, 14), (5980, 18)))
    return x * 0.3 + filt(rng.standard_normal(len(t)), lo=3000) * np.exp(-t * 120) * 0.3


# ---------- harmony ----------
C4, D4, E4, F4, G4, A4, B4 = 60, 62, 64, 65, 67, 69, 71
CHORDS = {  # pan voicings and bass roots
    'C': ([E4, G4, C4 + 12], 48), 'Am': ([E4, A4, C4 + 12], 45),
    'F': ([F4, A4, C4 + 12], 41), 'G': ([D4, G4, B4], 43),
}
PROG = ['C', 'Am', 'F', 'G']


def groove(b0, b1, drums=True, pan_on=True, bass_on=True):
    """Calypso bar pattern over beats b0..b1 (two beats a chord)."""
    for b in range(int(b0), int(b1)):
        ch = PROG[(b // 2) % 4]
        voices, root = CHORDS[ch]
        # pan comping: strum on the 'and' of each beat, plus a pickup on 'a' of the 2nd beat
        if pan_on:
            for off, v in ((0.5, 0.55), (0.75, 0.35) if b % 2 else (0.0, 0.0)):
                if v:
                    for j, m in enumerate(voices):
                        add('band', T(b + off) + j * 0.01, pan(m, 0.5, v / 1.6), pan=-0.25 + 0.25 * j)
        if bass_on:
            if b % 2 == 0:
                add('band', T(b), bass(root, BEAT * 1.4), 0.75)
            else:
                add('band', T(b + 0.5), bass(root + 7, BEAT * 0.5), 0.55)
        if drums:
            add('drums', T(b), kick(), 0.8)
            if b % 2 == 1:
                add('drums', T(b + 0.5), woodblock(), 0.35, pan=0.3)
            for q in range(4):
                add('drums', T(b + q * 0.25), shaker(0.45 if q % 2 else 0.25), pan=0.4)
            if b % 4 in (0, 3):
                add('drums', T(b + (0.75 if b % 4 == 0 else 0.0)), cowbell(), 0.22, pan=-0.35)


# ---------- the score ----------
# hook: lonely muted pan line in A minor, shaker only
hook_line = [(0.0, A4), (0.75, C4 + 12), (1.5, E4), (2.5, A4), (3.25, G4), (4.0, E4), (5.0, D4), (5.5, E4)]
for b, m in hook_line:
    add('band', T(s_hook + b), pan(m, 0.6, 0.55, muted=True), pan=-0.1)
for b in range(int(s_hook), int(s_turn)):
    for q in range(2):
        add('drums', T(b + q * 0.5 + 0.5 * (q == 0) * 0), shaker(0.25), pan=0.4)
for k, b in enumerate([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]):               # six NOVA tags pop in
    add('fx', T(s_hook + b), ting(84 + [0, 2, 4, 7, 9, 12][k]), 0.55, pan=-0.3 + 0.12 * k)
add('fx', T(s_hook + 3.5), thud(), 0.7)                               # "Nobody noticed."
add('band', T(s_hook + 3.5), bass(45, BEAT * 2), 0.6)

# turn: the pin clinks, a glissando climbs into C
add('fx', T(s_turn + 0.25), clink(), 0.6)
for k, m in enumerate([C4, D4, E4, G4, A4, C4 + 12, D4 + 12, E4 + 12]):
    add('band', T(s_turn) + k * 0.1, pan(m, 0.5, 0.45), pan=-0.4 + 0.1 * k)
add('band', T(s_turn + 1.5), bass(43, BEAT * 0.5), 0.6)

# commands, sync, live: full groove
groove(s_cmd, s_fact)
for a, b in ((s_cmd + 0.25, s_cmd + 2.0), (s_cmd + 3.25, s_cmd + 5.0), (s_sync + 0.25, s_sync + 1.5)):
    tb = a
    while tb < b - 0.05:
        add('fx', T(tb) + rng.uniform(-0.01, 0.01), tick(), 0.5, pan=rng.uniform(-0.2, 0.2))
        tb += 0.125 + rng.uniform(-0.02, 0.03)
for b in (s_cmd + 2.0, s_cmd + 5.0, s_sync + 1.5):                    # sends
    add('fx', T(b), blip(88), 0.4)
for b in (s_cmd + 2.5, s_cmd + 5.5, s_sync + 2.0):                    # Kaivor replies
    add('fx', T(b), pan(C4 + 24, 0.7, 0.5), pan=0.2)
    add('fx', T(b) + 0.09, pan(G4 + 12, 0.7, 0.45), pan=0.2)
for k, b in enumerate([2.5, 2.75, 3.0, 3.25, 3.5, 3.75]):            # names turn pink
    add('fx', T(s_sync + b), pan([C4, D4, E4, G4, A4, C4 + 12][k] + 12, 0.6, 0.5), pan=-0.3 + 0.12 * k)
add('fx', T(s_live + 1.5), click(), 0.6)                              # the switch
add('fx', T(s_live + 1.5) + 0.04, ting(91), 0.45)                     # tag appears
for j, m in enumerate([E4 + 12, G4 + 12, C4 + 24]):                   # Kaivor's thank-you
    add('fx', T(s_live + 2.5) + j * 0.03, pan(m, 1.0, 0.45), pan=-0.2 + 0.2 * j)
add('fx', T(s_live + 2.5), cowbell(), 0.3)

# fact: drums out, pan and bass, two falling notes
groove(s_fact, s_end, drums=False)
add('fx', T(s_fact), pan(G4 + 12, 0.7, 0.55))
add('fx', T(s_fact + 0.5), pan(C4 + 12, 0.9, 0.55))
for b in range(int(s_fact), int(s_end)):
    add('drums', T(b + 0.5), shaker(0.3), pan=0.4)

# end: rolled Cmaj9 to the last frame
end_dur = DUR - T(s_end)
add('band', T(s_end), roll([E4, G4, B4, D4 + 12], end_dur - 0.3, vel=0.5), 0.9)
add('band', T(s_end), bass(36, end_dur), 0.7)
add('drums', T(s_end), kick(), 0.8)
add('fx', T(s_end), pan(C4 + 24, 2.5, 0.4))

# ---------- mix ----------
mix = buses['band'] * 0.9 + buses['drums'] * 0.85 + buses['fx'] * 0.95
# a short room: two early reflections
for d, g in ((0.023, 0.18), (0.041, 0.12)):
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
