#!/usr/bin/env python3
"""Original sound for the purge-user video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

A bright 130 BPM "spring clean" track that starts in B minor and opens up to D major once the server is clean.
Every shot in index.html is a whole number of beats, so cuts land on the grid:
hook 0-8, turn 8-11, action 11-21, result 21-32, fact 32-38, end 38-45 (beats).
The idea: the spam is heard as 18 short notification pings, two per beat, each a little higher, over a muted
plucked bass pulse and a ticking shaker, so the hook gets busier and more annoying. The turn ("Or one command.")
drops to silence, one dry low thud and a rising filtered sweep. The band arrives with the action shot: FM electric
bells on Bm / G / D / A, a round sine-and-square sub bass on the off-beats, a dry kick on every beat and finger
snaps on 2 and 4. Typing is soft key clicks, the send is a blip and Kaivor's reply is a three-note bell. In the
result the squeegee is a long rubbery swish (band-passed noise gliding down) with a bright pop on each of the six
channels as it clears, pitched up a D-major scale. The fact shot is half-time on D and G, and the end card
resolves on a Dmaj9 bell chord with a soft tail that reaches the last frame.
"""
import sys, wave
import numpy as np
from scipy.signal import lfilter, butter

SR = 48000
FPS = 60
BPM = 130
BEAT = 60 / BPM
SHOT_BEATS = [8, 3, 10, 11, 6, 7]
frames = [round(b * BEAT * FPS) for b in SHOT_BEATS]          # same rounding as tools/capture.mjs
starts = np.concatenate([[0], np.cumsum(frames)]) / FPS        # shot start times in seconds
DUR = starts[-1]
rng = np.random.default_rng(1114)
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


def bp(x, lo, hi):
    b, a = butter(2, [lo / (SR / 2), hi / (SR / 2)], btype='band')
    return lfilter(b, a, x)


def hz(m):
    return 440 * 2 ** ((m - 69) / 12)


def env(t, a, d, dur):
    return np.minimum(1, t / a) * np.exp(-t / d) * np.minimum(1, np.maximum(0, (dur - t) / 0.02))


# ---------- instruments ----------
def fm_bell(m, dur=1.2, index=2.2, decay=0.55):
    """Two-operator FM electric bell: carrier 1:1, modulator ratio 3.5, index falling with the envelope."""
    t = tt(dur)
    f = hz(m)
    e = env(t, 0.004, decay, dur)
    mod = index * np.exp(-t / 0.25) * np.sin(2 * np.pi * f * 3.5 * t)
    return np.sin(2 * np.pi * f * t + mod) * e


def chord(notes, dur, gain=0.18, **kw):
    return sum(fm_bell(n, dur, **kw) for n in notes) * gain


def pluck_bass(m, dur, cut=500):
    t = tt(dur)
    f = hz(m)
    sq = np.sign(np.sin(2 * np.pi * f * t))
    s = 0.7 * np.sin(2 * np.pi * f * t) + 0.25 * lp(sq, cut)
    return s * env(t, 0.004, 0.16, dur)


def kick():
    t = tt(0.32)
    f = 48 + 110 * np.exp(-t / 0.035)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-t / 0.11) + 0.15 * rng.standard_normal(len(t)) * np.exp(-t / 0.004)


def snap():
    t = tt(0.16)
    n = bp(rng.standard_normal(len(t)), 1800, 5200)
    return n * np.exp(-t / 0.028) * 1.4 + 0.25 * np.sin(2 * np.pi * 1400 * t) * np.exp(-t / 0.01)


def shaker(accent=1.0):
    t = tt(0.07)
    n = bp(rng.standard_normal(len(t)), 5000, 11000)
    return n * np.minimum(1, t / 0.012) * np.exp(-t / 0.022) * accent


def ping(m):
    """Notification ping for one spam message: a short two-partial blip."""
    t = tt(0.22)
    f = hz(m)
    s = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * f * 2.01 * t)
    return s * env(t, 0.002, 0.06, 0.22)


def thud():
    t = tt(0.5)
    f = 42 + 50 * np.exp(-t / 0.05)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.16)


def rise_sweep(dur):
    t = tt(dur)
    n = rng.standard_normal(len(t))
    out = np.zeros_like(n)
    seg = int(0.03 * SR)
    for i in range(0, len(n), seg):
        k = i / len(n)
        lo = 300 + 3000 * k ** 2
        out[i:i + seg] = bp(n[max(0, i - 2000):i + seg], lo, lo * 1.6)[-len(n[i:i + seg]):]
    return out * (t / dur) ** 2 * 0.8


def swish(dur):
    """Squeegee: band-passed noise gliding down with a rubbery wobble."""
    t = tt(dur)
    n = rng.standard_normal(len(t))
    out = np.zeros_like(n)
    seg = int(0.02 * SR)
    for i in range(0, len(n), seg):
        k = i / len(n)
        c = 2600 - 1700 * k + 180 * np.sin(2 * np.pi * 7 * i / SR)
        out[i:i + seg] = bp(n[max(0, i - 2000):i + seg], c * 0.8, c * 1.25)[-len(n[i:i + seg]):]
    shape = np.minimum(1, t / 0.15) * np.minimum(1, (dur - t) / 0.25)
    return out * shape * 0.55


def pop(m):
    t = tt(0.18)
    f = hz(m) * (1 + 0.6 * np.exp(-t / 0.012))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(t, 0.001, 0.05, 0.18)


def key():
    t = tt(0.035)
    return bp(rng.standard_normal(len(t)), 2500, 7000) * np.exp(-t / 0.006)


def blip():
    t = tt(0.12)
    f = 700 + 500 * (t / 0.12)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(t, 0.002, 0.04, 0.12)


# ---------- arrangement ----------
s_hook, s_turn, s_act, s_res, s_fact, s_end = starts[:6]

# hook: muted B pulse, shaker 16ths, 18 climbing pings (same times as SPAM_AT in index.html)
for k in range(32):
    add('band', s_hook + B(k * 0.25), pluck_bass(35, 0.2, cut=380), 0.22 if k % 2 == 0 else 0.12)
for k in range(32):
    add('drums', s_hook + B(k * 0.25), shaker(1.0 if k % 2 == 0 else 0.55), 0.22, pan=0.25)
for k in range(4):
    add('drums', s_hook + B(k * 2), kick(), 0.5)
spam_notes = [71, 74, 73, 76, 74, 78, 76, 79, 78, 81, 79, 83, 81, 83, 85, 83, 86, 86]
for k in range(18):
    add('fx', s_hook + B(0.9 + k * 0.25), ping(spam_notes[k]), 0.32, pan=(-0.5 if k % 2 else 0.5) * 0.6)
add('band', s_hook, chord([59, 62, 66], B(8), gain=0.07, decay=1.6))

# turn: cut, one thud, sweep up into the action
add('drums', s_turn, thud(), 0.9)
add('fx', s_turn + B(0.8), rise_sweep(B(2.2)), 0.5)

# band: Bm / G / D / A (2 beats each) through action and result
prog_minor = [[59, 62, 66, 69], [55, 59, 62, 66], [62, 66, 69, 73], [57, 61, 64, 69]]
roots_minor = [35, 31, 38, 33]
band_start, band_end = s_act, s_fact
nb = int(round((band_end - band_start) / BEAT))
for b in range(nb):
    at = band_start + B(b)
    ci = (b // 2) % 4
    if b % 2 == 0:
        add('band', at, chord(prog_minor[ci], B(2.2), gain=0.12, decay=0.6), pan=-0.1)
    add('band', at + B(0.5), pluck_bass(roots_minor[ci], B(0.45)), 0.42)
    add('band', at + B(0.75), pluck_bass(roots_minor[ci] + 12, B(0.2)), 0.18)
    add('drums', at, kick(), 0.62)
    if b % 2 == 1:
        add('drums', at, snap(), 0.42, pan=0.1)
    for h in range(2):
        add('drums', at + B(0.5 * h + 0.25), shaker(0.8), 0.16, pan=0.3)
# bell melody hook on top of the result (short motif, every 4 beats)
motif = [(0, 78), (0.5, 76), (1, 74), (1.5, 78), (2.5, 81)]
for rep in range(2):
    for off, m in motif:
        add('band', s_res + B(4 * rep + off + 0.0), fm_bell(m, 0.6, index=1.4, decay=0.25), 0.08, pan=0.2)

# action: typing clicks (beats 0.4..4.0), send blip at 4.4, reply bell at 5.3
for i, b in enumerate(np.arange(0.45, 4.0, 0.25)):
    jitter = rng.uniform(-0.05, 0.05)
    add('fx', s_act + B(b + jitter), key(), 0.35, pan=rng.uniform(-0.2, 0.2))
add('fx', s_act + B(4.4), blip(), 0.35)
for j, m in enumerate([74, 78, 81]):
    add('fx', s_act + B(5.3 + j * 0.25), fm_bell(m, 0.8, index=1.2, decay=0.3), 0.26)

# result: squeegee swish (WIPE_A = 1.6 beats, WIPE_D = 5.2 beats) and a pop per channel at CLEAR_AT
add('fx', s_res + B(1.6), swish(B(5.2)), 0.6)
pops = [74, 76, 78, 79, 81, 86]
for i in range(6):
    add('fx', s_res + B(1.6) + B(5.2) * (i + 0.5) / 6, pop(pops[i]), 0.42, pan=-0.3 + 0.12 * i)
add('fx', s_res + B(8.2), fm_bell(86, 1.0, index=1.0, decay=0.4), 0.16)

# fact: half time on D / G
prog_major = [[62, 66, 69, 73], [55, 59, 62, 66, 69]]
roots_major = [38, 31]
add('drums', s_fact, thud(), 0.6)
for b in range(6):
    at = s_fact + B(b)
    ci = (b // 3) % 2
    if b % 3 == 0:
        add('band', at, chord(prog_major[ci], B(3.2), gain=0.12, decay=0.9))
    add('band', at + B(0.5), pluck_bass(roots_major[ci], B(0.45)), 0.36)
    if b % 2 == 0:
        add('drums', at, kick(), 0.55)
    else:
        add('drums', at, snap(), 0.36)
    add('drums', at + B(0.5), shaker(0.7), 0.13, pan=0.3)

# end: Dmaj9 resolve, one more bar of light kick and shaker, tail to the end
add('band', s_end, chord([50, 57, 62, 66, 69, 76], DUR - s_end, gain=0.12, decay=3.0))
add('band', s_end, pluck_bass(38, B(2)), 0.4)
for b in range(7):
    if b < 4:
        add('drums', s_end + B(b), kick(), 0.4 if b < 3 else 0.25)
    add('drums', s_end + B(b + 0.5), shaker(0.6), 0.1, pan=0.3)
add('band', s_end + B(4), pluck_bass(38, B(3)), 0.25)
for j, m in enumerate([81, 85, 88]):
    add('fx', s_end + B(2.3 + j * 0.5), fm_bell(m, 1.4, index=1.0, decay=0.5), 0.1, pan=0.2 - 0.2 * j)

# ---------- mix ----------
mix = buses['band'] * 0.9 + buses['drums'] * 0.85 + buses['fx'] * 0.95
fade = int(0.25 * SR)
mix[-fade:] *= np.linspace(1, 0.4, fade)[:, None]
mix /= max(1e-9, np.max(np.abs(mix))) / 0.8

out = sys.argv[1] if len(sys.argv) > 1 else 'sound.wav'
with wave.open(out, 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype('<i2').tobytes())
print(f'wrote {out}: {DUR:.3f}s, shots start at {np.round(starts, 3).tolist()}')
