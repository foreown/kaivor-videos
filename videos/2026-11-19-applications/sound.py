#!/usr/bin/env python3
"""Original sound for the applications video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

A 90 BPM (40 video frames a beat) pizzicato-strings and upright-bass groove in D major (D - Bm - G - A), with
brushes on a snare and a swung ride. Every shot in index.html is a whole number of beats, so cuts land on the grid:
hook 0-4, turn 4-6, create 6-10, panel 10-13, modal 13-17, review 17-22, result 22-25, fact 25-28, end 28-32 (beats).
The idea: the hook is a lonely B-minor pizzicato line over a cold pad, with a DM "send" for each of mira's three
messages that gets quieter every time, then nothing back but a low hum. The turn cuts to the empty checkbox on a low
pluck, and a quick bright pen scratch draws the tick, landing on a plucked D-major chord and a harp-like
run up. From the create shot the full groove runs: walking upright bass, brushed snare swishes on 2 and 4, a soft
kick, offbeat pizzicato chords. Typing is soft key ticks, a send is a blip, Kaivor's replies a two-note glass
chime. The pop-up opens with a paper whoosh; the Apply, Submit and Approve presses are mouse clicks. Approval lands
a warm string swell and a rising three-note chime; the DM gets a soft ding. The fact shot strips back to bass and
pad; the end card rings out on Dmaj9 with a pizzicato arpeggio to the last frame.
"""
import sys, wave
import numpy as np
from scipy.signal import lfilter, butter

SR = 48000
FPS = 60
BEAT_FRAMES = 40
BEAT = BEAT_FRAMES / FPS                                      # 0.6667 s
SHOT_BEATS = [4, 2, 4, 3, 4, 5, 3, 3, 4]
frames = [round(b * BEAT_FRAMES) for b in SHOT_BEATS]         # same rounding as tools/capture.mjs
starts_b = np.concatenate([[0], np.cumsum(SHOT_BEATS)])
s_hook, s_turn, s_make, s_panel, s_modal, s_rev, s_res, s_fact, s_end, s_total = starts_b
DUR = sum(frames) / FPS
rng = np.random.default_rng(1119)
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


def env_end(e, ms=20):
    r = min(int(ms / 1000 * SR), len(e))
    e[-r:] *= np.linspace(1, 0, r)
    return e


# ---------- instruments ----------
def pizz(m, dur=0.5, bright=1.0):
    """Plucked string: harmonics that die faster the higher they are, a finger click, a wooden body resonance."""
    t = tt(dur)
    f = hz(m)
    x = np.zeros(len(t))
    for k in range(1, 9):
        if f * k > 9000:
            break
        x += (1.0 / k ** 1.3) * np.sin(2 * np.pi * f * k * t + rng.random() * 0.3) * np.exp(-t * (6 + 9 * k) / bright)
    click = filt(rng.standard_normal(len(t)), lo=1500, hi=5000) * np.exp(-t / 0.004) * 0.25
    body = filt(x, lo=180, hi=1400) * 0.6
    e = np.minimum(1, t / 0.002)
    return env_end((x + body + click) * e, 30)


def upright(m, dur):
    """Upright bass: a round fundamental with a thumpy attack that slides a hair down into pitch."""
    t = tt(dur)
    f = hz(m) * (1 + 0.025 * np.exp(-t / 0.02))
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph)
    e = np.exp(-t / 0.42) * np.minimum(1, t / 0.004)
    thump = filt(rng.standard_normal(len(t)), lo=80, hi=400) * np.exp(-t / 0.012) * 0.3
    return env_end(x * e + thump, 40)


def pad(notes, dur, gain=0.05, attack=0.3, hi=2000):
    t = tt(dur)
    x = np.zeros(len(t))
    for m in notes:
        for det in (-0.06, 0.06):
            ph = rng.random() * 6
            x += np.sin(2 * np.pi * hz(m + det) * t + ph) + 0.25 * np.sin(2 * np.pi * hz(m + det) * 2 * t + ph)
    x = filt(x, hi=hi)
    e = np.minimum(1, t / attack)
    r = min(int(0.4 * SR), len(t) // 3)
    e[-r:] *= np.linspace(1, 0, r)
    return x / len(notes) * e * gain


def swell(notes, dur, gain=0.1):
    """Bowed-string swell: detuned saws with vibrato through a low-pass, rising then easing off."""
    t = tt(dur)
    x = np.zeros(len(t))
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.2 * t)
    for m in notes:
        for det in (-0.09, 0.0, 0.09):
            f = hz(m + det) * vib
            x += 2 * ((np.cumsum(f) / SR + rng.random()) % 1.0) - 1
    x = filt(x / len(notes) / 3, hi=2600, order=2)
    e = np.minimum(1, t / 0.18) * np.exp(-np.maximum(0, t - 0.25) / (dur * 0.6))
    return env_end(e, 80) * x * gain


def kick(dur=0.28):
    t = tt(dur)
    f = 44 + 80 * np.exp(-t / 0.025)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR)
    return x * np.exp(-t / 0.11) * np.minimum(1, t / 0.001)


def brush_swish(dur=0.32):
    """A brush dragged across a snare: band-passed noise that swells in and fades."""
    t = tt(dur)
    x = filt(rng.standard_normal(len(t)), lo=1800, hi=9000)
    e = np.minimum(1, t / 0.07) * np.exp(-np.maximum(0, t - 0.07) / 0.09)
    return env_end(x * e, 20)


def brush_tap(dur=0.09):
    t = tt(dur)
    x = filt(rng.standard_normal(len(t)), lo=2500, hi=10000)
    return x * np.exp(-t / 0.018) * np.minimum(1, t / 0.0008)


def ride(dur=0.5):
    t = tt(dur)
    x = np.zeros(len(t))
    for f in (3150, 4270, 5530, 6810, 8130):
        x += np.sin(2 * np.pi * f * t + rng.random() * 6)
    x = x / 5 + 0.3 * filt(rng.standard_normal(len(t)), lo=6000)
    return x * np.exp(-t / 0.16) * np.minimum(1, t / 0.0008)


def key_tick():
    t = tt(0.03)
    x = filt(rng.standard_normal(len(t)), lo=2000, hi=7000)
    return x * np.exp(-t / 0.006) * (0.6 + 0.4 * rng.random())


def mouse_click():
    t = tt(0.05)
    x = filt(rng.standard_normal(len(t)), lo=1200, hi=6000) * np.exp(-t / 0.004)
    x += 0.5 * np.sin(2 * np.pi * 2300 * t) * np.exp(-t / 0.006)
    return x


def blip(f=1250, dur=0.09):
    t = tt(dur)
    fr = f * (1 + 0.5 * np.exp(-t / 0.015))
    return np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t / 0.03) * np.minimum(1, t / 0.002)


def glass(m, dur=0.9):
    t = tt(dur)
    f = hz(m)
    x = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t / 0.08) + 0.15 * np.sin(2 * np.pi * f * 5.4 * t) * np.exp(-t / 0.04)
    return env_end(x * np.exp(-t / 0.32) * np.minimum(1, t / 0.002), 40)


def reply_chime(beat):
    add('fx', T(beat), glass(86, 0.7), 0.13, pan=-0.15)
    add('fx', T(beat) + 0.09, glass(93, 0.8), 0.1, pan=0.15)


def pen(dur, bright=1.0):
    """A felt pen on paper: grainy noise with a wobbling band, for drawing the box and the tick."""
    t = tt(dur)
    x = rng.standard_normal(len(t))
    grain = 0.6 + 0.4 * np.sign(np.sin(2 * np.pi * (38 + 20 * rng.random()) * t))
    x = filt(x * grain, lo=900 * bright, hi=5200 * bright)
    e = np.minimum(1, t / 0.02) * np.minimum(1, (dur - t) / 0.03)
    return x * np.clip(e, 0, 1)


def whoosh(dur=0.35, up=True):
    t = tt(dur)
    x = rng.standard_normal(len(t))
    fc = np.linspace(500, 3200, len(t)) if up else np.linspace(3200, 500, len(t))
    y = np.zeros(len(t))
    acc = 0.0
    a = np.exp(-2 * np.pi * fc / SR)
    for i in range(len(t)):
        acc = (1 - a[i]) * x[i] + a[i] * acc
        y[i] = acc
    e = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 2
    return filt(y, lo=200) * e


def ding(m=88):
    t = tt(1.2)
    f = hz(m)
    x = np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * f * 2 * t) * np.exp(-t / 0.2) + 0.2 * np.sin(2 * np.pi * f * 3.01 * t) * np.exp(-t / 0.1)
    return env_end(x * np.exp(-t / 0.4) * np.minimum(1, t / 0.003), 60)


# ---------- harmony: D - Bm - G - A, two beats a chord ----------
CHORDS = [
    ([50, 57, 62, 66, 69], 38, [38, 42, 45, 47]),   # D
    ([47, 54, 59, 62, 66], 35, [35, 38, 42, 40]),   # Bm
    ([43, 50, 55, 59, 62], 31, [31, 35, 38, 40]),   # G
    ([45, 52, 57, 61, 64], 33, [33, 37, 40, 42]),   # A
]


def groove(b0, b1, brushes=True, ride_on=True, chords=True, first_chord=0):
    """Full band from beat b0 to b1: walking bass, brushes, kick, swung ride, offbeat pizzicato chords."""
    b = b0
    while b < b1 - 1e-6:
        k = (int(round(b - s_make)) // 2 + first_chord) % 4
        notes, root, walk = CHORDS[k]
        beat_in_chord = int(round(b - s_make)) % 2
        # walking bass, one note a beat
        add('band', T(b), upright(walk[beat_in_chord * 2 + 0] if beat_in_chord == 0 else walk[2], T(1)), 0.5)
        # kick on the chord, brushes on the other beat
        if beat_in_chord == 0:
            add('drums', T(b), kick(), 0.4)
        elif brushes:
            add('drums', T(b) - 0.06, brush_swish(), 0.13, pan=0.1)
        if brushes:
            add('drums', T(b + 0.67), brush_tap(), 0.07, pan=-0.2)   # swung eighth
        if ride_on:
            add('drums', T(b), ride(), 0.045, pan=0.35)
            add('drums', T(b + 0.67), ride(0.25), 0.028, pan=0.35)
        if chords:
            # offbeat pizzicato chord (swung "and")
            for j, m in enumerate(notes[2:]):
                add('band', T(b + 0.67) + j * 0.008, pizz(m + 12, 0.35, 0.8), 0.05, pan=(-0.25, 0.0, 0.25)[j])
        b += 1


# ---------- hook: lonely Bm pizzicato, three fading sends, a hum ----------
add('band', T(s_hook), pad([47, 54, 59, 62], T(4), gain=0.05, attack=0.2, hi=1400))
for at, m in ((0.0, 71), (0.5, 74), (1.0, 78), (1.5, 76), (2.0, 74), (2.5, 71), (3.0, 69), (3.5, 66)):
    add('band', T(s_hook + at), pizz(m, 0.6), 0.16 if at < 2.75 else 0.1, pan=(-0.2, 0.2)[int(at * 2) % 2])
add('band', T(s_hook), upright(35, T(2)), 0.45)
add('band', T(s_hook + 2), upright(31, T(2)), 0.4)
for i, at in enumerate((0.5, 1.25, 2.0)):
    add('fx', T(s_hook + at), whoosh(0.16), 0.12 * (1 - 0.25 * i))
    add('fx', T(s_hook + at) + 0.12, blip(1100 - 120 * i), 0.18 * (1 - 0.25 * i))
add('fx', T(s_hook + 2.75), pad([35, 47], T(1.25), gain=0.06, attack=0.1, hi=500))

# ---------- turn: pen draws the box, the tick, a plucked D chord and a run up ----------
add('band', T(s_turn), pizz(50, 0.5), 0.14)
add('drums', T(s_turn), brush_tap(), 0.08)
add('fx', T(s_turn + 0.75), pen(0.13, 1.4), 0.16, pan=0.1)
add('drums', T(s_turn + 1), kick(), 0.45)
for j, m in enumerate([50, 57, 62, 66, 69, 74]):
    add('band', T(s_turn + 1) + j * 0.012, pizz(m, 0.9, 1.2), 0.12)
add('band', T(s_turn + 1), upright(38, T(1)), 0.5)
for j, m in enumerate([74, 76, 78, 81, 83, 86, 88, 90]):
    add('band', T(s_turn + 1.25) + j * 0.06, pizz(m, 0.4, 1.4), 0.07, pan=-0.3 + 0.08 * j)

# ---------- create: groove, typing, send, reply ----------
groove(s_make, s_panel)
for k in range(int((T(2) - T(0.15) - 0.08) / 0.055)):
    add('fx', T(s_make + 0.15) + k * 0.055 + rng.random() * 0.012, key_tick(), 0.09, pan=0.1)
add('fx', T(s_make + 2), blip(1250), 0.16)
reply_chime(s_make + 2.5)

# ---------- panel: groove, a click on Apply ----------
groove(s_panel, s_modal)
add('fx', T(s_panel + 1.75), mouse_click(), 0.3)

# ---------- modal: whoosh in, three answers typed, submit, sent ----------
add('fx', T(s_modal) - 0.05, whoosh(0.3), 0.14)
groove(s_modal, s_rev, brushes=True, ride_on=False)
for a, b in ((0.25, 1.25), (1.375, 2.0), (2.125, 2.375)):
    n = int((T(b) - T(a)) / 0.05)
    for k in range(max(2, n)):
        add('fx', T(s_modal + a) + k * 0.05 + rng.random() * 0.01, key_tick(), 0.08, pan=-0.1)
add('fx', T(s_modal + 2.75), mouse_click(), 0.3)
reply_chime(s_modal + 3)

# ---------- review: groove, click Approve, swell and a rising chime ----------
groove(s_rev, s_rev + 2)
add('fx', T(s_rev + 2), mouse_click(), 0.3)
add('drums', T(s_rev + 2.25), kick(), 0.5)
add('band', T(s_rev + 2.25), swell([50, 57, 62, 66, 69, 74], T(2.6), gain=0.12))
add('band', T(s_rev + 2.25), upright(38, T(1.5)), 0.5)
for j, m in enumerate((81, 86, 90)):
    add('fx', T(s_rev + 2.25) + j * 0.1, glass(m, 1.0), 0.11)
groove(s_rev + 3, s_res, first_chord=3)
reply_chime(s_rev + 2.5)

# ---------- result: lighter groove, DM ding ----------
groove(s_res, s_fact, ride_on=False, first_chord=0)
add('fx', T(s_res + 0.75), ding(88), 0.16)
add('fx', T(s_res + 0.75) + 0.13, ding(93), 0.1)

# ---------- fact: bass and pad ----------
add('band', T(s_fact), pad([43, 50, 55, 59, 62], T(1.6), gain=0.06, attack=0.05))
add('band', T(s_fact + 1.5), pad([45, 52, 57, 61, 64], T(1.6), gain=0.06, attack=0.05))
add('band', T(s_fact), upright(31, T(1.5)), 0.5)
add('band', T(s_fact + 1.5), upright(33, T(1.5)), 0.45)
add('drums', T(s_fact), kick(), 0.4)
add('drums', T(s_fact + 1) - 0.06, brush_swish(), 0.12)
add('drums', T(s_fact + 2.5) - 0.06, brush_swish(), 0.1)
for at, m in ((0.5, 78), (1.0, 74), (1.75, 76), (2.25, 73)):
    add('band', T(s_fact + at), pizz(m, 0.45), 0.1)

# ---------- end: Dmaj9 rings, pizzicato arpeggio to the last frame ----------
end_len = DUR - T(s_end)
add('band', T(s_end), pad([38, 50, 57, 61, 64, 66], end_len, gain=0.07, attack=0.05))
add('band', T(s_end), swell([50, 57, 61, 64, 66], T(2.5), gain=0.08))
add('band', T(s_end), upright(38, T(2)), 0.55)
add('band', T(s_end + 2), upright(38, T(2)), 0.4)
add('drums', T(s_end), kick(), 0.5)
add('drums', T(s_end + 1) - 0.06, brush_swish(), 0.12)
add('drums', T(s_end + 3) - 0.06, brush_swish(), 0.1)
arp = [74, 78, 81, 85, 88, 85, 81, 78]
for k in range(8):
    add('band', T(s_end + 0.25 + k * 0.5), pizz(arp[k] , 0.6), 0.1, pan=(-0.3, 0.3)[k % 2])

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
