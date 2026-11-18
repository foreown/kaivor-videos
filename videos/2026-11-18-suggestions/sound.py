#!/usr/bin/env python3
"""Original sound for the suggestions video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

A 112.5 BPM (32 video frames a beat) clavinet funk groove in Ab major (Ab - Fm - Db - Eb).
Every shot in index.html is a whole number of beats, so cuts land on the grid:
hook 0-6, turn 6-9, post 9-15, votes 15-21, approve 21-26, result 26-32, fact 32-35, end 35-41 (beats).
The idea: the hook opens on one warm "idea" bell for mira's message over a soft F-minor pad, then a flood of
notification blips piles in faster and faster until it all drops into a falling whoosh and a low hum
("Nobody voted. Nobody answered."). The turn draws five tally strokes as five pencil scratches, then a bright
Ab chord stab and a short riser. Post and votes run the full groove: a muted clavinet with a wah-like filter, a
round sub bass, a dry kick, claps on 2 and 4 and swung hats; typing is soft key ticks, the send a blip, Kaivor's
reply a two-note marimba. Each vote is a woodblock that climbs the pentatonic scale (nine for, one lower thunk
against), and the cursor's press is a mouse click. Approve drops the hats while staff type; the result lands a
rubber-stamp thud with a brass-like Ab chord when the suggestion turns green, and a soft DM ding. The fact shot
strips back to pad and bass; the end card rings out on Abmaj9 with a clav arpeggio to the last frame.
"""
import sys, wave
import numpy as np
from scipy.signal import lfilter, butter

SR = 48000
FPS = 60
BEAT_FRAMES = 32
BEAT = BEAT_FRAMES / FPS                                      # 0.5333 s
SHOT_BEATS = [6, 3, 6, 6, 5, 6, 3, 6]
frames = [round(b * BEAT_FRAMES) for b in SHOT_BEATS]         # same rounding as tools/capture.mjs
starts_b = np.concatenate([[0], np.cumsum(SHOT_BEATS)])
s_hook, s_turn, s_post, s_vote, s_appr, s_res, s_fact, s_end, s_total = starts_b
DUR = sum(frames) / FPS
rng = np.random.default_rng(1118)
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


def sweep_lp(x, f0, f1, tau):
    """One-pole low-pass whose cutoff falls from f0 to f1 (a cheap wah / filter envelope)."""
    t = np.arange(len(x)) / SR
    fc = f1 + (f0 - f1) * np.exp(-t / tau)
    a = np.exp(-2 * np.pi * fc / SR)
    y = np.zeros_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = (1 - a[i]) * x[i] + a[i] * acc
        y[i] = acc
    return y


# ---------- instruments ----------
def clav(m, dur=0.22, bright=1.0):
    """Muted clavinet: a narrow pulse wave through a falling filter, short and percussive."""
    t = tt(dur)
    f = hz(m)
    ph = (f * t) % 1.0
    x = np.where(ph < 0.22, 1.0, -0.28) - 0.0
    x = sweep_lp(x, 4200 * bright, 500, 0.035)
    e = np.exp(-t / 0.08) * np.minimum(1, t / 0.0015)
    return filt(x, lo=120) * env_end(e)


def sub_bass(m, dur):
    t = tt(dur)
    f = hz(m)
    x = np.sin(2 * np.pi * f * t) + 0.22 * np.sin(2 * np.pi * 2 * f * t)
    e = np.exp(-t / 0.35) * np.minimum(1, t / 0.006)
    return env_end(e, 30) * x


def pad(notes, dur, gain=0.05, attack=0.25):
    t = tt(dur)
    x = np.zeros(len(t))
    for m in notes:
        for det in (-0.07, 0.07):
            ph = rng.random() * 6
            x += np.sin(2 * np.pi * hz(m + det) * t + ph) + 0.3 * np.sin(2 * np.pi * hz(m + det) * 3 * t + ph)
    x = filt(x, hi=2400)
    e = np.minimum(1, t / attack)
    r = min(int(0.35 * SR), len(t) // 3)
    e[-r:] *= np.linspace(1, 0, r)
    return x / len(notes) * e * gain


def brass(notes, dur, gain=0.12):
    """Saw stack with a filter that opens and settles: a soft synth-brass chord."""
    t = tt(dur)
    x = np.zeros(len(t))
    for m in notes:
        for det in (-0.08, 0.0, 0.08):
            x += 2 * ((hz(m + det) * t + rng.random()) % 1.0) - 1
    x = sweep_lp(x / len(notes) / 3, 3800, 900, 0.18)
    e = np.minimum(1, t / 0.012) * np.exp(-t / (dur * 0.7))
    return env_end(e, 60) * x * gain * 3


def kick(dur=0.3):
    t = tt(dur)
    f = 46 + 110 * np.exp(-t / 0.03)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR)
    return x * np.exp(-t / 0.13) * np.minimum(1, t / 0.001)


def clap(dur=0.22):
    t = tt(dur)
    n = rng.standard_normal(len(t))
    e = np.zeros(len(t))
    for d in (0.0, 0.009, 0.018):
        e += (t >= d) * np.exp(-np.maximum(0, t - d) / (0.006 if d < 0.018 else 0.07))
    return filt(n, lo=900, hi=5200) * e * 0.6


def hat(dur=0.05, open_=False):
    t = tt(0.18 if open_ else dur)
    n = rng.standard_normal(len(t))
    return filt(n, lo=7000) * np.exp(-t / (0.06 if open_ else 0.012)) * 0.5


def woodblock(m, dur=0.16):
    t = tt(dur)
    f = hz(m)
    x = np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * f * 2.9 * t) * np.exp(-t / 0.01)
    return x * np.exp(-t / 0.035) * np.minimum(1, t / 0.0008)


def marimba(m, dur=0.5):
    t = tt(dur)
    f = hz(m)
    x = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * f * 4 * t) * np.exp(-t / 0.03)
    return env_end(np.exp(-t / 0.18) * np.minimum(1, t / 0.002)) * x


def bell(m, dur=1.2):
    """Soft FM bell for the idea and the DM."""
    t = tt(dur)
    f = hz(m)
    x = np.sin(2 * np.pi * f * t + 1.6 * np.exp(-t / 0.25) * np.sin(2 * np.pi * f * 1.4 * t))
    return env_end(np.exp(-t / 0.45) * np.minimum(1, t / 0.002), 60) * x


def blip(m, dur=0.09):
    t = tt(dur)
    f = hz(m) * (1 + 0.5 * np.exp(-t / 0.01))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR)
    return x * np.exp(-t / 0.03) * np.minimum(1, t / 0.001)


def key_tick(dur=0.02):
    t = tt(dur)
    return filt(rng.standard_normal(len(t)), lo=2500, hi=8000) * np.exp(-t / 0.004)


def mouse_click():
    t = tt(0.04)
    x = filt(rng.standard_normal(len(t)), lo=1500, hi=6000) * np.exp(-t / 0.003)
    x[int(0.018 * SR):] += 0.6 * filt(rng.standard_normal(len(t) - int(0.018 * SR)), lo=1500, hi=6000) * np.exp(-t[: len(t) - int(0.018 * SR)] / 0.003)
    return x


def pencil(dur=0.15):
    """A pencil stroke: band-passed noise with a scratchy, jittering level."""
    t = tt(dur)
    n = filt(rng.standard_normal(len(t)), lo=1800, hi=6500)
    grain = 0.55 + 0.45 * np.abs(filt(rng.standard_normal(len(t)), hi=60) * 6).clip(0, 1)
    e = np.minimum(1, t / 0.01) * np.minimum(1, (dur - t) / 0.03).clip(0, 1)
    return n * grain * e * 0.7


def whoosh_down(dur=1.0):
    t = tt(dur)
    n = rng.standard_normal(len(t))
    lo = filt(n, lo=200, hi=3000)
    x = sweep_lp(lo, 3000, 150, dur / 3)
    e = np.minimum(1, t / 0.05) * np.exp(-t / (dur * 0.5))
    return x * e


def riser(dur):
    t = tt(dur)
    n = filt(rng.standard_normal(len(t)), lo=600, hi=9000)
    return n * (t / dur) ** 2.2 * 0.5


def stamp():
    """Rubber stamp: a low thump, a slap of noise and a wooden knock."""
    t = tt(0.4)
    f = 70 + 60 * np.exp(-t / 0.02)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.09)
    slap = filt(rng.standard_normal(len(t)), lo=400, hi=3500) * np.exp(-t / 0.012)
    return body + 0.6 * slap + 0.3 * woodblock(62, 0.4)


# ---------- groove ----------
CHORDS = [  # (bass root, clav voicing, pad voicing)
    (44, [60, 63, 68], [56, 60, 63, 67]),   # Ab(maj7)
    (41, [60, 65, 68], [53, 56, 60, 63]),   # Fm7
    (37, [61, 65, 68], [49, 56, 60, 65]),   # Db(maj7)
    (39, [58, 63, 67], [51, 55, 58, 63]),   # Eb
]
CLAV_RHYTHM = [(0, 1.0), (0.75, 0.7), (1.25, 0.6), (1.5, 0.9)]   # within each 2-beat chord
SWING = 0.08


def groove(b0, b1, drums=True, hats=True, clav_on=True, chord0=0):
    b = b0
    k = chord0
    while b < b1 - 1e-6:
        root, cv, _ = CHORDS[k % 4]
        span = min(2, b1 - b)
        add('band', T(b), sub_bass(root, T(min(1.5, span))), 0.5)
        if span > 1.4:
            add('band', T(b + 1.5), sub_bass(root + 12, T(0.4)), 0.25)
        if clav_on:
            for at, g in CLAV_RHYTHM:
                if at < span:
                    for j, m in enumerate(cv):
                        add('band', T(b + at) + j * 0.004, clav(m, 0.2, 0.7 + 0.3 * g), 0.07 * g, pan=(-0.35, 0, 0.35)[j])
        if drums:
            for q in range(int(round(span))):
                add('drums', T(b + q), kick(), 0.55 if q == 0 else 0.4)
                if (b + q) % 2 == 1:
                    add('drums', T(b + q), clap(), 0.3)
                if hats:
                    add('drums', T(b + q + 0.5) + SWING * BEAT, hat(), 0.18, pan=0.3)
                    add('drums', T(b + q + 0.25), hat(), 0.08, pan=0.3)
                    add('drums', T(b + q + 0.75) + SWING * BEAT, hat(), 0.08, pan=0.3)
            if span >= 2:
                add('drums', T(b + 1.75), kick(), 0.25)
        b += 2
        k += 1


def typing_ticks(b0, b1, chars):
    for k in range(chars):
        q = k / max(1, chars - 1)
        # same ease-in-out as the page's typing()
        w = 8 * q ** 4 if q < 0.5 else 1 - (-2 * q + 2) ** 4 / 2
        add('fx', T(b0) + w * (T(b1) - T(b0) - 0.08), key_tick(), 0.18 + 0.08 * rng.random(), pan=rng.uniform(-0.2, 0.2))


def reply_chime(b, notes=(80, 87)):
    add('fx', T(b), marimba(notes[0]), 0.2)
    add('fx', T(b) + 0.09, marimba(notes[1]), 0.2)


# ---------- hook: one idea, buried by a flood of blips ----------
add('band', 0, pad([53, 56, 60, 65], T(4.5), gain=0.05, attack=0.15))
add('fx', 0.02, bell(80, 1.6), 0.22)                         # mira's idea
add('band', 0.02, sub_bass(41, T(1)), 0.35)
flood = [1, 1.5, 2, 2.5, 3, 3.5, 4]
for k, at in enumerate(flood):
    add('fx', T(at), blip(84 + (k * 5) % 12, 0.09), 0.2, pan=rng.uniform(-0.6, 0.6))
    add('fx', T(at) + 0.06, blip(91 + (k * 7) % 9, 0.07), 0.1, pan=rng.uniform(-0.6, 0.6))
    for extra in range(k // 2):                               # it gets busier
        add('fx', T(at + 0.25) + extra * 0.04, blip(86 + (extra * 3 + k) % 10, 0.06), 0.08, pan=rng.uniform(-0.8, 0.8))
for q in range(1, 4):
    add('drums', T(q), kick(), 0.3)
    add('drums', T(q + 0.5), hat(), 0.08)
add('fx', T(4.2), whoosh_down(1.1), 0.35)
add('band', T(4.5), pad([41, 53, 56], T(1.6), gain=0.05, attack=0.05))   # the low hum under "Nobody voted."

# ---------- turn: five tally strokes, then the stab ----------
for i in range(5):
    add('fx', T(s_turn + 0.1 + i * 0.25), pencil(0.13 if i < 4 else 0.2), 0.35, pan=(-0.3 + 0.15 * i))
add('band', T(s_turn), brass([56, 60, 63, 68], T(1.4), gain=0.1))
add('band', T(s_turn), sub_bass(44, T(1.2)), 0.5)
add('drums', T(s_turn), kick(), 0.55)
add('fx', T(s_turn + 1.5), riser(T(1.5)), 0.3)
add('drums', T(s_turn + 2.5), clap(), 0.2)
add('drums', T(s_turn + 2.75), clap(), 0.25)

# ---------- post: the full groove ----------
groove(s_post, s_vote)
typing_ticks(s_post + 0.25, s_post + 2.5, 36)
add('fx', T(s_post + 2.5), blip(79, 0.1), 0.2)
reply_chime(s_post + 3)

# ---------- votes: a woodblock per vote, climbing the scale ----------
groove(s_vote, s_appr, chord0=3)
add('fx', T(s_vote + 1) - 0.01, mouse_click(), 0.35)
add('fx', T(s_vote + 1.25), marimba(87, 0.3), 0.12)
penta = [68, 70, 72, 75, 77, 80, 82, 84, 87]
ups = [1, 1.5, 2.25, 2.75, 3.25, 3.5, 3.75, 4.25, 4.75]
for k, at in enumerate(ups):
    add('fx', T(s_vote + at), woodblock(penta[k] + 12), 0.26, pan=(-0.25, 0.25)[k % 2])
add('fx', T(s_vote + 3), woodblock(56), 0.3)                  # the one vote against

# ---------- approve: staff type, hats out ----------
groove(s_appr, s_res, hats=False, chord0=2)
typing_ticks(s_appr + 0.25, s_appr + 2.25, 41)
add('fx', T(s_appr + 2.25), blip(79, 0.1), 0.2)
reply_chime(s_appr + 2.75)
add('fx', T(s_appr + 4), riser(T(1)), 0.22)

# ---------- result: stamp, brass, DM ding ----------
add('drums', T(s_res), kick(), 0.5)
add('band', T(s_res), clav(68, 0.2), 0.08)
add('fx', T(s_res + 0.75), stamp(), 0.5)
add('band', T(s_res + 0.75), brass([56, 60, 63, 67, 72], T(2.2), gain=0.11))
groove(s_res + 1, s_fact, chord0=1)
add('band', T(s_res + 0.75), sub_bass(44, T(0.25)), 0.4)
add('fx', T(s_res + 2.75), bell(87, 1.2), 0.18)
add('fx', T(s_res + 2.75) + 0.12, bell(92, 1.0), 0.1)

# ---------- fact: pad and bass ----------
add('band', T(s_fact), pad([49, 56, 60, 65, 68], T(3), gain=0.06, attack=0.05))
add('band', T(s_fact), sub_bass(37, T(1.4)), 0.5)
add('band', T(s_fact + 2), sub_bass(39, T(1)), 0.45)
add('drums', T(s_fact), kick(), 0.45)
add('drums', T(s_fact + 1), clap(), 0.18)
for at, m in ((0.5, 80), (1.0, 77), (1.75, 75), (2.5, 72)):
    add('band', T(s_fact + at), clav(m, 0.25, 0.6), 0.09)

# ---------- end: Abmaj9 rings, clav arpeggio to the last frame ----------
end_len = DUR - T(s_end)
add('band', T(s_end), pad([44, 56, 60, 63, 67, 70], end_len, gain=0.07, attack=0.05))
add('band', T(s_end), brass([56, 60, 63, 67, 70], T(2.5), gain=0.09))
add('band', T(s_end), sub_bass(44, T(1.8)), 0.55)
add('band', T(s_end + 4), sub_bass(44, T(1.6)), 0.4)
add('drums', T(s_end), kick(), 0.55)
add('drums', T(s_end + 1), clap(), 0.2)
add('drums', T(s_end + 3), clap(), 0.2)
add('drums', T(s_end + 5), clap(), 0.16)
arp = [68, 72, 75, 79, 82, 79, 75, 72]
for k in range(11):
    add('band', T(s_end + 0.5 + k * 0.5), clav(arp[k % len(arp)] + 12, 0.3, 0.8), 0.08, pan=(-0.3, 0.3)[k % 2])

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
