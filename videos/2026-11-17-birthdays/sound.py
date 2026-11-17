#!/usr/bin/env python3
"""Original sound for the birthdays video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

A 128.57 BPM (28 video frames a beat) music-box groove in F major that starts as a sparse, slightly sad D-minor
music box in the hook ("Her birthday was yesterday.").
Every shot in index.html is a whole number of beats, so cuts land on the grid:
hook 0-8, turn 8-11, setup 11-20, member 20-26, result 26-36, fact 36-40, end 40-47 (beats).
The idea: the hook is a lone music box over a soft Dm - Bb pad, each chat message a small felt pop. The turn
("Save it once. Kaivor remembers.") is a match strike and a candle flame catching, then a bright celesta chord.
Setup and member run the full groove (F - Dm - Bb - C): music-box eighths, a round plucked bass, a dry kick, claps on
2 and 4 and a shaker; typing is soft key ticks, each send a blip and each Kaivor reply a three-note music-box chime.
The result holds its breath for a beat on a ticking clock, the 13 page tears off as a paper rip, and the birthday
message lands with a short synthesised party horn and a double clap as the groove comes back in; the role is a
bright pop. The fact shot strips back to pad and a slow music box; the end card rings out on Fmaj9 with an arpeggio
to the last frame.
"""
import sys, wave
import numpy as np
from scipy.signal import lfilter, butter

SR = 48000
FPS = 60
BEAT_FRAMES = 28
BEAT = BEAT_FRAMES / FPS                                      # 0.4667 s
SHOT_BEATS = [8, 3, 9, 6, 10, 4, 7]
frames = [round(b * BEAT_FRAMES) for b in SHOT_BEATS]         # same rounding as tools/capture.mjs
starts_b = np.concatenate([[0], np.cumsum(SHOT_BEATS)])
DUR = sum(frames) / FPS
rng = np.random.default_rng(1117)
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
def mbox(m, dur=0.9, bright=1.0):
    """Music box tine: a sine with inharmonic upper partials that die fast, and a tiny pluck click."""
    t = tt(dur)
    f = hz(m)
    x = np.sin(2 * np.pi * f * t)
    x += 0.35 * bright * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t / 0.09)
    x += 0.18 * bright * np.sin(2 * np.pi * f * 5.40 * t) * np.exp(-t / 0.025)
    e = np.exp(-t / (dur * 0.38)) * np.minimum(1, t / 0.0015)
    return x * env_end(e)


def celesta(notes, dur, gain=0.1):
    t = tt(dur)
    x = np.zeros(len(t))
    for m in notes:
        f = hz(m)
        x += np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.2)
    e = np.exp(-t / (dur * 0.5)) * np.minimum(1, t / 0.003)
    return x / len(notes) * env_end(e, 40) * gain * 2


def pluck_bass(m, dur):
    t = tt(dur)
    f = hz(m)
    x = np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.05)
    e = np.exp(-t / 0.24) * np.minimum(1, t / 0.004)
    return filt(x, hi=1000) * env_end(e)


def pad(notes, dur, gain=0.05, attack=0.25):
    t = tt(dur)
    x = np.zeros(len(t))
    for m in notes:
        for det in (-0.06, 0.06):
            x += np.sin(2 * np.pi * hz(m + det) * t + rng.random() * 6)
            x += 0.25 * np.sin(2 * np.pi * hz(m + det) * 2 * t)
    e = np.minimum(1, t / attack)
    r = min(int(0.3 * SR), len(t) // 3)
    e[-r:] *= np.linspace(1, 0, r)
    return x / len(notes) * e * gain


def kick(dur=0.28):
    t = tt(dur)
    f = 50 + 90 * np.exp(-t / 0.028)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.11)


def clap(dur=0.2):
    t = tt(dur)
    n = filt(rng.standard_normal(len(t)), lo=1000, hi=5200)
    e = np.zeros(len(t))
    for k, d in enumerate((0.0, 0.008, 0.017)):
        i = int(d * SR)
        e[i:] += np.exp(-(t[: len(t) - i]) / (0.007 if k < 2 else 0.05))
    return n * e * 0.6


def shaker(dur=0.06, accent=False):
    t = tt(dur)
    n = filt(rng.standard_normal(len(t)), lo=5500, hi=12000)
    return n * np.sin(np.pi * np.minimum(1, t / dur)) ** (1.5 if accent else 3)


def pop(m=84, dur=0.12):
    t = tt(dur)
    f = hz(m) * (1 + 0.6 * np.exp(-t / 0.01))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.03)


def felt(m=72, dur=0.16):
    """A soft, low message pop for the hook."""
    t = tt(dur)
    return filt(np.sin(2 * np.pi * hz(m) * t) * np.exp(-t / 0.04), hi=2500)


def tick(dur=0.018):
    t = tt(dur)
    return filt(rng.standard_normal(len(t)), lo=2500, hi=9000) * np.exp(-t / 0.003)


def clock_tick(dur=0.03, hi=True):
    t = tt(dur)
    x = filt(rng.standard_normal(len(t)), lo=1800 if hi else 1200, hi=6000) * np.exp(-t / 0.004)
    return x + 0.4 * np.sin(2 * np.pi * (2200 if hi else 1700) * t) * np.exp(-t / 0.006)


def match_strike(dur=0.5):
    """A scratch, then the head catching: a quick noise rasp into a soft swelling hiss."""
    t = tt(dur)
    n = rng.standard_normal(len(t))
    rasp = filt(n, lo=2000, hi=9000) * (np.abs(filt(rng.standard_normal(len(t)), hi=60)) * 6) * (t < 0.11)
    catch = filt(n, lo=600, hi=4000) * np.exp(-(t - 0.12).clip(0) / 0.12) * (t >= 0.1) * 0.6
    return (rasp * 0.7 + catch) * env_end(np.ones(len(t)))


def flame(dur=0.6):
    """The wick catching: a low, round 'fwoop' of filtered noise that swells and settles."""
    t = tt(dur)
    n = filt(rng.standard_normal(len(t)), lo=120, hi=900)
    e = np.sin(np.pi * np.minimum(1, t / 0.18) / 2) * np.exp(-(t - 0.18).clip(0) / 0.2)
    return n * env_end(e, 40)


def paper_rip(dur=0.36):
    """Tearing paper: band-passed noise chopped by random crackle bursts, rising in pitch."""
    t = tt(dur)
    n = rng.standard_normal(len(t))
    out = np.zeros(len(t))
    chunks = 12
    for k in range(chunks):
        a, b = k * len(t) // chunks, (k + 1) * len(t) // chunks
        c = 900 * (3.5 ** (k / chunks))
        out[a:b] = filt(n[a:b], lo=c * 0.6, hi=min(c * 3, 16000))
    crackle = (rng.random(len(t)) < 0.004).astype(float)
    crackle = lfilter([1.0], [1.0, -0.995], crackle)
    e = (0.35 + np.clip(crackle, 0, 3)) * np.minimum(1, t / 0.01) * np.minimum(1, (dur - t) / 0.06)
    return out * e


def party_horn(dur=0.62):
    """A short paper party horn: a buzzy reed tone that jumps up and wobbles, through a nasal band-pass."""
    t = tt(dur)
    f = hz(67) * (1 + 0.5 * np.minimum(1, t / 0.07)) * (1 + 0.012 * np.sin(2 * np.pi * 7 * t))
    ph = np.cumsum(f) / SR % 1.0
    x = np.where(ph < 0.3, 1.0, -0.43)
    x = filt(x - x.mean(), lo=500, hi=3200) + 0.25 * filt(rng.standard_normal(len(t)), lo=2000, hi=6000)
    e = np.minimum(1, t / 0.02) * np.minimum(1, (dur - t) / 0.08)
    return x * e


def sub(dur=0.9, m=29):
    t = tt(dur)
    return np.sin(2 * np.pi * hz(m) * t) * np.exp(-t / 0.5) * np.minimum(1, t / 0.005)


# ---------- arrangement ----------
s_hook, s_turn, s_set, s_mem, s_res, s_fact, s_end, s_out = starts_b

# F - Dm - Bb - C
prog = [(41, [65, 69, 72]), (38, [62, 65, 69]), (34, [62, 65, 70]), (36, [64, 67, 72])]
ARP = [0, 1, 2, 1, 2, 3, 2, 1]                                 # music-box eighths through the chord (+ octave)


def groove(b0, b1, start_bar=0, drums=True, full=True):
    b, bar = b0, start_bar
    while b < b1 - 1e-9:
        root, notes = prog[bar % 4]
        length = min(4, b1 - b)
        add('band', T(b), pad(notes, T(length), gain=0.035, attack=0.04))
        tones = notes + [notes[0] + 12]
        for s in range(int(length * 2)):
            m = tones[ARP[s % 8]] + 12
            add('band', T(b + s * 0.5), mbox(m, 0.6, bright=0.9), 0.09 if s % 2 == 0 else 0.065, pan=(-0.25, 0.25)[s % 2])
        for at, iv in ((0, 0), (1.5, 0), (2, 7), (3, 12)):
            if at < length:
                add('band', T(b + at), pluck_bass(root + iv, T(min(0.9, length - at))), 0.5)
        if drums:
            for k in (0, 2):
                if k < length:
                    add('drums', T(b + k), kick(), 0.5 if k == 0 else 0.38)
            for c in (1, 3):
                if c < length:
                    add('drums', T(b + c), clap(), 0.3)
            if full:
                for s in range(int(length * 4)):
                    add('drums', T(b + s * 0.25), shaker(accent=s % 2 == 1), 0.04 if s % 2 == 0 else 0.07, pan=0.35)
        b += 4
        bar += 1


# hook: lone music box over Dm - Bb, slow and a little sad; a felt pop for each message
add('band', T(0), pad([50, 57, 62, 65], T(4), gain=0.08, attack=0.3))
add('band', T(4), pad([46, 53, 62, 65], T(4.2), gain=0.08, attack=0.2))
add('band', T(0), pluck_bass(38, T(2)), 0.45)
add('band', T(4), pluck_bass(34, T(2)), 0.45)
hook_tune = [(0, 81), (1, 77), (1.5, 74), (2, 76), (3, 77), (4, 74), (5, 70), (5.5, 72), (6, 74), (7, 69)]
for at, m in hook_tune:
    add('band', T(at), mbox(m, 1.1), 0.19, pan=-0.1)
for at, m in ((1, 72), (2.5, 69), (4, 67)):
    add('fx', T(at), felt(m), 0.3)

# turn: match strike on the cut, the wick catches on beat 8.75, a bright celesta chord
add('fx', T(s_turn) - 0.02, match_strike(), 0.35)
add('fx', T(s_turn + 0.75) - 0.05, flame(), 0.5)
add('band', T(s_turn + 0.75), celesta([65, 69, 72, 76, 79], T(2.4), gain=0.12))
add('fx', T(s_turn + 0.75), sub(1.0, 29), 0.4)

# setup + member: the full groove
groove(s_set, s_mem, start_bar=0)
groove(s_mem, s_res, start_bar=1)


def typing_ticks(b0, b1, chars):
    for k in range(chars):
        at = T(b0) + (T(b1) - T(b0) - 0.08) * (k / chars) ** 0.95 + rng.uniform(-0.008, 0.008)
        add('fx', at, tick(), 0.24, pan=rng.uniform(-0.2, 0.2))


def reply_chime(b, notes=(84, 88, 91)):
    for j, m in enumerate(notes):
        add('fx', T(b + j * 0.25), mbox(m, 0.8, bright=1.1), 0.15, pan=-0.2 + 0.2 * j)


typing_ticks(s_set + 0.25, s_set + 1.75, 26)
add('fx', T(s_set + 1.75), pop(90, 0.09), 0.18)
reply_chime(s_set + 2.25)
typing_ticks(s_set + 3, s_set + 4.5, 24)
add('fx', T(s_set + 4.5), pop(90, 0.09), 0.18)
reply_chime(s_set + 5, (86, 89, 93))
typing_ticks(s_mem + 0.5, s_mem + 2.5, 22)
add('fx', T(s_mem + 2.5), pop(90, 0.09), 0.18)
reply_chime(s_mem + 3, (84, 88, 93))

# result: a clock ticks for two beats, the page tears on beat 27, the birthday lands on 28 with a horn
for k in range(4):
    add('fx', T(s_res + k * 0.5), clock_tick(hi=k % 2 == 0), 0.3, pan=0.1)
add('band', T(s_res), pad([60, 65, 67, 70], T(2), gain=0.04, attack=0.05))
add('fx', T(s_res + 1) - 0.03, paper_rip(), 0.42, pan=0.25)
add('fx', T(s_res + 2), party_horn(), 0.2, pan=-0.15)
add('drums', T(s_res + 2), clap(), 0.36)
add('drums', T(s_res + 2) + 0.06, clap(), 0.26)
add('fx', T(s_res + 2), sub(0.8, 29), 0.35)
add('band', T(s_res + 2), celesta([69, 72, 77, 81], T(2), gain=0.09))
groove(s_res + 2, s_fact, start_bar=0)
add('fx', T(s_res + 2.5), pop(93, 0.1), 0.22)
add('fx', T(s_res + 4), pop(86), 0.16)
add('fx', T(s_res + 6), pop(88), 0.16)

# fact: strip back to pad, bass and a slow music box
add('band', T(s_fact), pad([46, 58, 62, 65, 69], T(4), gain=0.06, attack=0.05))
add('band', T(s_fact), pluck_bass(34, T(1.6)), 0.55)
add('drums', T(s_fact), kick(), 0.45)
for at, m in ((0, 81), (1, 77), (1.5, 74), (2.5, 77), (3, 79)):
    add('band', T(s_fact + at), mbox(m, 1.0), 0.12)

# end: Fmaj9 rings, a music-box arpeggio to the last frame
end_len = DUR - T(s_end)
add('band', T(s_end), pad([41, 53, 57, 60, 64, 67], end_len, gain=0.07, attack=0.05))
add('band', T(s_end), celesta([65, 69, 72, 76, 79], T(3.5), gain=0.1))
add('band', T(s_end), pluck_bass(41, T(1.8)), 0.55)
add('band', T(s_end + 4), pluck_bass(41, T(1.6)), 0.4)
add('drums', T(s_end), kick(), 0.5)
add('drums', T(s_end + 1), clap(), 0.2)
add('drums', T(s_end + 3), clap(), 0.2)
arp = [77, 81, 84, 88, 91, 88, 84, 81]
for k in range(13):
    add('band', T(s_end + 0.5 + k * 0.5), mbox(arp[k % len(arp)], 0.9), 0.1, pan=(-0.3, 0.3)[k % 2])

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
