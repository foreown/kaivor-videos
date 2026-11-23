#!/usr/bin/env python3
"""Original sound for the starboard video (all synthesised, no samples, no music licence needed).

Usage: python3 sound.py out.wav
Then:  python3 tools/finish_audio.py silent.mp4 out.wav final.mp4

A 150 BPM (24 video frames a beat) 8-bit chiptune in F major (F - Dm - Bb - C): two pulse-wave channels (a 12.5 %
sixteenth-note arpeggio and a 25 % lead), a stepped 4-bit triangle bass, and noise-channel drums (kick, snare, hats).
Every shot in index.html is a whole number of beats, so cuts land on the grid: hook 0-6, buried 6-10, turn 10-13,
command 13-20, stars 20-27, board 27-34, fact 34-40, end 40-48 (beats).
The idea: the hook is a sparse D-minor pulse line with echo; juno's message is a bright pop and the flood of eleven
messages is a scramble of chip blips at random pitches. "Buried" lands on a falling pitch slide and a low thud, then a
held, muffled D-minor chord. The turn is a power-up sweep into a sticker slap with a sparkle, then a drum fill opens
the F-major groove. Typing is soft noise ticks, the send a blip, Kaivor's reply a two-note coin chime. Each of the three
stars is a coin, climbing in pitch, the third one a longer 1-up. The board post is a level-up fanfare with a crash. The
fact shot drops the drums; the end card plays one more bar and rings out on a rolled Fmaj9 to the last frame.
"""
import sys, wave
import numpy as np
from scipy.signal import lfilter, butter

SR = 48000
FPS = 60
BEAT_FRAMES = 24
BEAT = BEAT_FRAMES / FPS                                      # 0.4 s
SHOT_BEATS = [6, 4, 3, 7, 7, 7, 6, 8]
frames = [round(b * BEAT_FRAMES) for b in SHOT_BEATS]         # same rounding as tools/capture.mjs
starts_b = np.concatenate([[0], np.cumsum(SHOT_BEATS)])
s_hook, s_buried, s_turn, s_cmd, s_stars, s_board, s_fact, s_end, s_total = starts_b
DUR = sum(frames) / FPS
rng = np.random.default_rng(1123)
n_all = int(SR * DUR) + 1
buses = {k: np.zeros((n_all, 2)) for k in ('band', 'muff', 'drums', 'fx', 'echo')}


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


def adsr(n, a=0.003, d=0.06, s=0.6, r=0.03):
    e = np.full(n, s)
    na, nd, nr = max(1, int(a * SR)), max(1, int(d * SR)), max(1, int(r * SR))
    e[:na] = np.linspace(0, 1, na)
    e[na:na + nd] = np.linspace(1, s, len(e[na:na + nd]))
    e[-nr:] *= np.linspace(1, 0, nr)
    return e


# ---------- chip instruments ----------
def pulse(m, dur, duty=0.25, vel=1.0, s=0.55, d=0.08, slide=0.0, vib=0.0):
    """NES-style pulse wave. slide: semitones of pitch glide over the note; vib: vibrato depth in semitones."""
    t = tt(dur)
    semis = m + slide * (t / max(dur, 1e-6)) + vib * np.sin(2 * np.pi * 6 * t) * np.clip((t - 0.08) / 0.1, 0, 1)
    ph = np.cumsum(440.0 * 2 ** ((semis - 69) / 12)) / SR
    w = np.where((ph % 1.0) < duty, 1.0, -1.0)
    # 4-bit volume steps, like the hardware
    e = np.round(adsr(len(t), d=d, s=s) * 15) / 15
    return w * e * vel * 0.32


def tri(m, dur, vel=1.0):
    """Stepped 4-bit triangle (bass channel)."""
    t = tt(dur)
    ph = (t * hz(m)) % 1.0
    w = np.abs(ph * 4 - 2) - 1
    w = np.round(w * 7.5) / 7.5
    e = adsr(len(t), a=0.002, d=0.02, s=0.9, r=0.02)
    return w * e * vel * 0.55


def noise(dur, period=1, vel=1.0, decay=20.0):
    n = int(dur * SR)
    raw = rng.choice([-1.0, 1.0], size=n // period + 1)
    w = np.repeat(raw, period)[:n]
    e = np.exp(-np.arange(n) / SR * decay)
    return w * e * vel


def kick():
    t = tt(0.16)
    f = 150 * np.exp(-t * 28) + 42
    ph = np.cumsum(f) / SR
    w = np.abs((ph % 1.0) * 4 - 2) - 1
    return w * np.exp(-t * 16) * 0.9


def snare():
    t = tt(0.16)
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30) * 0.4
    return tone + noise(0.16, period=2, vel=0.5, decay=24)


def hat(open_=False):
    return filt(noise(0.12 if open_ else 0.04, period=1, vel=0.28, decay=14 if open_ else 70), lo=6000)


def crash():
    return filt(noise(1.2, period=1, vel=0.32, decay=3.2), lo=3500)


def blip(m=84, vel=1.0):
    return pulse(m, 0.06, duty=0.5, vel=0.8 * vel, s=0.4, d=0.03)


def coin(m1, m2, long=False, vel=1.0):
    a = pulse(m1, 0.07, duty=0.5, vel=vel, s=1.0, d=0.01)
    b = pulse(m2, 0.42 if long else 0.24, duty=0.5, vel=vel, s=0.5, d=0.18)
    return np.concatenate([a, b])


def keytick():
    return filt(noise(0.02, period=1, vel=0.22, decay=180), lo=2500)


def thud():
    t = tt(0.4)
    f = 90 * np.exp(-t * 9) + 38
    ph = np.cumsum(f) / SR
    w = np.abs((ph % 1.0) * 4 - 2) - 1
    return w * np.exp(-t * 7) * 1.0 + noise(0.4, period=8, vel=0.2, decay=14)


def slap():
    """Sticker slap: a short low noise burst with a tonal body."""
    t = tt(0.12)
    body = np.sin(2 * np.pi * 220 * t * (1 - t * 2)) * np.exp(-t * 30)
    return filt(noise(0.12, period=3, vel=0.7, decay=40), hi=3500) + body * 0.6


def sparkle(at, base=88):
    for k, m in enumerate([0, 4, 7, 12, 16, 19]):
        add('fx', at + k * 0.035, pulse(base + m, 0.09, duty=0.125, vel=0.45, s=0.3, d=0.05), pan=0.5 - 0.2 * k)


def sweep(at, dur, m0, m1):
    steps = int(dur / 0.025)
    for k in range(steps):
        m = m0 + (m1 - m0) * k / steps
        add('fx', at + k * 0.025, pulse(round(m), 0.03, duty=0.25, vel=0.55, s=0.8, d=0.01))


# ---------- groove ----------
F, Dm, Bb, C = 53, 50, 46, 48            # roots (F3, D3, Bb2, C3)
CHORDS = {F: [0, 4, 7], Dm: [0, 3, 7], Bb: [0, 4, 7], C: [0, 4, 7]}
PROG = [F, Dm, Bb, C]
LEAD = {  # per bar index in the groove: (beat offset, midi, beats)
    0: [(0, 77, 1), (1, 76, 0.5), (1.5, 74, 0.5), (2, 72, 1.5)],
    1: [(0, 74, 1), (1, 72, 0.5), (1.5, 69, 0.5), (2, 72, 2)],
    2: [(0, 70, 1), (1, 72, 0.5), (1.5, 74, 0.5), (2, 77, 1.5)],
    3: [(0, 76, 1.5), (1.5, 74, 0.5), (2, 72, 2)],
}


def groove(b0, b1, drums=True, lead=True, arp_vel=0.5, bar0=0):
    b = b0
    bar = bar0
    while b < b1 - 1e-9:
        root = PROG[bar % 4]
        tones = [root + 12 + x for x in CHORDS[root]]
        bar_end = min(b + 4, b1)
        # bass: eighths, octave jump on the off-beats
        e = 0.0
        while b + e < bar_end - 1e-9:
            m = root - 12 + (12 if int(e * 2) % 2 else 0)
            add('band', T(b + e), tri(m, T(0.45)), 0.9)
            e += 0.5
        # arp: sixteenths, 12.5 % pulse
        s = 0.0
        k = 0
        while b + s < bar_end - 1e-9:
            m = tones[k % 3] + (12 if (k // 3) % 2 else 0)
            add('echo', T(b + s), pulse(m + 12, T(0.24), duty=0.125, vel=arp_vel, s=0.35, d=0.05), pan=0.35 if k % 2 else -0.35)
            s += 0.25
            k += 1
        if lead:
            for off, m, ln in LEAD[bar % 4]:
                if b + off < bar_end - 1e-9:
                    add('echo', T(b + off), pulse(m, T(min(ln, bar_end - b - off)) - 0.01, duty=0.25, vel=0.62, s=0.6, d=0.1, vib=0.25), pan=-0.1)
        if drums:
            for q in range(int(round((bar_end - b) * 2))):
                at = T(b + q * 0.5)
                if q % 4 == 0:
                    add('drums', at, kick(), 0.95)
                if q % 4 == 2:
                    add('drums', at, snare(), 0.75)
                add('drums', at, hat(open_=(q % 8 == 7)), 0.8, pan=0.25)
                if q % 8 == 5:
                    add('drums', at, kick(), 0.6)
        b = bar_end
        bar += 1


# ---------- hook (beats 0-6): sparse D minor with echo, juno's pop, the flood ----------
hook_line = [(0, 62), (1, 65), (2, 69), (3, 65), (4, 62), (5, 60)]
for bb, m in hook_line:
    add('echo', T(s_hook + bb), pulse(m + 12, T(0.8), duty=0.125, vel=0.42, s=0.4, d=0.2), pan=-0.2)
    add('band', T(s_hook + bb), tri(38 if bb < 4 else 36, T(0.9)), 0.5)
for q in range(12):
    add('drums', T(s_hook + q * 0.5), hat(), 0.45, pan=0.3)
add('fx', T(s_hook + 0.5), coin(83, 88), 0.75)                         # juno's message
for i in range(11):                                                   # the flood, one every quarter beat
    m = int(rng.integers(76, 96))
    add('fx', T(s_hook + 2.5 + i * 0.25), blip(m, 0.8), pan=float(rng.uniform(-0.7, 0.7)))

# ---------- buried (beats 6-10) ----------
add('fx', T(s_buried), pulse(84, 0.5, duty=0.5, vel=0.7, s=0.8, d=0.05, slide=-30))
add('fx', T(s_buried) + 0.05, thud(), 0.9)
for m in (62, 65, 69):
    add('muff', T(s_buried + 0.25), pulse(m, T(3.5), duty=0.5, vel=0.35, s=0.7, d=0.2))
add('band', T(s_buried + 0.25), tri(38, T(3.5)), 0.45)
add('fx', T(s_buried + 2), blip(69, 0.5))                             # "Nobody scrolls up that far."

# ---------- turn (beats 10-13): power-up, sticker slap, fill ----------
sweep(T(s_turn), T(0.5), 60, 96)
add('fx', T(s_turn + 0.5), slap(), 0.9)
sparkle(T(s_turn + 0.55))
add('band', T(s_turn + 0.5), tri(41, T(1.4)), 0.7)
for m in (65, 69, 72):
    add('echo', T(s_turn + 0.5), pulse(m + 12, T(1.4), duty=0.25, vel=0.32, s=0.5, d=0.2))
for q in range(8):                                                    # snare fill into the groove
    add('drums', T(s_turn + 2 + q * 0.125), snare(), 0.35 + 0.06 * q)
add('drums', T(s_turn + 2), kick(), 0.8)

# ---------- command, stars, board (beats 13-34): the groove ----------
groove(s_cmd, s_fact, drums=True, lead=True, arp_vel=0.42)
add('drums', T(s_cmd), crash(), 0.6)
for k in range(24):                                                   # typing ,starboard set #starboard
    at = T(s_cmd + 0.5) + k * (T(2.5) - 0.1) / 24 * (0.6 + 0.8 * rng.random())
    add('fx', at, keytick(), 0.9, pan=float(rng.uniform(-0.3, 0.3)))
add('fx', T(s_cmd + 3), blip(79), 0.8)                                 # send
add('fx', T(s_cmd + 3.75), coin(88, 93), 0.7)                          # Kaivor's reply
add('fx', T(s_cmd + 5), blip(72, 0.4))                                 # "3 stars and it's in."
for i, (m1, m2) in enumerate([(83, 88), (86, 91), (88, 96)]):          # the three stars
    add('fx', T(s_stars + [1, 2.5, 4][i]), coin(m1, m2, long=(i == 2)), 0.85, pan=[-0.4, 0, 0.4][i])
sparkle(T(s_stars + 4.1), base=91)
fan = [(0, 72), (0.25, 76), (0.5, 79), (0.75, 84), (1.0, 88), (1.5, 84), (1.75, 88)]
for off, m in fan:                                                    # the board post: level-up fanfare
    add('fx', T(s_board + 0.5 + off), pulse(m, T(0.5 if off < 1.5 else 1.4), duty=0.5, vel=0.55, s=0.6, d=0.08, vib=0.2 if off >= 1.5 else 0))
add('drums', T(s_board + 0.5), crash(), 0.8)
add('fx', T(s_board + 3.5), blip(76, 0.4))                             # "Now everyone can find it."

# ---------- fact (beats 34-40): drums out ----------
groove(s_fact, s_end, drums=False, lead=False, arp_vel=0.38, bar0=1)
for k, at in enumerate([0, 2, 3, 4]):
    add('fx', T(s_fact + at), blip([72, 76, 79, 84][k], 0.45))

# ---------- end (beats 40-48): one more bar, then a rolled Fmaj9 ----------
groove(s_end, s_end + 4, drums=True, lead=True, arp_vel=0.42, bar0=0)
add('drums', T(s_end), crash(), 0.55)
ring = T(s_total - (s_end + 4)) + 0.1
for k, m in enumerate([53, 60, 64, 67, 69, 72, 76, 79]):
    add('echo', T(s_end + 4) + k * 0.05, pulse(m + 12 if m < 60 else m, ring - k * 0.05, duty=0.25 if k % 2 else 0.125,
                                               vel=0.36, s=0.55, d=0.3, vib=0.12), pan=-0.5 + k / 7)
add('band', T(s_end + 4), tri(41, ring), 0.8)
add('drums', T(s_end + 4), kick(), 0.9)
add('drums', T(s_end + 4), crash(), 0.6)

# ---------- mix ----------
def echo(x, delay=T(0.75), fb=0.32, mix=0.28):
    d = int(delay * SR)
    y = x.copy()
    for k in range(1, 4):
        sh = np.zeros_like(x)
        sh[d * k:] = x[:-d * k] * (fb ** k)
        # ping-pong
        if k % 2:
            sh = sh[:, ::-1]
        y += sh * mix / 0.32
    return y


band = filt(buses['band'], hi=9000)
echo_bus = filt(echo(buses['echo']), hi=8000)
muff = filt(buses['muff'], hi=900, order=4)
drums = buses['drums']
fx = filt(buses['fx'], hi=11000)
mix = band * 0.85 + echo_bus * 0.8 + muff * 0.9 + drums * 0.75 + fx * 0.9
mix = filt(mix, lo=30)
mix = filt(mix, hi=7000, order=4)                                       # round off the square edges (true peak)
mix /= max(1e-9, np.max(np.abs(mix))) / 0.85

out = sys.argv[1] if len(sys.argv) > 1 else 'sound.wav'
with wave.open(out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.clip(mix, -1, 1) * 32767).astype('<i2').tobytes())
print(out, f'{DUR:.3f}s')
