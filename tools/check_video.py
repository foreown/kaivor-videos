#!/usr/bin/env python3
"""Check a finished Kaivor video against VIDEO_RULES.md (format, sound, centring).

Usage: python3 tools/check_video.py out.mp4 [cover.jpg]

- Prints codec, profile, size, fps, frame count, duration, file size, colour tags and audio streams.
- Requires an AAC 48 kHz audio track at -14 LUFS integrated (+/-1) with true peak at or below -1.5 dBTP.
- Measures the horizontal centre of the content on ~12 frames: the bounding box of pixels that
  differ from that frame's own edge colour. It should land on ~540.
- With a second argument, exports the cover (a frame 1 s before the end) as JPEG.
Exits non-zero if a hard rule fails.
"""
import json, subprocess, sys
import numpy as np

src = sys.argv[1]
probe = json.loads(subprocess.check_output([
    "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", src]))
v = next(s for s in probe["streams"] if s["codec_type"] == "video")
audio = [s for s in probe["streams"] if s["codec_type"] == "audio"]
num, den = map(int, v["r_frame_rate"].split("/"))
fps = num / den
dur = float(probe["format"]["duration"])
size_mb = int(probe["format"]["size"]) / 1e6
print(f"codec={v['codec_name']} profile={v.get('profile')} {v['width']}x{v['height']} fps={fps:.2f} "
      f"frames={v.get('nb_read_frames')} duration={dur:.2f}s size={size_mb:.1f}MB pix_fmt={v['pix_fmt']} "
      f"colorspace={v.get('color_space')} audio_streams={len(audio)}")

fails = []
if (v["width"], v["height"]) != (1080, 1920): fails.append("not 1080x1920")
if abs(fps - 60) > 0.01: fails.append("not 60 fps")
if v["codec_name"] != "h264" or v.get("profile") != "High": fails.append("not H.264 High")
if v["pix_fmt"] != "yuv420p": fails.append("not yuv420p")
if v.get("color_space") != "bt709": fails.append("not tagged bt709")
if not audio:
    fails.append("no audio stream (every video needs sound)")
else:
    a = audio[0]
    if a["codec_name"] != "aac" or a.get("sample_rate") != "48000":
        fails.append("audio is not AAC 48 kHz")
    import re
    log = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", src, "-map", "0:a:0",
                          "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True).stderr
    summary = log[log.rfind("Summary:"):]
    lufs = float(re.search(r"I:\s+(-?[\d.]+) LUFS", summary).group(1))
    tp = float(re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", summary).group(1))
    print(f"audio: {a['codec_name']} {a.get('sample_rate')} Hz, integrated {lufs:.1f} LUFS, true peak {tp:.1f} dBTP")
    if abs(lufs + 14) > 1: fails.append(f"loudness {lufs:.1f} LUFS, should be -14")
    if tp > -1.5: fails.append(f"true peak {tp:.1f} dBTP, should be -2 or lower")
if size_mb > 30: fails.append("over 30 MB")
if not 15 <= dur <= 25: fails.append(f"duration {dur:.1f}s is outside ~18-22 s")

W, H = 1080, 1920
centres = []
for i in range(12):
    t = dur * (i + 0.5) / 12
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", src, "-frames:v", "1",
                                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    img = np.frombuffer(raw, np.uint8).reshape(H, W, 3).astype(int)
    edge = np.median(np.concatenate([img[:, :8].reshape(-1, 3), img[:, -8:].reshape(-1, 3)]), axis=0)
    diff = np.abs(img - edge).sum(axis=2) > 40
    cols = np.where(diff.sum(axis=0) > 3)[0]
    if len(cols) == 0:
        continue
    c = (cols[0] + cols[-1]) / 2
    centres.append((t, c))
    print(f"  t={t:5.2f}s content x {cols[0]}..{cols[-1]} centre={c:.1f}")
off = [abs(c - 540) for _, c in centres]
if off and max(off) > 12:
    fails.append(f"content centre off by up to {max(off):.0f}px")

if len(sys.argv) > 2:
    subprocess.check_call(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0, dur - 1):.3f}", "-i", src,
                           "-frames:v", "1", "-q:v", "2", sys.argv[2]])
    print(f"cover written to {sys.argv[2]}")

print("FAIL: " + "; ".join(fails) if fails else "PASS")
sys.exit(1 if fails else 0)
