#!/usr/bin/env python3
"""Add a sound track to a finished (silent) Kaivor video, following VIDEO_RULES.md rule 3.

Usage: python3 tools/finish_audio.py silent.mp4 sound.wav out.mp4

- Two-pass loudnorm to -14 LUFS integrated, true peak -2 dBTP, then a limiter at -3 dBFS (AAC encoding adds up to ~1 dB of true peak).
- AAC 48 kHz stereo, 192 kb/s. The video stream is copied untouched (-c:v copy).
- The audio is padded or trimmed to exactly the video's length.
"""
import json, os, re, subprocess, sys

silent, wav, out = sys.argv[1:4]
dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                     "-of", "default=nw=1:nk=1", silent]).decode().strip())
TARGET = "I=-14:TP=-2:LRA=11"
pre = f"aresample=48000,aformat=channel_layouts=stereo,apad,atrim=0:{dur:.3f}"

# pass 1: measure
log = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", wav, "-af",
                      f"{pre},loudnorm={TARGET}:print_format=json", "-f", "null", "-"],
                     capture_output=True, text=True).stderr
m = json.loads(re.findall(r"\{[^{}]*\}", log)[-1])
measured = (f"measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:"
            f"measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")

# pass 2: apply, limit, mux
tmp = out[:-4] + ".part.mp4"
fade_out = max(0.0, dur - 0.08)
subprocess.check_call([
    "ffmpeg", "-y", "-v", "error", "-i", silent, "-i", wav,
    "-filter_complex",
    f"[1:a]{pre},loudnorm={TARGET}:{measured},alimiter=limit=0.708:level=false,"
    f"aresample=48000,afade=t=out:st={fade_out:.3f}:d=0.08[a]",
    "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
    "-t", f"{dur:.3f}", "-movflags", "+faststart", tmp])
os.replace(tmp, out)
print(out)
