#!/usr/bin/env python3
"""Kaivor daily short-video renderer.

Usage: python3 render.py spec.json out.mp4

Renders a 1080x1920, 30 fps vertical video (~13 s) from a JSON spec:
{
  "tag": "ANTINUKE",                     # small pill label
  "hook": "Someone just nuked your server?",
  "sub": "Kaivor stops it in seconds.",
  "command": ",antinuke on",             # what the user types
  "reply_title": "Antinuke is on",
  "reply_lines": ["Mass bans blocked", "Rogue admins stripped", "Every action logged"],
  "cta": "Add Kaivor to your server"     # optional
}
All content is horizontally centered on the true center of the frame.
"""
import json, math, os, subprocess, sys, tempfile, shutil
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS = 1080, 1920, 30
CX = W // 2
HERE = os.path.dirname(os.path.abspath(__file__))
F = lambda name, size: ImageFont.truetype(os.path.join(HERE, "fonts", name), size)

NAVY = (8, 14, 34)
NAVY2 = (14, 28, 66)
AZURE = (46, 168, 255)
AZURE_DK = (24, 96, 196)
WHITE = (255, 255, 255)
MUTED = (160, 176, 205)
CARD = (22, 27, 44)
CARD2 = (30, 36, 58)
DISCORD_TEXT = (220, 224, 235)

S1, S2, S3 = 3.0, 7.0, 3.0  # scene lengths (s)


def ease_out(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def ease_back(t):
    t = max(0.0, min(1.0, t))
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def prog(t, start, dur):
    return max(0.0, min(1.0, (t - start) / dur))


_bg_cache = {}


def background(t):
    """Navy vertical gradient with a slowly drifting azure glow."""
    key = int(t * 10)  # glow moves smoothly enough at 10 steps/s
    if key in _bg_cache:
        return _bg_cache[key].copy()
    base = Image.new("RGB", (W, H), NAVY)
    grad = Image.linear_gradient("L").resize((W, H))
    base = Image.composite(Image.new("RGB", (W, H), NAVY2), base, grad.point(lambda v: v // 2))
    glow = Image.new("L", (W // 4, H // 4), 0)
    gd = ImageDraw.Draw(glow)
    gy = (H // 4) * (0.30 + 0.04 * math.sin(key / 10 * 0.9))
    gx = (W // 4) * (0.5 + 0.10 * math.sin(key / 10 * 0.6))
    r = 130
    gd.ellipse([gx - r, gy - r, gx + r, gy + r], fill=110)
    glow = glow.filter(ImageFilter.GaussianBlur(45)).resize((W, H))
    base = Image.composite(Image.new("RGB", (W, H), AZURE_DK), base, glow)
    _bg_cache[key] = base
    return base.copy()


def draw_logo(img, cx, cy, size, alpha=1.0):
    """Kaivor mark: navy rounded tile, white K with an azure diagonal stroke."""
    if size < 4 or alpha <= 0:
        return
    s = int(size)
    tile = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=int(s * 0.24), fill=(12, 22, 52, 255),
                        outline=(46, 168, 255, 255), width=max(2, s // 40))
    sw = s * 0.13
    x0, x1 = s * 0.30, s * 0.72
    top, bot, mid = s * 0.24, s * 0.76, s * 0.53
    d.rectangle([x0, top, x0 + sw, bot], fill=WHITE + (255,))
    # azure upper diagonal
    d.polygon([(x0 + sw, mid - sw * 0.2), (x1, top), (x1 + sw * 1.15, top), (x0 + sw * 1.1, mid + sw * 0.75)],
              fill=AZURE + (255,))
    # white lower leg
    d.polygon([(x0 + sw * 1.1, mid - sw * 0.25), (x1 + sw * 1.15, bot), (x1, bot), (x0 + sw * 0.6, mid + sw * 0.55)],
              fill=WHITE + (255,))
    if alpha < 1:
        a = tile.getchannel("A").point(lambda v: int(v * alpha))
        tile.putalpha(a)
    img.paste(tile, (int(cx - s / 2), int(cy - s / 2)), tile)


def wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if draw.textlength(test, font=font) <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def text_layer(size_xy):
    return Image.new("RGBA", size_xy, (0, 0, 0, 0))


def centered_text(img, text, font, y, fill, alpha=1.0, dy=0.0, max_w=900, spacing=1.12):
    """Draw wrapped text centered on CX. Returns bottom y."""
    layer = text_layer((W, H))
    d = ImageDraw.Draw(layer)
    lines = wrap(d, text, font, max_w)
    asc, desc = font.getmetrics()
    lh = int((asc + desc) * spacing)
    yy = y + dy
    for ln in lines:
        d.text((CX, yy), ln, font=font, fill=fill + (int(255 * alpha),), anchor="mt")
        yy += lh
    img.paste(layer, (0, 0), layer)
    return y + lh * len(lines)


def pill(img, text, font, cy, alpha=1.0, fg=WHITE, bg=AZURE, scale=1.0):
    d0 = ImageDraw.Draw(img)
    tw = d0.textlength(text, font=font)
    pw, ph = tw + 64, font.size + 34
    layer = text_layer((W, H))
    d = ImageDraw.Draw(layer)
    pw, ph = pw * scale, ph * scale
    d.rounded_rectangle([CX - pw / 2, cy - ph / 2, CX + pw / 2, cy + ph / 2], radius=ph / 2,
                        fill=bg + (int(255 * alpha),))
    if scale > 0.6:
        d.text((CX, cy), text, font=font, fill=fg + (int(255 * alpha),), anchor="mm")
    img.paste(layer, (0, 0), layer)


def fade(img, t, dur):
    """Dip from/to navy over 0.25 s at both ends of a scene."""
    a = 1.0
    if t < 0.25:
        a = t / 0.25
    elif t > dur - 0.25:
        a = (dur - t) / 0.25
    if a >= 1:
        return img
    return Image.blend(Image.new("RGB", (W, H), NAVY), img, max(0.0, a))


# ---------------- scenes ----------------

def scene_hook(spec, t):
    img = background(t)
    p = ease_back(prog(t, 0.05, 0.55))
    draw_logo(img, CX, 560, 260 * p, alpha=min(1, p * 1.5))
    pa = ease_out(prog(t, 0.35, 0.4))
    pill(img, spec["tag"].upper(), F("Inter-Bold.otf", 34), 790, alpha=pa, scale=0.85 + 0.15 * pa)
    ha = ease_out(prog(t, 0.55, 0.5))
    bottom = centered_text(img, spec["hook"], F("InterDisplay-Black.otf", 96), 890, WHITE,
                           alpha=ha, dy=(1 - ha) * 50, max_w=920, spacing=1.08)
    sa = ease_out(prog(t, 1.15, 0.5))
    centered_text(img, spec["sub"], F("Inter-SemiBold.otf", 52), bottom + 40, AZURE,
                  alpha=sa, dy=(1 - sa) * 30, max_w=880)
    return fade(img, t, S1)


def scene_demo(spec, t):
    img = background(S1 + t)
    ta = ease_out(prog(t, 0.1, 0.4))
    pill(img, spec["tag"].upper(), F("Inter-Bold.otf", 34), 400, alpha=ta)
    centered_text(img, "Watch this", F("InterDisplay-ExtraBold.otf", 72), 460, WHITE, alpha=ta)

    # chat card
    ca = ease_out(prog(t, 0.2, 0.45))
    cw, cx0 = 940, CX - 470
    cy0 = 610 + (1 - ca) * 60
    lines = spec["reply_lines"][:4]
    ch = 426 + 70 * len(lines) + 10
    layer = text_layer((W, H))
    d = ImageDraw.Draw(layer)
    A = int(255 * ca)
    d.rounded_rectangle([cx0, cy0, cx0 + cw, cy0 + ch], radius=36, fill=CARD + (A,),
                        outline=(48, 58, 92, A), width=2)

    name_f, msg_f, mono_f = F("Inter-Bold.otf", 38), F("Inter-Medium.otf", 40), F("DejaVuSansMono.ttf", 42)
    # user message
    ux, uy = cx0 + 50, cy0 + 50
    d.ellipse([ux, uy, ux + 84, uy + 84], fill=(88, 101, 242, A))
    d.text((ux + 42, uy + 42), "Y", font=F("Inter-Bold.otf", 40), fill=WHITE + (A,), anchor="mm")
    d.text((ux + 110, uy + 2), "you", font=name_f, fill=WHITE + (A,))
    cmd = spec["command"]
    n = int(len(cmd) * prog(t, 0.7, 1.3))
    typed = cmd[:n]
    caret = "▌" if (n < len(cmd) or int(t * 3) % 2 == 0) and t < 2.4 else ""
    d.text((ux + 110, uy + 50), typed + caret, font=mono_f, fill=DISCORD_TEXT + (A,))

    # bot reply
    ra = ease_out(prog(t, 2.3, 0.4))
    if ra > 0:
        RA = int(255 * ra * ca)
        by = cy0 + 200 + (1 - ra) * 30
        bx = cx0 + 50
        avatar = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        layer.alpha_composite(avatar)
        draw_logo(layer, bx + 42, by + 42, 84, alpha=ra * ca)
        d.text((bx + 110, by + 2), "Kaivor", font=name_f, fill=WHITE + (RA,))
        nx = bx + 110 + d.textlength("Kaivor", font=name_f) + 14
        d.rounded_rectangle([nx, by + 8, nx + 74, by + 44], radius=8, fill=(88, 101, 242, RA))
        d.text((nx + 37, by + 26), "APP", font=F("Inter-Bold.otf", 24), fill=WHITE + (RA,), anchor="mm")
        # embed
        ex, ey = bx + 110, by + 66
        ew = cw - 110 - 100
        eh = 110 + 70 * len(lines)
        d.rounded_rectangle([ex, ey, ex + ew, ey + eh], radius=14, fill=CARD2 + (RA,))
        d.rounded_rectangle([ex, ey, ex + 10, ey + eh], radius=5, fill=AZURE + (RA,))
        d.text((ex + 40, ey + 30), spec["reply_title"], font=F("Inter-Bold.otf", 42), fill=WHITE + (RA,))
        for i, ln in enumerate(lines):
            la = ease_out(prog(t, 2.8 + i * 0.45, 0.35))
            if la <= 0:
                continue
            LA = int(255 * la * ra * ca)
            ly = ey + 100 + i * 70
            lx = ex + 40 + (1 - la) * 20
            d.ellipse([lx, ly + 10, lx + 30, ly + 40], fill=(67, 181, 129, LA))
            d.text((lx + 15, ly + 25), "✓", font=F("Inter-Bold.otf", 22), fill=WHITE + (LA,), anchor="mm")
            d.text((lx + 50, ly + 4), ln, font=F("Inter-Medium.otf", 38), fill=DISCORD_TEXT + (LA,))
    img.paste(layer, (0, 0), layer)
    return fade(img, t, S2)


def scene_cta(spec, t):
    img = background(S1 + S2 + t)
    p = ease_back(prog(t, 0.05, 0.5))
    pulse = 1 + 0.03 * math.sin(t * 5) * prog(t, 0.6, 0.3)
    draw_logo(img, CX, 640, 300 * p * pulse, alpha=min(1, p * 1.5))
    a1 = ease_out(prog(t, 0.35, 0.45))
    bottom = centered_text(img, spec.get("cta") or "Add Kaivor to your server",
                           F("InterDisplay-Black.otf", 88), 880, WHITE, alpha=a1, dy=(1 - a1) * 40,
                           max_w=900, spacing=1.08)
    a2 = ease_out(prog(t, 0.75, 0.45))
    centered_text(img, "kaivor.space", F("Inter-Bold.otf", 64), bottom + 50, AZURE, alpha=a2)
    a3 = ease_out(prog(t, 1.05, 0.45))
    pill(img, "discord.gg/kaivor", F("Inter-Bold.otf", 44), bottom + 230, alpha=a3,
         bg=(88, 101, 242), scale=0.85 + 0.15 * a3)
    return fade(img, t, S3)


def render_scene(fn, spec, dur, out):
    tmp = out + ".part.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS), tmp]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(int(round(dur * FPS))):
        proc.stdin.write(fn(spec, i / FPS).convert("RGB").tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg failed for {out}")
    os.replace(tmp, out)


def main():
    spec_path, out = sys.argv[1], sys.argv[2]
    spec = json.load(open(spec_path))
    work = tempfile.mkdtemp(prefix="kv_")
    parts = []
    for i, (fn, dur) in enumerate([(scene_hook, S1), (scene_demo, S2), (scene_cta, S3)]):
        p = os.path.join(work, f"s{i}.mp4")
        render_scene(fn, spec, dur, p)
        parts.append(p)
    lst = os.path.join(work, "list.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{p}'\n" for p in parts)
    silent = os.path.join(work, "silent.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c", "copy", silent], check=True)
    total = S1 + S2 + S3
    # soft generated pad so platforms treat it as a normal video with audio
    audio = (f"aevalsrc='0.05*sin(2*PI*110*t)+0.035*sin(2*PI*164.8*t)+0.03*sin(2*PI*220*t)"
             f"+0.02*sin(2*PI*277.2*t)*(0.5+0.5*sin(2*PI*0.5*t))':s=44100:d={total}")
    tmp = out + ".part.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-f", "lavfi", "-i", audio,
                    "-af", f"afade=t=in:d=0.8,afade=t=out:st={total - 1.2}:d=1.2",
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                    "-shortest", "-movflags", "+faststart", tmp], check=True)
    os.replace(tmp, out)
    shutil.rmtree(work, ignore_errors=True)
    print(out)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[3] == "--frames":
        # preview: write key frames as PNG for checking layout
        spec = json.load(open(sys.argv[1]))
        os.makedirs(sys.argv[2], exist_ok=True)
        for name, fn, t in [("hook", scene_hook, 2.5), ("demo", scene_demo, 6.5), ("cta", scene_cta, 2.5)]:
            fn(spec, t).save(os.path.join(sys.argv[2], f"{name}.png"))
        print("frames written")
    else:
        main()
