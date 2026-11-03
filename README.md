# Kaivor daily videos

Generator for Kaivor's daily TikTok / YouTube Shorts.

- `topics.json` — content rotation (one entry per day) and dated overrides
- `pick.py [YYYY-MM-DD]` — prints the spec for a date (default: today, Europe/Oslo)
- `render.py spec.json out.mp4` — renders a 1080×1920, 30 fps, ~13 s video, centered on the frame
- Rendered videos live on the `media` branch (only the most recent ~7 are kept) and are served from
  `https://raw.githubusercontent.com/foreown/kaivor-videos/media/<date>.mp4`

Edit `topics.json` to change what gets posted. Fonts: Inter (SIL OFL), DejaVu Sans Mono.
