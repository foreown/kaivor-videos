# Kaivor daily videos

Daily promo videos for Kaivor (TikTok + YouTube Shorts), posted through Metricool.

**Read `VIDEO_RULES.md` first. Every video must follow it.**

## Layout

- `VIDEO_RULES.md`: the production rules (hard rules, look, structure, captions, checked facts)
- `ANALYTICS.md`: what the best-performing videos had in common, one dated entry per video (VIDEO_RULES.md section 9)
- `LOG.md`: every video made so far, with its feature, look and fonts. Check it so you don't repeat a feature or a look, and add a line for each new video
- `videos/<date>-<feature>/index.html`: one page per video, where `render(shot, frame)` sets the exact state of that frame
- `tools/capture.mjs`: captures a page with Playwright at 1080×1920, 60 fps, into one file per shot, then joins them with `-c copy`
  - `node tools/capture.mjs videos/<dir>/index.html out/<dir>.mp4`
  - `node tools/capture.mjs videos/<dir>/index.html out/stills --stills 1.5,6,11` writes stills for a contact sheet
- `tools/finish_audio.py silent.mp4 sound.wav final.mp4`: adds the sound track at −14 LUFS / −2 dBTP and copies the video stream untouched
- `tools/check_video.py out.mp4 cover.jpg`: checks format, fps, colour tags, sound and loudness, duration and size, measures the content's horizontal centre, and exports the cover
- `videos/2026-11-04-sticky-messages/sound.py`: example of an original sound track synthesised in code and timed to the shot list
- `assets/`: the real Kaivor logo (`kaivor-logo.png`, transparent) and app icon (`icon-512.png`), copied from aurora-docs and never altered
- Fonts come from npm `@fontsource` (`npm install`)

## Hosting

The finished MP4 and cover go on the `media` branch, which keeps only the newest ~10 videos and is force-pushed so history stays small. Metricool reads them from:

`https://raw.githubusercontent.com/foreown/kaivor-videos/media/<date>-<feature>.mp4`
