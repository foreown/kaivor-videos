# Kaivor promo videos: instructions for Claude

You make short promo videos for **Kaivor**, my Discord bot (kaivor.space, GitHub repo `foreown/aurora-bot`, docs repo `foreown/aurora-docs`). Follow every rule below without asking me again. If something here conflicts with a new request, the new request wins.

---

## 1. Hard rules (never break)

1. **Centred.** Every video MUST be horizontally centred on the true middle of the frame (x = 540 of 1080). Never shift content to dodge TikTok/Reels/Shorts UI. Check it on the finished file by measuring frames, not just by trusting the layout.
2. **Format.** 9:16, 1080×1920, **60 fps**, H.264 High, yuv420p, bt709, `-movflags +faststart`. Around 18–22 s unless I say otherwise. Keep the file under ~30 MB (CRF 16–17, `-maxrate 9M -bufsize 18M`).
3. **Every video needs sound.** Silent videos are not allowed. In this order of priority:
   1. **Licensed music first.** Use a track from `music/` (see `music/TRACKS.md`) that matches the video's tempo and mood, and cut the edit to its beat. Every track there must be licensed for commercial/brand use on every platform the video goes to (e.g. CC0, Pixabay Content License, YouTube Audio Library "no attribution", or another royalty-free licence that allows commercial use). Prefer tracks that are trending or popular *within* those licensed libraries (e.g. Pixabay Music sorted by trending). If a track needs attribution, put the credit in every caption. Don't reuse the same track two videos in a row. Write the track in the video's LOG.md row.
   2. **If no licensed track fits (or `music/` is empty), make your own:** original sound effects plus an original music bed, all synthesised in code (e.g. numpy), timed to the cuts and on-screen actions (key presses, message pops, the end card).
   - Either way: AAC 48 kHz stereo, normalised to −14 LUFS integrated, true peak −2 dBTP, with a limiter so nothing hits 0 dBFS (`tools/finish_audio.py` does this). Sound must start within the first second and last until the end. Light UI sound effects may sit on top of a licensed track.
4. **No copyrighted trending songs in the file.** Trending TikTok/chart songs are not licensed for brand accounts: baked into the file they get muted, claimed or taken down, and TikTok business accounts may only use its Commercial Music Library. So "trending" always means trending within licensed libraries. If I want an actual trending TikTok sound, I add it myself inside the app (or pick one from TikTok's Commercial Music Library there).
5. **Generic Discord-style UI.** Make it look like Discord (dark chat, menus, embeds, buttons) but use no Discord logo, wordmark or branding.
6. **Real Kaivor logo.** Use the real logo (I'll attach it, or get it from the repo). Never redraw or alter it.
7. **Made-up but realistic users.** Usernames like `mira`, `teo`, `gift_drop4821`. No real people's accounts.
8. **Every fact must be real.** Every command, flag, number and bot message shown on screen must be checked in the aurora-bot code first. Copy Kaivor's real reply text word for word. The default prefix is `,` (slash commands also exist). Never invent features or numbers. Make sure counts add up (e.g. "0 roles and 11 channels created, 29 already there").
9. **Platform-guideline safe.** Nothing that could get the video removed (see section 6).

---

## 2. Picking the feature

- **Before picking anything, do section 9** (learn from the best-performing video so far) and let what you find steer the feature choice, the hook, the length and the caption.
- "A new feature" means: run `git pull` and `git log` on aurora-bot and pick from the newest update. Read the commit and the code to see exactly how it behaves.
- Prefer features that are easy to *show* in under 20 s as a before → action → result story.
- Features already made into videos (don't repeat unless asked): honeypot channels, antinuke, `/server setup`, server backups, right-click (context-menu) commands, a 30 s security demo, and several contest videos.

---

## 3. Avoid "AI slop" (always research briefly and apply)

**Never use:**
- gradient text, violet-to-blue gradients, neon/cyan on black, glow halos
- glassmorphism, soft drop shadows, rounded cards everywhere
- Inter or other default fonts
- monospace for non-code text (monospace only for commands)
- a row of hero stats/metrics
- emoji or icons above every section
- pure #000 or #fff backgrounds
- bounce/elastic easing, blur-in reveals, floaty constant-speed camera moves
- every shot the same length
- the same lighting/background in every shot
- hype copy ("revolutionary", "game-changer"), subheadings that just repeat the heading
- visibly repeating textures (if you add grain, use many random tiles)
- over-saturated colours

**Do instead:**
- One restrained palette per video: an off-black or off-white base plus **one** accent colour.
- A distinctive typeface. Ones used so far: Archivo variable (condensed or wide), Instrument Serif, JetBrains Mono for code only. Fonts come from npm `@fontsource`.
- Hard cuts. Shots of different lengths. Switch between light and dark backgrounds between shots.
- One easing everywhere: fast out, no overshoot (`1-(1-t)^4`). Text reveals as lines sliding up out of a mask.
- A cursor that speeds up and slows down (ease-in-out), with a small press on clicks.
- Plain, human copy in short sentences. Hook within the first 2 seconds.
- **Give every video a different look** from the earlier ones. Looks used so far: paper + navy editorial with crop marks; charcoal + bone + orange with a serif; stone + cobalt with wide heavy sans; a notebook/handmade style; a retro talking logo.

---

## 4. Structure that works

1. **Hook (0–2.5 s):** the problem, shown (spam flood, channels deleted, a raid).
2. **Turn (~1 s):** one short line on a contrasting background.
3. **Action (3–4 s):** the real command or click, with Kaivor's real confirmation message.
4. **Result (3–4 s):** the problem undone on screen, plus Kaivor's real success message.
5. **The key fact (2 s):** e.g. "Nothing gets deleted." or "Same checks as typing the command."
6. **End card (3 s):** "New in Kaivor", feature name, the command, real logo, `kaivor.space`.

---

## 5. How to build and check

- Build each video as an HTML page where `render(frame)` sets the exact state for that frame. Capture it with Playwright headless Chromium at 1080×1920.
- Render **one encoded file per shot** by piping JPEG screenshots (quality 92) into ffmpeg. Write each to a temp `.mp4` and rename it only when ffmpeg succeeds. Then join with the concat demuxer and `-c copy`.
- **Sound:** build the audio track as a WAV the exact length of the video, timed to the shot list. Then add it with `python3 tools/finish_audio.py silent.mp4 sound.wav final.mp4`, which normalises the loudness and copies the video stream untouched.
- Hide inactive shots with `display:none` (not `visibility`), so nothing leaks into other shots.
- **Before the full render:** render stills at the busiest moments, put them on a contact sheet and look at them. Fix layout, overlap, empty areas and centring first.
- **After the render:** check frame count, fps, duration and size with ffprobe, and that there is an audio track at −14 LUFS / −2 dBTP (`tools/check_video.py` checks all of this). Listen through the timing by looking at the waveform against the shot list. Measure the horizontal centre of the content on several frames (it should land on ~540). Export a cover frame (JPEG) from the end card.
- Deliver the .mp4 and the cover. Then give a short shot-by-shot summary of what's in the video and which facts were checked in the code.

---

## 6. Captions (when I ask for descriptions)

Write one each for **TikTok, YouTube Shorts, X and Instagram Reels**, in English, matching each platform's tone. Put each in a code block so I can copy it.

- **TikTok:**
  - Tell me to turn on *Content disclosure → Your brand*.
  - No reward-for-engagement ("comment to win", "like and follow to enter").
  - No cash-forward hooks.
  - No links or invites to Discord.
  - A short lowercase tone works. 3–5 hashtags.
- **YouTube Shorts:** a title line, a few lines or bullets on what the feature does, `kaivor.space`, and `#shorts`. For contests, add "YouTube is not a sponsor of this contest."
- **X:** short, under 280 characters, no hashtag spam.
- **Instagram Reels:** a hook line, 2–3 sentences, "Link in bio → kaivor.space", 5–7 hashtags. For contests, add "This contest is not sponsored, endorsed or administered by, or associated with, Instagram."
- For contests on X: the rules should say multiple accounts are disqualified.
- If someone comments asking for proof, reply calmly with something they can check: the command, the docs at kaivor.space, or offer a screen recording.

---

## 7. Kaivor facts already checked (still re-check before use)

- **Honeypot:** `,honeypot add <channel> <ban|softban|jail>`. Staff and admins are exempt. The log reason reads "Honeypot triggered in #channel".
- **Antinuke:** default limit is 3 actions per 60 s, default punishment is ban. `,antinuke channeldelete on --threshold 3 --do ban`. The owner gets a DM saying "went past the limit in **server**, so I stepped in."
- **/server setup:** creates 16 roles, 8 categories and 24 channels, and sets up verification, tickets, welcome messages, starboard, logs and media perks. Its confirm message says "Nothing is deleted." It ends with an embed titled "Community setup complete".
- **Server backups:** `,backup schedule 24 --keep 7` replies "This server is backed up every **24 hours**, keeping the newest **7**." `,backup restore <name>` adds only what's missing; its confirm message ends "Nothing is deleted or changed." The result reads "Restored into **X**: N roles and N channels created, N already there." Restoring needs real Administrator.
- **Right-click commands:**
  - On a member: Warn, Time out, Notes, Cases, User info.
  - On a message: Translate, Purge after this, Open a ticket about this.
  - Purge replies "Deleted N messages after that one." (only you can see it).
  - Time out opens a form: "How long (for example 10m, 1h, 1d)" plus Reason.
  - Every action runs through the same permission checks and cooldowns as the typed command.
- Kaivor has 539 commands across 25 categories (re-check, this grows).

---

## 8. The contest (only if I ask for contest content)

- $25 for the best promo video about a Kaivor feature.
- Winner is the entry with the most interaction.
- Post it on socials, then put the link in **#contest-entries** on the Discord.
- Runs 15 Oct to 15 Nov 2026. It ends at Discord timestamp `<t:1794783540:F>`.
- TikTok has removed contest posts before. On TikTok, keep the money and "post to win" out of the video and caption.

---

## 9. Learn from the best-performing video (before every new video)

Before you pick the feature or write a single shot, look at how the videos on the Kaivor accounts have done, find the best one, and work out what made it work.

1. **Pull the numbers from Metricool** (brand / blogId `7853528`, timezone Europe/Oslo) with `getAnalyticsDataByMetrics`, from 30 days ago to today:
   - TikTok: `TKPO02` (published), `TKPO03` (url), `TKPO05` (caption), `TKPO07` views, `TKPO08` likes, `TKPO09` comments, `TKPO10` shares, plus `TKPO13` full-watch rate and `TKPO15` average watch time when they aren't empty.
   - YouTube Shorts: `YTVP02` (published), `YTVP17` (title), `YTVP05` (url), `YTVP06` views, `YTVP08` average view duration, `YTVP09` likes, `YTVP11` comments, `YTVP12` shares.
   - Leave out videos less than 48 hours old: they haven't had time yet.
2. **Pick the best performer.** Rank by views. When views are close (within about 20 %), the one with more interactions (likes + comments + shares) wins. If the full-watch rate or average watch time is available, a clearly better watch rate beats a few more views. Do this for TikTok and YouTube separately; if YouTube has no numbers yet, use TikTok alone and say so.
3. **Work out what was good about it.** Match it to its `LOG.md` row and its `videos/<dir>/index.html` (open the page or render a couple of stills; the cover may still be on the `media` branch). Compare it with the weakest videos, not just in isolation. Look at:
   - the hook: the first line on screen and the first line of the caption (a concrete problem? a number? a question? second person?)
   - the kind of feature (something members use and see, or a staff/admin tool)
   - length, pacing, how fast the action starts
   - the look and the sound
   - the posting time and the hashtags
4. **Apply it, within the rules.** Carry over the lesson (e.g. "open on a concrete, relatable problem with a number in it"), never the video itself: the feature, look and sound must still be new, and every other rule in this file still applies. Don't overfit to small differences: with a few hundred views, a gap of a few dozen is noise, so say how strong the signal is.
5. **Write it down.** Add a dated entry at the top of `ANALYTICS.md`: the best video (date, feature, link, numbers), what you think made it work, how strong the signal is, and what today's video does with it. Commit it with the video. Put one line about it in the final summary.

If Metricool returns nothing useful (no rows, or every number empty), say so in `ANALYTICS.md` and the summary and carry on with the normal rules.

