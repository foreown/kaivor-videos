#!/usr/bin/env node
// Capture a Kaivor video page into an MP4, following VIDEO_RULES.md section 5.
//
// Page contract (see videos/*/index.html):
//   window.VIDEO = { fps: 60, shots: [{ name, seconds }, ...] }
//   window.render(shotIndex, frameInShot)  -> sets the exact state for that frame (may return a Promise)
//   Inactive shots are hidden with display:none by render().
//
// Usage:
//   node tools/capture.mjs page.html out.mp4                 full render, one file per shot, joined with -c copy
//   node tools/capture.mjs page.html stills/ --stills 1.2,5,9.5   JPEG stills at global seconds (for a contact sheet)
import { createRequire } from 'node:module';
import { spawn, execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { pathToFileURL } from 'node:url';

// The workspace ships a global Playwright with its browser; CommonJS resolution finds it from anywhere.
const { chromium } = createRequire(import.meta.url)('playwright');

const [, , pagePath, outPath, flag, flagValue] = process.argv;
if (!pagePath || !outPath) {
  console.error('usage: capture.mjs page.html out.mp4 | capture.mjs page.html dir --stills 1,2,3');
  process.exit(1);
}

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1080, height: 1920 }, deviceScaleFactor: 1 });
await page.goto(pathToFileURL(path.resolve(pagePath)).href);
await page.evaluate(() => document.fonts.ready);
const video = await page.evaluate(() => window.VIDEO);
const fps = video.fps ?? 60;
const shots = video.shots.map((s) => ({ ...s, frames: Math.round(s.seconds * fps) }));

async function frame(shot, f) {
  await page.evaluate(([s, i]) => window.render(s, i), [shot, f]);
  return page.screenshot({ type: 'jpeg', quality: 92 });
}

if (flag === '--stills') {
  fs.mkdirSync(outPath, { recursive: true });
  const times = flagValue.split(',').map(Number);
  for (const t of times) {
    let left = Math.round(t * fps);
    let s = 0;
    while (s < shots.length - 1 && left >= shots[s].frames) left -= shots[s++].frames;
    left = Math.min(left, shots[s].frames - 1);
    fs.writeFileSync(path.join(outPath, `t${t.toFixed(2)}.jpg`), await frame(s, left));
  }
  console.log(`stills written to ${outPath}`);
  await browser.close();
  process.exit(0);
}

const work = fs.mkdtempSync(path.join(os.tmpdir(), 'kvcap-'));
const ENCODE = [
  '-c:v', 'libx264', '-profile:v', 'high', '-preset', 'veryfast', '-crf', '17',
  '-maxrate', '9M', '-bufsize', '18M',
  '-vf', 'scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p',
  '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv',
  '-r', String(fps), '-an',
];
const parts = [];
for (let s = 0; s < shots.length; s++) {
  const part = path.join(work, `shot${String(s).padStart(2, '0')}.mp4`);
  const tmp = part.replace(/\.mp4$/, '.part.mp4');
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps),
    '-c:v', 'mjpeg', '-i', '-', ...ENCODE, tmp], { stdio: ['pipe', 'inherit', 'inherit'] });
  const done = new Promise((resolve) => ff.on('close', resolve));
  for (let f = 0; f < shots[s].frames; f++) {
    const buf = await frame(s, f);
    if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once('drain', r));
  }
  ff.stdin.end();
  const code = await done;
  if (code !== 0) throw new Error(`ffmpeg failed on shot ${s} (${shots[s].name})`);
  fs.renameSync(tmp, part);
  parts.push(part);
  console.log(`shot ${s + 1}/${shots.length} ${shots[s].name}: ${shots[s].frames} frames`);
}
await browser.close();

const list = path.join(work, 'list.txt');
fs.writeFileSync(list, parts.map((p) => `file '${p}'\n`).join(''));
const tmpOut = outPath.replace(/\.mp4$/, '.part.mp4');
execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', list,
  '-c', 'copy', '-movflags', '+faststart', tmpOut]);
fs.renameSync(tmpOut, outPath);
fs.rmSync(work, { recursive: true, force: true });
console.log(outPath);
