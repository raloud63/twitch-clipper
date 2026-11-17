#!/usr/bin/env python3
"""clip_watcher: watch a live Twitch stream and cut a captioned 9:16 clip around every chat spike.

How it works
  1. Streamlink records the stream to disk from the moment it goes live.
  2. Twitch chat is read anonymously (read-only, no account, no token) and counted every 10 seconds.
  3. When a 10-second count jumps well above the recent normal, that moment is a spike.
  4. FFmpeg cuts 30 seconds around the spike from the recording, while the stream keeps going.
  5. Whisper writes word-timed captions, which are burned into a 1080x1920 vertical clip.

Usage
  python clip_watcher.py CHANNEL                 watch a channel live (waits until it goes live)
  python clip_watcher.py --file REC.ts --at 754  cut one captioned clip from a saved recording at 12:34

It only works while the stream is live: it can't go back and clip a stream that already ended.
"""
import argparse
import csv
import random
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from queue import Queue

IRC_WS_URL = 'wss://irc-ws.chat.twitch.tv:443'


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


# ---------- Chat: anonymous read-only IRC over WebSocket ----------
class ChatCounter(threading.Thread):
    """Counts chat messages via Twitch IRC over WebSocket (port 443).
    Twitch accepts a 'justinfan' nickname as an anonymous, read-only login."""

    def __init__(self, channel, stop):
        super().__init__(daemon=True)
        self.channel, self.stop = channel.lower(), stop
        self.count, self.total, self.lock = 0, 0, threading.Lock()
        self.connected = threading.Event()

    def take(self):
        with self.lock:
            n, self.count = self.count, 0
        return n

    def run(self):
        from websockets.sync.client import connect
        while not self.stop.is_set():
            try:
                with connect(IRC_WS_URL, open_timeout=15) as ws:
                    ws.send(f'NICK justinfan{random.randint(10000, 99999)}')
                    ws.send(f'JOIN #{self.channel}')
                    while not self.stop.is_set():
                        msg = ws.recv(timeout=60)
                        for line in msg.split('\r\n'):
                            if line.startswith('PING'):
                                ws.send(line.replace('PING', 'PONG', 1))
                            elif ' PRIVMSG #' in line:
                                with self.lock:
                                    self.count += 1
                                    self.total += 1
                            elif ' 366 ' in line and not self.connected.is_set():
                                self.connected.set()
                                log(f'chat: reading #{self.channel} anonymously')
            except TimeoutError:
                continue
            except Exception as e:  # noqa: BLE001 - keep reconnecting whatever went wrong
                if not self.stop.is_set():
                    log(f'chat: {e}; reconnecting in 5 s')
                    time.sleep(5)


# ---------- Recording: Streamlink writes the live stream to disk ----------
class Recorder(threading.Thread):
    """Runs Streamlink. It waits for the channel to go live, records, and starts a new file if the stream drops."""

    def __init__(self, channel, quality, folder, stop):
        super().__init__(daemon=True)
        self.channel, self.quality, self.folder, self.stop = channel, quality, Path(folder), stop
        self.path, self.started = None, None  # current file and the moment its first bytes arrived
        self.proc = None

    def offset(self):
        """Seconds of stream in the current file, by the clock."""
        return None if self.started is None else time.monotonic() - self.started

    def run(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        while not self.stop.is_set():
            path = self.folder / f'{self.channel}_{time.strftime("%Y%m%d_%H%M%S")}.ts'
            cmd = ['streamlink', '--retry-streams', '30', '--retry-open', '3',
                   f'https://twitch.tv/{self.channel}', f'{self.quality},best', '-o', str(path)]
            log('recorder: waiting for the stream to go live')
            errlog = self.folder / 'streamlink.log'
            with open(errlog, 'w') as err:
                self.proc = subprocess.Popen(cmd, stdout=err, stderr=subprocess.STDOUT)
                while self.proc.poll() is None and not self.stop.is_set():
                    if self.started is None and path.exists() and path.stat().st_size > 0:
                        self.path, self.started = path, time.monotonic()
                        log(f'recorder: live, recording to {path.name}')
                    time.sleep(0.5)
                if self.proc.poll() is None:
                    self.proc.terminate()
            if self.started is not None:
                log('recorder: stream ended or dropped')
            elif not self.stop.is_set():
                last = [l for l in errlog.read_text(errors='replace').splitlines() if l.strip()][-1:]
                log(f'recorder: streamlink stopped: {last[0] if last else "no output"}; retrying in 10 s')
                time.sleep(10)
            self.path, self.started = None, None


# ---------- Clip making: FFmpeg cut, Whisper captions, vertical render ----------
_model = None
DEVICE = 'auto'  # auto: try the GPU, fall back to the CPU


def whisper_model(name, cpu=False):
    global _model
    if _model is None or cpu:
        from faster_whisper import WhisperModel
        if DEVICE == 'cuda' or (DEVICE == 'auto' and not cpu):
            try:
                _model = WhisperModel(name, device='cuda', compute_type='float16')
                return _model
            except Exception:  # noqa: BLE001 - no usable GPU
                if DEVICE == 'cuda':
                    raise
        _model = WhisperModel(name, device='cpu', compute_type='int8')
    return _model


def srt_time(t):
    ms = int(round(t * 1000))
    return f'{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}'


def write_captions(media, srt, model_name, words_per_cue=3):
    """Transcribes the clip with word timestamps and writes short, punchy caption cues."""
    import numpy as np
    # Decode with FFmpeg rather than Whisper's own loader, which breaks with some PyAV versions.
    pcm = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', str(media), '-vn', '-ac', '1', '-ar', '16000', '-f', 's16le', '-'],
                         check=True, capture_output=True).stdout
    audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
    def run(model):
        segments, _ = model.transcribe(audio, word_timestamps=True, vad_filter=True)
        return [w for s in segments for w in (s.words or [])]
    try:
        words = run(whisper_model(model_name))
    except RuntimeError as e:  # a GPU without the CUDA libraries fails here, not when loading
        if DEVICE != 'auto':
            raise
        log(f'captions: GPU not usable ({str(e)[:60]}), using the CPU')
        words = run(whisper_model(model_name, cpu=True))
    # Up to 3 words a caption, and a pause of more than 0.6 s always starts a new one.
    cues = []
    for w in words:
        if cues and len(cues[-1]) < words_per_cue and w.start - cues[-1][-1].end <= 0.6:
            cues[-1].append(w)
        else:
            cues.append([w])
    with open(srt, 'w', encoding='utf-8') as f:
        for n, c in enumerate(cues, 1):
            text = ' '.join(w.word.strip() for w in c).upper()
            f.write(f'{n}\n{srt_time(c[0].start)} --> {srt_time(c[-1].end)}\n{text}\n\n')
    return len(words)


def make_clip(src, start, length, out, model_name='base', captions=True, crop=False):
    """Cuts [start, start+length) from src and renders a 1080x1920 clip with burned-in captions."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_suffix('.raw.mp4')
    srt = out.with_suffix('.srt')
    # 1. Fast cut, no re-encode: works on a recording that is still being written.
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{max(0, start):.2f}', '-i', str(src), '-t', f'{length:.2f}',
                    '-map', '0:v:0', '-map', '0:a:0?', '-c', 'copy', str(raw)], check=True)
    # 2. Vertical frame: either a centre crop, or the full frame over a blurred fill.
    if crop:
        vf = 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920'
    else:
        vf = ('split[a][b];[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=24:2[bg];'
              '[b]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2')
    n_words = 0
    if captions:
        n_words = write_captions(raw, srt, model_name)
        if n_words:
            style = 'FontName=Arial,Fontsize=16,Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Shadow=0,Alignment=2,MarginV=60'
            vf += f",subtitles={srt.name}:force_style='{style}'"
    # Run from the clip's folder so the subtitles path needs no escaping on Windows.
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', raw.name, '-filter_complex', vf,
                    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart', out.name], check=True, cwd=out.parent)
    raw.unlink(missing_ok=True)
    return n_words


# ---------- The live loop ----------
def watch(args):
    stop = threading.Event()
    chat = ChatCounter(args.channel, stop)
    rec = Recorder(args.channel, args.quality, Path(args.out) / 'recordings', stop)
    chat.start()
    rec.start()
    jobs = Queue()
    clips = Path(args.out) / 'clips'
    clips.mkdir(parents=True, exist_ok=True)
    log_file = clips / 'spikes.csv'
    new_log = not log_file.exists()
    with open(log_file, 'a', newline='') as f:
        if new_log:
            csv.writer(f).writerow(['time', 'messages_in_window', 'normal', 'clip'])

    def cutter():
        while True:
            path, start, end, name, msgs, base = jobs.get()
            while rec.path == path and (rec.offset() or 0) < end + 3:  # wait until the recording has the whole clip
                time.sleep(1)
            out = clips / f'{name}.mp4'
            try:
                words = make_clip(path, start, end - start, out, args.model, not args.no_captions, args.crop)
                log(f'clip: {out.name} ({words} caption words)')
                with open(log_file, 'a', newline='') as f:
                    csv.writer(f).writerow([time.strftime('%Y-%m-%d %H:%M:%S'), msgs, f'{base:.1f}', out.name])
            except Exception as e:  # noqa: BLE001 - one bad clip must never stop the watcher
                log(f'clip failed: {e}')

    threading.Thread(target=cutter, daemon=True).start()

    history = deque(maxlen=max(6, int(args.baseline / args.window)))
    last_clip = -1e9
    next_tick = time.monotonic() + args.window
    log(f'watching #{args.channel}: a spike is {args.ratio}x the normal {args.window:g}-second count '
        f'(at least {args.min_msgs} messages). Ctrl+C stops.')
    try:
        while True:
            time.sleep(max(0, next_tick - time.monotonic()))
            next_tick += args.window
            n = chat.take()
            if rec.started is None:  # not live yet: don't learn "normal" from the waiting room
                history.clear()
                if args.verbose:
                    log(f'{n} messages in the last {args.window:g} s (waiting for the recording)')
                continue
            normal = statistics.median(history) if len(history) >= 6 else None
            history.append(n)
            if args.verbose:
                log(f'{n} messages in the last {args.window:g} s (normal {normal})')
            if normal is None or n < max(args.min_msgs, args.ratio * max(normal, 1)):
                continue
            off = rec.offset()
            if off - last_clip < args.gap:
                continue
            last_clip = off
            # Chat reacts a few seconds after the moment, so most of the clip comes before the spike.
            start = max(0, off - args.window - args.before)
            end = start + args.length
            name = f'{args.channel}_{time.strftime("%H%M%S")}'
            log(f'spike: {n} messages vs a normal {normal:g}; cutting {args.length:g} s')
            jobs.put((rec.path, start, end, name, n, normal))
    except KeyboardInterrupt:
        log('stopping')
        stop.set()
        if rec.proc and rec.proc.poll() is None:
            rec.proc.terminate()


def main():
    p = argparse.ArgumentParser(description='Clip a live Twitch stream around every chat spike.')
    p.add_argument('channel', nargs='?', help='Twitch channel name, e.g. the part after twitch.tv/')
    p.add_argument('--out', default='output', help='folder for recordings and clips (default: output)')
    p.add_argument('--quality', default='best', help='Streamlink quality, e.g. best, 1080p60, 720p (default: best)')
    p.add_argument('--window', type=float, default=10, help='seconds per chat count (default: 10)')
    p.add_argument('--ratio', type=float, default=3.0, help='a spike is this many times the normal count (default: 3)')
    p.add_argument('--min-msgs', type=int, default=20, help='ignore spikes smaller than this many messages (default: 20)')
    p.add_argument('--baseline', type=float, default=180, help='seconds of chat that set "normal" (default: 180)')
    p.add_argument('--length', type=float, default=30, help='clip length in seconds (default: 30)')
    p.add_argument('--before', type=float, default=15, help='seconds kept before the spiking window (default: 15)')
    p.add_argument('--gap', type=float, default=60, help='at least this many seconds between clips (default: 60)')
    p.add_argument('--model', default='base', help='Whisper model: tiny, base, small, medium (default: base)')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='where Whisper runs (default: auto)')
    p.add_argument('--no-captions', action='store_true', help='skip Whisper captions')
    p.add_argument('--crop', action='store_true', help='centre-crop to 9:16 instead of a blurred fill')
    p.add_argument('--verbose', action='store_true', help='print every chat count')
    p.add_argument('--file', help='make one clip from a saved recording instead of watching live')
    p.add_argument('--at', type=float, help='with --file: the moment to clip, in seconds')
    args = p.parse_args()
    global DEVICE
    DEVICE = args.device

    missing = [t for t in ('ffmpeg',) + (() if args.file else ('streamlink',)) if not shutil.which(t)]
    if missing:
        sys.exit(f'Not found on PATH: {", ".join(missing)}. See README.md, step 1.')
    if args.file:
        if args.at is None:
            sys.exit('--file needs --at SECONDS')
        out = Path(args.out) / 'clips' / f'{Path(args.file).stem}_{int(args.at)}.mp4'
        start = max(0, args.at - args.before - args.window)
        words = make_clip(args.file, start, args.length, out, args.model, not args.no_captions, args.crop)
        log(f'clip: {out} ({words} caption words)')
        return
    if not args.channel:
        p.error('give a channel name, or --file and --at')
    watch(args)


if __name__ == '__main__':
    main()