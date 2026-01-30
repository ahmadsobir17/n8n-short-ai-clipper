#!/usr/bin/env python3
"""
Face Detection Auto-Crop for Podcast Videos with Whisper Transcription, Dynamic Zoom, and Viral Hooks.
"""

from pathlib import Path
import os
import tempfile
import sys
import argparse
import subprocess
import json
import wave

# Suppress MediaPipe/TF/OpenCV logging and force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_GPU_DISABLED'] = '1'
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

try:
    import cv2
    import mediapipe as mp
    import whisper
    import gc
    import numpy as np
except ImportError as e:
    print(f"Error: Required packages not installed: {e}")
    sys.exit(1)

class FaceDetector:
    def __init__(self, min_confidence=0.4):
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=min_confidence
        )
    def close(self):
        if hasattr(self, 'detector'): self.detector.close()
    def detect_faces(self, frame):
        if frame is None: return []
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try: results = self.detector.process(rgb_frame)
        except: return []
        faces = []
        if results.detections:
            for d in results.detections:
                bbox = d.location_data.relative_bounding_box
                faces.append({
                    'center_x': bbox.xmin + bbox.width / 2,
                    'center_y': bbox.ymin + bbox.height / 2,
                    'rel_w': bbox.width, 'rel_h': bbox.height
                })
        return faces

def get_video_info(video_path):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,duration', '-of', 'json', video_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(res.stdout)
        s = data['streams'][0]
        return int(s['width']), int(s['height']), float(s.get('duration', 0))
    except: return None, None, None

def extract_segment_h264(video_path, start_time, duration):
    temp = tempfile.NamedTemporaryFile(suffix='_seg.mp4', delete=False)
    temp.close()
    cmd = ['ffmpeg', '-y', '-ss', str(start_time), '-t', str(duration), '-i', video_path, '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23', '-an', temp.name]
    if subprocess.run(cmd, capture_output=True).returncode == 0: return temp.name
    return None

def analyze_video(video_path, start_time, duration):
    print("  Analyzing video frames...")
    detector = FaceDetector()
    w, h, total_dur = get_video_info(video_path)
    if w is None: return None
    is_ultrawide = (w / h) > 2.0
    temp_seg = extract_segment_h264(video_path, start_time, duration)
    if not temp_seg: return None
    cap = cv2.VideoCapture(temp_seg)
    frame_data = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        faces = detector.detect_faces(frame)
        primary = faces[0] if faces else {'center_x': 0.5, 'center_y': 0.5, 'rel_w': 0, 'rel_h': 0}
        frame_data.append({
            'timestamp': ts, 'face_count': len(faces), 'primary_x': primary['center_x'],
            'mode': 'wide' if len(faces) >= 2 or (is_ultrawide and (len(faces)==0 or (primary['rel_w']*primary['rel_h'] < 0.025))) else 'closeup'
        })
    cap.release()
    os.unlink(temp_seg)
    detector.close()
    segments = []
    if frame_data:
        curr_mode, start_t = frame_data[0]['mode'], 0
        for i in range(1, len(frame_data)):
            if frame_data[i]['mode'] != curr_mode:
                segments.append({'start': start_t, 'end': frame_data[i]['timestamp'], 'mode': curr_mode})
                start_t, curr_mode = frame_data[i]['timestamp'], frame_data[i]['mode']
        segments.append({'start': start_t, 'end': duration, 'mode': curr_mode})
    for seg in segments:
        seg_frames = [f for f in frame_data if seg['start'] <= f['timestamp'] <= seg['end']]
        seg['avg_x'] = sum(f['primary_x'] for f in seg_frames) / len(seg_frames) if seg_frames else 0.5
    return {'video_width': w, 'video_height': h, 'segments': segments, 'adjusted_start_time': start_time}

def analyze_audio_emphasis(video_path, start_time, duration):
    print("  Analyzing audio emphasis...")
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_wav.close()
    try:
        cmd = ['ffmpeg', '-y', '-ss', str(start_time), '-t', str(duration), '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', temp_wav.name]
        subprocess.run(cmd, capture_output=True)
        with wave.open(temp_wav.name, 'rb') as wf:
            audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)
        if len(audio_data) == 0: return []
        win = 1600 # 100ms
        rms = np.array([np.sqrt(np.mean(audio_data[i*win:(i+1)*win]**2)) for i in range(len(audio_data)//win)])
        if len(rms) == 0: return []
        avg_rms = np.mean(rms)
        thresh = avg_rms * 1.7
        emphasis = []
        is_emp, start_win = False, 0
        for i, val in enumerate(rms):
            if val > thresh and not is_emp: is_emp, start_win = True, i
            elif val <= thresh and is_emp:
                is_emp = False
                if (i - start_win) >= 5: emphasis.append({'start': start_win * 0.1, 'end': min(i * 0.1, start_win * 0.1 + 1.5)})
        return emphasis
    except Exception as e:
        print(f"  Warning: Audio emphasis detection failed: {e}")
        return []
    finally:
        if os.path.exists(temp_wav.name): os.unlink(temp_wav.name)

def generate_crop_filter(analysis, duration, words, emphasis):
    w, h = analysis['video_width'], analysis['video_height']
    segments = analysis['segments']
    out_w, out_h, panel_h = 1080, 1920, 960
    aspect_9_16, aspect_panel = 9/16, 1080/960
    fade_duration = 0.3 if len(segments) > 1 else 0
    filter_parts, v_names, a_names = [], [], []
    for i, seg in enumerate(segments):
        start, end, mode = seg['start'], seg['end'], seg['mode']
        v_seg, a_seg = f"v{i}", f"a{i}"
        trim_v, trim_a = f"trim=start={start}:end={end},setpts=PTS-STARTPTS", f"atrim=start={start}:end={end},asetpts=PTS-STARTPTS"
        
        # Build zoom expression (Punch Zoom + Snap Hook Zoom)
        z_expr = "1.0"
        # Hook Snap Zoom: 25% zoom for 1.5s at the very beginning of the clip
        z_expr += "+0.25*between(time,0,1.5)*sin(PI*time/1.5)"
        
        for emp in emphasis:
             es, ee = emp['start'] - start, emp['end'] - start
             if ee > 0 and es < (end - start):
                 d = max(0.01, min(end-start, ee) - max(0, es))
                 z_expr += f"+0.15*between(time,{max(0, es):.3f},{min(end-start, ee):.3f})*sin(PI*(time-{max(0,es):.3f})/{d:.3f})"
        
        zoom_filter = f"zoompan=z='{z_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={out_w}x{out_h}:fps=24"
        zoom_filter_panel = f"zoompan=z='{z_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={out_w}x{panel_h}:fps=24"

        if mode == 'wide':
            cw = min(w // 2 - 20, int(h * aspect_panel))
            ch = int(cw / aspect_panel)
            lx, rx, y = max(0, int(w*0.25)-cw//2), min(w-cw, int(w*0.75)-cw//2), (h - ch) // 2
            part = (f"[0:v]{trim_v},split=2[t{i}][b{i}];"
                    f"[t{i}]crop={cw}:{ch}:{lx}:{y},scale={out_w}:{panel_h},setsar=1,fps=24,format=yuv420p,{zoom_filter_panel}[top{i}];"
                    f"[b{i}]crop={cw}:{ch}:{rx}:{y},scale={out_w}:{panel_h},setsar=1,fps=24,format=yuv420p,{zoom_filter_panel}[bot{i}];"
                    f"[top{i}][bot{i}]vstack=inputs=2[{v_seg}];[0:a]{trim_a}[{a_seg}]")
        else:
            ch, cw = h, int(h * aspect_9_16)
            xb = max(0, min(int(seg['avg_x'] * w) - cw // 2, w - cw))
            part = f"[0:v]{trim_v},crop={cw}:{ch}:{xb}:0,scale={out_w}:{out_h},setsar=1,fps=24,format=yuv420p,{zoom_filter}[{v_seg}];[0:a]{trim_a}[{a_seg}]"
        filter_parts.append(part)
        v_names.append(f"[{v_seg}]"); a_names.append(f"[{a_seg}]")
    cv, ca, cum_t, chains = v_names[0], a_names[0], segments[0]['end']-segments[0]['start'], []
    for i in range(1, len(segments)):
        vo, ao = (f"[vx{i}]", f"[ax{i}]") if i < len(segments)-1 else ("[stacked]", "[stackeda]")
        off = max(0, cum_t - fade_duration)
        chains.append(f"{cv}{v_names[i]}xfade=transition=dissolve:duration={fade_duration}:offset={off:.3f}{vo}")
        chains.append(f"{ca}{a_names[i]}acrossfade=duration={fade_duration}{ao}")
        cum_t = off + (segments[i]['end']-segments[i]['start'])
        cv, ca = vo, ao
    if len(segments) > 1:
        f_str = ";".join(filter_parts) + ";" + ";".join(chains)
        ts, cur = 0, 0
        for wd in words:
            while cur < len(segments)-1 and wd['start'] > segments[cur]['end']:
                cur += 1; ts += fade_duration
            wd['start'] = max(0, wd['start'] - ts); wd['end'] = max(wd['start'] + 0.1, wd['end'] - ts)
        return f_str, "[stackeda]"
    return filter_parts[0].replace(v_names[0], "[stacked]").replace(a_names[0], "[stackeda]"), "[stackeda]"

def transcribe_audio(video_path, start_time, duration):
    print("  Transcribing audio with Whisper (base)...")
    temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp.close()
    try:
        subprocess.run(['ffmpeg', '-y', '-ss', str(start_time), '-t', str(duration), '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', temp.name], capture_output=True)
        model = whisper.load_model('medium')
        res = model.transcribe(temp.name, word_timestamps=True, language='id')
        words = []
        for s in res.get('segments', []):
            for w in s.get('words', []): words.append({'word': w['word'].strip(), 'start': w['start'], 'end': w['end']})
        return words
    finally: os.unlink(temp.name)

def format_ass_time(s): return f"{int(s//3600)}:{int((s%3600)//60):02d}:{int(s%60):02d}.{int((s%1)*100):02d}"

def generate_ass_subtitle(words, analysis, title=None):
    if not words: return ""
    segments, phrases, cur_p, cur_s = analysis['segments'], [], [], None
    for w in words:
        if cur_s is None: cur_s = w['start']
        cur_p.append(w['word'])
        if len(cur_p) >= 4 or w['word'].endswith(('.', '?', '!', ',')):
            phrases.append({'text': ' '.join(cur_p), 'start': cur_s, 'end': w['end']})
            cur_p, cur_s = [], None
    if cur_p: phrases.append({'text': ' '.join(cur_p), 'start': cur_s, 'end': words[-1]['end']})
    
    header = ("[Script Info]\nPlayResX: 1080\nPlayResY: 1920\n[V4+ Styles]\n"
              "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
              "Style: Default,Liberation Sans,54,&H00FFFFFF,&H000000FF,&H00000000,&HC0000000,-1,0,0,0,100,100,2,0,1,4,2,2,40,40,120,1\n"
              "Style: Hook,Liberation Sans,80,&H0000FFFF,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,2,0,3,10,0,6,60,60,150,1\n"
              "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    
    events = ""
    # Add Title Hook Banner if provided (first 3 seconds)
    if title:
        events += f"Dialogue: 1,0:00:00.00,0:00:03.00,Hook,,0,0,0,,{{\\fad(200,200)\\b1}}{title.upper()}\n"

    for p in phrases:
        mode = 'closeup'
        for s in segments:
            if s['start'] <= p['start'] <= s['end']: mode = s['mode']; break
        y, fs = (960, 48) if mode == 'wide' else (1800, 54)
        events += f"Dialogue: 0,{format_ass_time(p['start'])},{format_ass_time(p['end'])},Default,,0,0,0,,{{\\fs{fs}\\pos(540,{y})\\fad(100,100)}}{p['text']}\n"
    return header + events

def process_video(input_file, output_file, start_time, duration, ass_file=None, hook_title=None):
    print(f"  Starting video processing for {output_file}...")
    analysis = analyze_video(input_file, start_time, duration)
    if not analysis:
        print("  Error: Video analysis failed."); return
    actual_start = analysis['adjusted_start_time']
    words = transcribe_audio(input_file, actual_start, duration)
    emphasis = analyze_audio_emphasis(input_file, actual_start, duration)
    f_complex, a_stream = generate_crop_filter(analysis, duration, words, emphasis)
    if ass_file and words:
        with open(ass_file, 'w') as f: f.write(generate_ass_subtitle(words, analysis, title=hook_title))
        if os.path.exists(ass_file) and os.path.getsize(ass_file) > 0:
            f_complex += f";[stacked]ass={ass_file}[outv]"
            v_stream = "[outv]"
        else: v_stream = "[stacked]"
    else: v_stream = "[stacked]"
    
    print("  Executing FFmpeg...")
    cmd = ['ffmpeg', '-y', '-ss', str(actual_start), '-t', str(duration), '-i', input_file, '-filter_complex', f_complex, '-map', v_stream, '-map', a_stream, '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', output_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FFmpeg failed with exit code {result.returncode}\n  FFmpeg Error: {result.stderr}")
    else:
        if os.path.exists(output_file): print(f"  Successfully created {output_file} ({os.path.getsize(output_file)} bytes)")
        else: print(f"  Error: FFmpeg reported success but {output_file} was not created.")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', required=True)
    p.add_argument('-o', '--output', required=True)
    p.add_argument('-s', '--start', type=float, default=0)
    p.add_argument('-d', '--duration', type=float, default=60)
    p.add_argument('--ass-file')
    p.add_argument('--title')
    args = p.parse_args()
    process_video(args.input, args.output, args.start, args.duration, args.ass_file, args.title)
    print("Processing complete!")
