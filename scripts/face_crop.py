#!/usr/bin/env python3
"""
Face Detection Auto-Crop for Podcast Videos with Whisper Transcription
Uses MediaPipe to detect faces and Whisper to generate word-timed subtitles.
"""

from pathlib import Path
import os
import tempfile
import sys
import argparse
import subprocess

# Suppress MediaPipe/TF/OpenCV logging and force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_GPU_DISABLED'] = '1'  # Corrected from MEDIAPIPE_DISABLE_GPU
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Also for EGL/GL errors in headless environment
os.environ['MESA_LOADER_DRIVER_OVERRIDE'] = 'swrast'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ["GLOG_minloglevel"] = "2"

try:
    import cv2
    import mediapipe as mp
    import whisper
    import gc
except ImportError as e:
    print(f"Error: Required packages not installed: {e}")
    print("Install with: pip install opencv-python-headless mediapipe openai-whisper")
    sys.exit(1)


class FaceDetector:
    """MediaPipe-based face detector for video frames."""
    
    def __init__(self, min_confidence=0.4):
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0, # Try short range first as it's often more robust in podcasts
            min_detection_confidence=min_confidence
        )
    
    def close(self):
        """Clean up MediaPipe detector."""
        if hasattr(self, 'detector'):
            self.detector.close()
    
    def detect_faces(self, frame):
        if frame is None:
            return []
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = self.detector.process(rgb_frame)
        except Exception as e:
            print(f"  Debug: MediaPipe error during process: {e}")
            return []
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                # Convert relative to pixel coordinates
                fx = int(bbox.xmin * w)
                fy = int(bbox.ymin * h)
                fw = int(bbox.width * w)
                fh = int(bbox.height * h)
                print(f"      Face at rel_x={bbox.xmin:.2f}, rel_y={bbox.ymin:.2f}, rel_w={bbox.width:.2f}, rel_h={bbox.height:.2f}")
                faces.append((fx, fy, fw, fh))
        
        return faces


def extract_segment_h264(video_path, start_time, duration):
    """
    Extract a video segment and re-encode to H.264 for better OpenCV compatibility.
    This fixes issues with AV1 and other codecs that OpenCV struggles with.
    """
    temp_segment = tempfile.NamedTemporaryFile(suffix='_segment.mp4', delete=False)
    temp_segment.close()
    
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-t', str(duration),
        '-i', video_path,
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '23',
        '-an',  # No audio needed for face detection
        temp_segment.name
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Warning: Failed to extract H.264 segment: {result.stderr}")
        return None
    
    return temp_segment.name


def sample_frames(video_path, start_time, duration, sample_interval=2):
    """Sample frames from video at regular intervals."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int((start_time + duration) * fps)
    sample_frames_count = int(fps * sample_interval)
    
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = current_frame / fps
        frames.append((timestamp, frame))
        
        current_frame += sample_frames_count
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    
    cap.release()
    return frames


def sample_frames_from_segment(segment_path, sample_interval=2):
    """Sample frames from a pre-extracted H.264 segment (starts at 0)."""
    cap = cv2.VideoCapture(segment_path)
    if not cap.isOpened():
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_frames_count = int(fps * sample_interval)
    
    frames = []
    current_frame = 0
    
    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = current_frame / fps
        frames.append((timestamp, frame))
        current_frame += sample_frames_count
    
    cap.release()
    return frames


def analyze_video(video_path, start_time, duration):
    """Analyze video to determine face positions and camera angles."""
    detector = FaceDetector()
    temp_segment = None
    
    try:
        # First, get video dimensions and duration from original
        cap = cv2.VideoCapture(video_path)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        print(f"  Video stats: {video_width}x{video_height}, Duration: {video_duration:.1f}s")
        if start_time >= video_duration:
            print(f"  Warning: Start time ({start_time}s) is beyond video duration ({video_duration:.1f}s)!")
            # Fallback for out-of-bounds: try to sample the last few seconds instead of failing
            start_time = max(0, video_duration - duration - 1)
            print(f"  Adjusted start time to: {start_time:.1f}s")

        # Try to extract H.264 segment for better compatibility
        print("  Extracting H.264 segment for face detection...")
        temp_segment = extract_segment_h264(video_path, start_time, duration)
        
        sample_interval = 1.0 # Every 1 second for scene switching
        
        if temp_segment and os.path.exists(temp_segment):
            # Use the H.264 segment (timestamps start at 0)
            frames = sample_frames_from_segment(temp_segment, sample_interval=sample_interval)
            print(f"  Sampled {len(frames)} frames from H.264 segment")
        else:
            # Fallback to original video
            print("  Falling back to original video for sampling")
            frames = sample_frames(video_path, start_time, duration, sample_interval=sample_interval)
        
        all_faces = []
        print(f"  Debug: Starting face detection on {len(frames)} frames...")
        
        # Temporal analysis: classify each frame
        scene_data = []
        
        for i, (timestamp, frame) in enumerate(frames):
            if frame is not None:
                faces = detector.detect_faces(frame)
                
                # Podcast detection at frame level
                # Look at bounding box positions instead of center
                frame_has_left = any(face[0] / video_width < 0.48 for face in faces)
                frame_has_right = any((face[0] + face[2]) / video_width > 0.52 for face in faces)
                
                # Mode for this specific frame
                is_wide_frame = len(faces) >= 2 or (frame_has_left and frame_has_right)
                
                # Find most prominent face position (for close-up mode)
                if faces:
                    # Sort faces by size (width * height) and take the largest
                    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    fx, fy, fw, fh = faces_sorted[0]
                    prominent_x = (fx + fw // 2) / video_width
                else:
                    prominent_x = 0.5
                
                scene_data.append({
                    'timestamp': timestamp,
                    'is_wide': is_wide_frame,
                    'face_count': len(faces),
                    'prominent_x': prominent_x
                })
                
                if len(faces) > 0:
                    print(f"    Frame {i} at {timestamp:.1f}s: Detected {len(faces)} faces (Wide: {is_wide_frame})")
            else:
                scene_data.append({
                    'timestamp': timestamp,
                    'is_wide': False,
                    'face_count': 0,
                    'prominent_x': 0.5
                })
        
        if not scene_data:
             # Failsafe if no frames were read
             scene_data.append({'timestamp': 0, 'is_wide': False, 'face_count': 0, 'prominent_x': 0.5})
        
        # Smooth the mode decisions to prevent rapid flickering (hysteresis/smoothing)
        # We'll use a simple window of 3 samples (1s before, current, 1s after)
        final_scenes = []
        for i in range(len(scene_data)):
            window = scene_data[max(0, i-1):min(len(scene_data), i+2)]
            wide_votes = sum(1 for s in window if s['is_wide'])
            is_wide_smooth = wide_votes >= len(window) / 2
            
            # Weighted average of prominent_x for smooth tracking
            total_weight = 0
            weighted_x = 0
            for j, s in enumerate(window):
                weight = 2 if j == 1 else 1 # Current frame has more weight
                weighted_x += s['prominent_x'] * weight
                total_weight += weight
            avg_x_smooth = weighted_x / total_weight
            
            final_scenes.append({
                'start': scene_data[i]['timestamp'],
                'end': scene_data[i]['timestamp'] + sample_interval,
                'is_wide': is_wide_smooth,
                'avg_x': avg_x_smooth
            })

        # Calculate overall stats for backward compatibility if needed
        face_counts = [f['face_count'] for f in scene_data]
        avg_faces = sum(face_counts) / len(face_counts) if face_counts else 0
        
        return {
            'video_width': video_width,
            'video_height': video_height,
            'avg_face_count': avg_faces,
            'scenes': final_scenes,
            'is_wide_shot': any(s['is_wide'] for s in final_scenes) # If any part is wide, we might need special handling
        }
        
    finally:
        detector.close()
        # Cleanup temp segment
        if temp_segment and os.path.exists(temp_segment):
            try:
                os.unlink(temp_segment)
            except:
                pass


def transcribe_audio(video_path, start_time, duration, model_name='tiny'):
    """
    Transcribe audio from video segment using Whisper.
    Returns list of word segments with timestamps.
    """
    print(f"  Transcribing audio with Whisper ({model_name})...")
    
    # Extract audio segment to temp file
    temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_audio.close()
    
    try:
        # Extract audio using ffmpeg
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-t', str(duration),
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',
            '-ar', '16000',  # Whisper expects 16kHz
            '-ac', '1',  # Mono
            temp_audio.name
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        
        # Load Whisper model and transcribe
        model = whisper.load_model(model_name)
        # Try to transcribe with auto-detect (language=None)
        result = model.transcribe(
            temp_audio.name,
            word_timestamps=True,
            language=None  # Auto-detect language
        )
        
        # Extract word-level timestamps
        words = []
        for segment in result.get('segments', []):
            for word_info in segment.get('words', []):
                words.append({
                    'word': word_info['word'].strip(),
                    'start': word_info['start'],
                    'end': word_info['end']
                })
        
        print(f"  Transcribed {len(words)} words")
        
        # Explicitly free memory
        del model
        gc.collect()
        
        return words
        
    finally:
        # Cleanup temp file
        if os.path.exists(temp_audio.name):
            os.unlink(temp_audio.name)


def generate_word_timed_ass(words, is_wide_shot, duration=60):
    """
    Generate ASS subtitle with word-by-word timing from Whisper.
    Groups words into phrases (2-4 words) for readable display.
    """
    if not words:
        return generate_fallback_ass(is_wide_shot, duration)
    
    # Group words into phrases (3-5 words each for readability)
    phrases = []
    current_phrase = []
    current_start = None
    
    for word_info in words:
        if current_start is None:
            current_start = word_info['start']
        
        current_phrase.append(word_info['word'])
        
        # Create new phrase every 4-5 words or at natural breaks
        if len(current_phrase) >= 4 or word_info['word'].endswith(('.', '?', '!', ',')):
            phrases.append({
                'text': ' '.join(current_phrase),
                'start': current_start,
                'end': word_info['end']
            })
            current_phrase = []
            current_start = None
    
    # Add remaining words
    if current_phrase:
        phrases.append({
            'text': ' '.join(current_phrase),
            'start': current_start,
            'end': words[-1]['end']
        })
    
    # Generate ASS content
    if is_wide_shot:
        margin_v = 880  # Middle between panels
        font_size = 52
        style_name = "WideStyle"
    else:
        margin_v = 80  # Bottom
        font_size = 58
        style_name = "CloseUpStyle"
    
    ass_content = f"""[Script Info]
Title: Whisper Synced Subtitle
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: {style_name},Arial Black,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&HC0000000,-1,0,0,0,100,100,2,0,1,4,3,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    for phrase in phrases:
        start_time = format_ass_time(phrase['start'])
        end_time = format_ass_time(phrase['end'])
        
        # Add fade and pop animation
        animated_text = r"{\fad(150,100)\t(0,100,\fscx102\fscy102)\t(100,200,\fscx100\fscy100)}" + phrase['text']
        
        ass_content += f"Dialogue: 0,{start_time},{end_time},{style_name},,0,0,0,,{animated_text}\n"
    
    return ass_content


def format_ass_time(seconds):
    """Format seconds to ASS time format: H:MM:SS.CC"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


def generate_fallback_ass(is_wide_shot, duration):
    """Generate fallback ASS when transcription fails."""
    if is_wide_shot:
        margin_v = 880
        font_size = 52
        style_name = "WideStyle"
    else:
        margin_v = 80
        font_size = 58
        style_name = "CloseUpStyle"
    
    return f"""[Script Info]
Title: Fallback Subtitle
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: {style_name},Arial Black,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&HC0000000,-1,0,0,0,100,100,2,0,1,4,3,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:01.00,{style_name},,0,0,0,,
"""


def generate_smart_crop_filter(analysis):
    """Generate dynamic FFmpeg filter based on scene analysis."""
    w = analysis['video_width']
    h = analysis['video_height']
    scenes = analysis['scenes']
    
    out_w = 1080
    out_h = 1920
    panel_h = out_h // 2
    
    # Estimate stable left/right positions from all wide scenes
    # Better: use fixed 0.25 and 0.75 for split-screen stability
    crop_w_wide = min(w // 2, int(h * 1.125))
    crop_h_wide = int(crop_w_wide * 960 / 1080)
    
    left_x = max(0, min(int(0.25 * w) - crop_w_wide // 2, w // 2 - crop_w_wide))
    right_x = max(w // 2, min(int(0.75 * w) - crop_w_wide // 2, w - crop_w_wide))
    
    # Prepare Close-up parameters
    closeup_x_values = [s['avg_x'] for s in scenes if not s['is_wide']]
    avg_closeup_x = sum(closeup_x_values) / len(closeup_x_values) if closeup_x_values else 0.5
    
    crop_w_close = int(h * 9 / 16)
    crop_h_close = h
    crop_x_close = int(avg_closeup_x * w) - crop_w_close // 2
    crop_x_close = max(0, min(crop_x_close, w - crop_w_close))
    
    # 3. Combine using overlay and enable
    wide_intervals = []
    current_start = None
    
    for s in scenes:
        if s['is_wide']:
            if current_start is None:
                current_start = s['start']
        else:
            if current_start is not None:
                wide_intervals.append((current_start, s['start']))
                current_start = None
    if current_start is not None:
        wide_intervals.append((current_start, scenes[-1]['end']))
    
    if not wide_intervals:
        # No wide shots at all
        return f"[0:v]crop={crop_w_close}:{crop_h_close}:{crop_x_close}:0,scale={out_w}:{out_h}[stacked]", False
    
    if len(wide_intervals) == 1 and wide_intervals[0][0] == scenes[0]['start'] and wide_intervals[0][1] == scenes[-1]['end']:
        # All wide shots
        filter_all_wide = (
            f"[0:v]split=2[v_split_l][v_split_r];"
            f"[v_split_l]crop={crop_w_wide}:{crop_h_wide}:{left_x}:0,scale={out_w}:{panel_h}[left];"
            f"[v_split_r]crop={crop_w_wide}:{crop_h_wide}:{right_x}:0,scale={out_w}:{panel_h}[right];"
            f"[left][right]vstack=inputs=2[stacked]"
        )
        return filter_all_wide, True
    
    # Build enable expression for wide shots
    enable_expr = "+".join([f"between(t,{start:.2f},{end:.2f})" for start, end in wide_intervals])
    
    # DYNAMIC FILTER: requires splitting the input because it's used for left, right, and closeup
    full_filter = (
        f"[0:v]split=3[vwide1][vwide2][vclose];"
        f"[vwide1]crop={crop_w_wide}:{crop_h_wide}:{left_x}:0,scale={out_w}:{panel_h}[left];"
        f"[vwide2]crop={crop_w_wide}:{crop_h_wide}:{right_x}:0,scale={out_w}:{panel_h}[right];"
        f"[left][right]vstack=inputs=2[split_v];"
        f"[vclose]crop={crop_w_close}:{crop_h_close}:{crop_x_close}:0,scale={out_w}:{out_h}[closeup_v];"
        f"[closeup_v][split_v]overlay=enable='{enable_expr}'[stacked]"
    )
    
    return full_filter, True


def process_video(input_file, output_file, start_time, duration, ass_file=None, use_whisper=True):
    """Process video with face detection, smart cropping, and Whisper subtitles."""
    print(f"Processing video: {input_file}")
    print(f"  Start: {start_time}s, Duration: {duration}s")
    
    # Analyze video for face positions
    analysis = analyze_video(input_file, start_time, duration)
    print(f"  Detected avg faces: {analysis['avg_face_count']:.1f}")
    print(f"  Wide shot: {analysis['is_wide_shot']}")
    
    # Generate smart crop filter
    crop_filter, is_wide = generate_smart_crop_filter(analysis)
    print(f"  Mode: {'WIDE (2-speaker)' if is_wide else 'CLOSE-UP (1-speaker)'}")
    
    # Transcribe audio with Whisper and generate timed subtitles
    if use_whisper and ass_file:
        try:
            words = transcribe_audio(input_file, start_time, duration, model_name='tiny')
            ass_content = generate_word_timed_ass(words, is_wide, duration)
            print(f"  Generated word-timed subtitle")
        except Exception as e:
            print(f"  Whisper error: {e}, using fallback")
            ass_content = generate_fallback_ass(is_wide, duration)
        
        with open(ass_file, 'w', encoding='utf-8') as f:
            f.write(ass_content)
        print(f"  Subtitle saved to: {ass_file}")
    
    # Add ASS overlay to filter
    if ass_file and os.path.exists(ass_file):
        full_filter = f"{crop_filter};[stacked]ass={ass_file}[outv]"
        output_stream = "[outv]"
    else:
        full_filter = crop_filter
        output_stream = "[stacked]"
    
    print(f"  Debug: crop_filter={crop_filter}")
    print(f"  Debug: full_filter={full_filter}")
    print(f"  Debug: output_stream={output_stream}")
    
    # Build FFmpeg command
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-t', str(duration),
        '-i', input_file,
        '-filter_complex', full_filter,
        '-map', output_stream,
        '-map', '0:a?',  # Make audio optional to prevent failure if audio stream is missing
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        output_file
    ]
    
    print(f"  Running FFmpeg...")
    # Add env to subprocess for FFmpeg just in case
    ffmpeg_env = os.environ.copy()
    result = subprocess.run(cmd, capture_output=True, text=True, env=ffmpeg_env)
    
    if result.returncode != 0:
        print(f"  FFmpeg primary run failed (code {result.returncode})")
        print(f"  Error: {result.stderr}")
        # Try to run without ASS as fallback if it failed
        if ass_file:
            print("  Retrying without ASS overlay...")
            cmd_idx_filter = cmd.index('-filter_complex')
            cmd[cmd_idx_filter + 1] = crop_filter
            cmd_idx_map = cmd.index('-map')
            cmd[cmd_idx_map + 1] = "[stacked]"
            result_retry = subprocess.run(cmd, capture_output=True, text=True)
            if result_retry.returncode != 0:
                print(f"  FFmpeg retry also failed: {result_retry.stderr}")
                raise RuntimeError(f"FFmpeg failed: {result_retry.stderr}")
        else:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
    
    # Check if output file is valid (not just a tiny header)
    if os.path.exists(output_file) and os.path.getsize(output_file) < 1000:
        print(f"  Warning: Output file is very small ({os.path.getsize(output_file)} B). Process might have failed silently.")
        if ass_file:
            print("  Retrying without ASS overlay (due to small file size)...")
            cmd_idx_filter = cmd.index('-filter_complex')
            cmd[cmd_idx_filter + 1] = crop_filter
            cmd_idx_map = cmd.index('-map')
            cmd[cmd_idx_map + 1] = "[stacked]"
            result_retry = subprocess.run(cmd, capture_output=True, text=True)
            if result_retry.returncode != 0:
                 print(f"  FFmpeg small-file retry failed: {result_retry.stderr}")
    
    print(f"  Output saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Face detection auto-crop with Whisper subtitles')
    parser.add_argument('--input', '-i', required=True, help='Input video file')
    parser.add_argument('--output', '-o', required=True, help='Output video file')
    parser.add_argument('--start', '-s', type=float, default=0, help='Start time in seconds')
    parser.add_argument('--duration', '-d', type=float, default=60, help='Duration in seconds')
    parser.add_argument('--ass-file', help='Path to save ASS subtitle file')
    parser.add_argument('--no-whisper', action='store_true', help='Disable Whisper transcription')
    
    args = parser.parse_args()
    
    try:
        process_video(
            args.input,
            args.output,
            args.start,
            args.duration,
            args.ass_file,
            use_whisper=not args.no_whisper
        )
        print("Processing complete!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
