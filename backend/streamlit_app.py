"""CrickVision capstone demo: image/video object detection and RTMPose-S.

Uses the original checkpoints; RTMPose-S website inference runs via ONNX Runtime.
The training notebook's evaluation metrics are NOT inferred from uploaded media.
"""
from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
CONFIGS = BASE / "configs"
OBJECT_WEIGHTS = MODELS / "ijepa_yolov12n.pt"
RT_WEIGHTS = MODELS / "rtmpose_s.pth"
RT_ONNX = MODELS / "crickvision_rtmpose_s_13kpt.onnx"
PERSON_WEIGHTS = MODELS / "yolov8n.pt"
RT_CONFIG = CONFIGS / "rtmpose_s_crickvision_v3.py"
RT_META = CONFIGS / "crickvision_pose13.py"
TRACK_CONFIG = CONFIGS / "cricket_bytetrack_kalman_clean.yaml"

# 0=head, 1=left shoulder, 2=right shoulder, 3=left hip, 4=right hip,
# 5=left knee, 6=left ankle, 7=right knee, 8=right ankle,
# 9=left elbow, 10=left wrist, 11=right elbow, 12=right wrist.
EDGES = [(0,1),(0,2),(1,2),(1,9),(9,10),(2,11),(11,12),
         (1,3),(2,4),(3,4),(3,5),(5,6),(4,7),(7,8)]
STATIC_NAMES = {"boundaryline", "creaseline"}

st.set_page_config(page_title="CrickVision | AI Vision Studio", page_icon="🏏", layout="wide")
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #f7f9fc; }
  [data-testid="stMainBlockContainer"] { max-width: 1100px; padding-top: 20px; padding-bottom: 36px; }
  .cv-intro { display:flex; align-items:center; gap:19px; padding:21px 24px; margin:0 0 14px; background:#ffffff; border:1px solid #e0e8ed; border-radius:18px; }
  .cv-intro-icon { width:65px; height:65px; min-width:65px; display:grid; place-items:center; background:#e6f5ef; border:1px solid #cbe8dc; border-radius:17px; font-size:35px; }
  .cv-intro-label { color:#137f71; font-size:11px; font-weight:750; letter-spacing:1.4px; text-transform:uppercase; margin-bottom:3px; }
  .cv-intro h1 { color:#142338; font-size:29px; font-weight:780; letter-spacing:-.7px; line-height:1.2; margin:0 0 6px; padding:0; }
  .cv-intro p { color:#64748b; font-size:14px; line-height:1.5; margin:0; }
  .stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] { background:#0f8275; color:#fff; border-radius:12px; border:0; }
  .stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover { background:#0b6a60; color:#fff; border:0; }
  [data-testid="stFileUploader"] { border-radius:12px; }
  [data-testid="stFooter"] { display:none; }
  @media (max-width:720px) {
    .cv-intro { padding:16px; gap:12px; }
    .cv-intro-icon { width:49px; min-width:49px; height:49px; border-radius:13px; font-size:28px; }
    .cv-intro h1 { font-size:23px; }
    .cv-intro p { font-size:12px; }
  }
</style>
""", unsafe_allow_html=True)


def require(paths):
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing: " + ", ".join(str(p.relative_to(BASE)) for p in missing))


@st.cache_resource(show_spinner="Loading cricket-object detector...")
def get_object_model():
    require([OBJECT_WEIGHTS])
    return YOLO(str(OBJECT_WEIGHTS))


@st.cache_resource(show_spinner="Loading player detector...")
def get_person_model():
    require([PERSON_WEIGHTS])
    return YOLO(str(PERSON_WEIGHTS))


@st.cache_resource(show_spinner="Loading RTMPose-S (ONNX Runtime)...")
def get_pose_model():
    """Load same trained RTMPose-S model, exported to ONNX for website inference."""
    require([RT_ONNX])
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(RT_ONNX), sess_options=opts, providers=["CPUExecutionProvider"]
    )


def people_boxes(img_bgr, confidence=0.12):
    detector = get_person_model()
    results = detector.predict(img_bgr, conf=confidence, iou=0.5, classes=[0], max_det=20, verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return (results.boxes.xyxy.cpu().numpy().astype(np.float32),
            results.boxes.conf.cpu().numpy().astype(np.float32))


def get_poses(img_bgr, boxes):
    """Custom RTMPose-S ONNX inference with MMPose-matched preprocessing.

    The original config: 192x256 input, RGB normalization, bbox padding 1.25,
    13 keypoints, SimCC ratio 2, horizontal flip-test enabled.
    YOLOv8n supplies the same person boxes. Video tracking code is unchanged.
    """
    if len(boxes) == 0:
        return []

    session = get_pose_model()
    input_name = session.get_inputs()[0].name
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    w, h = 192.0, 256.0
    # Pair indices from the user's crickvision_pose13.py metadata.
    flip_indices = [0, 2, 1, 4, 3, 7, 8, 5, 6, 11, 12, 9, 10]
    extracted = []

    for box in boxes:
        x1, y1, x2, y2 = (float(v) for v in box)
        center = np.array([(x1 + x2)*.5, (y1 + y2)*.5], dtype=np.float32)
        scale = np.maximum(np.array([x2 - x1, y2 - y1], dtype=np.float32), 1.0) * 1.25
        aspect = w / h
        if scale[0] > scale[1] * aspect:
            scale[1] = scale[0] / aspect
        else:
            scale[0] = scale[1] * aspect

        # Equivalent to MMPose TopdownAffine -> get_warp_matrix at 0 degrees.
        warp_mat = np.array([
            [w/scale[0], 0.0, w*(0.5-center[0]/scale[0])],
            [0.0, h/scale[1], h*(0.5-center[1]/scale[1])],
        ], dtype=np.float32)
        crop = cv2.warpAffine(img_bgr, warp_mat, (192,256), flags=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        inp = np.ascontiguousarray(((rgb - mean) / std).transpose(2,0,1)[None], dtype=np.float32)

        sx, sy = session.run(None, {input_name: inp})
        if sx.shape != (1,13,384) or sy.shape != (1,13,512):
            raise RuntimeError(f"Unexpected ONNX SimCC output shapes: {sx.shape} and {sy.shape}")

        # Original model has test_cfg.flip_test=True. Mirror that at inference:
        # flip input, swap left/right keypoint indices, mirror x-axis logits,
        # then average predictions prior to SimCC decoding.
        inp_flip = np.ascontiguousarray(inp[:,:,:,::-1])
        sx_flip, sy_flip = session.run(None, {input_name: inp_flip})
        sx = (sx + sx_flip[:,flip_indices,::-1]) * 0.5
        sy = (sy + sy_flip[:,flip_indices,:]) * 0.5

        xloc = np.argmax(sx[0], axis=-1)
        yloc = np.argmax(sy[0], axis=-1)
        xval = sx[0].max(axis=-1)
        yval = sy[0].max(axis=-1)
        scores = np.minimum(xval, yval).astype(np.float32)
        pts = np.stack([xloc,yloc], axis=-1).astype(np.float32) / 2.0
        pts[:,0] = (pts[:,0] - w/2) * scale[0]/w + center[0]
        pts[:,1] = (pts[:,1] - h/2) * scale[1]/h + center[1]
        pts[scores <= 0.0] = -1.0
        extracted.append((pts,scores))

    return extracted


def draw_poses(image, boxes, poses, ids=None, scores=None, min_conf=0.25):
    out = image.copy()
    for i, (box, (pts, kp_scores)) in enumerate(zip(boxes, poses)):
        color = (166, 82, 20)  # dark blue, OpenCV BGR; RGB #1452A6
        x1,y1,x2,y2 = np.round(box).astype(int)
        cv2.rectangle(out,(x1,y1),(x2,y2),color,1,cv2.LINE_AA)
        tag = f"Player #{ids[i]}" if ids is not None else f"Player {i+1}"
        if scores is not None: tag += f" ({scores[i]:.2f})"
        cv2.putText(out,tag,(max(0,x1),max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
        for a,b in EDGES:
            if kp_scores[a]>=min_conf and kp_scores[b]>=min_conf:
                cv2.line(out,tuple(np.rint(pts[a]).astype(int)),tuple(np.rint(pts[b]).astype(int)),color,1,cv2.LINE_AA)
        for p,score in zip(pts,kp_scores):
            if score>=min_conf:
                cv2.circle(out,tuple(np.rint(p).astype(int)),2,color,-1,cv2.LINE_AA)
    return out


def detect_image(img_bgr, conf):
    result = get_object_model().predict(img_bgr, conf=conf, imgsz=640, verbose=False)[0]
    count = len(result.boxes) if result.boxes is not None else 0
    return result.plot(), count


def pose_image(img_bgr):
    boxes, scores = people_boxes(img_bgr)
    poses = get_poses(img_bgr, boxes)
    return draw_poses(img_bgr, boxes, poses, scores=scores), len(poses)


def resize_frame(frame, max_width=1280):
    h,w = frame.shape[:2]
    if w<=max_width: return frame
    return cv2.resize(frame, (max_width,round(h*max_width/w)))


def transcode_h264(src):
    """Convert OpenCV mp4v to browser-ready H.264 without cropping the frame.

    Use system FFmpeg when available, otherwise the imageio-ffmpeg binary.
    Raise an explicit error instead of silently serving unplayable MP4V.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg = get_ffmpeg_exe()
    dst = src.with_name("crickvision_h264.mp4")
    command = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-map", "0:v:0", "-an",
        # H.264/YUV420P require even dimensions; pad at most one pixel.
        # This preserves the entire image without cropping or stretching.
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-threads", "2", str(dst),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
        error = result.stderr.strip()[-1500:] or "no encoder output"
        raise RuntimeError("H.264 video encoding failed: " + error)
    return dst, True

def process_video(input_path, task, seconds, conf, progress):
    import supervision as sv
    capture = cv2.VideoCapture(str(input_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps<=0 or fps>240: fps=25.0
    total = int(min(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 999999, round(seconds*fps)))
    total = max(1,total)
    success,first = capture.read()
    if not success: capture.release(); raise RuntimeError("Could not read this MP4 video")
    first = resize_frame(first)
    height,width = first.shape[:2]
    raw = input_path.with_name("crickvision_output_raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    if not writer.isOpened(): capture.release(); raise RuntimeError("OpenCV could not create output MP4")

    if task == "Cricket object detection":
        require([TRACK_CONFIG])
        # New model per video: prevents tracking IDs leaking across uploads.
        detector = YOLO(str(OBJECT_WEIGHTS))
        names = detector.names
        if isinstance(names,list): names=dict(enumerate(names))
        moving = [int(k) for k,n in names.items() if "".join(ch for ch in str(n).lower() if ch.isalnum()) not in STATIC_NAMES]
        static = [int(k) for k in names if int(k) not in moving]
        history = defaultdict(lambda:deque(maxlen=25))
    else:
        detector = get_person_model()
        tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=45,
                               minimum_matching_threshold=0.80, frame_rate=max(1,int(round(fps))))
        get_pose_model()
        history = defaultdict(lambda:deque(maxlen=25))

    processed=0
    try:
        while processed < total:
            if processed==0: frame = first
            else:
                ret,frame = capture.read()
                if not ret: break
                frame = resize_frame(frame)
            if task == "Cricket object detection":
                results = detector.track(frame, tracker=str(TRACK_CONFIG), persist=True,
                                         conf=min(conf,0.05), iou=0.50, imgsz=640,
                                         classes=moving, verbose=False)[0]
                canvas = results.plot()
                if results.boxes is not None and results.boxes.id is not None:
                    for box,tid in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.id.int().cpu().tolist()):
                        x1,y1,x2,y2 = box
                        pt = (int((x1+x2)/2),int((y1+y2)/2))
                        history[tid].append(pt)
                        arr=np.array(history[tid],dtype=np.int32)
                        if len(arr)>1: cv2.polylines(canvas,[arr],False,(0,200,255),2)
                # Static geometry must still appear as detections, without persistent IDs.
                if static:
                    static_result = detector.predict(frame,classes=static,conf=conf,imgsz=640,verbose=False)[0]
                    if static_result.boxes is not None:
                        for box, cls in zip(static_result.boxes.xyxy.cpu().numpy(), static_result.boxes.cls.int().cpu().tolist()):
                            x1,y1,x2,y2 = np.rint(box).astype(int)
                            cv2.rectangle(canvas,(x1,y1),(x2,y2),(135,90,255),2)
                            cv2.putText(canvas,str(names[cls]),(x1,max(y1-5,15)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(135,90,255),2)
            else:
                boxes,scores=people_boxes(frame)
                dets = sv.Detections(xyxy=boxes, confidence=scores, class_id=np.zeros(len(boxes),dtype=int)) if len(boxes) else sv.Detections.empty()
                tracked=tracker.update_with_detections(dets)
                tracked_boxes=np.asarray(tracked.xyxy,dtype=np.float32).reshape(-1,4)
                poses=get_poses(frame,tracked_boxes)
                ids=[int(t) for t in tracked.tracker_id] if tracked.tracker_id is not None else list(range(len(tracked_boxes)))
                canvas=draw_poses(frame,tracked_boxes,poses,ids=ids,scores=tracked.confidence)
                for box,tid in zip(tracked_boxes,ids):
                    pt = tuple(np.rint([(box[0]+box[2])/2,(box[1]+box[3])/2]).astype(int))
                    history[tid].append(pt)
                    pts=np.array(history[tid],dtype=np.int32)
                    if len(pts)>1:cv2.polylines(canvas,[pts],False,(0,255,255),2)
            writer.write(canvas)
            processed += 1
            if processed % max(1,int(fps)//2)==0 or processed==total:
                progress.progress(min(1.0,processed/total),text=f"Processed {processed}/{total} frames")
    finally:
        capture.release();writer.release()
    if processed==0: raise RuntimeError("No video frames were processed")
    output,compatible=transcode_h264(raw)
    return output,processed,compatible


st.markdown("""
<div class="cv-intro">
  <span class="cv-intro-icon" aria-hidden="true">🏏</span>
  <div>
    <div class="cv-intro-label">Images · Videos · Tracking</div>
    <h1>Analysis Workspace</h1>
    <p>Select an analysis task and upload an image or MP4 to see model predictions.</p>
  </div>
</div>
""", unsafe_allow_html=True)
left,right=st.columns(2, gap="large")
with left:
    st.markdown("#### Input media")
    upload=st.file_uploader("Select an image or MP4 video",type=["jpg","jpeg","png","mp4"])
    if upload:
        is_video=upload.name.lower().endswith(".mp4")
        if is_video:st.video(upload)
        else:st.image(upload)
with right:
    st.markdown("#### Analysis settings")
    task=st.radio("Select task",["Cricket object detection","Player keypoint estimation (RTMPose-S)"])
    conf=st.slider("Object confidence threshold",0.05,0.95,0.25,0.05)
    max_sec=st.slider("Video seconds to process (demo limit)",2,20,5) if upload and upload.name.lower().endswith(".mp4") else 5
    run=st.button("▶ Run analysis",type="primary",use_container_width=True,disabled=upload is None)

if run and upload:
    started=time.perf_counter()
    try:
        if not is_video:
            data=np.frombuffer(upload.getvalue(),dtype=np.uint8)
            orig=cv2.imdecode(data,cv2.IMREAD_COLOR)
            if orig is None:raise ValueError("Not a readable image")
            with st.spinner("Running inference..."):
                out,count=(detect_image(orig,conf) if task=="Cricket object detection" else pose_image(orig))
            st.markdown("### Analysis results")
            a,b=st.columns(2)
            a.image(cv2.cvtColor(orig,cv2.COLOR_BGR2RGB),caption="Original image",use_container_width=True)
            b.image(cv2.cvtColor(out,cv2.COLOR_BGR2RGB),caption="Annotated output",use_container_width=True)
            c1,c2,c3=st.columns(3)
            c1.metric("Detections / players",count)
            c2.metric("Processing time",f"{time.perf_counter()-started:.2f}s")
            c3.metric("Model", "I-JEPA + YOLOv12n" if task=="Cricket object detection" else "RTMPose-S")
            ok,encoded=cv2.imencode('.png',out)
            if ok:st.download_button("Download annotated image",encoded.tobytes(),file_name="crickvision_result.png",mime="image/png")
        else:
            with tempfile.TemporaryDirectory(prefix="crickvision_") as temp:
                source=Path(temp)/"input.mp4"
                source.write_bytes(upload.getvalue())
                progress=st.progress(0,text="Preparing video...")
                with st.spinner("Processing video..."):
                    output,frames,h264=process_video(source,task,max_sec,conf,progress)
                progress.empty()
                st.success(f"Processed {frames} frames in {time.perf_counter()-started:.1f} seconds")
                final_video_bytes = output.read_bytes()
                st.video(final_video_bytes, format="video/mp4")
                st.download_button("Download annotated MP4", final_video_bytes,
                                   file_name="crickvision_result.mp4", mime="video/mp4")
    except Exception as exc:
        st.error(f"Inference could not run: {exc}")
        st.caption("Verify the checkpoint paths, RTMPose package versions, and available CPU/GPU memory. See deployment logs.")
