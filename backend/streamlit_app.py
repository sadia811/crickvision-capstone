"""CrickVision capstone demo: image/video object detection and RTMPose-S.

Requires your own checkpoints in backend/models and MMPose dependencies installed.
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


@st.cache_resource(show_spinner="Loading RTMPose-S...")
def get_pose_model():
    require([RT_WEIGHTS, RT_CONFIG, RT_META])
    # The exported training config references Kaggle-only metainfo paths.
    # Patch these without changing the trained architecture or keypoint order.
    from mmengine.config import Config
    from mmpose.utils import register_all_modules
    from mmpose.apis import init_model
    import torch

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(RT_CONFIG))
    for key in ("train_dataloader", "val_dataloader", "test_dataloader"):
        if key in cfg:
            cfg[key].dataset.metainfo = dict(from_file=str(RT_META))
    cfg.load_from = None
    cfg.resume = False
    cfg.work_dir = str(BASE / ".inference_workdir")
    # init_model accepts a Config instance (no need to create a new config file).
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = init_model(cfg, str(RT_WEIGHTS), device=device)
    if model.cfg.model.head.out_channels != 13:
        raise RuntimeError("RTMPose checkpoint/config must produce exactly 13 keypoints.")
    return model


def people_boxes(img_bgr, confidence=0.12):
    detector = get_person_model()
    results = detector.predict(img_bgr, conf=confidence, iou=0.5, classes=[0], max_det=20, verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return (results.boxes.xyxy.cpu().numpy().astype(np.float32),
            results.boxes.conf.cpu().numpy().astype(np.float32))


def get_poses(img_bgr, boxes):
    if len(boxes) == 0:
        return []
    from mmpose.apis import inference_topdown
    pose_samples = inference_topdown(get_pose_model(), img_bgr, boxes, bbox_format="xyxy")
    if len(pose_samples) != len(boxes):
        raise RuntimeError("Pose/bounding-box prediction count mismatch")
    extracted = []
    for sample in pose_samples:
        inst = sample.pred_instances
        pts = np.asarray(inst.keypoints, np.float32)
        scores = np.asarray(inst.keypoint_scores, np.float32)
        if pts.ndim == 3:
            pts = pts[0]
        if scores.ndim == 2:
            scores = scores[0]
        if pts.shape != (13, 2) or scores.shape != (13,):
            raise RuntimeError(f"Unexpected RTMPose output shape: {pts.shape}, {scores.shape}")
        extracted.append((pts, scores))
    return extracted


def draw_poses(image, boxes, poses, ids=None, scores=None, min_conf=0.25):
    out = image.copy()
    for i, (box, (pts, kp_scores)) in enumerate(zip(boxes, poses)):
        color = (56, 177, 52) if ids is None else ((43*i+80)%220+30, (97*i+50)%220+30, (139*i+90)%220+30)
        x1,y1,x2,y2 = np.round(box).astype(int)
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        tag = f"Player #{ids[i]}" if ids is not None else f"Player {i+1}"
        if scores is not None: tag += f" ({scores[i]:.2f})"
        cv2.putText(out,tag,(max(0,x1),max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
        for a,b in EDGES:
            if kp_scores[a]>=min_conf and kp_scores[b]>=min_conf:
                cv2.line(out,tuple(np.rint(pts[a]).astype(int)),tuple(np.rint(pts[b]).astype(int)),color,2,cv2.LINE_AA)
        for p,score in zip(pts,kp_scores):
            if score>=min_conf:
                cv2.circle(out,tuple(np.rint(p).astype(int)),3,color,-1,cv2.LINE_AA)
    return out


def detect_image(img_bgr, conf):
    result = get_object_model().predict(img_bgr, conf=conf, imgsz=640, verbose=False)[0]
    count = len(result.boxes) if result.boxes is not None else 0
    return result.plot(), count


def pose_image(img_bgr):
    boxes, scores = people_boxes(img_bgr)
    poses = get_poses(img_bgr, boxes)
    return draw_poses(img_bgr, boxes, poses, scores=scores), len(poses)


def resize_frame(frame, max_width=848):
    h,w = frame.shape[:2]
    if w<=max_width: return frame
    return cv2.resize(frame, (max_width,round(h*max_width/w)))


def transcode_h264(src):
    if not shutil.which("ffmpeg"):
        return src, False
    dst = src.with_name("crickvision_h264.mp4")
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-an", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    run = subprocess.run(command, capture_output=True, text=True)
    return (dst, True) if run.returncode == 0 and dst.is_file() else (src, False)


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
                if not h264:
                    st.warning("FFmpeg/H.264 conversion was unavailable. Downloaded video may not play in every browser.")
                st.video(str(output))
                st.download_button("Download annotated MP4",output.read_bytes(),file_name="crickvision_result.mp4",mime="video/mp4")
    except Exception as exc:
        st.error(f"Inference could not run: {exc}")
        st.caption("Verify the checkpoint paths, RTMPose package versions, and available CPU/GPU memory. See README.md.")
