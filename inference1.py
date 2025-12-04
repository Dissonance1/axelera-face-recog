#!/usr/bin/env python
# Copyright Axelera AI, 2025
# Voyager SDK EdgeX Integration Example (SenML Payload Version)
# Modified to send payloads only when new/reappearing faces are detected

import os
import sys
import time
import json
import datetime
import requests
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# --- Environment check ---
if not os.environ.get('AXELERA_FRAMEWORK'):
    sys.exit("Please activate the Axelera environment using: source venv/bin/activate")

from axelera.app import config, display, inf_tracers, logging_utils, statistics, yaml_parser
from axelera.app.stream import create_inference_stream

LOG = logging_utils.getLogger(__name__)
PBAR = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

# --- Load class ID mapping ---
CLASS_MAP_FILE = "/data/voyager-sdk/class_map.json"
EMBEDDING_MAP = {}

if os.path.exists(CLASS_MAP_FILE):
    with open(CLASS_MAP_FILE, "r") as f:
        EMBEDDING_MAP = json.load(f)
    print(f"✅ Loaded class ID mapping from {CLASS_MAP_FILE}")
else:
    print(f"⚠️ Class map file not found at {CLASS_MAP_FILE}. Names default to 'unknown'.")

CONFIDENCE_THRESHOLD = 70.0
SEND_EXECUTOR = ThreadPoolExecutor(max_workers=8)

# ------------------------------------------------------------------
#                   ENDPOINT DEFINITIONS (3 ENDPOINTS)
# ------------------------------------------------------------------

EDGE_X_EVENT_ENDPOINT = {
    "url": "http://localhost:4000/core-data/api/v3/event/device-rest/face_recog/face_rcog/all",
    "headers": {
        "Content-Type": "application/json",
    }
}

RAPID_ENDPOINT = {
    "url": "https://rapid.meridiandatalabs.com/core-data/api/v3/event/device-rest/face_recog/_Facerecog/all",
    "headers": {
        "Content-Type": "application/json",
        "Authorization": "Bearer eyJhbGciOiJFUzM4NCIsImtpZCI6ImMxNTg2NDgxLWU5ZTEtZWJlOC1mNjhkLTQ3NTkwNjQ4ZTE5OSJ9.eyJhdWQiOiJlZGdleCIsImV4cCI6MTc2NDY1NTEzNSwiaWF0IjoxNzY0NTY4NzM1LCJpc3MiOiIvdjEvaWRlbnRpdHkvb2lkYyIsIm5hbWUiOiJtZGxfYWRtaW40IiwibmFtZXNwYWNlIjoicm9vdCIsInN1YiI6ImQwMjVhN2I4LWRmNjUtNjJkMC0wYWE5LTNkZDFlNGFjOGUxMCJ9.7u8xYK_byQUSFTrwCprTbYM_3VvQcCUS-Y6slY7T3Mz6QoZ2mOLT9wYhgxCL0BTYNKP4D0FbsCkgoij9KT4gSfTjMoODIJSUvDngRLSJT6N3u7-77zASxp4_UyBjZ6dq"
    }
}

IOT_CHANNEL_ENDPOINT = {
    "url": "https://iot.meridiandatalabs.com/http/channels/af4aa8cf-1222-48a5-9081-e7f6fc1ea4c3/messages/",
    "headers": {
        "Content-Type": "application/json",
        "Authorization": "Thing 8a3c6884-84fa-43e8-94d1-e4984a5570ff"
    }
}

ALL_ENDPOINTS = [EDGE_X_EVENT_ENDPOINT, RAPID_ENDPOINT, IOT_CHANNEL_ENDPOINT]
# Remove endpoints you don't want:
# ALL_ENDPOINTS = [EDGE_X_EVENT_ENDPOINT]

# ------------------------------------------------------------------
#                       SEND HELPERS
# ------------------------------------------------------------------

def send_payload(payload, url, headers):
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=5)
        print(f"➡️  POST {url} → {resp.status_code}")
    except Exception as e:
        print(f"❌ Send failed ({url}): {e}")

def send_to_all(payload):
    for ep in ALL_ENDPOINTS:
        SEND_EXECUTOR.submit(send_payload, payload, ep["url"], ep["headers"])

# ------------------------------------------------------------------
#                   BUILD SENML PAYLOAD
# ------------------------------------------------------------------

def build_senml_payload_from_yaml(device_name, detection_info):
    iso_timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    base_name = device_name
    senml_payload = []

    senml_payload.append({
        "bn": base_name,
        "timestamp": iso_timestamp,
        "n": "DeviceName",
        "u": "name",
        "vs": base_name
    })

    if "face_count" in detection_info:
        senml_payload.append({
            "bn": base_name,
            "timestamp": iso_timestamp,
            "n": "face_count",
            "u": "count",
            "v": int(detection_info["face_count"])
        })

    if "person_name" in detection_info:
        senml_payload.append({
            "bn": base_name,
            "timestamp": iso_timestamp,
            "n": "recognized_names",
            "u": "person",
            "vs": detection_info["person_name"]
        })

    if "confidence" in detection_info:
        senml_payload.append({
            "bn": base_name,
            "timestamp": iso_timestamp,
            "n": "confidence",
            "u": "%",
            "v": float(detection_info["confidence"])
        })

    if "bbox_coordinates" in detection_info:
        senml_payload.append({
            "bn": base_name,
            "timestamp": iso_timestamp,
            "n": "bbox_coordinates",
            "u": "pixel",
            "vs": json.dumps(detection_info["bbox_coordinates"])
        })

    if "embeddings" in detection_info:
        senml_payload.append({
            "bn": base_name,
            "timestamp": iso_timestamp,
            "n": "embeddings",
            "u": "vector512d",
            "vs": detection_info["embeddings"]
        })

    if "frame_timestamp" in detection_info:
        senml_payload.append({
            "bn": base_name,
            "timestamp": iso_timestamp,
            "n": "frame_timestamp",
            "u": "ISO8601",
            "vs": detection_info["frame_timestamp"]
        })

    if "cam_temp" in detection_info:
        senml_payload.append({
            "bn": base_name,
            "timestamp": iso_timestamp,
            "n": "cam_temp",
            "u": "degrees C",
            "v": float(detection_info["cam_temp"])
        })

    return senml_payload

# ------------------------------------------------------------------
#            DETECTION METADATA EXTRACTION
# ------------------------------------------------------------------

def extract_detection_data(meta, device_base_name):
    detection_info = {
        "face_count": 0,
        "person_name": "unknown",
        "confidence": 0.0,
        "frame_timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    }

    if not meta or not hasattr(meta, "_meta_map"):
        return detection_info

    detections = getattr(meta._meta_map.get("detections"), "_secondary_metas", {}).get("recognitions", [])
    recognized_names, confidences = [], []

    for recog_meta in detections:
        person_name, confidence, class_id = "unknown", 0.0, None

        if hasattr(recog_meta, "_class_ids") and recog_meta._class_ids:
            class_id = str(int(recog_meta._class_ids[0][0]))

        if hasattr(recog_meta, "_scores") and recog_meta._scores:
            confidence = float(recog_meta._scores[0][0]) * 100.0

        if confidence >= CONFIDENCE_THRESHOLD and class_id in EMBEDDING_MAP:
            person_name = EMBEDDING_MAP[class_id]
        else:
            person_name = "unknown"

        recognized_names.append(person_name)
        confidences.append(confidence)

    detection_info.update({
        "face_count": len(recognized_names),
        "person_name": ",".join(recognized_names) if recognized_names else "unknown",
        "confidence": max(confidences) if confidences else 0.0,
    })

    return detection_info

# ------------------------------------------------------------------
#                     INFERENCE LOOP
# ------------------------------------------------------------------

def inference_loop(args, log_file_path, stream, app, wnd, tracers=None):
    device_base_name = "FaceDetector_2cf7f12052608e69_"
    last_seen = {}
    last_sent_state = set()
    REAPPEAR_TIMEOUT = 5.0

    for frame_result in tqdm(stream, desc="Detecting...", unit="frames", leave=False, bar_format=PBAR):
        image, meta = frame_result.image, frame_result.meta

        if image:
            wnd.show(image, meta, frame_result.stream_id)

        if wnd.is_closed:
            break

        if not meta:
            continue

        detection_info = extract_detection_data(meta, device_base_name)
        current_people = set(detection_info["person_name"].split(",")) if detection_info["person_name"] else set()
        now = time.time()

        expired = {p for p, t in last_seen.items() if now - t > REAPPEAR_TIMEOUT}

        for p in expired:
            last_seen.pop(p, None)
            if p in last_sent_state:
                last_sent_state.remove(p)

        for person in current_people:
            last_seen[person] = now

        new_people = {p for p in current_people if p not in last_sent_state}

        if new_people:
            print(f"📡 Sending event for: {new_people}")
            last_sent_state |= new_people

            senml_payload = build_senml_payload_from_yaml(device_base_name, detection_info)
            if detection_info["face_count"] > 0:
                send_to_all(senml_payload)

# ------------------------------------------------------------------
#                            MAIN
# ------------------------------------------------------------------

if __name__ == "__main__":
    network_yaml_info = yaml_parser.get_network_yaml_info()
    parser = config.create_inference_argparser(
        network_yaml_info,
        description="Voyager SDK Inference → EdgeX Integration (SenML Payload)"
    )
    parser.add_argument("--save-tracers", type=str, default=None)
    args = parser.parse_args()
    tracers = inf_tracers.create_tracers_from_args(args)

    try:
        log_file, log_file_path = None, None
        if args.show_stats:
            log_file, log_file_path = statistics.initialise_logging()

        stream = create_inference_stream(
            config.SystemConfig.from_parsed_args(args),
            config.InferenceStreamConfig.from_parsed_args(args),
            config.PipelineConfig.from_parsed_args(args),
            config.LoggingConfig.from_parsed_args(args),
            config.DeployConfig.from_parsed_args(args),
            tracers=tracers,
        )

        with display.App(
            visible=args.display,
            opengl=stream.hardware_caps.opengl,
            buffering=not stream.is_single_image(),
        ) as app:
            wnd = app.create_window("Inference demo", size=args.window_size)
            app.start_thread(inference_loop, (args, log_file_path, stream, app, wnd, tracers), name="InferenceThread")
            app.run(interval=1 / 10)

    except KeyboardInterrupt:
        LOG.exit_with_error_log()
    except logging_utils.UserError as e:
        LOG.exit_with_error_log(e.format())
    except Exception as e:
        LOG.exit_with_error_log(e)
    finally:
        if "stream" in locals():
            stream.stop()
        time.sleep(3)
        SEND_EXECUTOR.shutdown(wait=True)
