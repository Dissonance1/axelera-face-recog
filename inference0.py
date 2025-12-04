#!/usr/bin/env python
# Copyright Axelera AI, 2025
import os
import sys
import time
import json
import requests
import datetime
import threading

if not os.environ.get('AXELERA_FRAMEWORK'):
    sys.exit("Please activate the Axelera environment with source venv/bin/activate and run again")

from tqdm import tqdm
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
    print("🔹 Class ID mapping preview (first 10):")
    for k, v in list(EMBEDDING_MAP.items())[:10]:
        print(f"   {k} → {v}")
else:
    print(f"⚠️ Class map file not found at {CLASS_MAP_FILE}. Names will default to 'unknown'.")

CONFIDENCE_THRESHOLD = 50.0  # Confidence threshold to recognize a known person

# --- Async send helper ---
def send_async(payload, url, headers):
    def _send():
        try:
            print(f"➡️ Sending payload to {url}")
            response = requests.post(url, json=payload, headers=headers, timeout=5)
            print(f"Response: {response.status_code} {response.text[:200]}")
            response.raise_for_status()
            print(f"✅ Successfully sent to {url}")
        except Exception as e:
            print(f"❌ Failed to send to {url}: {e}")
    threading.Thread(target=_send, daemon=True).start()

# --- Detection endpoints ---
DETECTION_ENDPOINTS = [
    {
        "url": "https://rapid.meridiandatalabs.com/core-data/api/v3/event/device-rest/face_recognition/_Facerecog/all",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer eyJhbGciOiJFUzM4NCIsImtpZCI6IjcyOTg4NTk2LTc1YTgtZGJiOS1iNDhiLWY2NjM5MTBlZTJkZCJ9.eyJhdWQiOiJlZGdleCIsImV4cCI6MTc2MTIwMjgyNSwiaWF0IjoxNzYxMTE2NDI1LCJpc3MiOiIvdjEvaWRlbnRpdHkvb2lkYyIsIm5hbWUiOiJtZGxfYWRtaW4zIiwibmFtZXNwYWNlIjoicm9vdCIsInN1YiI6ImY3ZTliOWZmLThiYzUtYmU3Ni1lYzQ4LTAxMGMzNTVkMThjYyJ9.K9VmeqNAU9qd9B_TalyuXwBM3dJr7bKoHK45L_9J7CwMJLoo3McsIV3WwNhnqJsp_33ysR6Yjv9vnZJmZcyFaf--xPTiv1Ut462QWVvb3YpxNFTj36VTtuR9T_Q8ae_B"
        }
    },
    {
        "url": "https://iot.meridiandatalabs.com/http/channels/af4aa8cf-1222-48a5-9081-e7f6fc1ea4c3/messages/",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Thing 8a3c6884-84fa-43e8-94d1-e4984a5570ff"
        }
    }
]

# --- Health check endpoint (only this) ---
HEALTH_CHECK_ENDPOINT = {
    "url": "https://iot.meridiandatalabs.com/http/channels/af4aa8cf-1222-48a5-9081-e7f6fc1ea4c3/messages/",
    "headers": {
        "Content-Type": "application/json",
        "Authorization": "Thing 8a3c6884-84fa-43e8-94d1-e4984a5570ff"
    }
}

# --- Send SenML payload to all detection endpoints ---
def send_senml_to_edgex(senml_payload):
    for ep in DETECTION_ENDPOINTS:
        send_async(senml_payload, ep["url"], ep["headers"])

# --- Extract detection info from meta ---
def extract_detection_data(meta, device_base_name):
    senml_payload = []
    iso_timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Device info
    senml_payload.append({
        "bn": device_base_name,
        "timestamp": iso_timestamp,
        "n": "DeviceName",
        "u": "name",
        "vs": device_base_name
    })

    try:
        detections = meta._meta_map.get("detections")
        if not detections:
            return senml_payload

        recognitions = getattr(detections, "_secondary_metas", {}).get("recognitions", [])

        for recog_meta in recognitions:
            class_id = None
            confidence = 0.0
            person_name = "unknown"

            if hasattr(recog_meta, "_class_ids") and recog_meta._class_ids:
                # --- Add +1 to fix off-by-one mapping ---
                class_id = str(int(recog_meta._class_ids[0][0]) + 1)

            if hasattr(recog_meta, "_scores") and recog_meta._scores:
                confidence = float(recog_meta._scores[0][0]) * 100.0

            # Map class_id → person name, but check confidence
            if confidence >= CONFIDENCE_THRESHOLD and class_id in EMBEDDING_MAP:
                person_name = EMBEDDING_MAP[class_id]
            else:
                person_name = "unknown"

            senml_payload.append({
                "bn": device_base_name,
                "timestamp": iso_timestamp,
                "n": "Authorized",
                "u": "person",
                "vs": person_name
            })
            senml_payload.append({
                "bn": device_base_name,
                "timestamp": iso_timestamp,
                "n": "confidence",
                "u": "%",
                "v": confidence
            })

    except Exception as e:
        print(f"❌ Error extracting data: {e}")

    return senml_payload

# --- Health check loop ---
def health_check_loop(device_base_name, active_people, interval_seconds=3600):
    while True:
        try:
            now_iso = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            heartbeat_payload = [
                {
                    "bn": device_base_name,
                    "timestamp": now_iso,
                    "n": "HealthCheck",
                    "u": "status",
                    "vs": "alive"
                },
                {
                    "bn": device_base_name,
                    "timestamp": now_iso,
                    "n": "ActivePeopleCount",
                    "u": "count",
                    "v": len(active_people)
                }
            ]
            print(f"💓 Sending health check — active people: {len(active_people)}")
            send_async(heartbeat_payload, HEALTH_CHECK_ENDPOINT["url"], HEALTH_CHECK_ENDPOINT["headers"])
        except Exception as e:
            print(f"❌ Health check failed: {e}")
        time.sleep(interval_seconds)

# --- Inference loop with presence tracking ---
def inference_loop(args, log_file_path, stream, app, wnd, tracers=None):
    device_base_name = "FaceDetector_2cf7f12052608e69_"
    active_people = {}  # Tracks currently visible people
    timeout_seconds = 2.0  # Person is "gone" if not seen for >2 seconds

    # Start health check thread
    threading.Thread(
        target=health_check_loop,
        args=(device_base_name, active_people, 3600),  # 1 hour
        daemon=True
    ).start()

    for frame_result in tqdm(
        stream,
        desc="Detecting...",
        unit="frames",
        leave=False,
        bar_format=PBAR,
        disable=None,
    ):
        image, meta = frame_result.image, frame_result.meta
        if image:
            wnd.show(image, meta, frame_result.stream_id)
        if wnd.is_closed:
            break

        if meta is None:
            continue

        senml_payload = extract_detection_data(meta, device_base_name)
        now = time.time()

        # Current detections (include "unknown")
        current_frame_people = set(
            entry.get("vs") for entry in senml_payload if entry.get("n") == "Authorized"
        )

        # Remove people who left
        for person in list(active_people.keys()):
            if now - active_people[person] > timeout_seconds:
                print(f"👋 {person} left the frame")
                del active_people[person]

        # Send for new arrivals
        for person in current_frame_people:
            if person not in active_people:  # include unknown
                print(f"🚶 {person} entered view — sending payload...")
                send_senml_to_edgex(senml_payload)
            active_people[person] = now

# --- Main ---
if __name__ == "__main__":
    network_yaml_info = yaml_parser.get_network_yaml_info()
    parser = config.create_inference_argparser(
        network_yaml_info, description="Perform inference on an Axelera platform"
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
            app.start_thread(
                inference_loop,
                (args, log_file_path, stream, app, wnd, tracers),
                name="InferenceThread",
            )
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
        time.sleep(3)  # allow async sends to finish
