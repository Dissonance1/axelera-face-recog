#!/usr/bin/env python3
"""
Axelera Person Detection (Metis AIPU)
Runs YOLO-style person/animal detection using Axelera AI runtime.

This is a backend adapter mirroring windows_person_detection.py structure.
Replace the AxeleraRuntime placeholder with actual Voyager SDK runtime calls.
"""

import cv2
import numpy as np
import time
import logging
import argparse
from datetime import datetime
import os
from typing import List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AxeleraRuntime:
    """Voyager SDK adapter wrapper for an Axelera-compiled model.

    This class attempts to import and use the Voyager SDK if available. The exact
    API names can vary by SDK version; adjust the marked sections to your setup.
    """

    def __init__(self, compiled_model_path: str, input_size: Tuple[int, int] = (640, 640)):
        self.compiled_model_path = compiled_model_path
        self.input_size = input_size
        self._sdk_loaded = False
        self._session = None
        if not os.path.exists(compiled_model_path):
            raise FileNotFoundError(f"Compiled model not found: {compiled_model_path}")

        # Attempt to import Voyager SDK (adjust module name if different)
        self._voyager_mod = None
        possible_modules = [
            # Add alternative names here if your install uses a different top-level
            # package. Example names below are placeholders.
            'axelera.voyager',
            'voyager',
            'axelera_sdk',
        ]
        for mod_name in possible_modules:
            try:
                self._voyager_mod = __import__(mod_name)
                self._sdk_loaded = True
                break
            except Exception:
                continue

        if not self._sdk_loaded:
            logger.warning("Voyager SDK not detected. Inference will be unavailable until installed.")

        # Initialize session if SDK present
        if self._sdk_loaded:
            try:
                # EDIT HERE: replace with actual session/context creation API
                # Example (pseudo): self._session = self._voyager_mod.Session(self.compiled_model_path)
                # Ensure input/output bindings are prepared once.
                self._session = None  # Placeholder until actual SDK code is filled
            except Exception as exc:
                logger.error(f"Failed to initialize Voyager session: {exc}")
                raise

        logger.info(f"📦 Compiled model: {compiled_model_path}")
        logger.info(f"🧮 Input size: {self.input_size}")

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, int, int]]:
        h, w = frame.shape[:2]
        ih, iw = self.input_size
        scale = min(iw / w, ih / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((ih, iw, 3), dtype=np.uint8)
        top = (ih - nh) // 2
        left = (iw - nw) // 2
        canvas[top:top+nh, left:left+nw] = resized
        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # CHW
        blob = np.expand_dims(blob, axis=0)  # NCHW
        return blob, (scale, left, top, w, h)

    def run(self, input_blob: np.ndarray):
        if not self._sdk_loaded:
            raise RuntimeError(
                "Voyager SDK is not installed or could not be imported. "
                "Install Axelera Voyager SDK and re-run."
            )
        if self._session is None:
            raise RuntimeError(
                "Voyager SDK session not initialized. Replace the session creation placeholder "
                "with actual SDK code in AxeleraRuntime.__init__."
            )
        # EDIT HERE: replace with actual inference invocation and output retrieval
        # Example (pseudo): outputs = self._session.run({'input': input_blob})
        outputs = None
        return outputs

    @staticmethod
    def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
        cx, cy, w, h = xywh.T
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.stack([x1, y1, x2, y2], axis=1)

    @staticmethod
    def postprocess(raw_outputs, scale_info, conf_threshold: float, class_names: List[str]):
        # Support several common shapes:
        # 1) Nx6: [x1,y1,x2,y2,conf,cls]
        # 2) Nx(5+C): [cx,cy,w,h,obj_conf, class_probs...]
        # 3) Dict or tuple/list containing arrays → take the first array-like
        if raw_outputs is None:
            return []

        # Normalize raw to a numpy array of detections
        arr = None
        if isinstance(raw_outputs, np.ndarray):
            arr = raw_outputs
        elif isinstance(raw_outputs, (list, tuple)) and len(raw_outputs) > 0:
            # assume first output tensor contains detections
            first = raw_outputs[0]
            if isinstance(first, np.ndarray):
                arr = first
        elif isinstance(raw_outputs, dict) and len(raw_outputs) > 0:
            # take first array-like value
            for v in raw_outputs.values():
                if isinstance(v, np.ndarray):
                    arr = v
                    break

        if arr is None or arr.size == 0:
            return []

        if arr.ndim == 3:
            # e.g., [batch, num, dims]
            arr = arr[0]

        scale, left, top, orig_w, orig_h = scale_info

        detections = []
        num_dims = arr.shape[1]

        if num_dims >= 6 and num_dims <= 7:
            # Assume [x1,y1,x2,y2,conf,cls(,optional extra)] in model input space
            boxes_xyxy = arr[:, :4]
            confs = arr[:, 4]
            cls_ids = arr[:, 5].astype(np.int32)
        elif num_dims > 6:
            # Assume YOLO head: [cx,cy,w,h,obj_conf, class_probs...]
            boxes_xywh = arr[:, :4]
            obj_conf = arr[:, 4]
            class_scores = arr[:, 5:]
            if class_scores.size == 0:
                return []
            cls_ids = class_scores.argmax(axis=1)
            cls_conf = class_scores.max(axis=1)
            confs = obj_conf * cls_conf
            boxes_xyxy = AxeleraRuntime._xywh_to_xyxy(boxes_xywh)
        else:
            return []

        # Confidence filter
        keep = confs >= conf_threshold
        boxes_xyxy = boxes_xyxy[keep]
        confs = confs[keep]
        cls_ids = cls_ids[keep]

        # De-letterbox back to original image space
        if boxes_xyxy.size:
            boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - left) / max(scale, 1e-6)
            boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - top) / max(scale, 1e-6)
            boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w - 1)
            boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h - 1)

        results = []
        for (x1, y1, x2, y2), conf, cid in zip(boxes_xyxy, confs, cls_ids):
            results.append([float(x1), float(y1), float(x2), float(y2), float(conf), int(cid)])
        return results


class AxeleraPersonDetection:
    def __init__(self, camera_id=0, confidence_threshold=0.5, display=True,
                 compiled_model: Optional[str] = None, input_size: Tuple[int, int] = (640, 640)):
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        self.display = display
        self.input_size = input_size
        self.cap = None
        self.runtime: Optional[AxeleraRuntime] = None
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow']

        self.setup_camera()
        self.setup_axelera_backend(compiled_model)

    def setup_camera(self):
        logger.info(f"🎥 Initializing camera {self.camera_id}...")
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"✅ Camera opened: {width}x{height} @ {fps} FPS")

    def setup_axelera_backend(self, compiled_model: Optional[str]):
        if compiled_model is None:
            logger.error("❌ Please provide --compiled-model path to an Axelera compiled artifact")
            raise ValueError("compiled model path required")
        try:
            self.runtime = AxeleraRuntime(compiled_model, input_size=self.input_size)
        except Exception as e:
            logger.error(f"❌ Failed to initialize Axelera runtime: {e}")
            raise

    def detect_objects(self, frame: np.ndarray):
        assert self.runtime is not None
        blob, scale_info = self.runtime.preprocess(frame)
        try:
            raw = self.runtime.run(blob)
        except NotImplementedError as e:
            logger.error(str(e))
            return [], []
        detections = self.runtime.postprocess(raw, scale_info, self.confidence_threshold, self.class_names)
        persons = []
        animals = []
        for x1, y1, x2, y2, conf, cls_id in detections:
            cls_id = int(cls_id)
            class_name = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else str(cls_id)
            if class_name == 'person':
                persons.append({'bbox': (int(x1), int(y1), int(x2), int(y2)), 'confidence': float(conf), 'class': class_name})
            elif class_name in ['cat', 'dog', 'bird', 'horse', 'cow', 'sheep']:
                animals.append({'bbox': (int(x1), int(y1), int(x2), int(y2)), 'confidence': float(conf), 'class': class_name})
        return persons, animals

    @staticmethod
    def draw(frame: np.ndarray, persons, animals):
        for p in persons:
            x1, y1, x2, y2 = p['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person: {p['confidence']:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        for a in animals:
            x1, y1, x2, y2 = a['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{a['class'].title()}: {a['confidence']:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        return frame

    @staticmethod
    def draw_stats(frame: np.ndarray, persons, animals, conf: float):
        y = 30
        for text in [f"Persons: {len(persons)}", f"Animals: {len(animals)}", f"Confidence: {conf}"]:
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            y += 25
        return frame

    def run(self):
        logger.info("🎯 Starting Axelera Person Detection")
        logger.info("💡 Press 'q' to quit, 's' to save frame")
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    logger.error("Failed to read frame from camera")
                    break
                persons, animals = self.detect_objects(frame)
                frame = self.draw(frame, persons, animals)
                frame = self.draw_stats(frame, persons, animals, self.confidence_threshold)
                if self.display:
                    cv2.imshow('Axelera Person Detection', frame)
                    key = cv2.waitKey(1) & 0xFF
                else:
                    key = -1
                if key == ord('q'):
                    logger.info("🛑 Quit requested by user")
                    break
                elif key == ord('s'):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fn = f"axelera_detection_{ts}.jpg"
                    cv2.imwrite(fn, frame)
                    logger.info(f"💾 Saved frame as {fn}")
        except KeyboardInterrupt:
            logger.info("🛑 Interrupted by user")
        finally:
            if self.cap:
                self.cap.release()
            if self.display:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            logger.info("🧹 Cleanup completed")


def main():
    parser = argparse.ArgumentParser(description='Axelera Person Detection (Metis AIPU)')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')
    parser.add_argument('--compiled-model', type=str, required=True, help='Path to Axelera compiled model artifact')
    parser.add_argument('--input-size', type=str, default='640x640', help='Model input size WxH (e.g. 640x640)')
    args = parser.parse_args()

    try:
        w, h = [int(x) for x in args.input_size.lower().split('x')]
        det = AxeleraPersonDetection(
            camera_id=args.camera,
            confidence_threshold=args.confidence,
            display=not args.no_display,
            compiled_model=args.compiled_model,
            input_size=(h, w)
        )
        det.run()
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        logger.info("💡 Ensure Axelera Voyager SDK is installed and the compiled model path is correct")


if __name__ == '__main__':
    main()


