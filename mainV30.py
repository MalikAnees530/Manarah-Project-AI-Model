import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue,
    PointStruct, ScoredPoint, VectorParams,
)

FACE_DIM  = 512
BODY_DIM  = 768
LOCK_FILE = "reindex.lock"
VERSION   = "30.3-Production"

MAX_CACHE_SIZE = 300
_search_cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()
_cache_lock = threading.Lock()

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).flatten()
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n >= 1e-6 else np.zeros(v.size, dtype=np.float32)

def dot_sim(a, b) -> float:
    a = np.asarray(a, np.float32).flatten()
    b = np.asarray(b, np.float32).flatten()
    if a.size != b.size:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))

def _build_bank(member_vecs: list, max_bank: int = 16) -> list:
    deduped: list = []
    for v in member_vecs:
        v_arr = np.asarray(v, np.float32)
        if not deduped:
            deduped.append(v_arr.tolist())
        else:
            if max(dot_sim(v_arr, np.array(d)) for d in deduped) < 0.92:
                deduped.append(v_arr.tolist())
        if len(deduped) >= max_bank:
            break
    return deduped

def _face_in_person(face_bbox, person_box) -> bool:
    fx1, fy1, fx2, fy2 = face_bbox
    face_cx = (fx1 + fx2) / 2
    face_cy = (fy1 + fy2) / 2
    px1, py1, px2, py2 = person_box
    return px1 <= face_cx <= px2 and py1 <= face_cy <= py2

def _clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def clear_module_cache() -> None:
    for mod in list(sys.modules.keys()):
        if mod.startswith("titanium_core"):
            del sys.modules[mod]

class ClusterRegistry:
    PERSIST_FILE = "cluster_registry.json"

    def __init__(self):
        self._lock = threading.RLock()
        self._data: dict[str, list] = {}
        self._load()

    def _load(self) -> None:
        tmp = Path(self.PERSIST_FILE + ".tmp")
        if tmp.exists():
            tmp.unlink()
            print("[REGISTRY] Cleaned stale .tmp from previous crash.")

        p = Path(self.PERSIST_FILE)
        if not p.exists():
            return
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            self._data = {
                eid: [
                    [np.array(c, np.float32), [np.array(v, np.float32) for v in members]]
                    for c, members in clusters
                ]
                for eid, clusters in raw.items()
            }
            total = sum(len(v) for v in self._data.values())
            print(f"[REGISTRY] Loaded {total} clusters across {len(self._data)} events.")
        except Exception as e:
            print(f"[REGISTRY] Load failed ({e}), starting empty.")
            self._data = {}

    def _save(self) -> None:
        try:
            raw = {
                eid: [
                    [c.tolist(), [v.tolist() for v in members]]
                    for c, members in clusters
                ]
                for eid, clusters in self._data.items()
            }
            tmp = Path(self.PERSIST_FILE + ".tmp")
            tmp.write_text(json.dumps(raw), encoding="utf-8")
            tmp.replace(Path(self.PERSIST_FILE))
        except Exception as e:
            print(f"[REGISTRY] Save failed: {e}")

    def replace_event(self, event_id: str, clusters: list) -> None:
        with self._lock:
            self._data[event_id] = clusters
            self._save()

    def update_with_vector(self, event_id: str, vec: np.ndarray, merge_threshold: float = 0.65) -> list:
        with self._lock:
            clusters = self._data.setdefault(event_id, [])
            best_sim, best_idx = 0.0, -1
            for i, (c_vec, _) in enumerate(clusters):
                sim = dot_sim(vec, c_vec)
                if sim > best_sim:
                    best_sim, best_idx = sim, i

            if best_sim >= merge_threshold and best_idx >= 0:
                clusters[best_idx][1].append(vec)
                all_vecs = clusters[best_idx][1]
                clusters[best_idx][0] = l2_normalize(np.mean(all_vecs, axis=0))
                bank = _build_bank(clusters[best_idx][1])
            else:
                clusters.append([vec.copy(), [vec.copy()]])
                bank = [vec.tolist()]

            self._save()
            return bank

    def get_bank_for(self, event_id: str, vec: np.ndarray, match_threshold: float = 0.65) -> list:
        with self._lock:
            clusters = self._data.get(event_id, [])
            best_sim, best_idx = 0.0, -1
            for i, (c_vec, _) in enumerate(clusters):
                sim = dot_sim(vec, c_vec)
                if sim > best_sim:
                    best_sim, best_idx = sim, i
            if best_sim >= match_threshold and best_idx >= 0:
                return _build_bank(clusters[best_idx][1])
            return []

cluster_registry = ClusterRegistry()

def patch_vision_pipeline(project_root: Path) -> None:
    tc = project_root / "titanium_core"
    tc.mkdir(exist_ok=True)
    (tc / "__init__.py").touch()

    code = f'''"""
TitaniumVision — MANARAH V{VERSION}
Generated automatically by patch_vision_pipeline(). Do not edit by hand.
"""
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import insightface

FACE_DIM = {FACE_DIM}
BODY_DIM = {BODY_DIM}

KP_NOSE           = 0
KP_LEFT_EYE       = 1
KP_RIGHT_EYE      = 2
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP       = 11
KP_RIGHT_HIP      = 12

def l2_normalize(vec):
    v = np.asarray(vec, dtype=np.float32).flatten()
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n >= 1e-6 else np.zeros(v.size, dtype=np.float32)

def dot_sim(a, b) -> float:
    a, b = np.asarray(a, np.float32).flatten(), np.asarray(b, np.float32).flatten()
    if a.size != b.size:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))

def _face_in_person(face_bbox, person_box) -> bool:
    fx1, fy1, fx2, fy2 = face_bbox
    face_cx = (fx1 + fx2) / 2
    face_cy = (fy1 + fy2) / 2
    px1, py1, px2, py2 = person_box
    return px1 <= face_cx <= px2 and py1 <= face_cy <= py2

class TitaniumVision:
    def __init__(self):
        self.yolo = YOLO("yolo11n-pose.pt")

        self.face_app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        ctx = 0 if torch.cuda.is_available() else -1
        self.face_app.prepare(ctx_id=ctx, det_size=(640, 640))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.environ["TORCH_HOME"] = str(
            __import__("pathlib").Path.cwd() / "model_cache"
        )
        try:
            self.body_engine = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14",
                source="local", force_reload=False,
            )
        except Exception:
            print("[WARN] DINOv2 local cache not found — downloading (one-time)...")
            self.body_engine = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14"
            )
        self.body_engine.eval().to(self.device)
        self.dino_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _blur_score(self, crop: np.ndarray) -> float:
        if crop is None or crop.size == 0:
            return 0.0
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(
            np.clip(np.log1p(cv2.Laplacian(g, cv2.CV_64F).var()) / 6.0, 0.0, 1.0)
        )

    def _body_vec(self, crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0:
            return np.zeros(BODY_DIM, np.float32)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        t = self.dino_tf(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            v = self.body_engine(t)[0].cpu().numpy()
        return l2_normalize(v)

    def _torso_crop(self, img: np.ndarray, kp, person_box: list) -> np.ndarray:
        x1, y1, x2, y2 = person_box
        ih, iw = img.shape[:2]

        if kp is not None and kp.shape[0] > KP_RIGHT_HIP:
            ls = kp[KP_LEFT_SHOULDER]
            rs = kp[KP_RIGHT_SHOULDER]
            lh = kp[KP_LEFT_HIP]
            rh = kp[KP_RIGHT_HIP]

            valid = lambda p: not (np.allclose(p, 0) or (p[0] < 1 and p[1] < 1))
            shoulders = [p for p in [ls, rs] if valid(p)]
            hips      = [p for p in [lh, rh] if valid(p)]

            if shoulders and hips:
                all_pts = np.array(shoulders + hips)
                tx1 = int(max(0,  all_pts[:, 0].min() - 20))
                ty1 = int(max(0,  all_pts[:, 1].min() - 10))
                tx2 = int(min(iw, all_pts[:, 0].max() + 20))
                ty2 = int(min(ih, all_pts[:, 1].max() + 20))
                if tx2 > tx1 and ty2 > ty1:
                    crop = img[ty1:ty2, tx1:tx2]
                    if crop.size > 0:
                        return crop

        mid_y = y1 + int((y2 - y1) * 0.55)
        crop = img[max(0, y1):mid_y, max(0, x1):x2]
        return crop if crop.size > 0 else img[max(0, y1):y2, max(0, x1):x2]

    def _safe_yaw(self, face_obj, kp) -> float:
        try:
            pose = getattr(face_obj, "pose", None)
            if pose is not None and len(pose) >= 3:
                return float(abs(pose[1]))
        except Exception:
            pass

        if kp is not None and kp.shape[0] > KP_RIGHT_EYE:
            le, re = kp[KP_LEFT_EYE], kp[KP_RIGHT_EYE]
            if not (np.allclose(le, 0) or np.allclose(re, 0)):
                dx = re[0] - le[0]
                dy = re[1] - le[1]
                return float(abs(np.degrees(np.arctan2(dy, dx))))
        return 0.0

    def extract_features(self, image_path_or_mat) -> list:
        if isinstance(image_path_or_mat, str):
            img = cv2.imread(image_path_or_mat)
        else:
            img = image_path_or_mat.copy()

        if img is None:
            return []

        img_h, img_w = img.shape[:2]

        yolo_res = self.yolo(img, verbose=False)
        if not yolo_res or len(yolo_res[0].boxes) == 0:
            return []

        res      = yolo_res[0]
        kpts_all = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else None

        img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        all_faces = self.face_app.get(img_rgb)

        out = []
        for i in range(len(res.boxes)):
            box    = res.boxes[i].xyxy[0].cpu().numpy()
            conf   = float(res.boxes[i].conf[0])
            x1, y1, x2, y2 = [int(v) for v in box]
            person_box = [x1, y1, x2, y2]
            ba = max(1, (x2 - x1) * (y2 - y1))

            kp = kpts_all[i] if (kpts_all is not None and len(kpts_all) > i) else None

            torso    = self._torso_crop(img, kp, person_box)
            body_vec = self._body_vec(torso)
            blur     = self._blur_score(torso)

            face_vec       = np.zeros(FACE_DIM, np.float32)
            hr             = False
            fa             = 0
            yaw_deg        = 0.0
            best_face_obj  = None
            best_face_area = 0

            for face in all_faces:
                fb = face.bbox
                if _face_in_person(fb, person_box):
                    area = (fb[2] - fb[0]) * (fb[3] - fb[1])
                    if area > best_face_area:
                        best_face_area = area
                        best_face_obj  = face

            if best_face_obj is not None:
                fb = best_face_obj.bbox
                fa = max(0, int((fb[2] - fb[0]) * (fb[3] - fb[1])))
                if fa >= 1600:
                    emb = getattr(best_face_obj, "embedding", None)
                    if emb is not None and np.linalg.norm(emb) > 1e-6:
                        face_vec = l2_normalize(np.asarray(emb, np.float32))
                        hr       = True
                        yaw_deg  = self._safe_yaw(best_face_obj, kp)

            area_ratio = float(np.clip(fa / max(1, ba), 0.0, 1.0))
            fqs = 0.35 * conf + 0.35 * blur + 0.30 * area_ratio

            if not hr:
                fqs = min(fqs, 0.15)

            person_area_ratio = float(ba / max(1, img_h * img_w))

            out.append({{
                "face_vector":       face_vec.tolist(),
                "body_vector":       body_vec.tolist(),
                "fqs":               float(fqs),
                "blur_score":        float(blur),
                "has_reliable_face": hr,
                "yaw_deg":           float(yaw_deg),
                "person_area":       ba,
                "person_area_ratio": person_area_ratio,
                "person_idx":        i,
                "bbox":              person_box,
                "img_h":             img_h,
                "img_w":             img_w,
            }})

        return out
'''
    target = tc / "vision.py"
    if target.exists() and target.read_text(encoding="utf-8") == code:
        return
    target.write_text(code, encoding="utf-8")

class QdrantEngine:
    def __init__(self, path: str = "./qdrant_data", collection_name: str = "manarah_v30"):
        from qdrant_client import QdrantClient
        self.client          = QdrantClient(path=path)
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "face": VectorParams(size=FACE_DIM, distance=Distance.COSINE),
                    "body": VectorParams(size=BODY_DIM, distance=Distance.COSINE),
                },
            )

    def search(self, vector_name: str, query: list, event_id: str, limit: int = 100) -> list[ScoredPoint]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query,
            using=vector_name,
            query_filter=Filter(must=[
                FieldCondition(key="event_id", match=MatchValue(value=event_id))
            ]),
            with_payload=True,
            limit=limit,
        )
        return response.points

    def count_event(self, event_id: str) -> int:
        return self.client.count(
            collection_name=self.collection_name,
            count_filter=Filter(must=[
                FieldCondition(key="event_id", match=MatchValue(value=event_id))
            ]),
            exact=True,
        ).count

    def delete_event(self, event_id: str) -> int:
        from qdrant_client.models import FilterSelector
        before = self.count_event(event_id)
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(must=[
                    FieldCondition(key="event_id", match=MatchValue(value=event_id))
                ])
            ),
        )
        return before

def bank_sim(q_face: np.ndarray, bank: list) -> float:
    if not bank:
        return 0.0
    valid_bank = [b for b in bank if b is not None and len(b) > 0]
    if not valid_bank:
        return 0.0
    scores = [dot_sim(q_face, b) for b in valid_bank]
    if len(scores) == 1:
        return scores[0]
    spread = max(scores) - min(scores)
    if spread > 0.30:
        return float(np.percentile(scores, 75))
    return float(max(scores))

def adaptive_thresholds(pool_scores: list[float]) -> dict[str, float]:
    defaults = {
        "face_min":      0.18,
        "body_min":      0.42,
        "penalized_min": 0.32,
        "conf_min":      35,
    }
    if not pool_scores or len(pool_scores) < 2:
        return defaults

    arr = np.array(pool_scores, np.float32)
    p25 = float(np.percentile(arr, 25))
    p50 = float(np.percentile(arr, 50))
    p75 = float(np.percentile(arr, 75))
    iqr = p75 - p25
    return {
        "face_min":      float(np.clip(p50 - 0.5 * iqr, 0.12, 0.28)),
        "body_min":      float(np.clip(p25, 0.35, 0.55)),
        "penalized_min": float(np.clip(p25 - 0.5 * iqr, 0.20, 0.42)),
        "conf_min":      35,
    }

def score_candidates(
    candidates: list[dict],
    q_face_vec: np.ndarray,
    q_body_vec: np.ndarray,
    q_face_reliable: bool,
    thresholds: dict[str, float],
    db_client=None,
    collection_name: str | None = None,
) -> list[dict]:
    scored = []
    for c in candidates:
        payload   = c["payload"]
        rrf_score = c["rrf_score"]

        c_face_vec = np.array(
            payload.get("face_vector", np.zeros(FACE_DIM)), np.float32
        )
        c_body_vec = np.array(
            payload.get("body_vector", np.zeros(BODY_DIM)), np.float32
        )
        face_bank = payload.get("face_bank", [])
        c_fqs     = float(payload.get("fqs", 0.0))
        c_hr      = bool(payload.get("has_reliable_face", False))

        s_body = dot_sim(q_body_vec, c_body_vec)

        if q_face_reliable and c_hr:
            s_face     = dot_sim(q_face_vec, c_face_vec)
            s_bank     = bank_sim(q_face_vec, face_bank)
            s_face_eff = max(s_face, s_bank)
            combined   = 0.55 * s_face_eff + 0.45 * s_body
            min_thr    = thresholds["face_min"]
        elif q_face_reliable and not c_hr:
            s_face     = dot_sim(q_face_vec, c_face_vec)
            s_bank     = bank_sim(q_face_vec, face_bank)
            s_face_eff = max(s_face, s_bank)
            combined   = 0.40 * s_face_eff + 0.60 * s_body
            min_thr    = thresholds["penalized_min"]
        else:
            combined = s_body
            min_thr  = thresholds["body_min"]

        if combined < min_thr:
            continue

        evidence_bonus = 0.0
        if q_face_reliable and c_hr:
            evidence_bonus += 0.10
        if c_fqs > 0.60:
            evidence_bonus += 0.05
        if rrf_score > 0.02:
            evidence_bonus += 0.05

        raw_conf = combined + evidence_bonus

        if 0.55 <= raw_conf <= 0.80 and db_client and collection_name:
            try:
                neighbours = db_client.query_points(
                    collection_name=collection_name,
                    query=c_body_vec.tolist(),
                    using="body",
                    limit=6,
                    with_vectors=True,
                ).points
                for n in neighbours:
                    if n.id != c["id"]:
                        n_body = (n.vector or {}).get("body") if isinstance(n.vector, dict) else None
                        if n_body and dot_sim(q_body_vec, n_body) >= 0.52:
                            raw_conf += 0.05
                            break
            except Exception:
                pass

        # Translate Raw Cosine Math to a Continuous Business Confidence Curve
        base_pct = raw_conf * 100
        if base_pct >= 45:
            display_conf = 90 + ((base_pct - 45) / 25) * 9
        elif base_pct >= 25:
            display_conf = 70 + ((base_pct - 25) / 20) * 19
        else:
            display_conf = base_pct * (70 / 25)

        conf_pct = int(np.clip(display_conf, 0, 99))
        if base_pct > 75: 
            conf_pct = 99
            
        # Hard drop rule: If the final confidence is below 85%, discard it (Removes False Positives)
        if conf_pct < 85:
            continue

        scored.append({**c, "combined": combined, "confidence": conf_pct})

    scored.sort(key=lambda x: x["combined"], reverse=True)
    return scored

def _fuse_rrf(
    face_hits: list[ScoredPoint],
    body_hits: list[ScoredPoint],
    face_dominant: bool = False,
    k: int = 60,
) -> list[dict]:
    pd: dict[str, dict] = {}
    face_weight = 2.5 if face_dominant else 1.0

    for rank, h in enumerate(face_hits):
        rrf = face_weight / (k + rank + 1)
        if h.id not in pd:
            pd[h.id] = {"id": h.id, "payload": h.payload, "rrf_score": 0.0}
        pd[h.id]["rrf_score"] += rrf

    for rank, h in enumerate(body_hits):
        rrf = 1.0 / (k + rank + 1)
        if h.id not in pd:
            pd[h.id] = {"id": h.id, "payload": h.payload, "rrf_score": 0.0}
        pd[h.id]["rrf_score"] += rrf

    return list(pd.values())

def search_event(event_id: str, query_image_path: str, top_k: int = 20) -> dict:
    feats = vision_engine.extract_features(query_image_path)
    if not feats:
        return {
            "status": "no_person_detected", "matches_found": 0, "data": [],
            "query_persons": 0, "event_id": event_id, "thresholds": {},
        }

    query_feat      = max(feats, key=lambda f: f["fqs"])
    q_face_vec      = np.array(query_feat["face_vector"], np.float32)
    q_body_vec      = np.array(query_feat["body_vector"], np.float32)
    q_face_reliable = query_feat["has_reliable_face"]
    q_fqs           = query_feat["fqs"]

    img_h, img_w = query_feat["img_h"], query_feat["img_w"]
    x1, y1, x2, y2 = query_feat["bbox"]
    bbox_h_ratio = (y2 - y1) / max(1, img_h)
    area_ratio   = query_feat["person_area"] / max(1, img_h * img_w)
    face_dominant = bool(bbox_h_ratio > 0.50 or area_ratio > 0.35)

    if q_fqs < 0.60:
        img_mem    = cv2.imread(query_image_path)
        vecs_face  = [q_face_vec] if q_face_reliable else []
        vecs_body  = [q_body_vec]

        for aug_img in [cv2.flip(img_mem, 1), _clahe(img_mem)]:
            aug_feats = vision_engine.extract_features(aug_img)
            if aug_feats:
                aug_qp = max(aug_feats, key=lambda x: x["person_area"])
                if aug_qp["has_reliable_face"]:
                    vecs_face.append(np.asarray(aug_qp["face_vector"], np.float32))
                vecs_body.append(np.asarray(aug_qp["body_vector"], np.float32))

        if len(vecs_face) > 1:
            q_face_vec = l2_normalize(np.mean(vecs_face, axis=0))
        if len(vecs_body) > 1:
            q_body_vec = l2_normalize(np.mean(vecs_body, axis=0))

    face_hits = db_engine.search("face", q_face_vec.tolist(), event_id, limit=100)
    body_hits = db_engine.search("body", q_body_vec.tolist(), event_id, limit=100)

    candidates  = _fuse_rrf(face_hits, body_hits, face_dominant=face_dominant)
    pool_scores = [c["rrf_score"] for c in candidates]
    thresholds  = adaptive_thresholds(pool_scores)
    scored      = score_candidates(
        candidates, q_face_vec, q_body_vec, q_face_reliable, thresholds,
        db_client=db_engine.client,
        collection_name=db_engine.collection_name,
    )

    matches = []
    for item in scored[:top_k]:
        p = item["payload"]
        matches.append({
            "filename":         p.get("filename", ""),
            "event_id":         p.get("event_id", event_id),
            "confidence_score": item["confidence"],
            "image_url":        f"/static/{p.get('filename', '')}",
            "fqs":              p.get("fqs", 0.0),
            "yaw_deg":          p.get("yaw_deg", 0.0),
            "rrf_score":        item["rrf_score"],
        })

    return {
        "status":        "success",
        "matches_found": len(matches),
        "data":          matches,
        "query_persons": len(feats),
        "event_id":      event_id,
        "thresholds":    thresholds,
    }

def _process_upload(file_path: str, event_id: str) -> None:
    feats = vision_engine.extract_features(file_path)
    if not feats:
        print(f"[UPLOAD] No persons detected in {file_path}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return

    fname = os.path.basename(file_path)
    pts: list[PointStruct] = []

    for f in feats:
        bank: list = []
        if f["has_reliable_face"]:
            v    = np.array(f["face_vector"], np.float32)
            bank = cluster_registry.get_bank_for(event_id, v, match_threshold=0.65)

        pts.append(PointStruct(
            id=str(uuid.uuid4()),
            vector={"face": f["face_vector"], "body": f["body_vector"]},
            payload={
                "filename":          fname,
                "event_id":          event_id,
                "fqs":               f["fqs"],
                "has_reliable_face": f["has_reliable_face"],
                "blur_score":        f["blur_score"],
                "yaw_deg":           f.get("yaw_deg", 0.0),
                "person_area":       f["person_area"],
                "face_bank":         bank,
            },
        ))

    if pts:
        db_engine.client.upsert(
            collection_name=db_engine.collection_name,
            points=pts,
        )

    for f in feats:
        if f["has_reliable_face"]:
            v = np.array(f["face_vector"], np.float32)
            cluster_registry.update_with_vector(event_id, v, merge_threshold=0.65)

    print(f"[UPLOAD] {fname}: {len(pts)} person(s) indexed | event='{event_id}'")
    if os.path.exists(file_path):
        os.remove(file_path)

def do_reindex(event_id: str, images_dir: str) -> dict:
    lock_path = Path(LOCK_FILE)
    if lock_path.exists():
        return {"error": "Reindex already in progress.", "status": "locked"}

    lock_path.touch()
    try:
        deleted = db_engine.delete_event(event_id)
        print(f"[REINDEX] Cleared {deleted} existing points for event='{event_id}'")

        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_paths = [
            str(p) for p in Path(images_dir).rglob("*")
            if p.suffix.lower() in exts
        ]
        if not image_paths:
            return {"error": f"No images found in '{images_dir}'.", "status": "empty"}

        pts:       list[PointStruct] = []
        clusters:  list              = []
        errors = 0

        for img_path in image_paths:
            try:
                feats = vision_engine.extract_features(img_path)
            except Exception as ex:
                print(f"[REINDEX] Skipping {img_path}: {ex}")
                errors += 1
                continue

            fname = os.path.basename(img_path)
            for f in feats:
                bank: list = []
                if f["has_reliable_face"]:
                    v = np.array(f["face_vector"], np.float32)
                    best_sim, best_idx = 0.0, -1
                    for ci, (c_vec, _) in enumerate(clusters):
                        sim = dot_sim(v, c_vec)
                        if sim > best_sim:
                            best_sim, best_idx = sim, ci
                    if best_sim >= 0.65 and best_idx >= 0:
                        clusters[best_idx][1].append(v)
                        all_vecs = clusters[best_idx][1]
                        clusters[best_idx][0] = l2_normalize(np.mean(all_vecs, axis=0))
                        bank = _build_bank(clusters[best_idx][1])
                    else:
                        clusters.append([v.copy(), [v.copy()]])
                        bank = [v.tolist()]

                pts.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector={"face": f["face_vector"], "body": f["body_vector"]},
                    payload={
                        "filename":          fname,
                        "event_id":          event_id,
                        "fqs":               f["fqs"],
                        "has_reliable_face": f["has_reliable_face"],
                        "blur_score":        f["blur_score"],
                        "yaw_deg":           f.get("yaw_deg", 0.0),
                        "person_area":       f["person_area"],
                        "face_bank":         bank,
                    },
                ))

        for start in range(0, len(pts), 256):
            db_engine.client.upsert(
                collection_name=db_engine.collection_name,
                points=pts[start : start + 256],
            )

        cluster_registry.replace_event(event_id, clusters)
        print(f"[REINDEX] Done: {len(pts)} points | {len(clusters)} clusters | {errors} errors | event='{event_id}'")
        with _cache_lock:
            _search_cache.clear()
        return {
            "status":   "ok",
            "indexed":  len(pts),
            "clusters": len(clusters),
            "errors":   errors,
            "event_id": event_id,
        }

    finally:
        if lock_path.exists():
            lock_path.unlink()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

vision_engine: Any         = None
db_engine:     QdrantEngine = None  

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("ADMIN_KEY"):
        os.environ["ADMIN_KEY"] = "secret"
        print("[WARN] ADMIN_KEY not found in env. Defaulting to 'secret'")

    global vision_engine, db_engine

    lock = Path(LOCK_FILE)
    if lock.exists():
        age = datetime.now() - datetime.fromtimestamp(lock.stat().st_mtime)
        if age > timedelta(minutes=10):
            lock.unlink()
            print(f"[STARTUP] Removed stale lock file ({age} old).")

    project_root = Path(__file__).parent
    patch_vision_pipeline(project_root)
    clear_module_cache()

    from titanium_core.vision import TitaniumVision  
    vision_engine = TitaniumVision()
    db_engine     = QdrantEngine()
    print(f"[STARTUP] MANARAH V{VERSION} is ready.")
    yield
    print("[SHUTDOWN] MANARAH shutting down.")

app = FastAPI(title=f"MANARAH ReID V{VERSION}", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_dir = Path("wedding pictures") if Path("wedding pictures").exists() else Path("static")
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "version": VERSION, "time": datetime.utcnow().isoformat()}

@app.post("/api/v1/admin/upload")
async def upload(
    background_tasks: BackgroundTasks,
    file:      UploadFile = File(...),
    event_id:  str        = Form(default="default"),
    admin_key: str        = Form(...),
):
    if admin_key != os.getenv("ADMIN_KEY"):
        raise HTTPException(status_code=403, detail="Invalid admin key.")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    dest = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    background_tasks.add_task(_process_upload, str(dest), event_id)
    return {"status": "queued", "filename": file.filename, "event_id": event_id}

@app.post("/api/v1/search")
async def search(
    file:     UploadFile = File(...),
    event_id: str        = Form(default="default"),
    top_k:    int        = Form(default=20),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    raw_bytes = await file.read()
    cache_key = hashlib.sha256(raw_bytes + event_id.encode()).hexdigest()

    with _cache_lock:
        if cache_key in _search_cache:
            ts, cached_result = _search_cache[cache_key]
            if time.time() - ts < 300:
                _search_cache.move_to_end(cache_key)
                return cached_result

    tmp = UPLOAD_DIR / f"query_{uuid.uuid4()}{ext}"
    with tmp.open("wb") as fh:
        fh.write(raw_bytes)

    try:
        result = search_event(event_id=event_id, query_image_path=str(tmp), top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")
    finally:
        tmp.unlink(missing_ok=True)

    with _cache_lock:
        _search_cache[cache_key] = (time.time(), result)
        _search_cache.move_to_end(cache_key)
        while len(_search_cache) > MAX_CACHE_SIZE:
            _search_cache.popitem(last=False)

    return result

@app.post("/api/v1/admin/reindex")
async def reindex(
    background_tasks: BackgroundTasks,
    event_id:   str = Form(default="default"),
    images_dir: str = Form(default="images"),
    admin_key:  str = Form(...),
):
    if admin_key != os.getenv("ADMIN_KEY"):
        raise HTTPException(status_code=403, detail="Invalid admin key.")
    if not Path(images_dir).is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {images_dir}")

    background_tasks.add_task(do_reindex, event_id, images_dir)
    return {"status": "started", "event_id": event_id, "images_dir": images_dir}

@app.get("/api/v1/admin/stats/{event_id}")
async def stats(event_id: str):
    count    = db_engine.count_event(event_id)
    clusters = len(cluster_registry._data.get(event_id, []))
    return {
        "event_id": event_id,
        "indexed":  count,
        "clusters": clusters,
        "version":  VERSION,
    }

@app.delete("/api/v1/admin/event/{event_id}")
async def delete_event(event_id: str, admin_key: str = Form(...)):
    if admin_key != os.getenv("ADMIN_KEY"):
        raise HTTPException(status_code=403, detail="Invalid admin key.")
    deleted = db_engine.delete_event(event_id)
    with cluster_registry._lock:
        cluster_registry._data.pop(event_id, None)
        cluster_registry._save()
    return {"status": "deleted", "points_removed": deleted, "event_id": event_id}

def main() -> None:
    parser = argparse.ArgumentParser(description=f"MANARAH ReID Server V{VERSION}")
    parser.add_argument("--host",       default="0.0.0.0")
    parser.add_argument("--port",       default=8000, type=int)
    parser.add_argument("--reload",     action="store_true")
    parser.add_argument("--reindex",    action="store_true", help="Run offline reindex then exit")
    parser.add_argument("--event-id",   default="default", help="Event ID for --reindex mode")
    parser.add_argument("--images-dir", default="images", help="Image folder for --reindex mode")
    args = parser.parse_args()

    if args.reindex:
        if not os.getenv("ADMIN_KEY"):
            os.environ["ADMIN_KEY"] = "offline_reindex" 

        project_root = Path(__file__).parent
        patch_vision_pipeline(project_root)
        clear_module_cache()

        from titanium_core.vision import TitaniumVision
        global vision_engine, db_engine
        vision_engine = TitaniumVision()
        db_engine     = QdrantEngine()
        result        = do_reindex(event_id=args.event_id, images_dir=args.images_dir)
        print(f"[REINDEX] Result: {result}")
        return

    module_name = Path(__file__).stem
    print(f"[MANARAH {VERSION}] Starting → http://{args.host}:{args.port}/docs")
    uvicorn.run(
        f"{module_name}:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )

if __name__ == "__main__":
    main()
