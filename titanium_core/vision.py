"""
TitaniumVision — MANARAH V30.3-Production
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

FACE_DIM = 512
BODY_DIM = 768

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

            out.append({
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
            })

        return out
