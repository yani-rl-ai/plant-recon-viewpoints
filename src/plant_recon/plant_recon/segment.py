"""Plant segmentation: YOLO26-seg finds the plant (instance region), colour classification gives the pixels.

Why two stages: leaves are thin, a YOLO mask is a smooth region that also contains background between leaves.
The simulated plants have well separated colours (green foliage, terracotta pot, red apples), so inside the
YOLO region an HSV rule decides pixel by pixel.  Outside the region nothing counts (other plants, walls ...).
Classes of the per-pixel maps: 0 background, 1 pot, 2 plant (stem + leaves), 3 apple, 4 soil.
"""
from __future__ import annotations

import cv2
import numpy as np

BG, POT, FOLIAGE, FRUIT, SOIL = 0, 1, 2, 3, 4
CLASS_NAMES = {POT: 'pot', FOLIAGE: 'plant', FRUIT: 'apple', SOIL: 'soil'}


def hsv_classes(bgr: np.ndarray) -> np.ndarray:
    """Colour rule (OpenCV HSV: H 0..179).  Floor / walls / sky have low saturation and fall out automatically."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(np.int16), hsv[..., 1].astype(np.int16), hsv[..., 2].astype(np.int16)
    cls = np.zeros(H.shape, np.uint8)
    cls[(H >= 30) & (H <= 90) & (S >= 50) & (V >= 25)] = FOLIAGE
    cls[(H >= 8) & (H <= 20) & (S >= 90) & (V >= 25) & (V < 110)] = SOIL          # dark brown
    cls[(H >= 5) & (H <= 15) & (S >= 110) & (V >= 110)] = POT                      # terracotta
    cls[((H <= 4) | (H >= 175)) & (S >= 130) & (V >= 40)] = FRUIT
    return cls


def refine_in_region(bgr: np.ndarray, region: np.ndarray, grow: int = 7) -> np.ndarray:
    """Class map restricted to a (slightly grown) region, e.g. the YOLO plant instance."""
    roi = cv2.dilate(region.astype(np.uint8), np.ones((grow, grow), np.uint8)) > 0 if grow > 1 else region > 0
    cls = hsv_classes(bgr)
    cls[~roi] = BG
    return cls


class YoloSegmenter:
    """Thin wrapper around an ultralytics segmentation model (classes of the plant_seg dataset: 0 plant, 1 apple)."""

    def __init__(self, weights: str, conf: float = 0.25, device=0, imgsz: int = 640):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf, self.device, self.imgsz = conf, device, imgsz
        self.names = self.model.names

    def __call__(self, bgr: np.ndarray):
        """List of dicts (name, conf, mask bool HxW, box xyxy) for every detection."""
        r = self.model(bgr, conf=self.conf, device=self.device, imgsz=self.imgsz, verbose=False, retina_masks=True)[0]
        out = []
        if r.masks is None:
            return out
        h, w = bgr.shape[:2]
        for m, c, s, b in zip(r.masks.data.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                              r.boxes.xyxy.cpu().numpy()):
            mask = m > 0.5
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            out.append({'name': self.names[int(c)], 'conf': float(s), 'mask': mask, 'box': b})
        return out


def pick_instance(dets, uv, name='plant'):
    """The `name` instance that contains / is closest to the pixel `uv` (where the plant is expected); None if none."""
    best, best_d = None, 1e9
    u, v = int(round(uv[0])), int(round(uv[1]))
    for d in dets:
        if d['name'] != name:
            continue
        m = d['mask']
        if 0 <= v < m.shape[0] and 0 <= u < m.shape[1] and m[v, u]:
            return d
        ys, xs = np.nonzero(m)
        if len(xs):
            dist = np.min(np.hypot(xs[::7] - u, ys[::7] - v))
            if dist < best_d:
                best, best_d = d, dist
    return best if best_d < 60 else None


def iou_prf(pred: np.ndarray, gt: np.ndarray):
    tp = float((pred & gt).sum())
    p = tp / max(float(pred.sum()), 1.0)
    r = tp / max(float(gt.sum()), 1.0)
    iou = tp / max(float((pred | gt).sum()), 1.0)
    return {'precision': p, 'recall': r, 'iou': iou}


# --------------------------------------------------------------------------------------------- MobileSAM
def _ring(cx, cy, z, r, n=8):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a), np.full(n, z)], axis=1)


def _iou_box(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


class SamSegmenter:
    """Plant / pot / soil / apple masks from MobileSAM.

    SAM needs prompts.  They come from two places:
      * YOLO26 (COCO weights): `potted plant` -> box of the plant,  `apple` / `orange` -> apple boxes (at the 1-2 m
        scanning distance YOLO finds the potted plant in about half of the frames and almost never the 13 px apples);
      * the known geometry: where the plant stands (field layout) and where the camera is (odometry + tilt) give the
        image boxes of the plant volume, of the pot wall and of the soil disc; red blobs give the apple boxes.
    MobileSAM turns each box into a mask; overlaps are resolved by priority apple > soil > pot > plant.
    gate_colour: plant and apple masks additionally drop pixels whose colour is not green / red (SAM leaks into
    the background between thin leaves and around 13 px apples); pot and soil masks are pure SAM.
    """

    def __init__(self, sam_weights, yolo_weights=None, device=0, yolo_conf=0.10, use_yolo_box=True, gate_colour=True,
                 pot_r=0.11, pot_h=0.20, canopy_r=0.55, plant_h=1.55):
        from ultralytics.models.sam import Predictor as SAMPredictor
        self.pred = SAMPredictor(overrides=dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model=sam_weights,
                                                save=False, verbose=False, device=device))
        self.yolo = None
        if yolo_weights:
            from ultralytics import YOLO
            self.yolo = YOLO(yolo_weights)
        self.device, self.yolo_conf, self.use_yolo_box, self.gate = device, yolo_conf, use_yolo_box, gate_colour
        self.g = dict(pot_r=pot_r, pot_h=pot_h, canopy_r=canopy_r, plant_h=plant_h)
        self.last = {}

    def detect(self, bgr):
        if self.yolo is None:
            return []
        r = self.yolo(bgr, conf=self.yolo_conf, device=self.device, verbose=False)[0]
        return [{'name': self.yolo.names[int(c)], 'conf': float(s), 'box': b.tolist()}
                for c, s, b in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy())]

    @staticmethod
    def _box(project, pts, shape, pad=3):
        u, v, z = project(pts)
        ok = z > 0.1
        if ok.sum() < 3:
            return None
        h, w = shape
        b = [max(0.0, u[ok].min() - pad), max(0.0, v[ok].min() - pad), min(w - 1.0, u[ok].max() + pad), min(h - 1.0, v[ok].max() + pad)]
        return b if (b[2] - b[0] > 4 and b[3] - b[1] > 4) else None

    def __call__(self, bgr, project, prior_xy):
        """project(points (n,3) in the model frame) -> (u, v, depth_along_optical_axis).  Returns (class map, plant region)."""
        h, w = bgr.shape[:2]
        cx, cy = float(prior_xy[0]), float(prior_xy[1])
        g = self.g
        box_plant = self._box(project, np.vstack([_ring(cx, cy, z, g['canopy_r']) for z in (0.0, 0.5, 1.0, g['plant_h'])]
                                                 + [[[cx, cy, g['plant_h']]]]), (h, w), pad=6)
        box_pot = self._box(project, np.vstack([_ring(cx, cy, 0.0, g['pot_r']), _ring(cx, cy, g['pot_h'], g['pot_r'])]), (h, w))
        box_soil = self._box(project, np.vstack([_ring(cx, cy, g['pot_h'] + 0.003, g['pot_r'] - 0.01), [[cx, cy, g['pot_h'] + 0.003]]]),
                             (h, w), pad=2)
        cls = np.zeros((h, w), np.uint8)
        if box_plant is None:
            return cls, np.zeros((h, w), bool)
        dets = self.detect(bgr)
        yolo_box = None
        if self.use_yolo_box:
            cand = [d for d in dets if d['name'] == 'potted plant' and _iou_box(d['box'], box_plant) > 0.25]
            if cand:
                yolo_box = max(cand, key=lambda d: d['conf'])['box']
        plant_prompt = yolo_box if yolo_box is not None else box_plant
        # apple boxes: red blobs inside the plant volume + YOLO apple/orange detections
        red = (hsv_classes(bgr) == FRUIT).astype(np.uint8)
        x1, y1, x2, y2 = [int(v) for v in box_plant]
        roi = np.zeros_like(red)
        roi[y1:y2 + 1, x1:x2 + 1] = red[y1:y2 + 1, x1:x2 + 1]
        n, _, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
        apple_boxes = []
        for i in range(1, n):
            bx, by, bw, bh, area = stats[i]
            if area >= 6:
                pad = max(2, int(0.3 * max(bw, bh)))
                apple_boxes.append([max(0, bx - pad), max(0, by - pad), min(w - 1, bx + bw + pad), min(h - 1, by + bh + pad)])
        for d in dets:
            if d['name'] in ('apple', 'orange') and all(_iou_box(d['box'], b) < 0.4 for b in apple_boxes):
                bx = d['box']
                if x1 <= (bx[0] + bx[2]) / 2 <= x2 and y1 <= (bx[1] + bx[3]) / 2 <= y2 and (bx[2] - bx[0]) < 0.35 * (x2 - x1 + 1):
                    apple_boxes.append(bx)
        prompts = [plant_prompt, box_pot, box_soil] + apple_boxes
        present = [b is not None for b in prompts]
        boxes = [b for b in prompts if b is not None]
        self.pred.set_image(bgr)
        res = self.pred(bboxes=boxes)
        masks = res[0].masks.data.cpu().numpy() > 0.5 if res and res[0].masks is not None else np.zeros((0, h, w), bool)
        it = iter(range(len(boxes)))
        take = lambda ok: (masks[next(it)] if ok else None)                                           # noqa: E731
        m_plant, m_pot, m_soil = take(present[0]), take(present[1]), take(present[2])
        m_apples = [masks[next(it)] for _ in apple_boxes]
        clipbox = lambda m, b: self._clip(m, b)                                                       # noqa: E731
        hsv_cls = hsv_classes(bgr) if self.gate else None
        if m_plant is not None:
            mp = clipbox(m_plant, [x1, y1, x2, y2])
            if self.gate:                       # SAM keeps the shape; pixels that are plainly not green are leaks (floor, wall)
                mp &= hsv_cls == FOLIAGE
            cls[mp] = FOLIAGE
        if m_pot is not None:
            cls[clipbox(m_pot, box_pot)] = POT
        if m_soil is not None:
            cls[clipbox(m_soil, box_soil)] = SOIL
        for m, b in zip(m_apples, apple_boxes):
            mm = clipbox(m, b)
            if self.gate:
                mm = mm & (hsv_cls == FRUIT)
            if mm.sum() <= 2.5 * (b[2] - b[0]) * (b[3] - b[1]):                                       # SAM leaked beyond the blob
                cls[mm] = FRUIT
        self.last = {'yolo': dets, 'box_plant': box_plant, 'yolo_box_used': yolo_box is not None, 'n_apple_prompts': len(apple_boxes)}
        return cls, cls > 0

    @staticmethod
    def _clip(mask, box):
        out = np.zeros_like(mask)
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        out[max(0, y1):y2 + 1, max(0, x1):x2 + 1] = mask[max(0, y1):y2 + 1, max(0, x1):x2 + 1]
        return out
