#!/usr/bin/env python3
# maker.py - World fixed visualization with lane modes + PNG/MP4 export switch
#
# pip install numpy scipy opencv-python

from dataclasses import dataclass
import argparse, os, re, math
import numpy as np
import scipy.io as sio
import cv2


@dataclass
class Cfg:
    res: float = 0.2
    margin: float = 15.0
    stride: int = 5

    lane_thickness_px: int = 2
    ego_traj_thickness_px: int = 2
    ego_box_thickness_px: int = 2
    target_box_thickness_px: int = 2

    ego_length: float = 4.7
    ego_width: float = 1.9
    target_length: float = 4.7
    target_width: float = 1.9

    background_white: bool = True


# ---------- MAT helpers ----------
def series(data, name: str) -> np.ndarray:
    obj = getattr(data, name, None)
    if obj is None:
        raise KeyError(name)
    arr = np.array(obj.data).squeeze() if hasattr(obj, "data") else np.array(obj).squeeze()
    return arr.astype(np.float64)

def try_series(data, name: str):
    try:
        return series(data, name)
    except KeyError:
        return None

def fieldnames(data):
    return list(getattr(data, "_fieldnames", []))


# ---------- Geometry ----------
def rect_corners(cx, cy, length, width, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    lx, wy = length / 2.0, width / 2.0
    local = [( lx, wy), ( lx, -wy), (-lx, -wy), (-lx, wy)]
    out = []
    for px, py in local:
        rx = c * px - s * py
        ry = s * px + c * py
        out.append((cx + rx, cy + ry))
    return out


# ---------- Rasterizer (world -> image) ----------
class Rasterizer:
    # world x -> right, world y -> up
    def __init__(self, x_min, x_max, y_min, y_max, res):
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.y_min, self.y_max = float(y_min), float(y_max)
        self.res = float(res)
        self.W = int(math.ceil((self.x_max - self.x_min) / self.res)) + 1
        self.H = int(math.ceil((self.y_max - self.y_min) / self.res)) + 1

    def inside(self, x, y):
        return (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max)

    def xy_to_rc(self, x, y):
        c = (x - self.x_min) / self.res
        r = (self.y_max - y) / self.res
        return int(round(r)), int(round(c))

    def draw_poly(self, img, pts_xy, color_bgr, thickness=1, fill=False):
        pts = []
        for x, y in pts_xy:
            r, c = self.xy_to_rc(x, y)
            pts.append([c, r])
        pts = np.array([pts], dtype=np.int32)
        if fill:
            cv2.fillPoly(img, pts, color_bgr, lineType=cv2.LINE_AA)
        else:
            cv2.polylines(img, pts, True, color_bgr, thickness=thickness, lineType=cv2.LINE_AA)

    def draw_polyline(self, img, pts_xy, color_bgr, thickness=1):
        pts = []
        for x, y in pts_xy:
            r, c = self.xy_to_rc(x, y)
            pts.append([c, r])
        pts = np.array(pts, dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(img, [pts], False, color_bgr, thickness=thickness, lineType=cv2.LINE_AA)


# ---------- Lane reconstruction ----------
def find_traffic_ids(fields):
    ids = set()
    for f in fields:
        m = re.match(r"(Traffic_T\d\d)_tx$", f)
        if m:
            ids.add(m.group(1))
    return sorted(ids)

def road_heading_from_yaw_devang(ego_yaw, devAng):
    return ego_yaw - devAng

def lane_reference_mode(devDist, tRoad, tMidLane):
    err_road = np.median(np.abs(devDist - tRoad))
    err_lane = np.median(np.abs(devDist - (tRoad - tMidLane)))
    return "lane_center" if err_lane < err_road else "road_center"

def build_lane_points_world(data, idx_list):
    path_x = series(data, "Sensor_Road_Road_Path_tx")
    path_y = series(data, "Sensor_Road_Road_Path_ty")
    devAng = series(data, "Sensor_Road_Road_Path_DevAng")
    ego_yaw = series(data, "Car_Yaw")

    lane_w = try_series(data, "Sensor_Road_Road_Lane_Act_Width")
    if lane_w is None:
        lane_w = np.full_like(path_x, 3.5)

    tMidLane = try_series(data, "Sensor_Road_Road_Lane_Act_tMidLane")
    if tMidLane is None:
        tMidLane = np.zeros_like(path_x)

    devDist = series(data, "Sensor_Road_Road_Path_DevDist")
    tRoad = series(data, "Vhcl_tRoad")

    ref_mode = lane_reference_mode(devDist, tRoad, tMidLane)

    heading = road_heading_from_yaw_devang(ego_yaw, devAng)
    left_pts, right_pts = [], []

    for i in idx_list:
        psi = float(heading[i])
        nx, ny = -math.sin(psi), math.cos(psi)  # left normal
        half = float(lane_w[i]) / 2.0

        rx, ry = float(path_x[i]), float(path_y[i])

        # if path is road centerline -> shift to lane center using tMidLane
        if ref_mode == "road_center":
            tm = float(tMidLane[i])
            cx = rx + nx * tm
            cy = ry + ny * tm
        else:
            cx, cy = rx, ry

        left_pts.append((cx + nx * half,  cy + ny * half))
        right_pts.append((cx - nx * half, cy - ny * half))

    return left_pts, right_pts, ref_mode


# ---------- Video writer helper ----------
def open_mp4_writer(out_path: str, fps: float, frame_w: int, frame_h: int):
    """
    Try common MP4 codecs. Returns cv2.VideoWriter (opened) or raises RuntimeError.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Try a few fourcc codes that often work on Windows/macOS/Linux
    fourcc_candidates = ["mp4v", "avc1", "H264"]
    for cc in fourcc_candidates:
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*cc), fps, (frame_w, frame_h))
        if writer.isOpened():
            return writer, cc
        writer.release()

    raise RuntimeError(
        "Failed to open MP4 VideoWriter. "
        "Try installing a full OpenCV build (conda-forge opencv) or ensure codecs/ffmpeg are available."
    )


# ---------- Main ----------
def generate(mat_path, out_dir, cfg: Cfg, lane_mode: str, lane_window: int, export: str, video_name: str):
    os.makedirs(out_dir, exist_ok=True)

    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    data = mat["data"]
    fields = fieldnames(data)

    time = series(data, "Time")
    ego_x = series(data, "Car_Con_tx")
    ego_y = series(data, "Car_Con_ty")
    ego_yaw = series(data, "Car_Yaw")

    # Auto bounds: ego start~end visible
    x_min = float(np.min(ego_x) - cfg.margin)
    x_max = float(np.max(ego_x) + cfg.margin)
    y_min = float(np.min(ego_y) - cfg.margin)
    y_max = float(np.max(ego_y) + cfg.margin)
    rast = Rasterizer(x_min, x_max, y_min, y_max, cfg.res)

    render_idx = list(range(0, len(time), cfg.stride))

    # Ego trajectory
    ego_traj = [(float(ego_x[i]), float(ego_y[i])) for i in render_idx]

    # Lane points for all render_idx
    lane_left_all, lane_right_all, ref_mode = build_lane_points_world(data, render_idx)

    # ALWAYS draw traffic boxes if present
    tids = find_traffic_ids(fields)
    traffic_cache = {}
    for tid in tids:
        tx = try_series(data, f"{tid}_tx")
        ty = try_series(data, f"{tid}_ty")
        if tx is None or ty is None:
            continue
        tyaw = try_series(data, f"{tid}_rz")  # may be None
        traffic_cache[tid] = (tx, ty, tyaw)

    # Background base (ego trajectory + optional accum lanes)
    bg = (np.full((rast.H, rast.W, 3), 255, np.uint8)
          if cfg.background_white else np.zeros((rast.H, rast.W, 3), np.uint8))
    rast.draw_polyline(bg, ego_traj, (255, 0, 0), thickness=cfg.ego_traj_thickness_px)

    if lane_mode == "accum":
        rast.draw_polyline(bg, lane_left_all,  (0, 255, 0), thickness=cfg.lane_thickness_px)
        rast.draw_polyline(bg, lane_right_all, (0, 255, 0), thickness=cfg.lane_thickness_px)

    # Decide output mode
    if export == "png":
        # Summary PNG
        cv2.imwrite(os.path.join(out_dir, "summary_world.png"), bg)
        writer = None
        used_codec = None
    else:
        # MP4 only: no PNGs at all (including summary)
        # fps based on real time
        dt = np.diff(time)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        dt_med = float(np.median(dt)) if len(dt) else 0.01
        fps = 1.0 / (dt_med * cfg.stride)

        out_mp4 = os.path.join(out_dir, video_name)
        writer, used_codec = open_mp4_writer(out_mp4, fps, rast.W, rast.H)

    # Per-frame
    for k, i in enumerate(render_idx):
        img = bg.copy()

        # lane_mode=frame: draw only local lane around current frame
        if lane_mode == "frame":
            lo = max(0, k - lane_window)
            hi = min(len(render_idx) - 1, k + lane_window)
            local_left = lane_left_all[lo:hi+1]
            local_right = lane_right_all[lo:hi+1]
            rast.draw_polyline(img, local_left,  (0, 255, 0), thickness=cfg.lane_thickness_px)
            rast.draw_polyline(img, local_right, (0, 255, 0), thickness=cfg.lane_thickness_px)

        # ego bbox (outline)
        ego_box = rect_corners(float(ego_x[i]), float(ego_y[i]), cfg.ego_length, cfg.ego_width, float(ego_yaw[i]))
        rast.draw_poly(img, ego_box, (0, 0, 0), thickness=cfg.ego_box_thickness_px, fill=False)

        # traffic bboxes (outline, red) - always
        for tid, (tx, ty, tyaw) in traffic_cache.items():
            x = float(tx[i]); y = float(ty[i])
            # many logs use (0,0) when object not present
            if (abs(x) < 1e-6 and abs(y) < 1e-6) or not np.isfinite(x) or not np.isfinite(y):
                continue
            if not rast.inside(x, y):
                continue
            yaw = float(tyaw[i]) if tyaw is not None else 0.0
            box = rect_corners(x, y, cfg.target_length, cfg.target_width, yaw)
            rast.draw_poly(img, box, (0, 0, 255), thickness=cfg.target_box_thickness_px, fill=False)

        # Output
        if export == "png":
            cv2.imwrite(os.path.join(out_dir, f"sbev_{i:05d}.png"), img)
        else:
            writer.write(img)

    if export == "mp4":
        writer.release()

    print("[DONE]")
    print(f"  out_dir: {out_dir}")
    print(f"  export: {export}")
    print(f"  canvas: H={rast.H}, W={rast.W}, res={cfg.res} m/px")
    print(f"  bounds: x=[{x_min:.2f},{x_max:.2f}], y=[{y_min:.2f},{y_max:.2f}]")
    print(f"  lane_mode: {lane_mode} (window={lane_window} for frame mode)")
    print(f"  lane reference auto-detected: {ref_mode}")
    if len(traffic_cache) == 0:
        print("  traffic: no Traffic_Txx_* series found (nothing to draw).")
    else:
        print(f"  traffic: drawing {len(traffic_cache)} tracks: {list(traffic_cache.keys())}")
    if export == "mp4":
        dt = np.diff(time)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        dt_med = float(np.median(dt)) if len(dt) else 0.01
        fps = 1.0 / (dt_med * cfg.stride)
        print(f"  video: {video_name} (fps≈{fps:.3f}, codec={used_codec})")
    else:
        print("  outputs: summary_world.png, sbev_*.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--res", type=float, default=0.2)
    ap.add_argument("--margin", type=float, default=15.0)

    ap.add_argument("--lane_mode", choices=["frame", "accum"], required=True)
    ap.add_argument("--lane_window", type=int, default=30,
                    help="Only used when lane_mode=frame. Window size in *rendered frames* (stride applied).")

    ap.add_argument("--export", choices=["png", "mp4"], required=True,
                    help="png: write PNGs (and summary png). mp4: write MP4 only (no png at all).")
    ap.add_argument("--video_name", type=str, default="sbev.mp4",
                    help="Only used when export=mp4. Output mp4 file name inside --out directory.")

    args = ap.parse_args()

    cfg = Cfg(res=args.res, margin=args.margin, stride=args.stride)
    generate(args.mat, args.out, cfg, args.lane_mode, args.lane_window, args.export, args.video_name)


if __name__ == "__main__":
    main()