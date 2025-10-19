# ugly_doodle.py
import argparse, random, math, os
from tkinter import Tk, filedialog
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
C = mp_face_mesh.FACEMESH_CONTOURS  # high-level facial outlines
OVAL = mp_face_mesh.FACEMESH_FACE_OVAL
LIPS = mp_face_mesh.FACEMESH_LIPS
LEFT_EYE = mp_face_mesh.FACEMESH_LEFT_EYE
RIGHT_EYE = mp_face_mesh.FACEMESH_RIGHT_EYE
LEFT_EYEBROW = mp_face_mesh.FACEMESH_LEFT_EYEBROW
RIGHT_EYEBROW = mp_face_mesh.FACEMESH_RIGHT_EYEBROW
NOSE = mp_face_mesh.FACEMESH_NOSE
BODY = mp_pose.POSE_CONNECTIONS

def rng(seed=42):
    r = random.Random(seed)
    return r

def jitter_points(points, mag):
    # add small random offsets to simulate “doodle” wobble
    out = []
    for x, y in points:
        jx = x + np.random.normal(0, mag)
        jy = y + np.random.normal(0, mag)
        out.append((jx, jy))
    return out

def polyline(canvas, pts, thickness=3, alpha=1.0, close=False):
    pts = np.array(pts, dtype=np.int32)
    overlay = canvas.copy()
    if len(pts) >= 2:
        if close:
            cv2.polylines(overlay, [pts], True, (0,0,0), thickness, lineType=cv2.LINE_AA)
        else:
            cv2.polylines(overlay, [pts], False, (0,0,0), thickness, lineType=cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

def collect_chain(landmarks, image_w, image_h, connections):
    # Convert connection graph into ordered chains (rough but ok for doodle)
    adj = {}
    for a,b in connections:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    # Build chains by walking unvisited nodes with degree <= 2
    visited = set()
    chains = []
    for start in adj:
        if start in visited: 
            continue
        # try to start from an endpoint (deg 1), else any node
        deg = len(adj[start])
        if deg != 1: 
            continue
        chain = [start]
        visited.add(start)
        cur = start
        prev = None
        while True:
            neighbors = [n for n in adj[cur] if n != prev]
            if not neighbors: 
                break
            nxt = neighbors[0]
            if nxt in visited: 
                break
            chain.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt
        chains.append(chain)
    # Add any small cycles not covered
    for node in adj:
        if node not in visited:
            # walk a short cycle
            cycle = [node]
            visited.add(node)
            cur = node
            prev = None
            for _ in range(200):
                neighbors = [n for n in adj[cur] if n != prev]
                if not neighbors: break
                nxt = neighbors[0]
                if nxt in visited and nxt == cycle[0]:
                    cycle.append(nxt)
                    break
                cycle.append(nxt)
                visited.add(nxt)
                prev, cur = cur, nxt
            if len(cycle) > 2:
                chains.append(cycle)
    # Map to pixel coords
    out_chains = []
    for chain in chains:
        pts = []
        for idx in chain:
            lm = landmarks[idx]
            pts.append((lm.x * image_w, lm.y * image_h))
        out_chains.append(pts)
    return out_chains

def landmarks_to_xy(landmarks, w, h, indices):
    return [(landmarks[i].x*w, landmarks[i].y*h) for i in indices]

def conn_to_indices(conns):
    # Convert Mediapipe connections to sorted unique index pairs and flatten to a set of indices
    idxs = set()
    for a,b in conns:
        idxs.add(a)
        idxs.add(b)
    return sorted(list(idxs))

def exaggeration_factors(landmarks, w, h):
    # crude feature metrics to push caricature:
    # eye size vs face width, nose length vs face height, mouth width vs face width
    # returns (eye_scale, nose_scale, mouth_scale, brow_drop)
    # Use key points from Mesh indices (pick rough anchors from contours):
    # Approx: use left/right eye centers and mouth corners via provided sets
    def centroid(points):
        x = sum(p[0] for p in points)/len(points)
        y = sum(p[1] for p in points)/len(points)
        return (x,y)
    li = conn_to_indices(LEFT_EYE)
    ri = conn_to_indices(RIGHT_EYE)
    mi = conn_to_indices(LIPS)
    oi = conn_to_indices(OVAL)
    ni = conn_to_indices(NOSE)

    Leye = landmarks_to_xy(landmarks, w, h, li)
    Reye = landmarks_to_xy(landmarks, w, h, ri)
    Mouth = landmarks_to_xy(landmarks, w, h, mi)
    Oval = landmarks_to_xy(landmarks, w, h, oi)
    Nose = landmarks_to_xy(landmarks, w, h, ni)

    if not (Leye and Reye and Mouth and Oval and Nose):
        return (1.1, 1.1, 1.1, 5)  # more subtle defaults

    face_w = max(x for x,y in Oval) - min(x for x,y in Oval)
    face_h = max(y for x,y in Oval) - min(y for x,y in Oval)
    mouth_w = max(x for x,y in Mouth) - min(x for x,y in Mouth)

    le_c = centroid(Leye)
    re_c = centroid(Reye)
    eye_dist = abs(le_c[0] - re_c[0])

    nose_h = max(y for x,y in Nose) - min(y for x,y in Nose)

    # heuristics - toned down for a more "similar but ugly" look
    eye_scale = 1.0 + min(0.4, 0.3 * (eye_dist / (face_w + 1e-6)))
    nose_scale = 1.0 + min(0.5, 0.4 * (nose_h / (face_h + 1e-6)))
    mouth_scale = 1.0 + min(0.4, 0.35 * (mouth_w / (face_w + 1e-6)))
    brow_drop = int(3 + 15 * (eye_dist / (face_w + 1e-6)))
    return (eye_scale, nose_scale, mouth_scale, brow_drop)

def scale_shape(pts, center, sx, sy):
    out=[]
    for x,y in pts:
        out.append((center[0] + (x - center[0]) * sx,
                    center[1] + (y - center[1]) * sy))
    return out

def run(input_path, output_path, bg='white', seed=42, thickness=3):
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")
    h, w = img.shape[:2]
    canvas = np.full((h, w, 3), 255, np.uint8) if bg=='white' else np.zeros((h, w, 3), np.uint8)

    r = rng(seed)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process face
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    ) as fm:
        res = fm.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            eye_scale, nose_scale, mouth_scale, brow_drop = exaggeration_factors(lm, w, h)

            # Build doodle chains for key regions
            parts = [
                OVAL, LIPS, LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW, NOSE
            ]
            # draw multiple passes with jitter and variable alpha for messy look
            for conns in parts:
                chains = collect_chain(lm, w, h, conns)
                for ch in chains:
                    if len(ch) < 3: 
                        continue
                    pts = ch

                    # Exaggerate by part
                    cx = sum(p[0] for p in pts)/len(pts)
                    cy = sum(p[1] for p in pts)/len(pts)
                    center = (cx, cy)

                    if conns in (LEFT_EYE, RIGHT_EYE):
                        pts = scale_shape(pts, center, eye_scale, eye_scale)
                    elif conns in (LIPS,):
                        pts = scale_shape(pts, center, mouth_scale, mouth_scale)
                    elif conns in (NOSE,):
                        pts = scale_shape(pts, center, 1.0, nose_scale)
                    elif conns in (LEFT_EYEBROW, RIGHT_EYEBROW):
                        pts = [(x, y + brow_drop) for (x,y) in pts]

                    # jitter & draw a few layers to feel scribbly
                    for layer in range(3):
                        mag = 1.5 + layer * 0.8
                        thick = max(1, thickness - layer)
                        j = jitter_points(pts, mag)
                        canvas = polyline(canvas, j, thickness=thick, alpha=0.85, close=False)

    # Process body
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as pose:
        res = pose.process(rgb)
        if res.pose_landmarks:
            body_lm = res.pose_landmarks.landmark
            for con in BODY:
                p1_idx, p2_idx = con
                p1 = body_lm[p1_idx]
                p2 = body_lm[p2_idx]
                
                # only draw if both points are reasonably visible
                if p1.visibility > 0.4 and p2.visibility > 0.4:
                    pt1 = (p1.x * w, p1.y * h)
                    pt2 = (p2.x * w, p2.y * h)
                    pts = [pt1, pt2]
                    
                    for layer in range(3):
                        mag = 2.0 + layer * 1.0
                        thick = max(1, thickness - layer + 1)
                        j = jitter_points(pts, mag)
                        canvas = polyline(canvas, j, thickness=thick, alpha=0.8, close=False)

    # Fallback if no face or body was found
    if np.all(canvas == 255) or np.all(canvas == 0):
        edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 80, 160)
        ys, xs = np.where(edges > 0)
        pts = list(zip(xs, ys))
        r.shuffle(pts)
        for i in range(0, min(3000, len(pts)), 8):
            x,y = pts[i]
            x2 = int(x + r.uniform(-6,6))
            y2 = int(y + r.uniform(-6,6))
            cv2.line(canvas, (x,y), (x2,y2), (0,0,0), 1, cv2.LINE_AA)
        Image.fromarray(canvas[:,:,::-1]).save(output_path)
        return

    # Add random crosshatching on cheeks for “ugly” texture
    for _ in range(120):
        x = r.randint(int(0.2*w), int(0.8*w))
        y = r.randint(int(0.35*h), int(0.8*h))
        l = r.randint(5, 20)
        angle = r.uniform(-math.pi/3, math.pi/3)
        x2 = int(x + l*math.cos(angle))
        y2 = int(y + l*math.sin(angle))
        cv2.line(canvas, (x,y), (x2,y2), (0,0,0), 1, cv2.LINE_AA)

    Image.fromarray(canvas[:,:,::-1]).save(output_path)

if __name__ == "__main__":
    # Set up argument parser
    ap = argparse.ArgumentParser(description="Make an 'ugly doodle' from a face photo.")
    ap.add_argument("--bg", choices=["white", "black"], default="white", help="Background color")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--thickness", type=int, default=3, help="Stroke thickness")
    args = ap.parse_args()

    # Set up Tkinter root window
    root = Tk()
    root.withdraw()  # Hide the main window

    # Open file dialog to choose image
    input_path = filedialog.askopenfilename(
        title="Select a face photo",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )

    if not input_path:
        print("No image selected. Exiting.")
    else:
        # Create results directory if it doesn't exist
        if not os.path.exists("results"):
            os.makedirs("results")

        # Generate output path
        basename = os.path.basename(input_path)
        fname, ext = os.path.splitext(basename)
        output_path = os.path.join("results", f"{fname}_doodle.png")

        print(f"Processing {input_path} -> {output_path}")
        run(input_path, output_path, args.bg, args.seed, args.thickness)
        print("Done!")
