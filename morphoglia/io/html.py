
# morphoglia/io/html.py

from __future__ import annotations
from pathlib import Path
import base64
import warnings
import cv2
import numpy as np
from string import Template
import re
import json


from ..morphometrics.builtin import (
    # canonical masks & features
    make_qc_artifacts, QCArtifacts,
    # Sholl masks & metrics (circles + overlaps from soma-centroid)
    make_sholl_artifacts, ShollParams,
    make_branch_order_artifacts,
    # utilities / constants
    _largest_contour, DIST_FRACTION, SOMA_DILATE_ITERS,
)


WHITE=(200, 200, 200) 
# ---------------- helpers ----------------
def _to_png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode PNG")
    return buf.tobytes()

def _b64(img: np.ndarray) -> str:
    try:
        return base64.b64encode(_to_png_bytes(img)).decode("ascii")
    except Exception as e:
        warnings.warn(f"Failed to base64-encode image: {e}")
        return ""

# -------------- renderers (BGR) ---------------
def _render_cell(a: QCArtifacts) -> np.ndarray:
    vis = np.zeros((*a.bin_cell.shape, 3), dtype=np.uint8)
    vis[a.bin_cell > 0] = (40, 10, 230)  # red cell
    return vis

def _render_hull(a: QCArtifacts) -> np.ndarray:
    vis = np.zeros((*a.bin_cell.shape, 3), dtype=np.uint8)
    cnt = _largest_contour(a.bin_cell)
    if cnt is not None:
        vis[a.bin_cell > 0] = WHITE  # white cell for context

        hull = cv2.convexHull(cnt)
        color = (20, 255, 0)         # BGR (green)
        alpha = 0.30                 # fill opacity

        # 1) alpha fill on an overlay
        overlay = vis.copy()
        cv2.fillConvexPoly(overlay, hull, color)
        # blend only affects the green-filled region since overlay is black elsewhere
        cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

        # 2) crisp outline on top
        cv2.drawContours(vis, [hull], -1, color, 2)

    return vis


def _render_soma(a: QCArtifacts) -> np.ndarray:
    vis = np.zeros((*a.bin_cell.shape, 3), dtype=np.uint8)
    vis[a.bin_cell > 0] = WHITE  # white cell
    vis[a.soma > 0] = (40, 10, 230)          # red soma overlay
    return vis

def _render_branch_orders_skeleton(a: QCArtifacts) -> np.ndarray:
    bo = make_branch_order_artifacts(a, secondary_mode=">=2")
    vis = np.zeros((*a.bin_cell.shape, 3), dtype=np.uint8)
    vis[a.soma > 0] = (200, 200, 200)       # light soma
    vis[bo.primary_skel_mask > 0]       = (0, 255, 0)   # primary = green
    vis[bo.intermediate_skel_mask > 0]  = (0, 0, 255)   # intermediate(+)= red
    return vis


def _render_branch_orders_area(a: QCArtifacts) -> np.ndarray:
    bo = make_branch_order_artifacts(a, secondary_mode="rest")
    vis = np.zeros((*a.bin_cell.shape, 3), dtype=np.uint8)
    vis[a.bin_cell > 0] = WHITE
    vis[bo.intermediate_area_mask > 0] = (0, 255, 0)  # GREEN intermediate (rest)
    vis[bo.primary_area_mask  > 0]     = (0, 0, 255)  # RED primary (on top)
    return vis




def _render_branches(a: QCArtifacts) -> np.ndarray:
    vis = np.zeros((*a.bin_cell.shape, 3), dtype=np.uint8)
    vis[a.bin_cell > 0] = WHITE  # white cell
    vis[a.branches > 0] = (0, 0, 255)      # red branches
    return vis

def _render_skeleton_raw(a: QCArtifacts) -> np.ndarray:
    vis = np.zeros((*a.bin_cell.shape, 3), dtype=np.uint8)
    vis[a.skeleton_full > 0] = (255, 255, 255)  # white 1px skeleton
    return vis


def _render_skeleton_over_soma(a: QCArtifacts) -> np.ndarray:
    """
    Soma as light gray for context; 1px skeleton in bright white on top.
    """
    vis = np.zeros((*a.bin_cell.shape, 3), dtype=np.uint8)
    vis[a.soma > 0] = (200, 200, 200)       # light gray soma
    vis[a.skeleton_full > 0] = (255, 255, 255)  # bright white skeleton
    return vis


def _render_skeleton_features(a: QCArtifacts) -> np.ndarray:
    """
    Show soma + skeleton feature classes with per-class dilation:
      - slab (deg==2): no dilation (context)
      - junctions (deg>=3): dilate x1 (green)
      - endpoints (deg==1): dilate x2 (red)
      - initial endpoints (touch soma): dilate x3 (yellow, drawn last)
    """
    vis = np.zeros((*a.bin_cell.shape, 3), dtype=np.uint8)

    # base soma for context (white)
    vis[a.soma > 0] = WHITE

    # helper: dilate a {0,255} mask N times and clip to the cell area
    def dilate_n(mask_0255: np.ndarray, iters: int) -> np.ndarray:
        if iters <= 0:
            out = (mask_0255 > 0).astype(np.uint8) * 255
        else:
            k3 = np.ones((3, 3), np.uint8)
            out = cv2.dilate((mask_0255 > 0).astype(np.uint8), k3, iterations=iters) * 255
        # keep overlays inside the cell silhouette
        return (out & ((a.bin_cell > 0).astype(np.uint8) * 255)).astype(np.uint8)

    # --- base class (no dilation) ---
    vis[a.slab_mask > 0] = (240, 150, 0)      # blue (BGR)

    # --- dilated overlays (order matters; later draws on top) ---
    
    junc_dil = dilate_n(a.junctions_mask, 1)   # ×1
    endp_dil = dilate_n(a.endpoints_mask,  2)  # ×2
    init_dil = dilate_n(a.initial_points_mask, 3)  # ×3 (draw last)

    vis[junc_dil > 0] = (0, 255, 0)        # green
    vis[endp_dil > 0] = (0, 0, 255)        # red
    vis[init_dil > 0] = (0, 200, 255)      # yellow

    return vis


def _render_sholl(a_qc: QCArtifacts, roi: np.ndarray, sh_params: ShollParams) -> np.ndarray:
    """
    Render Sholl using builtin artifacts:
      - soma (white)
      - skeleton parts outside circles (white)
      - circle perimeters (blue)
      - overlaps of circles with skeleton minus soma (green)
    """
    sh = make_sholl_artifacts(
        roi,
        params=sh_params,
        soma_frac=DIST_FRACTION,
        soma_dilate_iters=SOMA_DILATE_ITERS,
    )

    vis = np.zeros((*a_qc.bin_cell.shape, 3), dtype=np.uint8)

    # white soma
    vis[a_qc.soma > 0] = WHITE

    # white skeleton parts that are NOT in overlaps
    skel_non_overlap = (a_qc.skeleton_full > 0) & (sh.overlap_mask == 0)
    vis[skel_non_overlap] = WHITE

    # blue circles
    vis[sh.circle_mask > 0] = (255, 0, 0)

    # green overlaps
    vis[sh.overlap_mask > 0] = (0, 0, 255)

    return vis

# FIX: sanitize more than spaces (colons, slashes, etc.)
def safe_id(name: str) -> str:
    return "btn-" + re.sub(r'[^a-zA-Z0-9_-]+', '-', name)



def _render_branches_pit(a: QCArtifacts) -> np.ndarray:
    bo = make_branch_order_artifacts(a, secondary_mode="rest")
    vis = np.zeros((*a.bin_cell.shape, 3), dtype=np.uint8)

    vis[a.soma > 0] = WHITE

    primary_color      = (30, 180, 220)  # BGR
    intermediate_color = (200, 50, 90)   # BGR
    terminal_color     = (240, 200, 40)  # BGR

    vis[bo.primary_area_mask > 0]        = primary_color
    vis[bo.intermediate_area_mask > 0]   = intermediate_color
    vis[bo.terminal_area_mask > 0]       = terminal_color
    return vis







# -------------- HTML writer ----------------



# --- helpers for HTML writer ---


def _neighbors_from_order(
    ordered_ids: list[str] | None,
    out_dir: Path,
    cell_id: str
) -> tuple[str, str]:
    """
    Prefer prev/next from the caller-provided ordered_ids (stable even if files
    don't exist yet). Fallback to filesystem glob within out_dir.
    """
    if ordered_ids:
        try:
            i = ordered_ids.index(cell_id)
            prev_id = ordered_ids[i-1] if i > 0 else ""
            next_id = ordered_ids[i+1] if i < len(ordered_ids)-1 else ""
            return (f"{prev_id}.html" if prev_id else "", f"{next_id}.html" if next_id else "")
        except ValueError:
            pass  # cell_id not in list → fall back

    # Fallback: glob whatever is present now
    current = f"{cell_id}.html"
    def key(name: str) -> tuple[int, str]:
        m = re.search(r'(\d+)(?=\.html$)', name)
        return (int(m.group(1)) if m else -1, name)

    names = [p.name for p in out_dir.glob("*.html")]
    if current not in names:
        names.append(current)
    names.sort(key=key)

    idx = names.index(current)
    prev_href = names[idx - 1] if idx > 0 else ""
    next_href = names[idx + 1] if idx < len(names) - 1 else ""
    return prev_href, next_href


def _build_views(a: QCArtifacts, roi: np.ndarray, sholl_params: ShollParams) -> list[tuple[str, np.ndarray]]:
    """
    Return an ordered list of (view_label, image) pairs that will be shown in the toolbar.
    We deliberately omit the raw 'Skeleton' view from the HTML (kept renderer in code).
    """
    return [
        ("Cell",                                  _render_cell(a)),
        ("Convex Hull",                           _render_hull(a)),
        ("Soma",                                  _render_soma(a)),
        ("Skeleton Features",                     _render_skeleton_features(a)),
        ("Branches: Primary/Intermediate/Terminal", _render_branches_pit(a)),
        ("Sholl",                                 _render_sholl(a, roi, sholl_params)),
    ]

def _images_json(views: list[tuple[str, np.ndarray]]) -> str:
    """Build a JSON map: label -> data URI."""
    mp = {label: "data:image/png;base64," + _b64(img) for label, img in views}
    # Back-compat: legacy keys map to the same PIT image
    if "Branches: Primary/Intermediate/Terminal" in mp:
        pit_uri = mp["Branches: Primary/Intermediate/Terminal"]
        mp["Branches (Primary/Intermediate/Terminal)"] = pit_uri
        mp["Branches (Primary/Secondary/Terminal)"] = pit_uri
        mp["Branches (Primary/Secondary)"] = pit_uri
    return json.dumps(mp, ensure_ascii=False)

def _names_json(views: list[tuple[str, np.ndarray]]) -> str:
    """Visible order of view labels (toolbar & keyboard cycling)."""
    return json.dumps([label for label, _ in views], ensure_ascii=False)

def _toolbar_html(views: list[tuple[str, np.ndarray]]) -> str:
    """
    Build the toolbar buttons. The button text for PIT is shortened, but the onclick
    uses the full view label to fetch the correct image.
    """
    parts: list[str] = []
    for i, (label, _) in enumerate(views):
        btn_text = "Branches (P/I/T)" if label == "Branches: Primary/Intermediate/Terminal" else label
        cls = ' class="active"' if i == 0 else ''
        # single-quoted attribute so json.dumps(label) (double quotes) is safe
        parts.append(
            f"<button onclick='setView({json.dumps(label)})' id='{safe_id(label)}'{cls}>{btn_text}</button>"
        )
    return "\n    ".join(parts)


def _prev_next_hrefs(out_dir: Path, cell_id: str) -> tuple[str, str]:
    """
    Natural numeric sort of neighbors by trailing integer in the filename.
    """
    current = f"{cell_id}.html"

    def key(name: str) -> tuple[int, str]:
        m = re.search(r'(\d+)(?=\.html$)', name)
        return (int(m.group(1)) if m else -1, name)

    names = [p.name for p in out_dir.glob("*.html")]
    if current not in names:
        names.append(current)
    names.sort(key=key)

    idx = names.index(current)
    prev_href = names[idx - 1] if idx > 0 else ""
    next_href = names[idx + 1] if idx < len(names) - 1 else ""
    return prev_href, next_href

def _pager_html(prev_href: str, next_href: str) -> tuple[str, str]:
    prev_html = f'<a href="{prev_href}">‹ Prev cell</a>' if prev_href else '<span class="disabled">‹ Prev cell</span>'
    next_html = f'<a href="{next_href}">Next cell ›</a>' if next_href else '<span class="disabled">Next cell ›</span>'
    return prev_html, next_html


def write_cell_html(
    cell_id: str,
    roi: np.ndarray,
    out_dir: Path,
    *,
    soma_frac: float = DIST_FRACTION,
    soma_dilate: int = SOMA_DILATE_ITERS,
    sholl_params: ShollParams = ShollParams(),
    ordered_ids: list[str] | None = None, 
) -> Path:
    """
    Build canonical masks and render views.
    - View buttons at TOP
    - « / » chevrons overlay the image to cycle CELLS (was views before)
    - Prev/Next CELL links at BOTTOM
    - Legend shows only for relevant views
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    a = make_qc_artifacts(
        roi,
        soma_frac=soma_frac,
        soma_dilate_iters=soma_dilate,
    )

    # Build view set + neighbors
    views = _build_views(a, roi, sholl_params)
    images_json  = _images_json(views)
    names_json   = _names_json(views)
    toolbar_html = _toolbar_html(views)
    
    # Use the caller-provided ordered_ids if available
    prev_href, next_href = _neighbors_from_order(ordered_ids, out_dir, cell_id)
    prev_link_html, next_link_html = _pager_html(prev_href, next_href)

    # Single template with minimal placeholders
    tmpl = Template("""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>${cell_id}</title>
<style>
  :root { --bg:#0f1115; --panel:#141821; --border:#2a3140; --text:#e8eaf0; --accent:#3b82f6; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:var(--bg); color:var(--text); }
  .wrap { max-width: 1200px; margin: 32px auto; padding: 0 16px; }
  h1 { font-weight:600; font-size:26px; margin:0 0 10px; text-align:center; }
  .subbar { text-align:center; opacity:0.8; font-size:12px; margin-bottom:10px; }

  .toolbar { margin: 18px 0 10px; display:flex; gap:12px; flex-wrap:wrap; justify-content:center; }
  button { padding:10px 14px; font-size:15px; border-radius:12px; border:1px solid var(--border); background:var(--panel); color:var(--text); cursor:pointer; }
  button.active { background:var(--accent); color:white; border-color:var(--accent); }

  .viewer { position:relative; display:flex; align-items:center; justify-content:center; min-height:60vh;
            background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:24px; }
  .viewer img { display:block; max-width:min(95vw, 1000px); max-height:75vh; height:auto; width:auto;
                image-rendering: pixelated; margin: 0 auto; object-fit: contain; }

  .chev { position:absolute; top:50%; transform:translateY(-50%); border-radius:999px; width:42px; height:42px;
          display:flex; align-items:center; justify-content:center; border:1px solid var(--border);
          background:rgba(20,24,33,0.75); cursor:pointer; user-select:none; font-size:20px; }
  .chev:hover { background:rgba(59,130,246,0.25); border-color:var(--accent); }
  .chev-left { left:10px; }
  .chev-right { right:10px; }

  .legend { margin-top:10px; display:flex; gap:12px; flex-wrap:wrap; justify-content:center; }
  .leg-item { display:flex; align-items:center; gap:8px; font-size:13px; opacity:0.95; }
  .swatch { width:16px; height:16px; border-radius:4px; border:1px solid var(--border); }

  .pager { margin-top:14px; display:flex; justify-content:space-between; gap:12px; }
  .pager a, .pager span { flex:1; text-align:center; padding:12px 14px; border-radius:10px; border:1px solid var(--border);
                          background:var(--panel); color:var(--text); text-decoration:none; font-size:18px; }
  .pager a:hover { background:rgba(59,130,246,0.15); border-color:var(--accent); }
  .pager .disabled { opacity:0.45; pointer-events:none; }
</style>
</head>
<body>
<div class="wrap">
  <h1>${cell_id}</h1>
  <div class="subbar">Tip: use ←/→ keys to switch cells</div>   <!-- ### CHANGED -->

  <div class="toolbar">
    ${toolbar_buttons}
  </div>

  <div class="viewer">
    <div class="chev chev-left"  onclick="gotoCell(-1)" title="Previous cell">‹</div>  <!-- ### CHANGED -->
    <img id="view" alt="cell view" decoding="async" />
    <div class="chev chev-right" onclick="gotoCell(+1)" title="Next cell">›</div>      <!-- ### CHANGED -->
  </div>

  <div class="legend" id="legend"></div>

  <div class="pager">
    ${prev_link}
    ${next_link}
  </div>
</div>

<script>
  // Data injected from Python
  const IMAGES = ${images_json};
  const NAMES  = ${names_json};

  // Neighbors injected from Python  ### NEW
  const PREV_CELL = ${prev_href_json};
  const NEXT_CELL = ${next_href_json};

  // Back-compat for older saved names → new label
  const SYNONYMS = {
    "Branches (Primary/Intermediate/Terminal)": "Branches: Primary/Intermediate/Terminal",
    "Branches (Primary/Secondary/Terminal)": "Branches: Primary/Intermediate/Terminal",
    "Branches (Primary/Secondary)": "Branches: Primary/Intermediate/Terminal"
  };

  const view   = document.getElementById('view');
  const legend = document.getElementById('legend');

  const LEGENDS = {
    "Skeleton Features": [
      { name: "Slab",       color: "rgb(0,150,240)" },
      { name: "Initial",    color: "rgb(255,200,0)" },
      { name: "Junctions",  color: "rgb(0,255,0)"   },
      { name: "Endpoints",  color: "rgb(255,0,0)"   },
    ],
    "Branches: Primary/Intermediate/Terminal": [
      { name: "Primary",      color: "rgb(220,180,30)" },
      { name: "Intermediate", color: "rgb(90,50,200)"  },
      { name: "Terminal",     color: "rgb(40,200,240)" }
    ],
    "default": []
  };

  function safeId(name) { return "btn-" + name.replace(/[^a-zA-Z0-9_-]+/g, "-"); }
  function renderLegend(name) {
    const items = LEGENDS[name] || LEGENDS["default"];
    legend.innerHTML = items.map(it =>
      '<div class="leg-item"><span class="swatch" style="background:'+it.color+';"></span>'+it.name+'</div>'
    ).join('');
  }
  function setView(name) {
    const real = SYNONYMS[name] || name;
    view.src = IMAGES[real];
    for (const id of NAMES) {
      const b = document.getElementById(safeId(id));
      if (b) b.classList.toggle('active', id === real);
    }
    renderLegend(real);
    try { localStorage.setItem('mg_last_view', real); } catch(e) {}
  }

  // ### NEW — replaces nudge()
  function gotoCell(dir) {
    if (dir < 0 && PREV_CELL) {
      window.location.href = PREV_CELL;
    } else if (dir > 0 && NEXT_CELL) {
      window.location.href = NEXT_CELL;
    }
  }

  // Restore last view if present
  const saved0 = (function(){ try { return localStorage.getItem('mg_last_view'); } catch(e) { return null; }})();
  const saved  = saved0 && (IMAGES[saved0] || IMAGES[SYNONYMS[saved0]]) ? (SYNONYMS[saved0] || saved0) : null;
  setView(saved || NAMES[0]);

  // Keyboard ← / →  ### CHANGED
  document.addEventListener('keydown', (e) => {
    if (e.target && (/input|textarea|select/i).test(e.target.tagName)) return;
    if (e.key === 'ArrowLeft')  gotoCell(-1);
    if (e.key === 'ArrowRight') gotoCell(+1);
  });
</script>
</body>
</html>""")

    html = tmpl.safe_substitute(
        cell_id=cell_id,
        toolbar_buttons=toolbar_html,
        images_json=images_json,
        names_json=names_json,
        prev_link=prev_link_html,
        next_link=next_link_html,
        prev_href_json=json.dumps(prev_href),   # ### NEW
        next_href_json=json.dumps(next_href),   # ### NEW
    )

    html_path = out_dir / f"{cell_id}.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path







