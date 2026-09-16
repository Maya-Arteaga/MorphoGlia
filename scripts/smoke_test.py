from __future__ import annotations

import importlib
import platform
import sys


CHECKS = [
    ("numpy", "NumPy"),
    ("pandas", "pandas"),
    ("scipy", "SciPy"),
    ("skimage", "scikit-image"),
    ("sklearn", "scikit-learn"),
    ("cv2", "OpenCV"),
    ("matplotlib", "Matplotlib"),
    ("seaborn", "seaborn"),
    ("PIL", "Pillow"),
    ("tifffile", "tifffile"),
    ("tqdm", "tqdm"),
    ("umap", "UMAP"),
    ("joblib", "joblib"),
    ("threadpoolctl", "threadpoolctl"),
    ("yaml", "PyYAML"),
    ("tkinter", "Tkinter"),
    ("morphoglia", "MorphoGlia"),
    ("morphoglia.gui.app", "MorphoGlia GUI"),
]

print()
print("=" * 68)
print("MORPHOGLIA INSTALLATION CHECK")
print("=" * 68)
print(f"Python:   {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")
print()

failed = []

for module_name, label in CHECKS:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        failed.append((label, exc))
        print(f"X {label:<24} {type(exc).__name__}: {exc}")
        continue

    version = getattr(module, "__version__", "")
    suffix = f" {version}" if version else ""
    print(f"OK {label:<22}{suffix}")

print()

if failed:
    print(f"FAILED: {len(failed)} import(s)")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Runtime compatibility check
# ---------------------------------------------------------------------------

print()
print("Runtime compatibility:")

try:
    import numpy as np
    import sklearn
    import umap

    rng = np.random.default_rng(42)
    X = rng.normal(size=(32, 4)).astype(np.float32)

    embedding = umap.UMAP(
        n_neighbors=5,
        n_components=2,
        min_dist=0.1,
        random_state=42,
        n_epochs=20,
    ).fit_transform(X)

    if embedding.shape != (32, 2):
        raise RuntimeError(
            f"Unexpected UMAP output shape: {embedding.shape}"
        )

    if not np.isfinite(embedding).all():
        raise RuntimeError(
            "UMAP compatibility test produced non-finite values."
        )

    print(
        "OK UMAP/scikit-learn runtime  "
        f"umap={getattr(umap, '__version__', '?')}  "
        f"sklearn={getattr(sklearn, '__version__', '?')}"
    )

except Exception as exc:
    print(
        "X UMAP/scikit-learn runtime   "
        f"{type(exc).__name__}: {exc}"
    )
    raise SystemExit(1)



# ---------------------------------------------------------------------------
# Topology primitive parity check
# ---------------------------------------------------------------------------

print()
print("Topology primitive:")

try:
    import numpy as np
    from morphoglia.morphometrics.builtin import _neighbor_degree_map

    rng = np.random.default_rng(918273)
    skeleton = (
        rng.random((37, 43)) > 0.82
    ).astype(np.uint8)

    reference = np.zeros_like(
        skeleton,
        dtype=np.int8,
    )

    h, w = skeleton.shape

    for y in range(h):
        for x in range(w):
            total = 0

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue

                    yy = y + dy
                    xx = x + dx

                    if (
                        0 <= yy < h
                        and 0 <= xx < w
                    ):
                        total += int(
                            skeleton[yy, xx] > 0
                        )

            reference[y, x] = total

    observed = _neighbor_degree_map(
        skeleton
    )

    if not np.array_equal(
        observed,
        reference,
    ):
        mismatch = int(
            np.count_nonzero(
                observed != reference
            )
        )
        raise RuntimeError(
            "8-neighbor degree map is not exact: "
            f"{mismatch} pixel(s) differ."
        )

    print(
        "OK exact 8-neighbor topology degree map"
    )

except Exception as exc:
    print(
        "X topology primitive             "
        f"{type(exc).__name__}: {exc}"
    )
    raise SystemExit(1)


print("MorphoGlia environment is ready.")
print("=" * 68)
