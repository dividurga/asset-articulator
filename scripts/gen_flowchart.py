import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

PASTEL   = "#b4a7d6ff"   # box fill
BORDER   = "#7B4FBF"   # box edge
ARROW    = "#7B4FBF"   # arrow colour
TITLE_FG = "#3D1A78"   # bold title text
BODY_FG  = "#3D1A78"   # bullet text

W, H = 6.0, 1.0        # box width / height (data units)
GAP  = 0.38            # vertical gap between boxes
STEP = H + GAP

NODES = [
    (
        "Oriented Bounding Volume",
        "cuboid · cylinder",
    ),
    (
        "Sutherland–Hodgman Clipping",
        "segment–plane intersection · half-space test",
    ),
    (
        "Boundary Loop Extraction",
        "directed half-edge test → ordered loops",
    ),
    (
        "BFS Clustering",
        "shared-edge adjacency → moving parts",
    ),
    (
        "Joint Inference",
        "infer extrusion direction, joint axis & frame",
    ),
    (
        "Extrude and Cap",
        "fan-triangulate · side-wall quads",
    ),
    (
        "URDF Assembly",
        "revolute / prismatic joint: origin, axis, limits",
    ),
]

N      = len(NODES)
FIG_H  = N * STEP + GAP
FIG_W  = W + 1.2

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor("white")

x0 = (FIG_W - W) / 2   # left edge of all boxes

for i, (title, body) in enumerate(NODES):
    # y counts from top
    y_top = FIG_H - GAP - i * STEP
    y_ctr = y_top - H / 2

    box = FancyBboxPatch(
        (x0, y_top - H), W, H,
        boxstyle="round,pad=0.04",
        linewidth=1.6,
        edgecolor=BORDER,
        facecolor=PASTEL,
    )
    ax.add_patch(box)

    ax.text(
        x0 + W / 2, y_ctr + 0.18,
        title,
        ha="center", va="center",
        fontsize=16, fontweight="bold", color=TITLE_FG,
    )
    ax.text(
        x0 + W / 2, y_ctr - 0.22,
        body,
        ha="center", va="center",
        fontsize=13, color=BODY_FG,
        linespacing=1.4,
    )

    if i < N - 1:
        y_arrow_start = y_top - H
        y_arrow_end   = y_top - H - GAP
        ax.annotate(
            "",
            xy=(FIG_W / 2, y_arrow_end),
            xytext=(FIG_W / 2, y_arrow_start),
            arrowprops=dict(
                arrowstyle="-|>",
                color=ARROW,
                lw=1.8,
                mutation_scale=14,
            ),
        )

out = "data/flowchart.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"saved → {out}")
