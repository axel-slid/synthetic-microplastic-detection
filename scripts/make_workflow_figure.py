from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "overleaf_microplastic_project" / "images" / "fig_workflow_complex.png"

SAMPLES = {
    "cohort1_img": ROOT / "benchmark" / "data" / "c1" / "imgs" / "117.png",
    "cohort1_mask": ROOT / "benchmark" / "data" / "c1" / "masks" / "117.png",
    "cohort2_img": ROOT / "benchmark" / "data" / "c2" / "imgs" / "0012.png",
    "synthetic_img": ROOT / "benchmark" / "data" / "c2" / "c2_sd2_inpaint" / "generated_04561.png",
    "synthetic_mask": ROOT / "benchmark" / "data" / "c2" / "c2_sd2_inpaint_masks" / "generated_04561.png",
    "cohort3_img": ROOT / "benchmark" / "data" / "c3" / "imgs" / "054.png",
    "cohort3_mask": ROOT / "benchmark" / "data" / "c3" / "masks" / "054.png",
}

INK = "#202a35"
MUTED = "#596879"
LINE = "#c8d0da"
BLUE = "#245f9f"
GREEN = "#2f745c"
PANEL_BG = "#fbfcfd"


def load_rgb(path):
    return Image.open(path).convert("RGB")


def load_binary(path):
    mask = Image.open(path).convert("L")
    arr = np.array(mask)
    out = np.where(arr > 0, 255, 0).astype(np.uint8)
    return Image.fromarray(out, mode="L").convert("RGB")


def crop_box_from_mask(mask_path, margin=0.62):
    arr = np.array(Image.open(mask_path).convert("L"))
    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        width, height = Image.open(mask_path).size
        return (0, 0, width, height)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    side = int(max(x1 - x0 + 1, y1 - y0 + 1) * (1 + 2 * margin))
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    height, width = arr.shape

    x0 = max(0, min(int(round(cx - side / 2)), width - side))
    y0 = max(0, min(int(round(cy - side / 2)), height - side))
    return (x0, y0, min(width, x0 + side), min(height, y0 + side))


def scaled_box(box, source_size, target_size):
    sx = target_size[0] / source_size[0]
    sy = target_size[1] / source_size[1]
    x0, y0, x1, y1 = box
    return (int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy))


def crop_pair(img_path, mask_path):
    mask_img = Image.open(mask_path)
    box = crop_box_from_mask(mask_path)
    img = load_rgb(img_path)
    mask = load_binary(mask_path)
    return img.crop(scaled_box(box, mask_img.size, img.size)), mask.crop(box)


def cover_crop(img, ratio):
    width, height = img.size
    current = width / height
    if current > ratio:
        new_width = int(height * ratio)
        x0 = (width - new_width) // 2
        return img.crop((x0, 0, x0 + new_width, height))
    new_height = int(width / ratio)
    y0 = max(0, (height - new_height) // 2)
    return img.crop((0, y0, width, y0 + new_height))


def add_image(fig, image, box, label=None, label_size=7.2):
    ax = fig.add_axes(box)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
        spine.set_linewidth(0.75)
    if label:
        x, y, w, _ = box
        fig.text(x + w / 2, y - 0.024, label, ha="center", va="top", fontsize=label_size, color=MUTED)


def add_panel(fig, x, y, w, h):
    panel = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.004,rounding_size=0.006",
        transform=fig.transFigure,
        facecolor=PANEL_BG,
        edgecolor=LINE,
        linewidth=0.75,
        zorder=-1,
    )
    fig.add_artist(panel)


def add_step_header(fig, x, y, number, title, subtitle):
    dot = Circle((x + 0.025, y - 0.001), radius=0.017, transform=fig.transFigure, facecolor=BLUE, edgecolor=BLUE)
    fig.add_artist(dot)
    fig.text(x + 0.025, y - 0.001, str(number), ha="center", va="center", fontsize=7.8, color="white", weight="bold")
    fig.text(x + 0.052, y + 0.007, title, ha="left", va="center", fontsize=9.1, color=INK, weight="bold")
    fig.text(x + 0.052, y - 0.027, subtitle, ha="left", va="center", fontsize=7.8, color=MUTED)


def add_arrow(fig, start, end, rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.15,
        color="#536273",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
        zorder=2,
    )
    fig.add_artist(arrow)


def add_inline_arrow(fig, x0, x1, y):
    arrow = FancyArrowPatch(
        (x0, y),
        (x1, y),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=9.5,
        linewidth=1.0,
        color="#536273",
        shrinkA=0,
        shrinkB=0,
        zorder=2,
    )
    fig.add_artist(arrow)


def add_pair(fig, x, y, w, h, img, mask):
    gap = 0.018
    image_w = (w - gap) / 2
    add_image(fig, img, [x, y, image_w, h], "image")
    add_image(fig, mask, [x + image_w + gap, y, image_w, h], "binary mask")


def add_arrow_pair(fig, x, y, w, h, img, mask):
    gap = 0.055
    image_w = (w - gap) / 2
    add_image(fig, img, [x, y, image_w, h], "image")
    add_inline_arrow(fig, x + image_w + 0.014, x + image_w + gap - 0.014, y + h / 2)
    add_image(fig, mask, [x + image_w + gap, y, image_w, h], "binary mask")


def add_triplet(fig, x, y, w, h, first, second, third):
    gap = 0.030
    image_w = (w - 2 * gap) / 3
    add_image(fig, first, [x, y, image_w, h], "original Cohort 2", label_size=6.3)
    fig.text(
        x + image_w + gap / 2,
        y + h / 2,
        "+",
        ha="center",
        va="center",
        fontsize=11.0,
        color="#536273",
        weight="bold",
    )
    add_image(fig, second, [x + image_w + gap, y, image_w, h], "binary mask", label_size=6.7)
    add_inline_arrow(fig, x + 2 * image_w + gap + 0.004, x + 2 * image_w + 2 * gap - 0.004, y + h / 2)
    add_image(fig, third, [x + 2 * (image_w + gap), y, image_w, h], "generated image", label_size=6.5)


def main():
    plt.rcParams.update(
        {
            "font.family": "Noto Sans",
            "font.size": 8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    cohort1_img, cohort1_mask = crop_pair(SAMPLES["cohort1_img"], SAMPLES["cohort1_mask"])
    synthetic_img, synthetic_mask = crop_pair(SAMPLES["synthetic_img"], SAMPLES["synthetic_mask"])
    cohort3_img, cohort3_mask = crop_pair(SAMPLES["cohort3_img"], SAMPLES["cohort3_mask"])
    cohort2_img = cover_crop(load_rgb(SAMPLES["cohort2_img"]), ratio=1.0)

    fig = plt.figure(figsize=(6.9, 4.65), dpi=600)
    fig.subplots_adjust(0, 0, 1, 1)

    panel_w, panel_h = 0.405, 0.345
    x_left, x_right = 0.055, 0.540
    y_top, y_bottom = 0.585, 0.120

    panels = [
        (x_left, y_top, "Train generator model", "Cohort 1 labeled source"),
        (x_right, y_top, "Apply generator to Cohort 2", "original image, mask, generated image"),
        (x_right, y_bottom, "Train segmenter model", "synthetic image and binary-mask pairs"),
        (x_left, y_bottom, "Evaluate and report metrics", "held-out Cohort 3 images"),
    ]

    for idx, (x, y, title, subtitle) in enumerate(panels, start=1):
        add_panel(fig, x, y, panel_w, panel_h)
        add_step_header(fig, x + 0.016, y + panel_h - 0.055, idx, title, subtitle)

    add_pair(fig, x_left + 0.036, y_top + 0.074, panel_w - 0.072, 0.168, cohort1_img, cohort1_mask)

    add_triplet(
        fig,
        x_right + 0.028,
        y_top + 0.082,
        panel_w - 0.056,
        0.140,
        cohort2_img,
        synthetic_mask,
        synthetic_img,
    )

    add_arrow_pair(fig, x_right + 0.036, y_bottom + 0.084, panel_w - 0.072, 0.145, synthetic_img, synthetic_mask)

    add_pair(fig, x_left + 0.036, y_bottom + 0.088, panel_w - 0.072, 0.145, cohort3_img, cohort3_mask)
    fig.text(
        x_left + panel_w / 2,
        y_bottom + 0.017,
        "Metrics: Dice   mask IoU   precision/recall",
        ha="center",
        va="center",
        fontsize=7.1,
        color=INK,
        weight="bold",
    )

    add_arrow(fig, (x_left + panel_w + 0.010, y_top + 0.205), (x_right - 0.010, y_top + 0.205))
    add_arrow(fig, (x_right + panel_w * 0.50, y_top - 0.006), (x_right + panel_w * 0.50, y_bottom + panel_h + 0.006))
    add_arrow(fig, (x_right - 0.010, y_bottom + 0.205), (x_left + panel_w + 0.010, y_bottom + 0.205))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
