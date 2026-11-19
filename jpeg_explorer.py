#!/usr/bin/env python3
"""JPEG Explorer — an interactive spatial/DCT view of JPEG compression.

Start with ``python jpeg_explorer.py``.
Use ``--image PATH`` to choose an initial image, 
    ``--quality 75`` for the quality factor, and optionally
    ``--sampling 4:2:0`` for chroma subsampling.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, CheckButtons, Slider
from PIL import Image, ImageOps, JpegImagePlugin, UnidentifiedImageError


DEFAULT_IMAGE = Path(__file__).resolve().parent / "images" / "redhat.ppm"
BLOCK_SIZES = (2, 4, 8, 16)
DCT_MODES = ("Original", "Quantized ≈", "Dequantized ≈", "JPEG decoded", "Difference")
DCT_DESCRIPTIONS = (
    "Before quantization: DCT of original pixels",
    "Original DCT ÷ quantization table, then rounded",
    "Estimated quantized coefficients × quantization table",
    "DCT recomputed from decoded JPEG pixels",
    "Decoded DCT − original DCT (reconstruction error)",
)
SAMPLING_MODES = ("4:4:4", "4:2:2", "4:2:0")
SAMPLING_CODES = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}
CHANNELS = (("Y", 0), ("Cr", 2), ("Cb", 1))  # Pillow orders channels Y, Cb, Cr.
DCT_COLORMAP = "Blues"
SAMPLING_VIS_COLORMAP = "tab20"


def _macos_image_file_dialog() -> Path | None:
    """Use the macOS-native file picker without requiring tkinter/PyObjC.

    AppleScript's ``choose file`` is a standard macOS open-file dialog.
    A canceled dialog is not an error.
    """
    script = (
        'set chosenFile to choose file with prompt "Open an image for JPEG Explorer"\n'
        'return POSIX path of chosenFile'
    )
    result = subprocess.run(
        ["/usr/bin/osascript", "-e", script],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        error = result.stderr.strip()
        if "User canceled" in error or "(-128)" in error:
            return None
        raise RuntimeError(f"macOS Open dialog failed: {error or result.returncode}")
    path = result.stdout.strip()
    return Path(path) if path else None


def _tk_image_file_dialog(initial_dir: Path) -> Path | None:
    """Fallback file picker for other platforms or failed macOS AppleScript."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("Tkinter file dialog is not available") from exc
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        if sys.platform != "darwin":
            root.attributes("-topmost", True)
        filename = filedialog.askopenfilename(
            parent=root, title="Open an image for JPEG Explorer",
            initialdir=str(initial_dir),
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.ppm *.pgm *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        return Path(filename) if filename else None
    finally:
        if root is not None:
            root.destroy()


def choose_image_file(initial_dir: Path) -> Path | None:
    """Select an image with a native dialog on macOS and a Tk fallback."""
    if sys.platform == "darwin":
        try:
            return _macos_image_file_dialog()
        except (OSError, RuntimeError) as mac_error:
            try:
                return _tk_image_file_dialog(initial_dir)
            except Exception as tk_error:
                raise RuntimeError(
                    f"Could not open macOS file picker ({mac_error}); "
                    f"Tkinter fallback also failed ({tk_error})"
                ) from tk_error
    try:
        return _tk_image_file_dialog(initial_dir)
    except Exception as exc:
        raise RuntimeError(f"File picker unavailable: {exc}") from exc


def chroma_group_map(sampling: str, size: int = 8) -> np.ndarray:
    """Return the shared-chroma group index for each luma pixel in an 8×8 block.

    Every luma pixel has its own Y sample. This map indicates which pixels
    share a single chroma (Cb/Cr) sample under the chosen subsampling mode.
    """
    if size != 8:
        raise ValueError("Sampling visualization is defined for an 8×8 luma block")
    if sampling == "4:4:4":
        return np.arange(size * size, dtype=np.int32).reshape(size, size)
    if sampling == "4:2:2":
        rows = np.arange(size, dtype=np.int32)[:, np.newaxis]
        cols = (np.arange(size, dtype=np.int32)[np.newaxis, :] // 2)
        return rows * (size // 2) + cols
    if sampling == "4:2:0":
        rows = (np.arange(size, dtype=np.int32)[:, np.newaxis] // 2)
        cols = (np.arange(size, dtype=np.int32)[np.newaxis, :] // 2)
        return rows * (size // 2) + cols
    raise ValueError(f"Unsupported sampling mode: {sampling}")


def dct_basis(n: int) -> np.ndarray:
    """Return the orthonormal DCT-II transform matrix of dimension n."""
    k = np.arange(n, dtype=np.float64)[:, np.newaxis]
    j = np.arange(n, dtype=np.float64)[np.newaxis, :]
    t = np.sqrt(2.0 / n) * np.cos(np.pi * (2.0 * j + 1.0) * k / (2.0 * n))
    t[0, :] = np.sqrt(1.0 / n)
    return t


def block_dct(values: np.ndarray) -> np.ndarray:
    """2-D orthonormal DCT of the level-shifted 8-bit intensity values.

    For an 8×8 input this matches JPEG's *mathematical* DCT normalization,
    but does not reproduce a codec's integer-arithmetic rounding exactly.
    Non-8×8 inputs are analysis DCTs, not JPEG coding blocks.
    """
    n, m = values.shape
    if n != m:
        raise ValueError("DCT input must be square")
    t = dct_basis(n)
    return t @ (values.astype(np.float64) - 128.0) @ t.T


def histogram_entropy(values: np.ndarray) -> float:
    hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / values.size
    return float(-np.sum(p * np.log2(p)))


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """Use Pillow's 8-bit YCbCr conversion (channels Y,Cb,Cr)."""
    return np.asarray(Image.fromarray(rgb, "RGB").convert("YCbCr")).copy()


def load_rgb(path: Path) -> np.ndarray:
    """Load via Pillow; honor EXIF orientation and flatten transparency on white."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            background.alpha_composite(rgba)
            im = background.convert("RGB")
        else:
            im = im.convert("RGB")
        return np.asarray(im, dtype=np.uint8).copy()


def jpeg_roundtrip(
    rgb: np.ndarray, quality: int, sampling: str = "4:4:4"
) -> tuple[np.ndarray, bytes, dict[int, np.ndarray]]:
    """Encode/decode a real baseline JPEG with selectable chroma sampling.

    The default is deliberately 4:4:4 for the introductory DCT lesson;
    4:2:0 is more typical of photographic JPEG encoder defaults. Actual
    quantization tables are taken from the newly encoded JPEG's DQT markers.
    """
    if sampling not in SAMPLING_CODES:
        raise ValueError(f"Unsupported JPEG chroma sampling: {sampling}")
    stream = BytesIO()
    Image.fromarray(rgb, "RGB").save(
        stream, format="JPEG", quality=int(quality),
        subsampling=SAMPLING_CODES[sampling], optimize=False,
    )
    contents = stream.getvalue()
    with Image.open(BytesIO(contents)) as im:
        actual_sampling = JpegImagePlugin.get_sampling(im)
        if actual_sampling != SAMPLING_CODES[sampling]:
            raise RuntimeError(f"JPEG encoder used unexpected sampling mode {actual_sampling}")
        quantizers = {
            int(key): np.array(value, dtype=np.float64).reshape(8, 8)
            for key, value in (im.quantization or {}).items()
        }
        recovered = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    return recovered, contents, quantizers


def channel_statistics(rgb: np.ndarray, ycbcr: np.ndarray) -> dict[str, np.ndarray]:
    # Display order is Y, Cr, Cb (requested for the three DCT panels).
    order = (0, 2, 1)
    entropies = np.array([histogram_entropy(ycbcr[:, :, i]) for i in order])
    return {
        "means": np.mean(ycbcr[:, :, order], axis=(0, 1)),
        "entropies": entropies,
        # Zero-order independent-channel estimate in bits per full-resolution
        # YCbCr pixel, NOT joint entropy or an actual JPEG coding rate.
        "entropy_bpp": float(np.sum(entropies)),
        "rgb_means": np.mean(rgb, axis=(0, 1)),
    }


def get_block(data: np.ndarray, x0: int, y0: int, n: int) -> np.ndarray:
    """Return an n×n tile. Repeat boundary pixels for partial edge blocks."""
    h, w = data.shape[:2]
    xs = np.clip(np.arange(x0, x0 + n), 0, w - 1)
    ys = np.clip(np.arange(y0, y0 + n), 0, h - 1)
    return data[np.ix_(ys, xs)].copy()


def center_block(width: int, height: int, n: int) -> tuple[int, int]:
    """Select the tile containing the pixel nearest the image center."""
    return (width // 2 // n) * n, (height // 2 // n) * n


def compression_statistics(original: np.ndarray, decoded: np.ndarray, jpeg_size: int) -> dict[str, float]:
    h, w = original.shape[:2]
    squared_error = (original.astype(np.float64) - decoded.astype(np.float64)) ** 2
    mse = float(np.mean(squared_error))
    return {
        "jpeg_kib": jpeg_size / 1024.0,
        "compression_ratio": original.nbytes / jpeg_size,
        "bits_per_pixel": 8.0 * jpeg_size / (w * h),
        "mse": mse,
        "psnr": float(10 * np.log10(255.0**2 / mse)) if mse > 0 else float("inf"),
    }


def approximate_quantized_coefficients(
    original_tile: np.ndarray, quantizers: dict[int, np.ndarray], channel_index: int
) -> np.ndarray:
    """Estimate integer JPEG coefficients for an 8×8 YCbCr tile.

    The actual file's quantization matrices are used. The input pixels are
    obtained with Pillow's RGB→YCbCr transform rather than the encoder's
    internal transform, so integer estimates can differ from the bitstream.
    """
    if original_tile.shape != (8, 8):
        raise ValueError("JPEG quantized-coefficient estimates require 8×8 tiles")
    table_key = 0 if channel_index == 0 else 1
    table = quantizers.get(table_key)
    if table is None:
        raise ValueError("JPEG quantization table not available")
    return np.rint(block_dct(original_tile) / table).astype(np.int32)


def approximate_dequantized_coefficients(
    original_tile: np.ndarray, quantizers: dict[int, np.ndarray], channel_index: int
) -> np.ndarray:
    """Estimate JPEG dequantization: round(DCT / Q) × Q (8×8 only)."""
    table_key = 0 if channel_index == 0 else 1
    table = quantizers.get(table_key)
    if table is None:
        raise ValueError("JPEG quantization table not available")
    return approximate_quantized_coefficients(original_tile, quantizers, channel_index) * table


@dataclass
class ExplorerState:
    quality: int = 75
    block_size: int = 8
    sampling: str = "4:4:4"
    select_point: bool = True
    dct_mode: int = 0
    x0: int = 0
    y0: int = 0


class JPEGExplorer:
    """Interactive Matplotlib GUI; reusable and testable without a GUI backend."""

    def __init__(
        self, image_path: Path = DEFAULT_IMAGE, quality: int = 75, sampling: str = "4:4:4"
    ):
        if sampling not in SAMPLING_MODES:
            raise ValueError(f"Unsupported JPEG chroma sampling: {sampling}")
        self.state = ExplorerState(quality=int(np.clip(quality, 1, 100)), sampling=sampling)
        self.image_path = Path(image_path).expanduser().resolve()
        self.original = load_rgb(self.image_path)
        self.source_ycbcr = rgb_to_ycbcr(self.original)
        self.source_statistics = channel_statistics(self.original, self.source_ycbcr)
        self.decoded: np.ndarray
        self.decoded_ycbcr: np.ndarray
        self.decoded_statistics: dict[str, np.ndarray]
        self.codec_statistics: dict[str, float]
        self.quantizers: dict[int, np.ndarray]
        self.jpeg_data: bytes
        self._compress()
        self.state.x0, self.state.y0 = center_block(
            self.original.shape[1], self.original.shape[0], self.state.block_size
        )
        self._suspend_widgets = False
        self._create_figure()
        self._refresh()

    def _compress(self) -> None:
        self.decoded, self.jpeg_data, self.quantizers = jpeg_roundtrip(
            self.original, self.state.quality, self.state.sampling
        )
        self.decoded_ycbcr = rgb_to_ycbcr(self.decoded)
        self.decoded_statistics = channel_statistics(self.decoded, self.decoded_ycbcr)
        self.codec_statistics = compression_statistics(
            self.original, self.decoded, len(self.jpeg_data)
        )

    def _create_figure(self) -> None:
        self.figure = plt.figure(figsize=(17.8, 15.0), facecolor="#f7f8fa")
        self.figure.canvas.manager.set_window_title("IMGS.361 — JPEG Explorer")

        # The entire input image appears at left and the entire decoded JPEG
        # appears at right. The source is never zoomed/cropped in these panels.
        # The quantization tables are directly under the Quality slider, and
        # the 8×8 chroma-sampling diagram sits directly under the Sampling
        # button. These relationships take priority over grouping all the
        # mathematical panels in a separate bottom analysis area.
        # Selected original/decoded tiles and DCTs remain together below.
        grid = self.figure.add_gridspec(
            6, 12, left=0.042, right=0.958, top=0.952, bottom=0.072,
            height_ratios=[5.0, 0.92, 1.32, 3.05, 2.30, 3.48],
            hspace=0.40, wspace=0.18,
        )
        self.ax_original = self.figure.add_subplot(grid[0, :6])
        self.ax_jpeg = self.figure.add_subplot(grid[0, 6:])
        self.ax_stats = self.figure.add_subplot(grid[1, :])
        self.ax_stats.set_axis_off()
        self.ax_controls = self.figure.add_subplot(grid[2, :])
        self.ax_controls.set_axis_off()
        # Closely linked JPEG controls and their immediate visual outputs.
        self.ax_quant = [self.figure.add_subplot(grid[3, 0:3]),
                         self.figure.add_subplot(grid[3, 3:6])]
        self.ax_sampling_vis = self.figure.add_subplot(grid[3, 6:10])
        self.ax_zoom_original = self.figure.add_subplot(grid[4, :6])
        self.ax_zoom_jpeg = self.figure.add_subplot(grid[4, 6:])
        self.ax_dct = [self.figure.add_subplot(grid[5, k:k + 4]) for k in (0, 4, 8)]

        # The row with the enlarged selected blocks and the DCT-mode selector
        # benefits from a little extra breathing room above it.  Shift that
        # row downward slightly (and trim its height just a bit) so it does not
        # feel crowded against the controls / quantization / sampling row.
        for ax in (self.ax_zoom_original, self.ax_zoom_jpeg):
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0 - 0.016, pos.width, pos.height * 0.95])

        self.original_im = self.ax_original.imshow(self.original, interpolation="nearest")
        self.jpeg_im = self.ax_jpeg.imshow(self.decoded, interpolation="nearest")
        for ax, title in ((self.ax_original, "Original"), (self.ax_jpeg, "JPEG (decoded)")):
            ax.set_title(title, fontsize=13, weight="semibold", pad=7)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#9ca3af")

        self.outlines = []
        self.crosshairs = []
        for ax in (self.ax_original, self.ax_jpeg):
            outline = Rectangle((0, 0), 8, 8, fill=False, linewidth=2.3,
                                edgecolor="#facc15", zorder=8)
            ax.add_patch(outline)
            self.outlines.append(outline)
            (cross,) = ax.plot([], [], marker="+", markersize=10,
                               markeredgewidth=1.7, color="#e11d48", linestyle="none", zorder=9)
            self.crosshairs.append(cross)

        self.zoom_original_im = self.ax_zoom_original.imshow(
            np.zeros((8, 8, 3), dtype=np.uint8), interpolation="nearest"
        )
        self.zoom_jpeg_im = self.ax_zoom_jpeg.imshow(
            np.zeros((8, 8, 3), dtype=np.uint8), interpolation="nearest"
        )
        for ax in (self.ax_zoom_original, self.ax_zoom_jpeg):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#cbd5e1")

        self.dct_images = []
        self.dct_texts: list[list] = []
        for ax in self.ax_dct:
            artist = ax.imshow(np.zeros((8, 8)), origin="upper", interpolation="nearest",
                               cmap=DCT_COLORMAP, vmin=0, vmax=1)
            self.dct_images.append(artist)
            self.dct_texts.append([])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#d1d5db")

        self.sampling_vis_image = self.ax_sampling_vis.imshow(
            np.zeros((8, 8)), origin="upper", interpolation="nearest",
            cmap=SAMPLING_VIS_COLORMAP, vmin=0, vmax=19
        )
        self.ax_sampling_vis.set_xticks([])
        self.ax_sampling_vis.set_yticks([])
        for spine in self.ax_sampling_vis.spines.values():
            spine.set_color("#d1d5db")
        self.sampling_note = self.ax_sampling_vis.text(
            0.5, -0.11, "", transform=self.ax_sampling_vis.transAxes, ha="center",
            va="top", fontsize=8.7, color="#475569"
        )

        # Both 8×8 matrices come directly from the encoded JPEG's DQT
        # markers. JPEG normally uses table 0 for Y and table 1 for Cb/Cr.
        # Keep the color normalization fixed across qualities so darkening
        # genuinely indicates larger quantization divisors.
        self.quant_images = []
        self.quant_texts: list[list] = []
        for ax in self.ax_quant:
            artist = ax.imshow(
                np.zeros((8, 8)), origin="upper", interpolation="nearest",
                cmap="Blues", vmin=0.0, vmax=np.log(255.0)
            )
            self.quant_images.append(artist)
            self.quant_texts.append([])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#d1d5db")

        self.stats_text = self.ax_stats.text(
            0.01, 0.98, "", va="top", ha="left", fontsize=9.5, fontfamily="monospace",
            color="#1f2937", transform=self.ax_stats.transAxes, linespacing=1.45,
        )
        self._create_controls(grid[2, :].get_position(self.figure))
        self.footer = self.figure.text(
            0.5, 0.016,
            "Click either full image to select a block  •  o: open  •  r: reset  •  q: quit  "
            "•  arrows: move block  •  +/-: quality  •  b: block size  •  d: DCT  •  s: sampling  •  h: help",
            ha="center", fontsize=8.8, color="#475569",
        )
        self.ui_status = self.figure.text(
            0.5, 0.060, "", ha="center", va="center", fontsize=9.5, color="#475569",
        )
        self._create_help_overlay()
        self.cid_click = self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.cid_key = self.figure.canvas.mpl_connect("key_press_event", self._on_key)

    def _create_help_overlay(self) -> None:
        """Help is an overlay, not another OS window that may lose focus."""
        self.help_ax = self.figure.add_axes(
            [0.16, 0.13, 0.68, 0.76], facecolor="white", zorder=100
        )
        self.help_ax.set_xticks([])
        self.help_ax.set_yticks([])
        for spine in self.help_ax.spines.values():
            spine.set_edgecolor("#94a3b8")
            spine.set_linewidth(1.6)
        self.help_ax.text(
            0.04, 0.975, "JPEG EXPLORER — HELP", transform=self.help_ax.transAxes,
            va="top", ha="left", fontsize=14, fontweight="semibold", color="#0f172a"
        )
        help_text = (
            "Click either full image to select an analysis block.\n"
            "The magnified blocks use nearest-neighbor interpolation.\n\n"
            "o   Open image                       r   Reset controls\n"
            "q   Quit                             h / ?   Show or hide help\n"
            "← ↑ ↓ →   Move selected block         + / -   JPEG quality\n"
            "b   Cycle block size                d   Cycle DCT view\n"
            "s   Cycle chroma sampling           Esc   Close help\n\n"
            "JPEG always uses 8×8 coding DCT blocks. Other analysis block\n"
            "sizes are for exploratory transforms, not JPEG coding.\n\n"
            "Quantized and dequantized DCT coefficients are estimates, not\n"
            "values extracted from the JPEG entropy-coded bitstream. With\n"
            "4:2:2 or 4:2:0 chroma subsampling, the full-resolution 8×8\n"
            "chrominance tile does not map directly to a JPEG coding block.\n\n"
            "Original entropy (bpp) = H(Y) + H(Cr) + H(Cb), assuming the\n"
            "full-resolution channels are independent. This is not the\n"
            "JPEG compressed file's actual bits per pixel.\n\n"
            "Open uses the native macOS file chooser when running on a Mac.\n"
            "Problems opening files will appear at the bottom of this window."
        )
        self.help_ax.text(
            0.045, 0.895, help_text, transform=self.help_ax.transAxes,
            va="top", ha="left", fontsize=10.2, fontfamily="monospace",
            color="#334155", linespacing=1.45
        )
        self.help_close_ax = self.figure.add_axes(
            [0.754, 0.825, 0.065, 0.036], facecolor="#dbeafe", zorder=101
        )
        self.help_close_button = Button(
            self.help_close_ax, "Close", color="#dbeafe", hovercolor="#bfdbfe"
        )
        self.help_close_button.on_clicked(lambda _evt: self.hide_help())
        self.help_ax.set_visible(False)
        self.help_close_ax.set_visible(False)

    def _create_controls(self, area) -> None:
        x, y, w, h = area.x0, area.y0, area.width, area.height
        # Block and Quality are vertically stacked with the same x-position
        # and width, above their two quantization tables. Sampling stays
        # centered above the independent chroma-sampling diagram.
        slider_x = x + 0.161*w
        slider_width = 0.204*w
        self.ax_block = self.figure.add_axes(
            [slider_x, y + 0.63*h, slider_width, 0.22*h], facecolor="#edf2f7"
        )
        self.block_slider = Slider(
            self.ax_block, "Block", valmin=2, valmax=16, valinit=self.state.block_size,
            valstep=list(BLOCK_SIZES), valfmt="%d px", color="#2563eb"
        )
        self.ax_quality = self.figure.add_axes(
            [slider_x, y + 0.14*h, slider_width, 0.22*h], facecolor="#edf2f7"
        )
        self.quality_slider = Slider(
            self.ax_quality, "Quality", valmin=1, valmax=100,
            valinit=self.state.quality, valstep=1, valfmt="%d", color="#2563eb"
        )
        self.ax_sampling = self.figure.add_axes([
            x + 0.597*w, y + 0.19*h, 0.133*w, 0.61*h
        ])
        self.sampling_button = Button(
            self.ax_sampling, f"Sampling: {self.state.sampling}",
            color="#dbeafe", hovercolor="#bfdbfe",
        )
        self.sampling_button.label.set_fontsize(8.7)
        self.sampling_button.on_clicked(lambda _evt: self.cycle_sampling())

        self.ax_check = self.figure.add_axes([x + 0.742*w, y + 0.18*h, 0.119*w, 0.68*h],
                                             facecolor="#f7f8fa")
        self.select_check = CheckButtons(self.ax_check, ["Select point on"], [True])
        for label in self.select_check.labels:
            label.set_fontsize(8.7)
        self.ax_check.set_xticks([])
        self.ax_check.set_yticks([])

        control_specs = (
            ("Open…", self.open_image_dialog, 0.866, 0.037),
            ("Reset", self.reset, 0.907, 0.029),
            ("Help", self.show_help, 0.941, 0.026),
            ("Quit", self.close, 0.973, 0.027),
        )
        self.buttons: list[Button] = []
        for label, callback, rx, rw in control_specs:
            ax = self.figure.add_axes([x + rx*w, y + 0.19*h, rw*w, 0.61*h])
            button = Button(ax, label, color="#e2e8f0", hovercolor="#bfdbfe")
            button.label.set_fontsize(9.5)
            button.on_clicked(lambda _evt, action=callback: action())
            self.buttons.append(button)

        # Place the DCT selector between the two enlarged blocks, not over them.
        zoom_box = self.ax_zoom_jpeg.get_position()
        self.figure.text(0.5, zoom_box.y0 + 0.121, "Coefficient display",
                         ha="center", va="center", fontsize=11, color="#334155")
        self.ax_dct_mode = self.figure.add_axes([
            0.398, zoom_box.y0 + 0.075, 0.204, 0.033
        ])
        self.dct_button = Button(self.ax_dct_mode, "DCT: Original",
                                 color="#dbeafe", hovercolor="#bfdbfe")
        self.dct_button.label.set_fontsize(9)
        self.dct_description = self.figure.text(0.5, zoom_box.y0 + 0.059, "",
                                                ha="center", va="center", fontsize=9,
                                                color="#334155")
        self.block_info = self.figure.text(0.5, zoom_box.y0 + 0.034, "",
                                          ha="center", va="center", fontsize=8.8,
                                          color="#64748b")
        self.dct_button.on_clicked(lambda _evt: self.cycle_dct_mode())
        self.quality_slider.on_changed(self._on_quality_change)
        self.block_slider.on_changed(self._on_block_change)
        self.select_check.on_clicked(self._on_select_change)

    def _on_quality_change(self, value: float) -> None:
        if self._suspend_widgets:
            return
        quality = int(round(value))
        if quality != self.state.quality:
            self.state.quality = quality
            self._compress()
            self._refresh()

    def _on_block_change(self, value: float) -> None:
        if self._suspend_widgets:
            return
        n = int(round(value))
        if n != self.state.block_size:
            # Preserve the approximate selected pixel, not the old tile origin.
            px = self.state.x0 + self.state.block_size // 2
            py = self.state.y0 + self.state.block_size // 2
            self.state.block_size = n
            self._select_pixel(px, py)

    def _on_select_change(self, _label: str) -> None:
        if self._suspend_widgets:
            return
        self.state.select_point = bool(self.select_check.get_status()[0])
        self._refresh()

    def _select_pixel(self, x: int, y: int) -> None:
        h, w = self.original.shape[:2]
        x = max(0, min(int(x), w - 1))
        y = max(0, min(int(y), h - 1))
        n = self.state.block_size
        self.state.x0 = x // n * n
        self.state.y0 = y // n * n
        self._refresh()

    def _on_click(self, event) -> None:
        if not self.state.select_point or event.inaxes not in (self.ax_original, self.ax_jpeg):
            return
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return
        self._select_pixel(round(event.xdata), round(event.ydata))

    def _on_key(self, event) -> None:
        key = (event.key or "").lower()
        if self.help_ax.get_visible():
            if key in ("h", "?", "escape"):
                self.hide_help()
            return
        if key == "q":
            self.close()
        elif key == "o":
            self.open_image_dialog()
        elif key == "r":
            self.reset()
        elif key == "b":
            i = (BLOCK_SIZES.index(self.state.block_size) + 1) % len(BLOCK_SIZES)
            self.block_slider.set_val(BLOCK_SIZES[i])
        elif key == "d":
            self.cycle_dct_mode()
        elif key == "s":
            self.cycle_sampling()
        elif key in ("+", "="):
            self.quality_slider.set_val(min(100, self.state.quality + 1))
        elif key in ("-", "_"):
            self.quality_slider.set_val(max(1, self.state.quality - 1))
        elif key == "h" or key == "?":
            self.show_help()
        elif key in ("left", "right", "up", "down"):
            dx = (1 if key == "right" else -1 if key == "left" else 0)
            dy = (1 if key == "down" else -1 if key == "up" else 0)
            self._select_pixel(
                self.state.x0 + dx * self.state.block_size,
                self.state.y0 + dy * self.state.block_size,
            )

    def cycle_dct_mode(self) -> None:
        self.state.dct_mode = (self.state.dct_mode + 1) % len(DCT_MODES)
        self._refresh()

    def cycle_sampling(self) -> None:
        """Switch the JPEG encoder's chroma sampling and immediately recompress."""
        idx = (SAMPLING_MODES.index(self.state.sampling) + 1) % len(SAMPLING_MODES)
        self.state.sampling = SAMPLING_MODES[idx]
        self._compress()
        self._refresh()

    def reset(self) -> None:
        """Restore initial controls and select the center block of this image."""
        self._suspend_widgets = True
        self.quality_slider.set_val(75)
        self.block_slider.set_val(8)
        if not self.select_check.get_status()[0]:
            self.select_check.set_active(0)
        self._suspend_widgets = False
        self.state.quality = 75
        self.state.block_size = 8
        self.state.sampling = "4:4:4"
        self.state.dct_mode = 0
        self.state.select_point = True
        self.state.x0, self.state.y0 = center_block(
            self.original.shape[1], self.original.shape[0], 8
        )
        self._compress()
        self._refresh()

    def load_image(self, path: Path) -> None:
        """Open new image; preserve quality and analysis block preferences."""
        rgb = load_rgb(path)
        self.original = rgb
        self.image_path = Path(path).expanduser().resolve()
        self.source_ycbcr = rgb_to_ycbcr(self.original)
        self.source_statistics = channel_statistics(self.original, self.source_ycbcr)
        self._compress()
        self.state.x0, self.state.y0 = center_block(
            self.original.shape[1], self.original.shape[0], self.state.block_size
        )
        self._refresh()

    def _set_status(self, message: str, *, error: bool = False) -> None:
        """Show feedback inside the explorer, including file-dialog failures."""
        self.ui_status.set_text(message)
        self.ui_status.set_color("#b91c1c" if error else "#475569")
        self.figure.canvas.draw_idle()

    def open_image_dialog(self) -> None:
        """Open a file with macOS's own chooser (or a Tk fallback)."""
        try:
            selected = choose_image_file(self.image_path.parent)
            if selected is None:
                self._set_status("Open canceled")
                return
            self.load_image(selected)
            self._set_status(f"Opened: {selected.name}")
        except (OSError, ValueError, UnidentifiedImageError, RuntimeError) as exc:
            self._set_status(f"Open failed: {exc}", error=True)
            print(f"JPEG Explorer: Open failed: {exc}", file=sys.stderr)

    def show_help(self) -> None:
        """Toggle an in-window Help overlay so macOS cannot hide it behind the plot."""
        if self.help_ax.get_visible():
            self.hide_help()
        else:
            self.help_ax.set_visible(True)
            self.help_close_ax.set_visible(True)
            self.figure.canvas.draw_idle()

    def hide_help(self) -> None:
        self.help_ax.set_visible(False)
        self.help_close_ax.set_visible(False)
        self.figure.canvas.draw_idle()

    def close(self) -> None:
        plt.close(self.figure)

    def _refresh(self) -> None:
        n, x0, y0 = self.state.block_size, self.state.x0, self.state.y0
        h, w = self.original.shape[:2]
        self.original_im.set_data(self.original)
        self.jpeg_im.set_data(self.decoded)
        # ``set_data`` alone preserves the previous image extent.  Explicitly
        # update both extents when a newly opened image has different dimensions.
        full_extent = (-0.5, w - 0.5, h - 0.5, -0.5)
        self.original_im.set_extent(full_extent)
        self.jpeg_im.set_extent(full_extent)
        self.ax_original.set_xlim(-0.5, w - 0.5)
        self.ax_original.set_ylim(h - 0.5, -0.5)
        self.ax_jpeg.set_xlim(-0.5, w - 0.5)
        self.ax_jpeg.set_ylim(h - 0.5, -0.5)
        for rect, cross in zip(self.outlines, self.crosshairs):
            rect.set_xy((x0 - 0.5, y0 - 0.5))
            rect.set_width(min(n, w - x0))
            rect.set_height(min(n, h - y0))
            cross.set_data([min(x0 + (n - 1)/2, w - 1)], [min(y0 + (n - 1)/2, h - 1)])
            rect.set_linestyle("-" if self.state.select_point else "--")

        original_tile = get_block(self.original, x0, y0, n)
        decoded_tile = get_block(self.decoded, x0, y0, n)
        for ax, artist, tile, label in (
            (self.ax_zoom_original, self.zoom_original_im, original_tile, "Original"),
            (self.ax_zoom_jpeg, self.zoom_jpeg_im, decoded_tile, "JPEG decoded"),
        ):
            artist.set_data(tile)
            artist.set_extent((-0.5, n - 0.5, n - 0.5, -0.5))
            ax.set_xlim(-0.5, n - 0.5)
            ax.set_ylim(n - 0.5, -0.5)
            ax.set_title(f"{label}  •  {n}×{n} pixel block", fontsize=11, pad=6)
            # Draw explicit pixel cell boundaries at every analysis resolution.
            ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
            ax.grid(which="minor", color="#0f172a", alpha=0.37, linewidth=0.45)
            ax.tick_params(which="minor", length=0)

        tile_mse = float(np.mean((original_tile.astype(float) - decoded_tile.astype(float)) ** 2))
        self.block_info.set_text(f"Block RGB MSE: {tile_mse:.2f}")
        self._refresh_stats()
        self._refresh_dct()
        self._refresh_sampling_visualization()
        self._refresh_quantization()
        self.dct_button.label.set_text(f"DCT: {DCT_MODES[self.state.dct_mode]}")
        self.dct_description.set_text(DCT_DESCRIPTIONS[self.state.dct_mode])
        self.sampling_button.label.set_text(f"Sampling: {self.state.sampling}")
        self.figure.canvas.draw_idle()

    def _refresh_stats(self) -> None:
        h, w = self.original.shape[:2]
        src, dst, c = self.source_statistics, self.decoded_statistics, self.codec_statistics

        def fmt(values: np.ndarray) -> str:
            return ", ".join(f"{val:6.2f}" for val in values)

        def rgb_fmt(values: np.ndarray) -> str:
            return ", ".join(f"{val:5.1f}" for val in values)

        psnr = "∞" if np.isinf(c["psnr"]) else f"{c['psnr']:.2f} dB"
        lines = [
            f"IMAGE  {self.image_path.name}   {w} × {h} pixels     "
            f"Selected block: ({self.state.x0}, {self.state.y0})  {self.state.block_size}×{self.state.block_size}     "
            f"JPEG: {self.state.sampling}, quality {self.state.quality}",
            f"ORIGINAL  mean [Y, Cr, Cb]: ({fmt(src['means'])})     "
            f"entropy [Y, Cr, Cb]: ({fmt(src['entropies'])}) bits/sample  "
            f"Σ entropy: {src['entropy_bpp']:.2f} bpp",
            f"JPEG      mean [Y, Cr, Cb]: ({fmt(dst['means'])})     "
            f"entropy [Y, Cr, Cb]: ({fmt(dst['entropies'])}) bits/sample",
            f"RGB MEANS original [R, G, B]: ({rgb_fmt(src['rgb_means'])})  "
            f"JPEG: ({rgb_fmt(dst['rgb_means'])})     "
            f"Size: {c['jpeg_kib']:.1f} KiB  |  {c['bits_per_pixel']:.2f} bpp  |  "
            f"{c['compression_ratio']:.1f}:1 RGB/JPEG  |  PSNR: {psnr}",
        ]
        self.stats_text.set_text("\n".join(lines))

    def _refresh_dct(self) -> None:
        n, x0, y0 = self.state.block_size, self.state.x0, self.state.y0
        source_block = get_block(self.source_ycbcr, x0, y0, n)
        decoded_block = get_block(self.decoded_ycbcr, x0, y0, n)

        for column, (name, component) in enumerate(CHANNELS):
            src_dct = block_dct(source_block[:, :, component])
            jpeg_dct = block_dct(decoded_block[:, :, component])
            mode = DCT_MODES[self.state.dct_mode]
            if mode == "Original":
                coefficients = src_dct
                method = "Original pixels"
            elif mode == "JPEG decoded":
                coefficients = jpeg_dct
                method = "Decoded pixels"
            elif mode == "Difference":
                coefficients = jpeg_dct - src_dct
                method = "JPEG − original"
            elif n != 8:
                # JPEG quantizes 8×8 blocks only, regardless of display size.
                coefficients = np.full((n, n), np.nan)
                method = "Requires 8×8"
            elif component != 0 and self.state.sampling != "4:4:4":
                # The JPEG encoder performs chroma downsampling before its 8×8
                # DCT. A full-resolution clicked chroma block no longer maps
                # 1:1 to an encoded 8×8 chroma block, so do not imply that it does.
                coefficients = np.full((n, n), np.nan)
                method = "4:4:4 chroma only"
            elif mode == "Quantized ≈":
                coefficients = approximate_quantized_coefficients(
                    source_block[:, :, component], self.quantizers, component
                )
                method = "Est. quantized"
            elif mode == "Dequantized ≈":
                coefficients = approximate_dequantized_coefficients(
                    source_block[:, :, component], self.quantizers, component
                )
                method = "Est. dequantized"
            else:
                raise ValueError(f"Unknown DCT view: {mode}")

            self._draw_dct_grid(
                self.ax_dct[column], self.dct_images[column], self.dct_texts[column],
                coefficients, f"{name} DCT  •  {method}"
            )

    def _refresh_sampling_visualization(self) -> None:
        """Visualize chroma-sharing groups over one full 8×8 luma block."""
        groups = chroma_group_map(self.state.sampling, 8)
        shading = np.mod(groups, 20)
        self.sampling_vis_image.set_data(shading)
        self.sampling_vis_image.set_extent((-0.5, 7.5, 7.5, -0.5))
        self.ax_sampling_vis.set_xlim(-0.5, 7.5)
        self.ax_sampling_vis.set_ylim(7.5, -0.5)
        unique_groups = int(np.unique(groups).size)
        self.ax_sampling_vis.set_title(
            f"8×8 Chroma sampling  •  {self.state.sampling}",
            fontsize=10.2, pad=7, fontweight="semibold"
        )
        self.ax_sampling_vis.set_xticks(np.arange(-0.5, 8, 1), minor=True)
        self.ax_sampling_vis.set_yticks(np.arange(-0.5, 8, 1), minor=True)
        self.ax_sampling_vis.grid(which="minor", color="#64748b", alpha=0.40, linewidth=0.55)
        self.ax_sampling_vis.tick_params(which="minor", length=0)
        # Remove prior grouping boundary artists or cell labels created by this method.
        for artist in list(self.ax_sampling_vis.lines) + list(self.ax_sampling_vis.texts):
            if artist is not self.sampling_note:
                artist.remove()
        # Thicker boundaries emphasize how chroma samples are shared.
        if self.state.sampling == "4:2:2":
            for x in np.arange(-0.5, 8.5, 2.0):
                self.ax_sampling_vis.plot([x, x], [-0.5, 7.5], color="#0f172a", linewidth=2.2, alpha=0.65)
            for y in np.arange(-0.5, 8.5, 1.0):
                self.ax_sampling_vis.plot([-0.5, 7.5], [y, y], color="#0f172a", linewidth=1.0, alpha=0.35)
        elif self.state.sampling == "4:2:0":
            for x in np.arange(-0.5, 8.5, 2.0):
                self.ax_sampling_vis.plot([x, x], [-0.5, 7.5], color="#0f172a", linewidth=2.2, alpha=0.65)
            for y in np.arange(-0.5, 8.5, 2.0):
                self.ax_sampling_vis.plot([-0.5, 7.5], [y, y], color="#0f172a", linewidth=2.2, alpha=0.65)
        else:
            for x in np.arange(-0.5, 8.5, 1.0):
                self.ax_sampling_vis.plot([x, x], [-0.5, 7.5], color="#0f172a", linewidth=1.0, alpha=0.28)
            for y in np.arange(-0.5, 8.5, 1.0):
                self.ax_sampling_vis.plot([-0.5, 7.5], [y, y], color="#0f172a", linewidth=1.0, alpha=0.28)
        # Light Y markers in every pixel cell underline that luma remains full resolution.
        for iy in range(8):
            for ix in range(8):
                self.ax_sampling_vis.text(ix, iy, "Y", ha="center", va="center",
                                          fontsize=7.7, color="#0f172a", alpha=0.78)
        self.sampling_note.set_text(
            f"64 Y; {unique_groups} Cb + {unique_groups} Cr samples"
        )

    def _refresh_quantization(self) -> None:
        """Draw the two 8×8 matrices extracted from the current JPEG file."""
        for i, (table_id, description) in enumerate(
            ((0, "Y (luminance)"), (1, "Cb / Cr (chrominance)"))
        ):
            table = self.quantizers.get(table_id)
            if table is None:
                raise ValueError(f"Encoded JPEG has no quantization table {table_id}")
            self._draw_quantization_grid(
                self.ax_quant[i], self.quant_images[i], self.quant_texts[i], table,
                f"{description}  •  Table {table_id}  •  Quality {self.state.quality}",
            )

    @staticmethod
    def _draw_quantization_grid(ax, image, texts: list, table: np.ndarray, title: str) -> None:
        """Render DQT values in natural-frequency order, DC at upper left."""
        for artist in texts:
            artist.remove()
        texts.clear()
        # JPEG quantization entries are integers in [1, 255] for baseline
        # 8-bit JPEG. Logging the displayed colors improves legibility, while
        # using a *fixed* range makes visual comparisons across Q meaningful.
        shading = np.log(np.clip(table.astype(float), 1, 255))
        image.set_data(shading)
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(7.5, -0.5)
        ax.set_title(title, fontsize=11, pad=7, fontweight="semibold")
        ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
        ax.grid(which="minor", color="#64748b", alpha=0.43, linewidth=0.55)
        ax.tick_params(which="minor", length=0)
        for y in range(8):
            for x in range(8):
                color = "white" if shading[y, x] / np.log(255.0) > 0.66 else "#111827"
                texts.append(ax.text(
                    x, y, f"{int(table[y, x])}", ha="center", va="center",
                    fontsize=9, color=color,
                ))

    @staticmethod
    def _draw_dct_grid(ax, image, texts: list, data: np.ndarray, title: str) -> None:
        for artist in texts:
            artist.remove()
        texts.clear()
        n = data.shape[0]
        absval = np.log1p(np.nan_to_num(np.abs(data), nan=0.0))
        maximum = max(float(absval.max()), 1e-10)
        image.set_data(absval)
        image.set_extent((-0.5, n - 0.5, n - 0.5, -0.5))
        image.set_clim(0, maximum)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_title(title, fontsize=11, pad=7, fontweight="semibold")
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="#64748b", alpha=0.40, linewidth=0.55)
        ax.tick_params(which="minor", length=0)
        # 16×16 is naturally denser; cell numbers are smaller by design.
        fontsize = {2: 11, 4: 11, 8: 8.1, 16: 5.1}[n]
        for iy in range(n):
            for ix in range(n):
                value = data[iy, ix]
                if np.isnan(value):
                    label = "—"
                elif abs(value) < 0.5:
                    label = "0"
                elif abs(value) < 9999:
                    label = str(int(np.rint(value)))
                else:
                    label = f"{value:.0e}"
                text_color = "white" if absval[iy, ix] > maximum * 0.65 else "#111827"
                texts.append(ax.text(ix, iy, label, ha="center", va="center",
                                     fontsize=fontsize, color=text_color))

    def show(self) -> None:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE,
                        help="Image file to open initially (default: images/redhat.ppm)")
    parser.add_argument("--quality", type=int, default=75, metavar="1..100",
                        help="Starting JPEG quality factor (default: 75)")
    parser.add_argument("--sampling", choices=SAMPLING_MODES, default="4:4:4",
                        help="JPEG chroma sampling (default: 4:4:4 for DCT teaching)")
    args = parser.parse_args()
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be between 1 and 100")
    if not args.image.is_file():
        parser.error(f"image not found: {args.image}")
    explorer = JPEGExplorer(args.image, args.quality, args.sampling)
    explorer.show()


if __name__ == "__main__":
    main()
