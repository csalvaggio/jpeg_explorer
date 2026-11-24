"""Headless checks for JPEG Explorer's image/DCT model and interactions."""
import os
os.environ["MPLBACKEND"] = "Agg"
import sys
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jpeg_explorer import (  # noqa: E402
    DEFAULT_IMAGE, DCT_MODES, DCT_DESCRIPTIONS, JPEGExplorer,
    approximate_dequantized_coefficients, approximate_quantized_coefficients,
    block_dct, center_block, channel_statistics, chroma_group_map, dct_basis, get_block,
    histogram_entropy, jpeg_roundtrip, load_rgb, rgb_to_ycbcr, scroll_viewport_size,
)


class ModelTests(unittest.TestCase):
    def test_scroll_viewport_keeps_natural_size_on_large_screens(self):
        self.assertEqual(scroll_viewport_size(1780, 1500, 3000, 2400),
                         (1814, 1600))

    def test_scroll_viewport_fits_small_screen_in_both_directions(self):
        width, height = scroll_viewport_size(1780, 1500, 1024, 768)
        self.assertLess(width, 1024)
        self.assertLess(height, 768)
        self.assertLess(width, 1780)
        self.assertLess(height, 1500)

    def test_scroll_viewport_rejects_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            scroll_viewport_size(0, 1200, 1024, 768)

    def test_dct_is_orthonormal(self):
        for n in (2, 4, 8, 16):
            t = dct_basis(n)
            np.testing.assert_allclose(t @ t.T, np.eye(n), atol=1e-14)

    def test_dc_level_shift(self):
        np.testing.assert_allclose(block_dct(np.full((8, 8), 128)), 0, atol=1e-13)
        result = block_dct(np.full((8, 8), 138))
        self.assertAlmostEqual(result[0, 0], 80)
        np.testing.assert_allclose(result[1:, :], 0, atol=1e-12)

    def test_default_loads_and_roundtrips(self):
        original = load_rgb(DEFAULT_IMAGE)
        self.assertEqual(original.shape, (480, 420, 3))
        decoded, data, quantizers = jpeg_roundtrip(original, 50)
        self.assertEqual(original.shape, decoded.shape)
        self.assertGreater(len(data), 100)
        self.assertIn(0, quantizers)
        self.assertIn(1, quantizers)
        self.assertEqual(quantizers[0].shape, (8, 8))
        self.assertTrue(np.any(decoded != original))
        result = approximate_quantized_coefficients(
            rgb_to_ycbcr(original)[0:8, 0:8, 0], quantizers, 0
        )
        self.assertEqual(result.shape, (8, 8))
        self.assertTrue(np.issubdtype(result.dtype, np.integer))

    def test_dequantization_multiplies_estimated_coefficients_by_actual_table(self):
        ycbcr = rgb_to_ycbcr(load_rgb(DEFAULT_IMAGE))
        _, _, q = jpeg_roundtrip(load_rgb(DEFAULT_IMAGE), 50)
        for component in (0, 1, 2):
            block = ycbcr[:8, :8, component]
            quantized = approximate_quantized_coefficients(block, q, component)
            dequantized = approximate_dequantized_coefficients(block, q, component)
            np.testing.assert_array_equal(dequantized, quantized * q[0 if component == 0 else 1])

    def test_chroma_sampling_choices_reencode_pixels(self):
        rgb = load_rgb(DEFAULT_IMAGE)
        results = {}
        for mode in ("4:4:4", "4:2:2", "4:2:0"):
            decoded, data, quantizers = jpeg_roundtrip(rgb, 75, sampling=mode)
            self.assertEqual(decoded.shape, rgb.shape)
            self.assertEqual(len(quantizers), 2)
            results[mode] = (decoded, len(data))
        self.assertGreater(results["4:4:4"][1], results["4:2:0"][1])
        self.assertFalse(np.array_equal(results["4:4:4"][0], results["4:2:0"][0]))
        with self.assertRaises(ValueError):
            jpeg_roundtrip(rgb, 75, sampling="4:1:1")

    def test_chroma_group_map_has_expected_group_counts(self):
        g444 = chroma_group_map("4:4:4")
        g422 = chroma_group_map("4:2:2")
        g420 = chroma_group_map("4:2:0")
        self.assertEqual(g444.shape, (8, 8))
        self.assertEqual(np.unique(g444).size, 64)
        self.assertEqual(np.unique(g422).size, 32)
        self.assertEqual(np.unique(g420).size, 16)
        self.assertEqual(g422[0, 0], g422[0, 1])
        self.assertNotEqual(g422[0, 1], g422[0, 2])
        self.assertEqual(g420[0, 0], g420[1, 1])
        self.assertNotEqual(g420[1, 1], g420[1, 2])

    def test_actual_quantization_tables_vary_with_quality(self):
        original = load_rgb(DEFAULT_IMAGE)
        _, _, q50 = jpeg_roundtrip(original, 50)
        self.assertEqual(int(q50[0][0, 0]), 16)
        self.assertEqual(int(q50[1][0, 0]), 17)
        self.assertEqual(int(q50[0][0, 1]), 11)
        _, _, q100 = jpeg_roundtrip(original, 100)
        np.testing.assert_array_equal(q100[0], np.ones((8, 8)))
        np.testing.assert_array_equal(q100[1], np.ones((8, 8)))
        _, _, q1 = jpeg_roundtrip(original, 1)
        np.testing.assert_array_equal(q1[0], np.full((8, 8), 255))
        np.testing.assert_array_equal(q1[1], np.full((8, 8), 255))

    def test_entropy_and_channel_statistics(self):
        self.assertEqual(histogram_entropy(np.zeros((5, 5), dtype=np.uint8)), 0)
        self.assertAlmostEqual(histogram_entropy(np.array([0, 1] * 10, dtype=np.uint8)), 1)
        rgb = np.full((4, 4, 3), [10, 30, 90], dtype=np.uint8)
        st = channel_statistics(rgb, rgb_to_ycbcr(rgb))
        np.testing.assert_allclose(st["rgb_means"], [10, 30, 90])
        self.assertEqual(len(st["entropies"]), 3)
        self.assertAlmostEqual(st["entropy_bpp"], 0)
        rand = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
        mixed = channel_statistics(rand, rgb_to_ycbcr(rand))
        self.assertAlmostEqual(mixed["entropy_bpp"], sum(mixed["entropies"]))
        self.assertAlmostEqual(mixed["entropy_bpp"], 1)

    def test_center_and_partial_edge_tile(self):
        self.assertEqual(center_block(420, 480, 8), (208, 240))
        src = np.arange(6 * 5).reshape(5, 6)
        crop = get_block(src, 4, 4, 4)
        self.assertEqual(crop.shape, (4, 4))
        np.testing.assert_array_equal(crop[0], [src[4, 4], src[4, 5], src[4, 5], src[4, 5]])
        np.testing.assert_array_equal(crop, np.tile(crop[0], (4, 1)))


class HeadlessInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.explorer = JPEGExplorer(DEFAULT_IMAGE)

    @classmethod
    def tearDownClass(cls):
        cls.explorer.close()

    def test_initial_selected_center(self):
        ex = self.explorer
        self.assertEqual((ex.state.x0, ex.state.y0), (208, 240))
        self.assertEqual(ex.state.block_size, 8)
        self.assertEqual(ex.state.quality, 75)

    def test_select_block_and_sizes(self):
        ex = self.explorer
        ex._select_pixel(15, 27)
        self.assertEqual((ex.state.x0, ex.state.y0), (8, 24))
        ex.block_slider.set_val(16)
        self.assertEqual(ex.state.block_size, 16)
        self.assertEqual((ex.state.x0, ex.state.y0), (0, 16))
        ex.cycle_dct_mode()
        ex.cycle_dct_mode()
        self.assertEqual(ex.state.dct_mode, DCT_MODES.index("Dequantized ≈"))
        self.assertEqual(ex.ax_dct[0].get_title(), "Y DCT  •  Requires 8×8")
        self.assertEqual(ex.dct_description.get_text(), DCT_DESCRIPTIONS[2])
        ex.reset()
        self.assertEqual((ex.state.x0, ex.state.y0), (208, 240))
        self.assertEqual(ex.state.dct_mode, 0)
        self.assertEqual(ex.state.block_size, 8)
        self.assertEqual(ex.state.sampling, "4:4:4")
        self.assertEqual(ex.ax_original.get_title(), "Original")
        self.assertEqual(ex.ax_jpeg.get_title(), "JPEG (decoded)")
        self.assertEqual(ex.dct_button.label.get_text(), "DCT: Original")
        self.assertNotIn("(d)", ex.dct_button.label.get_text())
        self.assertIn("Σ entropy:", ex.stats_text.get_text())
        self.assertIn("4:4:4", ex.ax_sampling_vis.get_title())
        self.assertIn("64 Cb + 64 Cr samples", ex.sampling_note.get_text())

    def test_dct_sequence_labels_and_description(self):
        ex = self.explorer
        ex.reset()
        for idx, mode in enumerate(DCT_MODES):
            self.assertEqual(ex.dct_button.label.get_text(), f"DCT: {mode}")
            self.assertEqual(ex.dct_description.get_text(), DCT_DESCRIPTIONS[idx])
            if mode == "Dequantized ≈":
                ytile = get_block(ex.source_ycbcr, ex.state.x0, ex.state.y0, 8)[:, :, 0]
                expected = approximate_dequantized_coefficients(ytile, ex.quantizers, 0)
                self.assertEqual(ex.dct_texts[0][0].get_text(), str(int(np.rint(expected[0, 0]))))
            ex.cycle_dct_mode()
        self.assertEqual(ex.state.dct_mode, 0)

    def test_sampling_button_recomputes_and_restricts_chroma_estimates(self):
        ex = self.explorer
        ex.reset()
        ex.cycle_dct_mode()  # estimated quantized
        self.assertTrue(ex.dct_texts[1][0].get_text() != "—")
        original_bytes = len(ex.jpeg_data)
        ex.cycle_sampling()  # 4:2:2
        self.assertEqual(ex.state.sampling, "4:2:2")
        self.assertIn("4:2:2", ex.sampling_button.label.get_text())
        self.assertIn("JPEG: 4:2:2", ex.stats_text.get_text())
        self.assertIn("32 Cb + 32 Cr samples", ex.sampling_note.get_text())
        self.assertEqual(ex.dct_texts[1][0].get_text(), "—")
        self.assertNotEqual(original_bytes, len(ex.jpeg_data))
        ex.cycle_sampling()  # 4:2:0
        self.assertEqual(ex.state.sampling, "4:2:0")
        self.assertIn("16 Cb + 16 Cr samples", ex.sampling_note.get_text())
        ex.cycle_sampling()  # 4:4:4
        self.assertEqual(ex.state.sampling, "4:4:4")
        self.assertNotEqual(ex.dct_texts[1][0].get_text(), "—")
        ex.reset()

    def test_click_selection_checkbox(self):
        ex = self.explorer
        ex.reset()
        click = SimpleNamespace(inaxes=ex.ax_jpeg, button=1, xdata=31, ydata=49)
        ex._on_click(click)
        self.assertEqual((ex.state.x0, ex.state.y0), (24, 48))
        ex.select_check.set_active(0)
        self.assertFalse(ex.state.select_point)
        ex._on_click(SimpleNamespace(inaxes=ex.ax_original, button=1, xdata=99, ydata=99))
        self.assertEqual((ex.state.x0, ex.state.y0), (24, 48))
        ex.reset()

    def test_loading_image_of_different_size(self):
        ex = self.explorer
        with TemporaryDirectory() as directory:
            filename = Path(directory) / "other.png"
            Image.new("RGB", (43, 35), (180, 30, 105)).save(filename)
            ex.load_image(filename)
            self.assertEqual(ex.decoded.shape, (35, 43, 3))
            self.assertEqual(tuple(ex.original_im.get_extent()), (-0.5, 42.5, 34.5, -0.5))
            self.assertEqual((ex.state.x0, ex.state.y0), (16, 16))
        ex.load_image(DEFAULT_IMAGE)
        ex.reset()

    def test_layout_puts_visuals_under_corresponding_controls(self):
        ex = self.explorer
        ex.figure.canvas.draw()
        quality = ex.ax_quality.get_position()
        sampling = ex.ax_sampling.get_position()
        q0, q1 = [ax.get_position() for ax in ex.ax_quant]
        chroma = ex.ax_sampling_vis.get_position()
        dcts = [ax.get_position() for ax in ex.ax_dct]
        zoom = ex.ax_zoom_original.get_position()

        self.assertLess(max(q0.y1, q1.y1), quality.y0)
        self.assertLess(chroma.y1, sampling.y0)
        self.assertAlmostEqual(quality.x0 + quality.width / 2,
                               (q0.x0 + q1.x1) / 2, delta=0.055)
        self.assertAlmostEqual(sampling.x0 + sampling.width / 2,
                               chroma.x0 + chroma.width / 2, delta=0.055)
        self.assertLess(q0.x1, q1.x0)
        self.assertLess(q1.x1, chroma.x0)
        self.assertLess(max(ax.y1 for ax in dcts), zoom.y0)
        self.assertGreater(chroma.y0, zoom.y1)
        self.assertGreater(min(q0.y0, q1.y0, chroma.y0) - zoom.y1, 0.055)

    def test_stacked_sliders_and_spacing(self):
        ex = self.explorer
        ex.reset()
        ex.figure.canvas.draw()
        block = ex.ax_block.get_position()
        quality = ex.ax_quality.get_position()
        self.assertGreater(block.y0, quality.y1)
        self.assertAlmostEqual(block.x0, quality.x0, delta=1e-8)
        self.assertAlmostEqual(block.width, quality.width, delta=1e-8)
        self.assertLess(ex.ax_original.get_position().y1, 0.97)
        zoom = ex.ax_zoom_jpeg.get_position()
        button = ex.ax_dct_mode.get_position()
        self.assertLess(ex.dct_description.get_position()[1], button.y0)
        self.assertGreater(ex.dct_description.get_position()[1], button.y0 - 0.025)
        caption = next(t for t in ex.figure.texts if t.get_text() == "Coefficient display")
        self.assertGreater(caption.get_position()[1], button.y1)
        self.assertLess(caption.get_position()[1] - button.y1, 0.02)
        self.assertGreater(button.y0, zoom.y0)

    def test_help_keyboard_and_button(self):
        ex = self.explorer
        ex.hide_help()
        ex._on_key(SimpleNamespace(key="h"))
        self.assertTrue(ex.help_ax.get_visible())
        ex._on_key(SimpleNamespace(key="escape"))
        self.assertFalse(ex.help_ax.get_visible())
        self.assertIn("Help", [button.label.get_text() for button in ex.buttons])
        # Matplotlib Button callback registry also serves the real GUI click.
        help_button = next(button for button in ex.buttons if button.label.get_text() == "Help")
        help_button._observers.process("clicked", None)
        self.assertTrue(ex.help_ax.get_visible())
        ex.help_close_button._observers.process("clicked", None)
        self.assertFalse(ex.help_ax.get_visible())

    def test_open_keyboard_and_button_route_to_dialog(self):
        ex = self.explorer
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "dialog_test.png"
            Image.new("RGB", (32, 32), (15, 25, 35)).save(image_path)
            with patch("jpeg_explorer.choose_image_file", return_value=image_path) as dialog:
                ex._on_key(SimpleNamespace(key="o"))
                self.assertEqual(ex.image_path, image_path)
                self.assertIn("Opened:", ex.ui_status.get_text())
                open_button = next(button for button in ex.buttons if button.label.get_text() == "Open…")
                open_button._observers.process("clicked", None)
                self.assertEqual(dialog.call_count, 2)
            with patch("jpeg_explorer.choose_image_file", side_effect=RuntimeError("simulated dialog failure")):
                ex.open_image_dialog()
                self.assertIn("simulated dialog failure", ex.ui_status.get_text())
            ex.load_image(DEFAULT_IMAGE)
            ex.reset()

    def test_quality_recompress(self):
        ex = self.explorer
        size_before = len(ex.jpeg_data)
        ex.quality_slider.set_val(20)
        self.assertEqual(ex.state.quality, 20)
        self.assertNotEqual(len(ex.jpeg_data), size_before)
        ex.reset()

    def test_quantization_panels_update_with_slider_and_reset(self):
        ex = self.explorer
        ex.reset()
        self.assertEqual(len(ex.quant_images), 2)
        for texts in ex.quant_texts:
            self.assertEqual(len(texts), 64)
        ex.quality_slider.set_val(50)
        self.assertEqual(ex.quant_texts[0][0].get_text(), "16")
        self.assertEqual(ex.quant_texts[1][0].get_text(), "17")
        self.assertIn("Quality 50", ex.ax_quant[0].get_title())
        np.testing.assert_allclose(ex.quant_images[0].get_array(), np.log(ex.quantizers[0]))
        ex.quality_slider.set_val(100)
        self.assertEqual(ex.quant_texts[0][0].get_text(), "1")
        self.assertEqual(ex.quant_texts[1][0].get_text(), "1")
        self.assertIn("Quality 100", ex.ax_quant[1].get_title())
        ex.reset()
        self.assertEqual(ex.state.quality, 75)
        self.assertEqual(ex.quant_texts[0][0].get_text(), "8")


if __name__ == "__main__":
    unittest.main()
