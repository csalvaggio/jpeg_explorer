# JPEG Explorer

JPEG Explorer is an interactive program that demonstrates JPEG compression, chroma subsampling, DCT coefficients, quantization, and reconstruction error.

## Requirements and running

- Python 3.10 or newer
- NumPy, Matplotlib, and Pillow (see `requirements.txt`)
- A Matplotlib interactive backend for the desktop interface

From the project directory:

```sh
python3 -m pip install -r requirements.txt
python3 jpeg_explorer.py
```

The default input image is `images/redhat.ppm`. To choose a different initial image or encoding settings:

```sh
python3 jpeg_explorer.py --image /path/to/image.png --quality 45 --sampling 4:2:0
```

`--quality` accepts integers from 1 to 100; `--sampling` accepts `4:4:4`, `4:2:2`, or `4:2:0`. The initial settings are quality **75**, sampling **4:4:4**, and an **8×8** analysis block centered in the image. The 4:4:4 default keeps chroma subsampling out of the introductory DCT comparison.

## Interface

The top row shows the full **Original** and **JPEG (decoded)** images. Clicking either image selects the corresponding analysis block, provided **Select point on** is enabled. Both images indicate the same selected region.

The **Block** and **Quality** sliders are grouped above the Y and Cb/Cr quantization tables. The **Sampling** control is located above the 8×8 chroma-sampling diagram. Two enlarged, nearest-neighbor displays show the selected original and reconstructed pixel blocks, with the DCT coefficient selector and its explanation between them. The Y, Cr, and Cb DCT coefficient matrices are shown together below.

The enlarged blocks have pixel-cell boundaries and a fixed display size. Blocks that extend beyond an image boundary repeat edge pixels for display and analysis.

## Controls

| Control | Action |
|---|---|
| Click an original or decoded full image | Select the block at the clicked point |
| **Select point on** | Enable or disable click selection |
| **Block** slider / `b` | Select or cycle the analysis size: 2×2, 4×4, 8×8, or 16×16 |
| **Quality** slider / `+`, `-` | Set JPEG quality (1–100); the keys change it by one |
| **Sampling** button / `s` | Cycle 4:4:4 → 4:2:2 → 4:2:0 |
| **Coefficient display** button / `d` | Cycle DCT coefficient representations |
| **Open…** / `o` | Choose another image |
| **Reset** / `r` | Restore default controls and center the selection |
| **Help** / `h` / `?` | Toggle in-window help; `Esc` closes it |
| Arrow keys | Move the selection one analysis block |
| **Quit** / `q` | Close the explorer |

On macOS, the Open dialog uses the system file chooser through AppleScript (`osascript`), with a Tkinter fallback. On other platforms, the chooser uses Tkinter. An image can also be opened with `--image` if a graphical chooser is unavailable. File-opening errors are displayed in the explorer.

## JPEG encoding and chroma sampling

Pillow encodes the source as an actual JPEG in memory and decodes it for comparison. Changing **Quality** or **Sampling** triggers another encoding and decoding operation.

JPEG retains full-resolution luminance (Y) samples and can use fewer chrominance (Cb and Cr) samples:

| Sampling | Y samples in 8×8 region | Cb samples | Cr samples | Chroma sampling pattern |
|---|---:|---:|---:|---|
| **4:4:4** | 64 | 64 | 64 | One chroma sample per pixel |
| **4:2:2** | 64 | 32 | 32 | One chroma sample per horizontal pair |
| **4:2:0** | 64 | 16 | 16 | One chroma sample per 2×2 group |

The 8×8 sampling diagram depicts an **8×8 luminance region**, not an 8×8 JPEG minimum coded unit. Each cell is a Y sample; matching color/grouping indicates pixels that share a Cb/Cr sample. The diagram changes with the sampling mode but not with the analysis block size.

## DCT coefficient displays

The coefficient selector has five modes. The selected mode and a short explanation appear above the channel matrices.

| Mode | Computation / meaning |
|---|---|
| **Original** | Orthonormal 2-D DCT-II of the original selected Y, Cr, and Cb pixels after subtracting 128 |
| **Quantized ≈** | Estimated integer coefficients: `round(original DCT / Q)` |
| **Dequantized ≈** | Estimated reconstructed coefficients: `round(original DCT / Q) × Q` |
| **JPEG decoded** | DCT computed from Y, Cr, and Cb values of the decoded JPEG image |
| **Difference** | `decoded DCT − original DCT`, showing reconstruction error in the transform domain |

`Q` is the appropriate luminance or chrominance JPEG quantization matrix read from the encoded file. The **Quantized ≈** and **Dequantized ≈** displays are estimates calculated from the selected pixels; they are **not** extracted directly from the JPEG entropy-coded coefficient stream. Encoder color conversion and integer arithmetic can cause differences.

JPEG encodes **8×8 DCT blocks**. The explorer also provides 2×2, 4×4, and 16×16 *analysis* DCTs, but these are not alternate JPEG encoding block sizes. Quantized and dequantized estimates are therefore available only when the analysis block is 8×8.

Under 4:2:2 or 4:2:0 subsampling, an 8×8 full-resolution chroma selection does not correspond to one encoded 8×8 chroma block. The explorer omits the estimated **Cr/Cb quantized and dequantized** coefficient matrices in those modes rather than implying a one-to-one correspondence. Y estimates remain available; Original, JPEG decoded, and Difference always refer to the selected full-resolution pixel region.

## Quantization tables

The two 8×8 numerical panels display the JPEG's **Y (table 0)** and **Cb/Cr (table 1)** quantization divisors extracted from the encoded JPEG. They update with the Quality slider.

Entries are arranged in natural two-dimensional frequency order, with the DC divisor in the upper-left corner. Background shading uses a fixed logarithmic scale from 1 to 255; a larger divisor generally increases quantization error and makes coefficients more likely to become zero. The explorer does not display the JPEG bitstream's zigzag ordering.

## Image statistics

The statistics describe the complete original and decoded images unless otherwise noted:

- **Mean [Y, Cr, Cb]:** Average value of each full-resolution YCbCr channel (0–255).
- **Entropy [Y, Cr, Cb]:** Independent 256-bin Shannon histogram entropy of each channel, in bits/sample.
- **Original Σ entropy (bpp):** `H(Y) + H(Cr) + H(Cb)`, a zero-order sum of full-resolution channel entropies in bits per original color pixel. It is **not** joint color entropy or the JPEG coding rate.
- **RGB means:** Average red, green, and blue intensities for the original and reconstructed images.
- **JPEG size (KiB) and bpp:** Number of bytes in the encoded JPEG (including headers), expressed in kibibytes and bits per original pixel.
- **Compression ratio:** Raw 24-bit RGB image size divided by the JPEG byte count; it is not a comparison to the input image's on-disk format.
- **PSNR:** Peak signal-to-noise ratio based on RGB mean-squared reconstruction error.
- **Block RGB MSE:** Mean-squared error in the currently selected displayed pixel block.

The original entropy estimate and actual compressed JPEG bpp measure different quantities and should not be interpreted as directly interchangeable.

## Code structure

- `jpeg_explorer.py`: Image loading and color conversion; JPEG encode/decode; entropy, DCT, and quantization calculations; Matplotlib interface, controls, and event handlers.
- `images/redhat.ppm`: Default example image.
- `requirements.txt`: Python dependencies.
- `tests/test_jpeg_explorer.py`: Automated mathematical, encoding, interaction, and layout checks.

## Tests

From the project directory:

```sh
python3 -m unittest discover -s tests -v
```

The tests use Matplotlib's noninteractive Agg backend and simulate dialog calls. They do not replace a manual check of the system's native file picker or keyboard focus behavior.

## License

This project is licensed under the GNU General Public License v3.0.
See `LICENSE` for details.

## Contact

### Author

Carl Salvaggio, Ph.D.  
Professor of Imaging Science  
Director, Digital Imaging and Remote Sensing (DIRS) Laboratory

### E-mail

carl.salvaggio@rit.edu

### Organization

Chester F. Carlson Center for Imaging Science  
Rochester Institute of Technology  
Rochester, New York, 14623  
United States
