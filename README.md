# LayerDiffuse for AUTOMATIC1111

Generate transparent PNG images natively using Latent Transparency. Initial testing shows it works roughly 70-80% of the time without a problem. Some images would require additional work due to muddy or incomplete transparency. 

## Attribution

This is a port of [sd-forge-layerdiffuse](https://github.com/lllyasviel/sd-forge-layerdiffuse) by [lllyasviel](https://github.com/lllyasviel) for use with standard AUTOMATIC1111 WebUI. All credit for LayerDiffuse technology and models goes to the original authors.

**⚠️ This port has received limited testing.**

---

## Usage

1. Enable **LayerDiffuse** in the accordion panel
2. Select your method:
   - `(SDXL) Attention Injection` - Best quality for SDXL
   - `(SDXL) Conv Injection` - Sharper edges
   - `(SD1.5) Attention Injection` - For SD 1.5 models **Untested**
3. Enable **Auto-add transparency prompts** (recommended)
4. Generate!

### Optimal Background Color

For best transparency results, the model should generate a **mid-gray background (RGB 128, 128, 128)**. The auto-prompt feature handles this automatically.

If transparency is incomplete, add to your prompts:
- **Positive:** `transparent_background, dark_gray_background, simple_background`
- **Negative:** `white background, black background, gradient_background`

---

## API

```json
"alwayson_scripts": {
  "LayerDiffuse": {
    "args": [true, "(SDXL) Only Generate Transparent Image (Attention Injection)", 1.0, 1.0, true, "Crop and Resize", false]
  }
}
```

Args: `[enabled, method, weight, ending_step, auto_prompt, resize_mode, output_origin]`
