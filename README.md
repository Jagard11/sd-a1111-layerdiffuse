# LayerDiffuse for AUTOMATIC1111

Generate transparent PNG images natively using Latent Transparency - ported from Forge to AUTOMATIC1111.

![Example transparent image generation](https://github.com/layerdiffusion/sd-forge-layerdiffusion/assets/161511761/36598904-ae5f-4578-87d3-4b496e11dcc5)

## Credits & Attribution

This extension is a port of [sd-forge-layerdiffuse](https://github.com/lllyasviel/sd-forge-layerdiffuse) by [lllyasviel](https://github.com/lllyasviel), originally designed for [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge).

All credit for the LayerDiffuse technology, models, and original implementation goes to the original authors. This port simply adapts the extension to work with the standard AUTOMATIC1111 WebUI.

For the academic paper and full technical details, see the original repository.

---

## ⚠️ Testing Status

**This port has received limited testing.** It has been verified to work with:
- ✅ SDXL models (Attention Injection method)
- ✅ SDXL models (Conv Injection method)  
- ✅ SD 1.5 models (Attention Injection method)
- ✅ Hires Fix with various upscalers
- ✅ API access via `alwayson_scripts`

**Not tested / may not work:**
- ❓ Layer composition modes (fg→bg, bg→fg, etc.)
- ❓ Batch generation with multiple images
- ❓ img2img workflows

Please report issues you encounter!

---

## Installation

1. Open AUTOMATIC1111 WebUI
2. Go to **Extensions** → **Install from URL**
3. Paste this repository URL
4. Click **Install**
5. Restart the WebUI

Models are downloaded automatically from HuggingFace on first use (~1GB total).

---

## Usage

### Basic Usage (UI)

1. Enable **LayerDiffuse** in the accordion panel
2. Select a method:
   - `(SDXL) Only Generate Transparent Image (Attention Injection)` - Best quality for SDXL
   - `(SDXL) Only Generate Transparent Image (Conv Injection)` - Sharper edges, may affect style
   - `(SD1.5) Only Generate Transparent Image (Attention Injection)` - For SD 1.5 models
3. Enable **Auto-add transparency prompts** (recommended)
4. Generate!

The output PNG will have a proper alpha channel. The preview may show black where transparency exists - this is normal.

---

## 🎨 Optimal Background Color for Best Results

The transparency decoder works best when the diffusion model generates a **neutral gray background** in areas that should be transparent.

### Optimal: Mid-Gray (RGB 128, 128, 128)

The decoder was trained on images with gray backgrounds representing "empty" areas. When enabled, the **Auto-add transparency prompts** option adds:

**Positive prompt prefix:**
```
transparent_background, simple_background, gray_background
```

**Negative prompt prefix:**
```
gradient_background, complex_background, white background, black background
```

### Why Gray Works Best

| Background Color | Transparency Quality |
|-----------------|---------------------|
| **Mid-gray (#808080)** | ✅ Best - ~98%+ clean alpha |
| Light gray | ✅ Good |
| White | ⚠️ May leave residual alpha |
| Black | ⚠️ May leave residual alpha |
| Colorful/gradient | ❌ Poor - splotchy transparency |

### Manual Prompt Tips

If auto-prompts aren't enough, try adding weighted keywords:
```
(gray_background:1.3), (solid_color:1.2), (uniform_background:1.2)
```

Or in negative:
```
(colorful_background:1.3), (detailed_background:1.2)
```

---

## API Usage

The extension is accessible via the standard A1111 API using `alwayson_scripts`:

```json
{
  "prompt": "a cat, high quality",
  "negative_prompt": "bad, ugly",
  "width": 1024,
  "height": 1024,
  "steps": 20,
  "cfg_scale": 7,
  "sampler_name": "DPM++ 2M SDE Karras",
  
  "alwayson_scripts": {
    "LayerDiffuse": {
      "args": [
        true,
        "(SDXL) Only Generate Transparent Image (Attention Injection)",
        1.0,
        1.0,
        true,
        "Crop and Resize",
        false
      ]
    }
  }
}
```

### Parameter Reference

| Index | Name | Type | Default | Description |
|-------|------|------|---------|-------------|
| 0 | `enabled` | bool | `false` | Enable LayerDiffuse |
| 1 | `method` | string | see below | Injection method |
| 2 | `weight` | float | `1.0` | LoRA weight (0.0-2.0) |
| 3 | `ending_step` | float | `1.0` | Stop at this % of steps |
| 4 | `auto_prompt` | bool | `true` | Auto-add transparency prompts |
| 5 | `resize_mode` | string | `"Crop and Resize"` | Resize mode |
| 6 | `output_origin` | bool | `false` | Output original (unused) |

### Method Values

```
"(SDXL) Only Generate Transparent Image (Attention Injection)"
"(SDXL) Only Generate Transparent Image (Conv Injection)"
"(SD1.5) Only Generate Transparent Image (Attention Injection)"
```

### Python Example

```python
import requests
import base64
from PIL import Image
from io import BytesIO

response = requests.post("http://localhost:7860/sdapi/v1/txt2img", json={
    "prompt": "a red apple, high quality",
    "width": 1024,
    "height": 1024,
    "steps": 20,
    "alwayson_scripts": {
        "LayerDiffuse": {
            "args": [True, "(SDXL) Only Generate Transparent Image (Attention Injection)", 
                     1.0, 1.0, True, "Crop and Resize", False]
        }
    }
})

# Decode and save transparent PNG
img_data = base64.b64decode(response.json()['images'][0])
img = Image.open(BytesIO(img_data))
img.save("transparent_output.png")
print(f"Saved with mode: {img.mode}")  # Should be RGBA
```

---

## Troubleshooting

### Black preview but file is transparent
This is normal. The WebUI preview doesn't display alpha. Save/download the PNG and open in an image editor to see transparency.

### Splotchy/incomplete transparency
- Enable **Auto-add transparency prompts**
- Add `gray_background` to your positive prompt
- Add `white background, black background, colorful background` to negative
- Try the Conv injection method for sharper edges

### Error about tensor size mismatch
Fixed in latest version. The decoder now pads images to compatible dimensions automatically.

### Transparency doesn't work with my LoRA
Some style LoRAs override background colors. Try:
- Increasing weight of background keywords: `(gray_background:1.4)`
- Lowering LoRA weight
- Using a different LoRA

---

## License

This port follows the same license as the original [sd-forge-layerdiffuse](https://github.com/lllyasviel/sd-forge-layerdiffuse).
