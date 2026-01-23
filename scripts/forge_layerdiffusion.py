"""
LayerDiffuse for Automatic1111 WebUI
Adapted from sd-forge-layerdiffuse for compatibility with A1111.

Generates transparent PNG images by injecting LoRA weights into the UNet.

KNOWN LIMITATION: Hires Fix is NOT supported - disable it to use LayerDiffuse.
"""

import gradio as gr
import os
import sys
import functools
import torch
import numpy as np

# Add extension directory to path for local imports
ext_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ext_dir not in sys.path:
    sys.path.insert(0, ext_dir)

from modules import scripts, shared, devices
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingImg2Img
from modules.paths import models_path
from modules.modelloader import load_file_from_url
from modules import images
from PIL import Image, ImageOps
from enum import Enum

from lib_layerdiffusion.enums import ResizeMode
from lib_layerdiffusion.utils import rgba2rgbfp32, to255unit8, crop_and_resize_image
from lib_layerdiffusion.models import TransparentVAEDecoder, TransparentVAEEncoder
from backend import utils as backend_utils


layer_model_root = os.path.join(models_path, 'layer_model')
os.makedirs(layer_model_root, exist_ok=True)

vae_transparent_encoder = None
vae_transparent_decoder = None

# Global storage for latent samples
_latent_storage = {}

# Store original weights for restoration
_original_weights = {}


class LayerMethod(Enum):
    FG_ONLY_ATTN_SD15 = "(SD1.5) Only Generate Transparent Image (Attention Injection)"
    FG_ONLY_ATTN = "(SDXL) Only Generate Transparent Image (Attention Injection)"
    FG_ONLY_CONV = "(SDXL) Only Generate Transparent Image (Conv Injection)"


@functools.lru_cache(maxsize=2)
def load_layer_model_state_dict(filename):
    return backend_utils.load_torch_file(filename, safe_load=True)


def get_unet_module(model, key_path):
    """Get a module from the UNet by key path."""
    try:
        parts = key_path.split('.')
        module = model.model.diffusion_model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    except (AttributeError, IndexError, KeyError):
        return None


def apply_layer_lora_weights(model, lora_state_dict, weight=1.0):
    """Apply LayerDiffuse LoRA weights to the model."""
    global _original_weights
    
    applied_count = 0
    _original_weights.clear()
    
    lora_pairs = {}
    
    for key in lora_state_dict.keys():
        if '::lora::' in key:
            base_key = key.split('::lora::')[0]
            lora_idx = int(key.split('::lora::')[1])
            
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            lora_pairs[base_key][lora_idx] = lora_state_dict[key]
    
    for base_key, lora_data in lora_pairs.items():
        if 0 not in lora_data or 1 not in lora_data:
            continue
        
        if not base_key.startswith('diffusion_model.'):
            continue
        
        key_without_prefix = base_key[len('diffusion_model.'):]
        
        if key_without_prefix.endswith('.weight'):
            module_path = key_without_prefix[:-len('.weight')]
        elif key_without_prefix.endswith('.bias'):
            continue
        else:
            continue
        
        module = get_unet_module(model, module_path)
        if module is None or not hasattr(module, 'weight'):
            continue
        
        lora_down = lora_data[0].to(device=module.weight.device, dtype=module.weight.dtype)
        lora_up = lora_data[1].to(device=module.weight.device, dtype=module.weight.dtype)
        
        weight_key = f"{module_path}.weight"
        if weight_key not in _original_weights:
            _original_weights[weight_key] = (module, module.weight.data.clone())
        
        try:
            delta = torch.mm(lora_down, lora_up) * weight
            if delta.shape == module.weight.shape:
                module.weight.data += delta
                applied_count += 1
            elif delta.T.shape == module.weight.shape:
                module.weight.data += delta.T
                applied_count += 1
        except:
            try:
                delta = torch.mm(lora_up, lora_down) * weight
                if delta.shape == module.weight.shape:
                    module.weight.data += delta
                    applied_count += 1
            except:
                pass
    
    print(f'[LayerDiffuse] Applied LoRA weights to {applied_count} layers')
    return applied_count > 0


def restore_original_weights():
    """Restore original UNet weights after generation."""
    global _original_weights
    
    for key, (module, original_weight) in _original_weights.items():
        module.weight.data.copy_(original_weight)
    
    restored_count = len(_original_weights)
    _original_weights.clear()
    
    if restored_count > 0:
        print(f'[LayerDiffuse] Restored {restored_count} original weights')


class LayerDiffusionForA1111(scripts.Script):
    def title(self):
        return "LayerDiffuse"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(label='Enabled', value=False)
            
            method = gr.Dropdown(
                choices=[e.value for e in LayerMethod], 
                value=LayerMethod.FG_ONLY_ATTN.value, 
                label="Method", 
                type='value'
            )
            
            gr.HTML('<p style="color: #888; font-size: 0.85em;">Generates transparent PNGs. Black areas in preview = transparent when saved.</p>')

            auto_prompt = gr.Checkbox(label='Auto-add transparency prompts', value=True)
            gr.HTML('<p style="color: #666; font-size: 0.8em; margin-top: -8px;">Adds "transparent_background, simple_background" to positive and "gradient_background, complex_background" to negative.</p>')

            with gr.Row():
                weight = gr.Slider(label="Weight", value=1.0, minimum=0.0, maximum=2.0, step=0.001)
                ending_step = gr.Slider(label="Stop At", value=1.0, minimum=0.0, maximum=1.0)

            resize_mode = gr.Radio(
                choices=[e.value for e in ResizeMode], 
                value=ResizeMode.CROP_AND_RESIZE.value, 
                label="Resize Mode", 
                type='value',
                visible=False
            )
            
            output_origin = gr.Checkbox(label='Output original mat for img2img', value=False, visible=False)

        return enabled, method, weight, ending_step, auto_prompt, resize_mode, output_origin

    def process(self, p: StableDiffusionProcessing, enabled, method, weight, ending_step, auto_prompt, resize_mode, output_origin):
        """Called before processing starts - inject LoRA weights."""
        global _latent_storage
        
        if not enabled:
            return

        # Auto-add transparency prompts if enabled
        if auto_prompt:
            positive_addition = "transparent_background, simple_background"
            negative_addition = "gradient_background, complex_background"
            
            # Modify all prompts in the batch
            if hasattr(p, 'all_prompts') and p.all_prompts:
                p.all_prompts = [f"{positive_addition}, {prompt}" for prompt in p.all_prompts]
            if hasattr(p, 'all_negative_prompts') and p.all_negative_prompts:
                p.all_negative_prompts = [f"{negative_addition}, {neg}" for neg in p.all_negative_prompts]
            
            # Also modify the main prompt for display
            if hasattr(p, 'prompt'):
                p.prompt = f"{positive_addition}, {p.prompt}"
            if hasattr(p, 'negative_prompt'):
                p.negative_prompt = f"{negative_addition}, {p.negative_prompt}"
            
            print(f'[LayerDiffuse] Added transparency prompts to positive and negative')

        job_id = id(p)
        _latent_storage[job_id] = {'latents': None}
        
        # Wrap decode_first_stage to capture latents before decoding
        if not hasattr(shared.sd_model, '_layerdiffuse_original_decode'):
            original_decode = shared.sd_model.decode_first_stage
            
            def wrapped_decode(z, *args, **kwargs):
                # Capture the latent before decoding (always keep the latest)
                if isinstance(z, torch.Tensor) and z.shape[1] == 4:  # Latent has 4 channels
                    _latent_storage[job_id]['latents'] = z.clone().detach().cpu()
                    print(f'[LayerDiffuse] Intercepted latents at decode: shape={z.shape}')
                return original_decode(z, *args, **kwargs)
            
            shared.sd_model._layerdiffuse_original_decode = original_decode
            shared.sd_model.decode_first_stage = wrapped_decode

        method_enum = LayerMethod(method)
        print(f'[LayerDiffuse] Enabled with method: {method_enum}')

        job_id = id(p)
        _latent_storage[job_id] = {'latents': None, 'hr_latents': None}

        p.layerdiffuse_enabled = True
        p.layerdiffuse_method = method_enum
        p.layerdiffuse_weight = weight
        p.layerdiffuse_ending_step = ending_step
        p.layerdiffuse_job_id = job_id

        # Load and apply LoRA weights based on method
        lora_applied = False
        
        if method_enum == LayerMethod.FG_ONLY_ATTN:
            model_path = load_file_from_url(
                url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_attn.safetensors',
                model_dir=layer_model_root,
                file_name='layer_xl_transparent_attn.safetensors'
            )
            lora_state_dict = load_layer_model_state_dict(model_path)
            lora_applied = apply_layer_lora_weights(shared.sd_model, lora_state_dict, weight)
            
        elif method_enum == LayerMethod.FG_ONLY_CONV:
            model_path = load_file_from_url(
                url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_conv.safetensors',
                model_dir=layer_model_root,
                file_name='layer_xl_transparent_conv.safetensors'
            )
            lora_state_dict = load_layer_model_state_dict(model_path)
            lora_applied = apply_layer_lora_weights(shared.sd_model, lora_state_dict, weight)
            
        elif method_enum == LayerMethod.FG_ONLY_ATTN_SD15:
            model_path = load_file_from_url(
                url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_transparent_attn.safetensors',
                model_dir=layer_model_root,
                file_name='layer_sd15_transparent_attn.safetensors'
            )
            lora_state_dict = load_layer_model_state_dict(model_path)
            lora_applied = apply_layer_lora_weights(shared.sd_model, lora_state_dict, weight)

        p.layerdiffuse_lora_applied = lora_applied

        p.extra_generation_params.update({
            'layerdiffusion_enabled': enabled,
            'layerdiffusion_method': method,
            'layerdiffusion_weight': weight,
            'layerdiffusion_ending_step': ending_step,
        })

    def post_sample(self, p, ps, enabled, method, weight, ending_step, auto_prompt, resize_mode, output_origin):
        """Called after sampling - latents are now captured via decode wrapper instead."""
        # Latents are captured by the wrapped decode_first_stage function
        pass

    def postprocess_image(self, p, pp, enabled, method, weight, ending_step, auto_prompt, resize_mode, output_origin):
        """Called after each image - decode transparency."""
        global vae_transparent_decoder, _latent_storage

        if not enabled:
            return

        method_enum = LayerMethod(method)
        need_process = False

        if method_enum in [LayerMethod.FG_ONLY_ATTN, LayerMethod.FG_ONLY_CONV]:
            need_process = True
            if vae_transparent_decoder is None:
                model_path = load_file_from_url(
                    url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_decoder.safetensors',
                    model_dir=layer_model_root,
                    file_name='vae_transparent_decoder.safetensors'
                )
                vae_transparent_decoder = TransparentVAEDecoder(backend_utils.load_torch_file(model_path))

        if method_enum == LayerMethod.FG_ONLY_ATTN_SD15:
            need_process = True
            if vae_transparent_decoder is None:
                model_path = load_file_from_url(
                    url='https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_vae_transparent_decoder.safetensors',
                    model_dir=layer_model_root,
                    file_name='layer_sd15_vae_transparent_decoder.safetensors'
                )
                vae_transparent_decoder = TransparentVAEDecoder(backend_utils.load_torch_file(model_path))

        if not need_process:
            return

        try:
            image = pp.image
            job_id = getattr(p, 'layerdiffuse_job_id', id(p))
            
            storage = _latent_storage.get(job_id, {})
            latent = storage.get('latents')
            
            if latent is None:
                print("[LayerDiffuse] WARNING: No latent data captured.")
                return
            
            index = getattr(pp, 'index', 0)
            
            if len(latent.shape) == 4:
                single_latent = latent[index] if index < latent.shape[0] else latent[0]
            else:
                single_latent = latent
            
            single_latent = single_latent.to(devices.get_optimal_device())
            
            lC, lH, lW = single_latent.shape
            expected_h, expected_w = image.height // 8, image.width // 8
            
            # Allow some tolerance for rounding differences
            if abs(lH - expected_h) > 2 or abs(lW - expected_w) > 2:
                print(f'[LayerDiffuse] Latent size mismatch: latent={lH}x{lW}, expected={expected_h}x{expected_w}')
                # Resize latent to match if needed
                single_latent = torch.nn.functional.interpolate(
                    single_latent.unsqueeze(0), 
                    size=(expected_h, expected_w), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                print(f'[LayerDiffuse] Resized latent to {expected_h}x{expected_w}')
            
            png, vis = vae_transparent_decoder.decode(single_latent, image)
            
            pp.image = png if png.mode == 'RGBA' else png.convert('RGBA')
            print(f'[LayerDiffuse] Decoded transparent image')
            
            if hasattr(p, 'extra_result_images'):
                p.extra_result_images.append(vis)
                    
        except Exception as e:
            print(f"[LayerDiffuse] Error: {e}")
            import traceback
            traceback.print_exc()

    def postprocess(self, p, processed, enabled, method, weight, ending_step, auto_prompt, resize_mode, output_origin):
        """Called after all processing - cleanup."""
        global _latent_storage
        
        restore_original_weights()
        
        # Restore original decode function
        if hasattr(shared.sd_model, '_layerdiffuse_original_decode'):
            shared.sd_model.decode_first_stage = shared.sd_model._layerdiffuse_original_decode
            delattr(shared.sd_model, '_layerdiffuse_original_decode')
        
        if not enabled:
            return
        
        job_id = getattr(p, 'layerdiffuse_job_id', id(p))
        if job_id in _latent_storage:
            del _latent_storage[job_id]
