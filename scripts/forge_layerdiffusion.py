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

from modules import scripts, shared, devices, errors
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingImg2Img, process_images
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

# Neutral grey #808080 — only applied when preparing refine-pass img2img inputs
NEUTRAL_GREY_RGBA = (128, 128, 128, 255)


def composite_rgba_on_neutral_grey(img: Image.Image) -> Image.Image:
    """Flatten RGBA onto opaque #808080; used only as refine img2img init (not on final gallery output)."""
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    bg = Image.new('RGBA', img.size, NEUTRAL_GREY_RGBA)
    return Image.alpha_composite(bg, img)

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


def _apply_transparency_prompts_batch(p: StableDiffusionProcessing, prompts: list, batch_number: int):
    """Prepend transparency tags to this batch's prompts and sync back to all_prompts / all_negative_prompts."""
    positive_addition = "transparent_background, simple_background, grey_background"
    negative_addition = "gradient_background, complex_background, white background, black background"
    bs = p.batch_size
    start = batch_number * bs
    for i in range(len(prompts)):
        prompts[i] = f"{positive_addition}, {prompts[i]}"
        if start + i < len(p.all_prompts):
            p.all_prompts[start + i] = prompts[i]
    if hasattr(p, 'negative_prompts') and p.negative_prompts:
        for i in range(len(p.negative_prompts)):
            p.negative_prompts[i] = f"{negative_addition}, {p.negative_prompts[i]}"
            if start + i < len(p.all_negative_prompts):
                p.all_negative_prompts[start + i] = p.negative_prompts[i]
    if batch_number == 0:
        if p.all_prompts:
            p.main_prompt = p.all_prompts[0]
        if p.all_negative_prompts:
            p.main_negative_prompt = p.all_negative_prompts[0]
        if hasattr(p, 'prompt') and isinstance(p.prompt, str):
            p.prompt = p.main_prompt
        if hasattr(p, 'negative_prompt') and isinstance(p.negative_prompt, str):
            p.negative_prompt = p.main_negative_prompt
    print("[LayerDiffuse] Added transparency prompts to positive and negative (batch)")


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
        return "Alpha Injector - LayerDiffuse"

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

            auto_prompt = gr.Checkbox(label='Auto-add transparency prompts', value=False)
            gr.HTML('<p style="color: #666; font-size: 0.8em; margin-top: -8px;">Adds "transparent_background, simple_background, grey_background" to positive and "gradient_background, complex_background, white background, black background" to negative.</p>')

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

            refine_second_pass = gr.Checkbox(
                label='Refine pass (img2img after first pass)',
                value=True,
                elem_id='layerdiffuse_refine_second_pass',
            )
            refine_denoise = gr.Slider(
                label='Refine denoise strength',
                value=0.1,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                elem_id='layerdiffuse_refine_denoise',
            )

        return enabled, method, weight, ending_step, auto_prompt, resize_mode, output_origin, refine_second_pass, refine_denoise

    def process(self, p: StableDiffusionProcessing, enabled, method, weight, ending_step, auto_prompt, resize_mode, output_origin, refine_second_pass, refine_denoise):
        """Called before processing starts - inject LoRA weights."""
        global _latent_storage
        
        if not enabled:
            return

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
            'layerdiffusion_refine_second_pass': refine_second_pass,
            'layerdiffusion_refine_denoise': refine_denoise,
        })

    def before_process_batch(
        self,
        p: StableDiffusionProcessing,
        enabled,
        method,
        weight,
        ending_step,
        auto_prompt,
        resize_mode,
        output_origin,
        refine_second_pass,
        refine_denoise,
        **kwargs,
    ):
        """Runs after each batch's prompts are sliced; must run before extra networks / conditioning."""
        if not enabled or not auto_prompt:
            return
        prompts = kwargs.get('prompts')
        if not prompts:
            return
        batch_number = kwargs.get('batch_number', 0)
        _apply_transparency_prompts_batch(p, prompts, batch_number)

    def post_sample(self, p, ps, enabled, method, weight, ending_step, auto_prompt, resize_mode, output_origin, refine_second_pass, refine_denoise):
        """Called after sampling - latents are now captured via decode wrapper instead."""
        # Latents are captured by the wrapped decode_first_stage function
        pass

    def postprocess_image(self, p, pp, enabled, method, weight, ending_step, auto_prompt, resize_mode, output_origin, refine_second_pass, refine_denoise):
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
            
            # The UNet1024 decoder has 7 down/up blocks, so image must be divisible by 128
            # Pad image and latent to compatible sizes, then crop result after
            orig_h, orig_w = image.height, image.width
            pad_h = (128 - (orig_h % 128)) % 128
            pad_w = (128 - (orig_w % 128)) % 128
            
            if pad_h > 0 or pad_w > 0:
                # Pad image (PIL uses left, top, right, bottom for expand)
                padded_w = orig_w + pad_w
                padded_h = orig_h + pad_h
                padded_image = Image.new(image.mode, (padded_w, padded_h), (128, 128, 128))  # Gray padding
                padded_image.paste(image, (0, 0))
                
                # Resize latent to match padded image
                latent_h, latent_w = padded_h // 8, padded_w // 8
                single_latent = torch.nn.functional.interpolate(
                    single_latent.unsqueeze(0),
                    size=(latent_h, latent_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                
                print(f'[LayerDiffuse] Padded image from {orig_w}x{orig_h} to {padded_w}x{padded_h}')
                image = padded_image
            else:
                # Still need to ensure latent matches image
                lC, lH, lW = single_latent.shape
                expected_h, expected_w = image.height // 8, image.width // 8
                
                if abs(lH - expected_h) > 2 or abs(lW - expected_w) > 2:
                    single_latent = torch.nn.functional.interpolate(
                        single_latent.unsqueeze(0), 
                        size=(expected_h, expected_w), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
            
            png, vis = vae_transparent_decoder.decode(single_latent, image)
            
            # Crop back to original size if we padded
            if pad_h > 0 or pad_w > 0:
                png = png.crop((0, 0, orig_w, orig_h))
                vis = vis.crop((0, 0, orig_w, orig_h))
            
            pp.image = png if png.mode == 'RGBA' else png.convert('RGBA')
            print(f'[LayerDiffuse] Decoded transparent image')
            
            if hasattr(p, 'extra_result_images'):
                p.extra_result_images.append(vis)
                    
        except Exception as e:
            print(f"[LayerDiffuse] Error: {e}")
            import traceback
            traceback.print_exc()

    def _run_refine_pass(self, p, processed, refine_denoise):
        """Run img2img on first-pass outputs; composite onto #808080 only for the refine init images."""
        if not processed.images:
            return

        scripts_img2img = scripts.scripts_img2img
        self_script = scripts_img2img.script(self.title().lower())
        if self_script is None:
            print('[LayerDiffuse] Refine pass skipped: script not found in img2img runner')
            return

        if getattr(p, 'script_args', None) is None:
            print('[LayerDiffuse] Refine pass skipped: script_args is None')
            return

        outer_args = list(p.script_args[self.args_from:self.args_to])
        inner_args = [getattr(control, 'value', None) if control is not None else None for control in scripts_img2img.inputs]

        if len(outer_args) != (self.args_to - self.args_from):
            print('[LayerDiffuse] Refine pass skipped: could not read outer script args')
            return

        if (self_script.args_to - self_script.args_from) != len(outer_args):
            print(
                '[LayerDiffuse] Refine pass skipped: LayerDiffuse arg slice mismatch '
                f'(outer={len(outer_args)}, inner={(self_script.args_to - self_script.args_from)})'
            )
            return

        inner_args[self_script.args_from:self_script.args_to] = outer_args

        # Last two controls in this script: refine_second_pass, refine_denoise
        inner_args[self_script.args_to - 2] = False

        init_images = []
        source_images = processed.images[processed.index_of_first_image:]
        if not source_images:
            print('[LayerDiffuse] Refine pass skipped: no source images after grid prefix')
            return

        for im in source_images:
            if not isinstance(im, Image.Image):
                print('[LayerDiffuse] Refine pass skipped: non-PIL image in result')
                return
            # Neutral grey behind transparent/semi-transparent pixels for img2img input only
            init_images.append(composite_rgba_on_neutral_grey(im).convert('RGB'))

        first = init_images[0]
        w, h = first.size

        # Refine pass uses img2img; copied txt2img+hires params include callable
        # "Hires prompt" / "Hires negative prompt" that require all_hr_* on txt2img only.
        refine_extra = dict(p.extra_generation_params)
        refine_extra.pop("Hires prompt", None)
        refine_extra.pop("Hires negative prompt", None)

        p2 = StableDiffusionProcessingImg2Img(
            sd_model=shared.sd_model,
            outpath_samples=p.outpath_samples,
            outpath_grids=p.outpath_grids,
            prompt=p.prompt,
            negative_prompt=p.negative_prompt,
            styles=p.styles,
            batch_size=len(init_images),
            n_iter=1,
            cfg_scale=p.cfg_scale,
            width=w,
            height=h,
            init_images=init_images,
            resize_mode=0,
            denoising_strength=refine_denoise,
            steps=p.steps,
            sampler_name=p.sampler_name,
            scheduler=p.scheduler,
            seed=p.seed,
            subseed=p.subseed,
            subseed_strength=p.subseed_strength,
            seed_resize_from_h=p.seed_resize_from_h,
            seed_resize_from_w=p.seed_resize_from_w,
            restore_faces=p.restore_faces,
            tiling=p.tiling,
            do_not_save_samples=p.do_not_save_samples,
            do_not_save_grid=p.do_not_save_grid,
            extra_generation_params=refine_extra,
            override_settings=dict(p.override_settings) if p.override_settings else None,
            eta=p.eta,
            ddim_discretize=p.ddim_discretize,
            s_churn=p.s_churn,
            s_tmin=p.s_tmin,
            s_tmax=p.s_tmax,
            s_noise=p.s_noise,
            s_min_uncond=p.s_min_uncond,
        )

        if hasattr(p, 'tag_enable'):
            p2.tag_enable = getattr(p, 'tag_enable', None)
            p2.tag_eta = getattr(p, 'tag_eta', None)
            p2.tag_enable_ctag = getattr(p, 'tag_enable_ctag', None)

        if getattr(p, 'all_prompts', None):
            p2.all_prompts = p.all_prompts
        if getattr(p, 'all_negative_prompts', None):
            p2.all_negative_prompts = p.all_negative_prompts

        p2.scripts = scripts_img2img
        p2.script_args = inner_args
        p2._layerdiffuse_refine_pass = True

        print(f'[LayerDiffuse] Running refine img2img pass on {len(init_images)} image(s) (denoise={refine_denoise})')
        result = process_images(p2)
        processed.images = list(result.images)
        processed.index_of_first_image = result.index_of_first_image
        if getattr(result, 'infotexts', None):
            processed.infotexts = result.infotexts
        if processed.info is not None and getattr(result, 'info', None):
            processed.info = result.info

    def postprocess(self, p, processed, enabled, method, weight, ending_step, auto_prompt, resize_mode, output_origin, refine_second_pass, refine_denoise):
        """Restore UNet / decode hook and latent storage. Refine img2img runs later via patched ScriptRunner (after all other postprocess hooks)."""
        global _latent_storage

        if getattr(p, '_layerdiffuse_refine_pass', False):
            restore_original_weights()
            if hasattr(shared.sd_model, '_layerdiffuse_original_decode'):
                shared.sd_model.decode_first_stage = shared.sd_model._layerdiffuse_original_decode
                delattr(shared.sd_model, '_layerdiffuse_original_decode')
            job_id = getattr(p, 'layerdiffuse_job_id', id(p))
            if job_id in _latent_storage:
                del _latent_storage[job_id]
            return

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

    def run_layerdiffuse_refine_after_postprocess_chain(
        self,
        p,
        processed,
        enabled,
        method,
        weight,
        ending_step,
        auto_prompt,
        resize_mode,
        output_origin,
        refine_second_pass,
        refine_denoise,
    ):
        """Called once after every script's postprocess() so Hires Fix and other refiners run first."""
        if getattr(p, '_layerdiffuse_refine_pass', False):
            return
        if not enabled or not refine_second_pass:
            return
        self._run_refine_pass(p, processed, refine_denoise)


def _patch_script_runner_for_layerdiffuse_refine_last():
    """Run LayerDiffuse refine after the full postprocess chain (other extensions, etc.)."""
    if getattr(scripts.ScriptRunner, '_layerdiffuse_refine_last_patched', False):
        return

    _orig_postprocess = scripts.ScriptRunner.postprocess

    def postprocess(self, p, processed):
        _orig_postprocess(self, p, processed)
        for script in self.alwayson_scripts:
            if script.__class__.__name__ != 'LayerDiffusionForA1111':
                continue
            try:
                if getattr(p, 'script_args', None) is None:
                    continue
                script_args = p.script_args[script.args_from:script.args_to]
                script.run_layerdiffuse_refine_after_postprocess_chain(p, processed, *script_args)
            except Exception:
                errors.report('Error in LayerDiffuse refine pass (after postprocess chain)', exc_info=True)

    scripts.ScriptRunner.postprocess = postprocess
    scripts.ScriptRunner._layerdiffuse_refine_last_patched = True


_patch_script_runner_for_layerdiffuse_refine_last()
