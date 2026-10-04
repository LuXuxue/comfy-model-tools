"""INT8 + ConvRot quantizer for comfy-kitchen — auto layer detection (no per-model recipe).

Quantizes the per-token block linears (attention + FFN), passes everything else through.
Recipe: fp32 upcast, block-Hadamard rotation at the per-layer best power-of-4 groupsize,
per-channel absmax scale, embedded `<layer>.comfy_quant` config.

    python quant_int8_auto.py SRC [DST.safetensors] [--dry-run] [options]

SRC: .safetensors (lazy) or torch pickle .pth/.pt/.ckpt (safe weights_only load, held in RAM).
DST optional: defaults to SRC with bf16/fp16/fp32 -> int8_convrot (or _int8_convrot appended).
Auto-detect can't see token count M (small-M/windowed/audio layers get quantized anyway — size
win, maybe not speed) or loader quirks (manual_cast/key-remap loaders -> loads but outputs garbage).
So run --dry-run on an unfamiliar arch and use --min-gemm / --exclude as needed.

Scale granularity is per-OUTPUT-CHANNEL everywhere (reduce over K -> [N,1]), for both int8 and
W4A8 -- note the `int8_tensorwise` string in the comfy_quant config is comfy-kitchen's layout
name, not "one scale for the tensor". A genuine single tensor-wide scale is ~8x worse (6.6% vs
0.83% relerr), so there is nothing to gain there. W4A8 is finer still: rowwise fp32 s_channel
times a per-group-of-16 fp8 s_rel.
"""
# ruff: noqa: T201  (print() is this CLI's output)
import argparse
import dataclasses
import json
import os
import re
import sys
import time
import collections
import torch
from safetensors import safe_open
from safetensors.torch import save_file
try:
    from comfy_kitchen.tensor.int8 import _build_hadamard, _rotate_weight
except ImportError:
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_weight

VALID_GS  = (256, 64, 16)                       # convrot Hadamard sizes; power-of-4, prefer largest
CLIP_GRID = torch.linspace(0.55, 1.0, 80)
FP8 = (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None))

def best_gs(k):
    return next((g for g in VALID_GS if k % g == 0), None)

# Token-embedding lookup tables -> per-row int8 (+rotation). Match real token tables
# (embed_tokens / embed_tokens_per_layer / wte / ...), not an `*.embedding_projection`.
EMBED_SEG = re.compile(r"^(embed_tokens(_per_layer)?|token_embedding|word_embeddings|tok_embeddings|wte|shared)$")

def is_token_embedding(key, shape):
    if len(shape) != 2:
        return False
    n, k = shape
    if n < 4096 or n <= k:            # vocab-sized table (rows >> cols)
        return False
    return any(EMBED_SEG.match(s) for s in key.split("."))

# safe_open-compatible reader for torch pickle checkpoints (weights_only=True -> safe load, no code
# execution; whole file into RAM since pickle has no lazy access).
_DTYPE_CODE = {torch.float16: "F16", torch.bfloat16: "BF16", torch.float32: "F32",
               torch.float64: "F64", torch.int8: "I8", torch.uint8: "U8",
               getattr(torch, "float8_e4m3fn", None): "F8_E4M3",
               getattr(torch, "float8_e5m2", None): "F8_E5M2"}

class _TorchSlice:                                  # mimics safetensors get_slice()
    def __init__(self, t): self._t = t
    def get_shape(self): return list(self._t.shape)
    def get_dtype(self): return _DTYPE_CODE.get(self._t.dtype, str(self._t.dtype))

class _TorchReader:
    def __init__(self, path):
        obj = torch.load(path, map_location="cpu", weights_only=True)
        sd = self._find_state_dict(obj)
        if sd is None:
            raise ValueError(f"no tensor state-dict found in {path}")
        self._sd = sd
    @staticmethod
    def _find_state_dict(obj):
        if isinstance(obj, dict):
            tensors = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
            if tensors:
                return tensors                       # this level holds the weights
            for key in ("state_dict", "model_state_dict", "model", "module", "net", "ema", "params"):
                if isinstance(obj.get(key), dict):
                    found = _TorchReader._find_state_dict(obj[key])
                    if found:
                        return found
        return None
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def metadata(self): return {}
    def keys(self): return list(self._sd.keys())
    def get_slice(self, k): return _TorchSlice(self._sd[k])
    def get_tensor(self, k): return self._sd[k]

def open_model(path):
    """safe_open for .safetensors; safe (weights_only) torch.load for .pth/.pt/.ckpt/.bin."""
    if path.lower().endswith(".safetensors"):
        return safe_open(path, framework="pt", device="cpu")
    return _TorchReader(path)

# ComfyUI infers the architecture from tensor SHAPES (comfy/model_detection.py). int8 keeps
# weight at [N, K] so shapes survive, but the PACKED formats store [N, K*bits/8], which
# shrinks K and makes detect_unet_config mismatch, so the file is rejected with "no match
# {...}" before a single layer loads. Two shapes matter on the UNet/DiT paths:
#   context_dim      <- <first transformer block>.attn2.to_k.weight          (line 37)
#   adm_in_channels  <- label_emb.0.0.weight / class_embedding.linear_1     (lines 1203, 1442)
# A half-size read gives 1024 instead of 2048, and SDXL then refuses to match.
#
# The list is explicit (as comfy/model_detection.py itself is) rather than inferred from
# repetition counts: `label_emb.0.0` and `label_emb.0.2` both normalize to `label_emb.N.N`,
# so "how often does this path pattern occur" cannot separate them. Extend when a new
# architecture lands.
DETECT_SENSITIVE = re.compile(
    r"attn2\.to_k$"                                     # context_dim
    r"|(?:^|[._])label_emb(?:\.|$)"                      # adm_in_channels, num_classes
    r"|class_embedding|add_embedding"                    # adm_in_channels
    r"|x_embedder|img_in|input_proj|conv_in|latent_in"   # in/out channels, patch size, depth
    r"|condition_proj|cond_in|cond_seq_linear"           # text/vec conditioning dims
    r"|to_global_embed|to_timestep_embed|time_embedder"  # timestep / global cond dims
    r"|input_embedder|vision_in")

# Detection = quantize every eligible 2-D block linear, minus a name denylist. No projection-name
# allowlist (fragile: every arch invents new names like to_qkv/add_q_proj/single_blocks.linear1).
# The rule of thumb is M, not size: a layer whose GEMM is per-token (M = sequence length or the
# spatial grid) averages its error down and quantizes cleanly; one driven by a per-sample
# conditioning vector (M = batch, 1-4 at inference) gets no averaging, so its error lands
# straight on the scale/shift or condition it feeds. Everything below is the M=batch kind:
# scale_shift buffers, rope/pos_embed, input embedders, gate/router logits, M=1 timestep MLPs,
# output head/final, adaLN MODULATION, and conditioning adapters. 1-D norms are dropped by
# not-2d already.
# Careful bits: `embedder` (not `embed`) keeps `*_embeddings_connector` in; bare `gate` stays
# (SwiGLU) — only gate_logits/router drop; `timestep`/`time` catch the M=1 embed but not the modulator.
# `adaln_modulation` (Anima) and `emb_layers` (SDXL) are the same adaLN path under two names;
# `label_emb` (SDXL) is the pooled-CLIP projection, also M=batch. Their weight MSE really is fine
# (~0.76-0.90%, measured) — excluding them is about the M=1 amplification, not the weight error.
# Cost: 8.5% of Anima's 2-D weights, 1.0% of SDXL's.
# Bare `modulation` (SD3/Flux/PixArt naming) is deliberately NOT listed: those archs are 18-33%
# modulation and that is a much bigger, untested change — pass `--exclude modulation` if wanted.
#
# `visual` = a VLM vision tower inside a text encoder (Qwen3-VL / Qwen3.5). Unlike every other
# entry this one is NOT about M: those GEMMs are per-patch and quantize cleanly by weight error
# (0.87% mean on qwen3-vl-4b), but the tower's OUTPUT amplifies that 5.5x. Measured on one real
# 768x768 image through ComfyUI's own preprocessing, fp16 source vs int8:
#     deepstack taps   1.21% -> 1.97% -> 2.48%   (grows with depth — a plain ViT stack has no
#                                                  per-block renormalisation to bound it, unlike
#                                                  a UNet whose adaLN rescales every block)
#     merged (LLM input)                  4.72%   cosine 0.99894
# Re-running with the tower left in bf16 gives exactly 0.00%, so it accounts for 100% of the
# vision-path error — which makes it worth the 200-280 MB that int8 would otherwise save.
# Same conclusion convert_to_quant's --qwen_vlm reaches via `visual.`.
# `mtp` = multi-token-prediction head, i.e. the lm_head pattern under another name; absent from
# every checkpoint measured here, added for symmetry with the `head` entry above.
# NB this list is matched per path SEGMENT (classify), unlike --exclude which matches the full
# key -- so a pattern spanning two segments (`.layers.0.`) cannot live here. EXCLUDE_KEY below
# is the full-key companion for exactly those.
EXCLUDE_SEG = re.compile(
    r"scale_shift|rope|rotary|rel_pos|pos_?embed|embedder|"
    r"gate_logits|router|routing|logit|temperature|"
    r"(?:^|_)time|temb|t_emb|guidance|register|refiner_blocks|adapter|"
    r"(?:^|_)(?:final|head|proj_out|out_layer)(?:_|$)|"
    r"adaln_modulation|emb_layers|label_emb|mtp|visual")
# Cross-segment exclusions, matched against the whole key. `(?:^|\.)` on the left and `(?:\.|$)`
# on the right mean `layers.0` cannot accidentally match `layers.10`/`layers.0x`, and no
# diffusion arch uses a segment literally named `layers`, so SDXL/Anima are untouched.
# The FIRST transformer/language layer: its input is the raw embedding output, before the
# stack has normalized anything, so its activation statistics are unlike every other layer and
# it is the one convert_to_quant's --qwen_vlm / --zimage filters also protect. 193-218M on the
# Qwen3 text encoders here, ~2.5% of the file. NOT the last layer: that one is already covered
# for text encoders by `lm_head` -> the `head` entry above.
EXCLUDE_KEY = re.compile(r"(?:^|\.)layers\.0(?:\.|$)")
# `refiner_blocks` = short-M text side-path (Krea txtfusion.refiner_blocks); main-stream refiners are
# `*_refiner` (Boogu/Z-Image), kept. `adapter` = conditioning injection modules (Wan-Animate
# face_adapter, Anima llm_adapter, ip/control adapters): tiny or M=short, identity/quality-critical,
# quantize worst -> leave bf16. `proj_out` is here for the output head sense (Flux); note SDXL's
# ResBlock `proj_in`/`proj_out` are ordinary per-token channel projections and this entry drops the
# latter but not the former — an asymmetry inherited from earlier revisions, kept so behaviour on
# already-converted models does not change.

def classify(key, shape):
    """Quantize every eligible 2-D block linear except the name denylist. Returns (bool, reason)."""
    if len(shape) != 2:
        return (False, "not-2d")
    n, k = shape
    if n < 8:
        return (False, "small-N")
    gs = best_gs(k)
    if gs is None:
        return (False, "ineligible-K")
    segs = key.split(".")
    # in a block = an integer segment with named structure after it (blocks.5.attn.q). A trailing
    # integer is a Sequential index on a top-level MLP (tmlp.0, img_emb.proj.1) -> not a block.
    if not any(segs[i].isdigit() for i in range(len(segs) - 1)):
        return (False, "not-in-indexed-block")
    if EXCLUDE_KEY.search(key):
        return (False, "denylist(first layer: layers.0)")
    if any(EXCLUDE_SEG.search(s) for s in segs):
        return (False, "denylist(scale_shift/embed/gate/time/head/adaln/adapter)")
    return (True, f"gs{gs}")

# ---------------------------------------------------------------------------
# Quantization (fp32 upcast + block-Hadamard rotation + MSE-optimal per-channel clip)
# ---------------------------------------------------------------------------
@torch.no_grad()
def quantize_convrot(w, gs, mseclip=True, device="cuda"):
    wf = w.to(device, torch.float32)
    h  = _build_hadamard(gs, device=wf.device, dtype=torch.float32)
    wr = _rotate_weight(wf, h, gs)
    absmax = wr.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
    if not mseclip:
        scale = (absmax / 127.0).clamp(min=1e-30)
        q = (wr / scale).round().clamp(-127, 127)
        return q.to(torch.int8), scale.to(torch.float32)
    best_mse = torch.full_like(absmax, float("inf"))
    best_scale = absmax / 127.0
    best_q = None
    for a in CLIP_GRID.tolist():
        scale = (absmax * a / 127.0).clamp(min=1e-30)
        q = (wr / scale).round().clamp(-127, 127)
        mse = ((q * scale - wr) ** 2).mean(dim=1, keepdim=True)
        better = mse < best_mse
        best_mse = torch.where(better, mse, best_mse)
        best_scale = torch.where(better, scale, best_scale)
        best_q = q.clone() if best_q is None else torch.where(better.expand_as(q), q, best_q)
    return best_q.to(torch.int8), best_scale.to(torch.float32)

@torch.no_grad()
def recon_metrics(qd, scale, w_ref, gs, device="cuda"):
    """Reconstruct (dequant + un-rotate) and return (cosine, relative_error_%)."""
    deq = qd.to(device).float() * scale.to(device)
    h = _build_hadamard(gs, device=device, dtype=torch.float32)
    deq = _rotate_weight(deq, h, gs)
    wf = w_ref.to(device).float()
    cos = torch.nn.functional.cosine_similarity(deq.flatten(), wf.flatten(), dim=0).item()
    relerr = ((deq - wf).norm() / wf.norm().clamp(min=1e-30)).item() * 100.0
    return cos, relerr

def cq_tensor(gs):
    cfg = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": gs}
    return torch.tensor(list(json.dumps(cfg).encode("utf-8")), dtype=torch.uint8)

# ---------------------------------------------------------------------------
# W4A8 fp8 group-scale refinement
#
# comfy-kitchen stores s_rel (the per-group relative scale) as fp8-e4m3: 3 mantissa bits,
# so each group's scale carries ~6% relative error that nothing compensates for. It already
# ships a fix -- pick, per group, the neighbouring fp8 value whose *decoded int8 levels* fit
# the group best (`_search_group_scale`) -- but gates it on 6-bit only, so --w4a8 never gets
# it. Re-enabling it at 4-bit cuts weight error ~1% at zero storage cost (the alternative,
# fp32 s_rel, buys ~4% for +0.19 B/weight = a 33% size increase, so it is not worth it).
#
# The math below mirrors comfy-kitchen's eager path exactly, reimplemented locally so it does
# not depend on private symbols: rotate in bf16 -> ALS codebook+group scale -> fp8 s_rel ->
# nearest decoded level -> pick the best of 3 fp8 neighbours -> repack. Kept numerically
# identical by construction (the candidate set is +-1 on the e4m3 bit pattern) and validated
# against the private `_search_group_scale` in tests.
# ---------------------------------------------------------------------------
W4A8_E4M3_MAX = 0x7E          # 0x7F is NaN in e4m3fn
W4A8_ROW_ELEMS = 1 << 22      # cap the fp32 working set per chunk (comfy-kitchen uses the same)

def _w4a8_grid_levels(codebook, s_rel):
    """int8 value each codebook level decodes to, per group: round(clamp(level * s_rel))."""
    return (codebook.view(1, 1, -1) * s_rel.float().unsqueeze(-1)).round_().clamp_(-127, 127)

def _w4a8_nearest(grouped, levels, target):
    """Index of the nearest decoded level per element. `target` = grouped / s_channel."""
    n, groups, gsize = grouped.shape
    last = levels.shape[-1] - 1
    lv = levels.reshape(n * groups, last + 1).contiguous()      # per-group sorted levels
    tg = target.reshape(n * groups, gsize).contiguous()
    pos = torch.searchsorted(lv, tg)
    lo = (pos - 1).clamp(0, last)
    hi = pos.clamp(0, last)
    dlo = tg.sub(torch.gather(lv, 1, lo)).abs_()
    dhi = tg.sub(torch.gather(lv, 1, hi)).abs_()
    return torch.where(dhi < dlo, hi, lo).to(torch.int32).reshape(n, groups, gsize)

@torch.no_grad()
def _w4a8_search_scales(grouped, s_rel, s_channel, codebook):
    """Per group, the fp8 neighbour of s_rel whose decoded levels minimize squared error."""
    target = grouped / s_channel.view(-1, 1, 1)
    raw = s_rel.view(torch.uint8)
    cands = [s_rel.float(),
             (raw - 1).clamp_(min=1).view(torch.float8_e4m3fn).float(),
             (raw + 1).clamp_(max=W4A8_E4M3_MAX).view(torch.float8_e4m3fn).float()]
    def score(c):
        levels = _w4a8_grid_levels(codebook, c)
        idx = _w4a8_nearest(grouped, levels, target)
        err = torch.gather(levels, 2, idx.long()).sub_(target).pow_(2).sum(-1)
        return c, err, idx
    best_scale, best_err, best_idx = score(cands[0])
    for c in cands[1:]:
        c, err, idx = score(c)
        better = err < best_err
        torch.where(better, err, best_err, out=best_err)
        torch.where(better, c, best_scale, out=best_scale)
        torch.where(better.unsqueeze(-1), idx, best_idx, out=best_idx)
    return best_scale.to(s_rel.dtype).contiguous(), best_idx

def _w4a8_pack4(codes):
    """int32 codes [N, K] -> int8 storage: two 4-bit codes per byte, even col in the low nibble."""
    return ((codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4)).to(torch.int8).contiguous()

@torch.no_grad()
def refine_w4a8_scales(p, wf, group_size, bits, convrot_groupsize=256):
    """Re-pick the fp8 group scales of an existing W4A8 quantization by grid-aware search.
    Returns (packed-weight, new-params), or None if the layout is not a searchable W4A8.

    Chunked over rows so a large layer never materializes a full extra fp32 copy. Bits==6 is
    already searched by comfy-kitchen, so it is left alone.
    """
    if bits != 4 or p.codebook is None or p.correction is not None:
        return None
    n, k = wf.shape
    h = _build_hadamard(convrot_groupsize, device=wf.device, dtype=torch.bfloat16)
    cb = p.codebook.float()
    s_rel, s_channel = p.scale, p.s_channel
    row = max(1, W4A8_ROW_ELEMS // max(k, 1))
    packed, srels = [], []
    for r0 in range(0, n, row):
        # bf16 rotation + .float() matches rotate_int8_convrot_weight -> grouped_weight exactly
        rot = _rotate_weight(wf[r0:r0 + row].contiguous(), h, convrot_groupsize)
        grouped = rot.float().view(rot.shape[0], k // group_size, group_size)
        sr, codes = _w4a8_search_scales(grouped, s_rel[r0:r0 + row], s_channel[r0:r0 + row], cb)
        packed.append(_w4a8_pack4(codes.view(-1, k).to(torch.int32)))
        srels.append(sr)
        del rot, grouped
    return torch.cat(packed), dataclasses.replace(p, scale=torch.cat(srels))

W4A4_GROUP = 64        # int4 MMA kernel requires exactly this; not configurable

@torch.no_grad()
def quantize_w4a4(w, gs=None, device="cuda"):
    """ConvRot W4A4: rotated int4 weight, one fp32 scale per row. 0.50 bytes/weight.

    EXPERIMENTAL -- the output is visibly degraded. Kept for measuring the format, not for use.

    Same 4-bit family as `--w4a8` but a different kernel (`convrot_w4a4`): uniform int4 with a
    per-row scale instead of a Lloyd-Max codebook plus fp8 per-group-of-16 scales. The coarser
    scale granularity is the problem, and `quant_group_size` cannot be tightened here because the
    int4 MMA kernel requires exactly 64. Measured on SDXL (full model, 711 layers):

        int8 convrot + absmax   0.78% weight error   2.957 GB   ~20 dB end-to-end
        w4a8 group16            ~5.8%                2.027 GB   15.3 dB
        w4a4 group64           13.14%                1.933 GB   11.9 dB

    The 11.9 dB sample had a heavy magenta cast and blocky coloured noise. Note that ComfyUI's
    `convrot_w4a4` layout carries no low-rank branch, so there is nothing to recover the ~13%
    with -- that compensation is what `svdquant_w4a4` adds in comfy-kitchen, at the cost of
    requiring offline calibration. The honest summary is that this format saves only ~4.6% over
    `--w4a8` while roughly doubling the error, so the size win does not pay for the quality loss.

    Unlike w4a8 this path keeps `gs` per layer, so layers whose K is not a multiple of the
    convrot groupsize fall back to int8 rather than being forced to gs=256.

    Returns (out-tensor dict, comfy_quant cfg, relerr%, cosine)."""
    from comfy_kitchen.tensor.convrot_w4a4 import (
        dequantize_convrot_w4a4_weight, quantize_convrot_w4a4_weight)
    gs = gs or 256
    wf = w.to(device, torch.float32)
    q, scale = quantize_convrot_w4a4_weight(
        wf, convrot_groupsize=gs, quant_group_size=W4A4_GROUP)
    deq = dequantize_convrot_w4a4_weight(
        q, scale, convrot_groupsize=gs, quant_group_size=W4A4_GROUP)
    cos = torch.nn.functional.cosine_similarity(deq.flatten(), wf.flatten(), dim=0).item()
    relerr = ((deq - wf).norm() / wf.norm().clamp(min=1e-30)).item() * 100.0
    tensors = {"weight": q.cpu(), "weight_scale": scale.cpu()}
    cfg = {"format": "convrot_w4a4", "convrot_groupsize": gs}
    return tensors, cfg, relerr, cos

@torch.no_grad()
def quantize_w4a8(w, bits=4, group_size=None, device="cuda", scale_search=True):
    """W4A8: ConvRot-rotated int4 weight with a Lloyd-Max codebook + fp8 group
    scales, or with bits=6 W6A8: uniform int6, no codebook, ~3x lower weight error for
    1.5x the bytes. Activations quantize to int8 at runtime. Needs K divisible by 256 and N>=64.
    `scale_search` re-picks the fp8 group scales by grid-aware search (~1% lower weight error at
    4-bit, no size cost); it is already built into comfy-kitchen for 6-bit. Returns
    (out-tensor dict, comfy_quant cfg, relerr%, relerr_before_search%)."""
    from comfy_kitchen.tensor import AsymW4A8Int8Layout, QuantizedTensor
    group_size = group_size or (32 if bits == 6 else 16)
    wf = w.to(device, torch.bfloat16)
    q, p = AsymW4A8Int8Layout.quantize(
        wf, group_size=group_size, convrot_groupsize=256,
        scale_dtype=torch.float8_e4m3fn, codebook=True, stochastic_rounding=0, bits=bits)
    def relerr(qq, pp):
        deq = QuantizedTensor(qq, "AsymW4A8Int8Layout", pp).dequantize().float()
        return ((deq - wf.float()).norm() / wf.float().norm().clamp(min=1e-30)).item() * 100.0
    err_base = relerr(q, p)
    if scale_search:
        # self-validating: keep the searched scales only if they actually dequantize better
        r = refine_w4a8_scales(p, wf, group_size, bits)
        if r is not None:
            q2, p2 = r
            if relerr(q2, p2) < err_base:
                q, p = q2, p2
    deq = QuantizedTensor(q, "AsymW4A8Int8Layout", p).dequantize()
    tensors = {"weight": q.cpu(), "weight_s_rel": p.scale.cpu(), "weight_s_channel": p.s_channel.cpu()}
    if p.codebook is not None:
        tensors["weight_codebook"] = p.codebook.cpu()
    cfg = {"format": "w6a8_int8" if bits == 6 else "asym_w4a8_int8", "group_size": group_size,
           "convrot": True, "convrot_groupsize": 256}
    err = ((deq.float() - wf.float()).norm() / wf.float().norm().clamp(min=1e-30)).item() * 100.0
    return tensors, cfg, err, err_base

@torch.no_grad()
def quantize_embedding(w, gs, device="cuda", chunk=32768):
    """Rotated per-row int8 for token-embedding tables. Rotation Gaussianizes each row, tightening
    its absmax (~half the error for free). The runtime un-rotates after the lookup, since a lookup
    has no GEMM to fold the inverse into. gs=None -> plain absmax per-row int8 (no rotation).
    Chunked over rows so a ~1B-element vocab table never lands fully on the GPU."""
    h = _build_hadamard(gs, device=device, dtype=torch.float32) if gs else None
    qs, ss = [], []
    for i in range(0, w.shape[0], chunk):
        wf = w[i:i + chunk].to(device, torch.float32)
        wr = _rotate_weight(wf, h, gs) if h is not None else wf
        scale = (wr.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-30)
        q = (wr / scale).round().clamp(-127, 127)
        qs.append(q.to(torch.int8).cpu())
        ss.append(scale.to(torch.float32).cpu())
        del wf, wr, q, scale
        torch.cuda.empty_cache()
    return torch.cat(qs), torch.cat(ss)

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?", help="output .safetensors; if omitted, derived from SRC by "
                    "replacing bf16/fp16/fp32 with int8_convrot (or appending _int8_convrot)")
    ap.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    fmt = ap.add_mutually_exclusive_group()
    fmt.add_argument("--w4a8", action="store_true",
                     help="quantize eligible block linears as W4A8 (int4 weight + codebook "
                          "+ fp8 scales, int8 activations) instead of int8. Layers with K not divisible by "
                          "256 or N<64 fall back to int8; embeddings stay int8.")
    fmt.add_argument("--w6a8", action="store_true",
                     help="like --w4a8 but uniform 6-bit weights: ~3x lower weight error than "
                          "W4A8 at 0.78 vs 0.56 bytes/weight, same speed")
    fmt.add_argument("--w4a4", action="store_true",
                     help="ConvRot W4A4: uniform int4, one fp32 scale per row, 0.50 B/weight. "
                          "!! EXPERIMENTAL, OUTPUT IS VISIBLY DEGRADED -- end-to-end on SDXL "
                          "this scored PSNR 11.9 dB with heavy colour cast and blocky coloured "
                          "noise, versus ~20 dB for --w4a8 and the plain int8 path at ~0.8%% "
                          "weight error. Weight error is 13.1%% mean, because the scale "
                          "granularity is group-64 with no codebook (quant_group_size is fixed "
                          "at 64 by the int4 MMA kernel, not tunable here) and ComfyUI's "
                          "convrot_w4a4 layout has NO low-rank correction branch. It saves "
                          "only ~4.6%% over --w4a8 (1.93 vs 2.03 GB on SDXL) while roughly "
                          "doubling the error, so the size win does not pay for the quality "
                          "loss. Use --w4a8 or --w6a8 unless you are measuring this format; "
                          "for real 4-bit quality see svdquant_w4a4 in comfy-kitchen, which "
                          "does carry the low-rank branch but needs offline calibration. "
                          "Layers whose K is not a multiple of the convrot groupsize fall "
                          "back to int8")
    ap.add_argument("--group-size", type=int, default=None,
                    help="columns per fp8 group scale for --w4a8/--w6a8 (default 16 for W4A8, 32 for W6A8)")
    ap.add_argument("--scale-search", action=argparse.BooleanOptionalAction, default=True,
                    help="re-pick the fp8 group scales of --w4a8 by grid-aware search: for each "
                         "group take the neighbouring fp8 value whose decoded levels fit best "
                         "(~1%% lower weight error, no size cost). comfy-kitchen only applies "
                         "this at 6-bit, so it is done here for 4-bit. --no-scale-search to skip")
    ap.add_argument("--exclude", default=None, help="regex; matching layers are FORCED to passthrough")
    ap.add_argument("--include", default=None, help="regex; matching eligible layers are FORCED to quantize")
    ap.add_argument("--min-gemm", type=int, default=256,
                    help="skip a layer if min(N,K) < this (default 256: a GEMM whose small side is "
                         "under ~256 never beats bf16 at any M, so int8 is pure overhead). --min-gemm 0 disables.")
    ap.add_argument("--quant-embeddings", action=argparse.BooleanOptionalAction, default=True,
                    help="quantize token-embedding tables (embed_tokens, embed_tokens_per_layer, ...) "
                         "per-ROW int8 (+rotation); ON by default (often the biggest size win). "
                         "--no-quant-embeddings keeps them bf16.")
    ap.add_argument("--mseclip", action="store_true", help="MSE-optimal clip instead of absmax for the CONVROT linears only (embeddings always absmax) (~2-3%% lower weight error, but a proxy — validate output before trusting it)")
    ap.add_argument("--downcast-fp32", action="store_true", help="downcast stray fp32 passthrough linears to compute dtype")
    ap.add_argument("--warn-thresh", type=float, default=None,
                    help="warn on any quantized layer whose relerr%% exceeds this (default 2.0 for int8, "
                         "4.0 for W6A8, 10.0 for W4A8: their normal weight error is ~2.4%% / ~7%%)")
    ap.add_argument("--verify-report", default=None, help="write the full per-layer (relerr, cos, gs) table to this path")
    args = ap.parse_args()
    bits = 6 if args.w6a8 else 4
    args.w4a8 = args.w4a8 or args.w6a8       # one grouped-int path below, parametrized by bits
    if args.warn_thresh is None:
        args.warn_thresh = {4: 10.0, 6: 4.0}[bits] if args.w4a8 else (20.0 if args.w4a4 else 2.0)
    if args.w4a4 and not args.dry_run:
        print("!! --w4a4 is EXPERIMENTAL and the output is visibly degraded "
              "(SDXL end-to-end PSNR 11.9 dB, heavy colour cast + blocky coloured noise). "
              "It beats --w4a8 on size by only ~4.6%. Prefer --w4a8 or --w6a8.\n",
              file=sys.stderr, flush=True)
    if not args.dst and not args.dry_run:
        # derive dst from src: swap dtype token for int8_convrot (else append), always .safetensors
        base = os.path.splitext(os.path.basename(args.src))[0]
        tag = "w4a4_convrot" if args.w4a4 else (f"w{bits}a8_convrot" if args.w4a8 else "int8_convrot")
        new = re.sub(r"(?i)(bf16|fp16|fp32)", tag, base)
        if new == base:
            new = base + "_" + tag
        args.dst = os.path.join(os.path.dirname(args.src), new + ".safetensors")
        print(f"auto dst -> {args.dst}")
    exc = re.compile(args.exclude) if args.exclude else None
    inc = re.compile(args.include) if args.include else None

    with open_model(args.src) as st:               # .safetensors (lazy) or .pth/.pt/.ckpt (safe load)
        src_meta = st.metadata() or {}
        keys = list(st.keys())
        scaled = {k[:-len(".weight_scale")] for k in keys if k.endswith(".weight_scale")}  # fp8 sources
        # compute/passthrough dtype = dominant non-fp8 float weight dtype
        dtc = collections.Counter(st.get_slice(k).get_dtype() for k in keys if k.endswith(".weight"))
        target = torch.float16 if dtc.get("F16", 0) >= dtc.get("BF16", 0) and dtc.get("F16", 0) else torch.bfloat16

        # ---- plan ----
        plan = []           # (base, shape, gs)  block linears -> int8+convrot
        eplan = []          # (base, shape)      token embeddings -> per-row int8
        skip = collections.Counter()
        for key in keys:
            if not key.endswith(".weight"):
                continue
            base = key[:-len(".weight")]
            shape = tuple(st.get_slice(key).get_shape())
            if is_token_embedding(base, shape):
                if args.quant_embeddings and not (exc and exc.search(base)):
                    eplan.append((base, shape))
                else:
                    skip["embedding(--no-quant-embeddings/excluded)"] += 1
                continue
            q, reason = classify(base, shape)
            if exc and exc.search(base):
                q, reason = False, "excluded(flag)"
            if inc and inc.search(base) and len(shape) == 2 and best_gs(shape[1]) and shape[0] >= 8:
                q, reason = True, f"gs{best_gs(shape[1])}(incl-flag)"
            if q and args.min_gemm and min(shape) < args.min_gemm:
                q, reason = False, f"below-min-gemm({min(shape)})"
            if q:
                plan.append((base, shape, best_gs(shape[1])))
            else:
                skip[reason] += 1

        # ---- report ----
        by_pat = collections.defaultdict(lambda: [0, None, None])
        qparams = 0
        for base, shape, gs in plan:
            pat = re.sub(r"\d+", "N", base)
            by_pat[pat][0] += 1
            by_pat[pat][1] = shape
            by_pat[pat][2] = gs
            qparams += shape[0] * shape[1]
        print(f"SRC {args.src}")
        print(f"compute/passthrough dtype: {target}")
        fmt_label = ("W4A4+convrot -- EXPERIMENTAL, visibly degraded (see --w4a4); "
                     "int8 fallback when K is not a multiple of the convrot groupsize"
                     if args.w4a4 else
                     f"W{bits}A8+convrot" + ("" if bits == 6 else
                                      f", fp8 scale-search {'on' if args.scale_search else 'off'}") +
                      " (int8 fallback for K%256!=0 or N<64)" if args.w4a8
                     else f"int8+convrot, {'MSE-clip' if args.mseclip else 'absmax'}")
        print(f"\nQUANTIZE {len(plan)} layers ({fmt_label}):")
        for pat in sorted(by_pat):
            c, shape, gs = by_pat[pat]
            print(f"  x{c:<4d} gs{gs:<3d} {str(shape):16s} {pat}")
        gsdist = collections.Counter(gs for _, _, gs in plan)
        print(f"  groupsizes: {dict(gsdist)}   quantized params: {qparams/1e9:.2f}B  (~{qparams/1e9:.1f} GB int8)")
        if eplan:
            ep = sum(s[0] * s[1] for _, s in eplan)
            print(f"\nEMBEDDINGS {len(eplan)} table(s) per-row int8 (+rotation):")
            for b, s in eplan:
                print(f"  {str(s):20s} {b}")
            print(f"  embedding params: {ep/1e9:.2f}B  (~{ep/1e9:.1f} GB int8, saves ~{ep/1e9:.1f} GB vs bf16)")
        print(f"\nLEAVE AS-IS ({sum(skip.values())} weights):")
        for reason, c in skip.most_common():
            print(f"  x{c:<4d} {reason}")
        if args.dry_run:
            print("\n[dry-run] nothing written.")
            return

        # ---- execute ----
        quant_set = {b for b, _, _ in plan}
        embed_set = {b for b, _ in eplan}
        out = {}
        nq = 0
        t0 = time.time()
        errs = []            # (relerr%, cos, gs, base) per quantized layer
        for key in keys:
            if key.endswith(".weight_scale"):
                continue                                  # fp8 source scale: consumed by dequant
            t = st.get_tensor(key)
            if not key.endswith(".weight"):
                out[key] = t
                continue
            base = key[:-len(".weight")]
            # materialize source weight (dequant fp8 rowwise if needed)
            if t.dtype in FP8 and base in scaled:
                sc = st.get_tensor(base + ".weight_scale").float()
                w = t.float() * sc.view(-1, 1)
            elif t.dtype in FP8:
                w = t.float()
            else:
                w = t
            if base in embed_set:
                egs = best_gs(w.shape[1])                  # rotate if the row dim allows it
                qd, scale = quantize_embedding(w, egs)
                # relerr, chunked with f64 accumulation (a single fp32 .norm() over a ~1B-element
                # vocab table accumulates enough error to misreport by >0.1pp)
                h_e = _build_hadamard(egs, device="cuda", dtype=torch.float32) if egs else None
                se = sw = 0.0
                for i in range(0, w.shape[0], 32768):
                    wf = w[i:i + 32768].to("cuda", torch.float32)
                    dq = qd[i:i + 32768].to("cuda").float() * scale[i:i + 32768].to("cuda")
                    if h_e is not None:
                        dq = _rotate_weight(dq, h_e, egs)   # un-rotate to compare against source
                    se += ((dq - wf) ** 2).sum().double().item()
                    sw += (wf ** 2).sum().double().item()
                    del wf, dq
                torch.cuda.empty_cache()
                relerr = (se / sw) ** 0.5 * 100.0
                rot = f"rotated gs{egs} + " if egs else ""
                print(f"  embedding {base} {tuple(w.shape)} {rot}per-row int8 relerr={relerr:.3f}%", flush=True)
                ecfg = {"format": "int8_tensorwise"}
                if egs is not None:
                    ecfg["convrot"] = True
                    ecfg["convrot_groupsize"] = egs
                out[key] = qd
                out[f"{base}.weight_scale"] = scale
                out[f"{base}.comfy_quant"] = torch.tensor(list(json.dumps(ecfg).encode("utf-8")), dtype=torch.uint8)
                torch.cuda.empty_cache()
                continue
            gs4 = best_gs(w.shape[1]) if len(w.shape) == 2 else None
            if (base in quant_set and args.w4a4 and gs4 is not None
                    and not DETECT_SENSITIVE.search(base)):
                tensors, cfg, relerr, cos = quantize_w4a4(w, gs4)
                assert cos > 0.95, f"BROKEN quant (rotation/format?) {base} cos={cos:.5f} relerr={relerr:.2f}%"
                if relerr > args.warn_thresh:
                    print(f"  WARN high error: {base} W4A4 relerr={relerr:.2f}%", flush=True)
                errs.append((relerr, cos, gs4, base))
                for suf, val in tensors.items():
                    out[f"{base}.{suf}"] = val
                out[f"{base}.comfy_quant"] = torch.tensor(list(json.dumps(cfg).encode("utf-8")), dtype=torch.uint8)
                nq += 1
                if nq % 100 == 0:
                    print(f"  {nq}/{len(plan)} ... {base} W4A4 relerr={relerr:.2f}%", flush=True)
            elif (base in quant_set and args.w4a8 and len(w.shape) == 2
                    and best_gs(w.shape[1]) == 256 and w.shape[0] >= 64
                    and not DETECT_SENSITIVE.search(base)):
                tensors, cfg, relerr, err_base = quantize_w4a8(
                    w, bits, args.group_size, scale_search=args.scale_search)
                gain = f" (fp8 scale search -{err_base - relerr:.3f}pp)" if relerr < err_base else ""
                if relerr > args.warn_thresh:
                    print(f"  WARN high error: {base} W{bits}A8 relerr={relerr:.2f}%{gain}", flush=True)
                errs.append((relerr, 1.0, 256, base))
                for suf, val in tensors.items():
                    out[f"{base}.{suf}"] = val
                out[f"{base}.comfy_quant"] = torch.tensor(list(json.dumps(cfg).encode("utf-8")), dtype=torch.uint8)
                nq += 1
                if nq % 100 == 0:
                    print(f"  {nq}/{len(plan)} ... {base} W{bits}A8 relerr={relerr:.2f}%{gain}", flush=True)
            elif base in quant_set:
                gs = best_gs(w.shape[1])
                qd, scale = quantize_convrot(w, gs, mseclip=args.mseclip)
                cos, relerr = recon_metrics(qd, scale, w, gs)
                assert cos > 0.99, f"BROKEN quant (rotation/format?) {base} cos={cos:.5f} relerr={relerr:.2f}%"
                if relerr > args.warn_thresh:
                    print(f"  WARN high error: {base} gs={gs} relerr={relerr:.2f}% cos={cos:.5f}", flush=True)
                errs.append((relerr, cos, gs, base))
                out[key] = qd.cpu()
                out[f"{base}.weight_scale"] = scale.cpu()
                out[f"{base}.comfy_quant"]  = cq_tensor(gs)
                nq += 1
                if nq % 100 == 0:
                    print(f"  {nq}/{len(plan)} ... {base} gs={gs} relerr={relerr:.2f}% cos={cos:.5f}", flush=True)
            else:
                # passthrough: fp8 must be de-fp8'd; fp32 optionally downcast; else keep source dtype
                if t.dtype in FP8:
                    out[key] = w.to(target)
                elif t.dtype == torch.float32 and args.downcast_fp32 \
                        and not (base.endswith(".scale") or EXCLUDE_SEG.search(base.split(".")[-1])):
                    out[key] = w.to(target)
                else:
                    out[key] = t
            torch.cuda.empty_cache()
        save_file(out, args.dst, metadata=dict(src_meta))
        print(f"DONE: quantized {nq} layers, {len(out)} tensors, {time.time()-t0:.1f}s -> {args.dst}")

        # ---- per-layer error report ----
        if errs:
            errs.sort(reverse=True)                        # worst relerr first
            rvals = [e[0] for e in errs]
            mean = sum(rvals) / len(rvals)
            over = [e for e in errs if e[0] > args.warn_thresh]
            per_gs = collections.defaultdict(list)
            for r, c, gs, b in errs:
                per_gs[gs].append(r)
            print("\n=== quant error (relerr = ||dequant-source|| / ||source||) ===")
            print(f"  mean {mean:.3f}%   min {min(rvals):.3f}%   max {max(rvals):.3f}%   layers {len(errs)}")
            print("  per groupsize: " + "  ".join(
                f"gs{gs}: mean {sum(v)/len(v):.3f}% max {max(v):.3f}% (x{len(v)})" for gs, v in sorted(per_gs.items())))
            print("  worst 8 layers:")
            for r, c, gs, b in errs[:8]:
                print(f"    {r:6.3f}%  cos {c:.5f}  gs{gs:<3d} {b}")
            if over:
                print(f"  !! {len(over)} layer(s) over --warn-thresh ({args.warn_thresh}%) — review above")
        if args.verify_report and errs:
            with open(args.verify_report, "w") as f:
                f.write("relerr_pct\tcosine\tgroupsize\tlayer\n")
                for r, c, gs, b in errs:
                    f.write(f"{r:.4f}\t{c:.6f}\t{gs}\t{b}\n")
            print(f"  full per-layer table -> {args.verify_report}")

if __name__ == "__main__":
    main()
