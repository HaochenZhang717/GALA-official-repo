# Third-party code

This release vendors code from several publicly released baseline
implementations so that all reported baselines can be reproduced from this
repository alone. Each item below lists what was taken, where it lives here,
and under which license. Modifications are noted; original license texts are
preserved where the upstream project ships one.

## T2S (IJCAI'25) — Apache License 2.0

Verbatim subset of the official T2S release, kept under its original package
layout so the baseline trainers can import it unchanged:

```
baseline_codes/T2S/
  LICENSE                            # upstream Apache-2.0 text, unmodified
  model/backbone/rectified_flow.py
  model/denoiser/transformer.py
  model/pretrained/vqvae.py
  model/pretrained/core.py           # BaseModel, required by vqvae.py
```

These four files are **unmodified**. Only the modules actually imported by
`t2ts_jit/pretrain_vae_t2s.py` and `t2ts_jit/train_cond_v7_t2s.py` are
included; the rest of the upstream repository (data pipeline, training
scripts, evaluation, figures) is not redistributed.

## ConTSG-Bench — MIT License

```
MIT License
Copyright (c) 2026 ConTSG Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Derived, **modified** single-file ports (Lightning / Registry / config
dependencies stripped, plus the short-series adaptations documented in each
file's module docstring):

- `t2ts_jit/cond_jit_basics/diffusets_modules.py` — DiffuSETS VAE, latent U-Net
  and Gaussian diffusion.
- `t2ts_jit/cond_jit_basics/text2motion.py` — Text2Motion (CVPR'22) movement
  autoencoder and text-conditioned CVAE generator.
- `t2ts_jit/cond_jit_basics/bridge_denoiser.py` and
  `t2ts_jit/cond_jit_basics/bridge_modules/` — BRIDGE (ICML'25) 1D UNet,
  Domain-Unified Prototyper, cross-attention and conditioning MLP.

## VerbalTS (ICML'25)

- `t2ts_jit/cond_jit_basics/verbalts.py` — modified single-file port of the
  VerbalTS diffusion-side architecture (multi-patch denoiser, text projector,
  diffusion embedding). Deviations from the original source are listed in the
  file's module docstring.

The upstream VerbalTS release does not ship a license file; the port is
included here solely for reproducing the reported baseline.

## Other borrowed components

- `t2ts_jit/cond_jit_basics/model_jit.py`, `model_util.py` — adapted from SiT
  (`willisma/SiT`) and Lightning-DiT (`hustvl/LightningDiT`), as noted in the
  file headers.
