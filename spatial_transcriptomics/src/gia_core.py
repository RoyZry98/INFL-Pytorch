from __future__ import annotations

import logging
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_CORE = Path("/deltadisk/zhangrongyu/PPML-Omics-master/Application/SpatialTranscriptomics/gia_fed_st_demo.py")


def _load_source_core() -> Dict:
    if not SOURCE_CORE.exists():
        raise FileNotFoundError(f"PPML-Omics GIA core not found: {SOURCE_CORE}")
    namespace = runpy.run_path(str(SOURCE_CORE))
    namespace["load_st_module"] = load_st_module
    return namespace


def load_st_module(args, device):
    mod = runpy.run_path(str(SCRIPT_DIR / "simulation_core.py"))
    setattr(sys.modules["__main__"], "IdentityDict", mod["IdentityDict"])
    st_args = SimpleNamespace(
        use_key_inr=0,
        key_inr_coord_seed=20260508,
        key_inr_coord_mode="uniform",
        key_inr_coord_points=128,
        key_inr_coord_dim=2,
        key_inr_coord_constant=1.0,
        key_inr_key_dim=16,
        key_inr_key_hidden=8,
        key_inr_strength=1.5,
        key_inr_init_std=0.02,
        key_inr_inject_scope="all_linear",
        key_inr_controlled_classifier=0,
        key_inr_key_only_classifier_bias=0,
        fixed_test_patients=args.fixed_test_patients,
        test=0.25,
        window=224,
        gene_filter="tumor",
        downsample=1,
        norm=None,
        gene_transform="log",
        model=args.model,
        batch=args.batch_size,
        workers=args.num_workers,
        gpu=device.type == "cuda",
        pretrained=True,
        task="gene",
        load=None,
        gene_mask=None,
        pred_root=None,
        average=None,
        checkpoint=None,
        checkpoint_every=10,
        keep_checkpoints=None,
    )
    globals_ref = mod["getSTDataset"].__globals__
    globals_ref["args"] = st_args
    globals_ref["device"] = device
    globals_ref["logger"] = logging.getLogger("spatial_transcriptomics_gia")
    return mod


_defs = _load_source_core()
globals().update({name: value for name, value in _defs.items() if not name.startswith("__")})
globals()["load_st_module"] = load_st_module
