import argparse
import csv
import logging
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


FIXED_TEST_PATIENTS = "BT23269,BT23277,BT23288,BT23377,BT23901,BT23944"
SCRIPT_DIR = Path(__file__).resolve().parent
TASK_ROOT = SCRIPT_DIR.parent


def script_path(name: str) -> str:
    return str(SCRIPT_DIR / name)


def task_path(path: str) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else TASK_ROOT / resolved


def unwrap_module(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def densenet_forward(base_model, x):
    features = base_model.features(x)
    out = torch.nn.functional.relu(features, inplace=False)
    out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    return base_model.classifier(out)


def model_forward(model, x, variant):
    if variant == "without_inr":
        inner = unwrap_module(model)
        if hasattr(inner, "forward_without_inr"):
            return inner.forward_without_inr(x)
        return densenet_forward(inner.base_model, x)
    return model(x)


def set_wrong_key(mod, model, args_model, device, seed):
    inner = unwrap_module(model)
    wrong_coords = mod["make_key_inr_coordinates"](
        seed=seed,
        mode=args_model.key_inr_coord_mode,
        num_points=args_model.key_inr_coord_points,
        coord_dim=args_model.key_inr_coord_dim,
        device=device,
        constant=args_model.key_inr_coord_constant,
    )
    inner.set_coordinates(wrong_coords)


def selected_named_parameters(model, scope):
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if scope == "all":
            params.append((name, param))
        elif scope == "classifier":
            if "classifier" in name or "feature_film" in name:
                params.append((name, param))
        else:
            raise ValueError(f"unknown parameter scope: {scope}")
    if not params:
        raise RuntimeError(f"no parameters selected for scope={scope}")
    return params


def gradients(loss, named_params, create_graph):
    params = [param for _, param in named_params]
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=True,
    )
    output = []
    for (_, param), grad in zip(named_params, grads):
        if grad is None:
            grad = torch.zeros_like(param)
        output.append(grad)
    return output


def gradient_loss(dummy_grads, target_grads):
    loss = None
    for dummy, target in zip(dummy_grads, target_grads):
        item = torch.mean((dummy - target) ** 2)
        loss = item if loss is None else loss + item
    return loss / max(1, len(dummy_grads))


def gradient_cosine_loss(dummy_grads, target_grads, eps=1e-12):
    dot = None
    dummy_norm = None
    target_norm = None
    for dummy, target in zip(dummy_grads, target_grads):
        item_dot = torch.sum(dummy * target)
        item_dummy = torch.sum(dummy * dummy)
        item_target = torch.sum(target * target)
        dot = item_dot if dot is None else dot + item_dot
        dummy_norm = item_dummy if dummy_norm is None else dummy_norm + item_dummy
        target_norm = item_target if target_norm is None else target_norm + item_target
    cosine = dot / (torch.sqrt(dummy_norm + eps) * torch.sqrt(target_norm + eps))
    return 1.0 - cosine


def gradient_match_loss(dummy_grads, target_grads, mode):
    mse = gradient_loss(dummy_grads, target_grads)
    if mode == "mse":
        return mse
    cosine = gradient_cosine_loss(dummy_grads, target_grads)
    if mode == "cosine":
        return cosine
    if mode == "mix":
        return mse + cosine
    raise ValueError(f"unknown gradient loss mode: {mode}")


def gradient_global_norm(grads):
    norm_sq = None
    for grad in grads:
        item = torch.sum(grad.detach() ** 2)
        norm_sq = item if norm_sq is None else norm_sq + item
    if norm_sq is None:
        return torch.zeros((), device="cpu")
    return torch.sqrt(norm_sq)


def release_target_gradients(grads, clip_norm, noise_multiplier, seed):
    clip_norm = float(clip_norm)
    noise_multiplier = float(noise_multiplier)
    grad_norm = gradient_global_norm(grads)
    clip_coef = min(1.0, clip_norm / (float(grad_norm.detach().cpu()) + 1e-12))
    generator = torch.Generator(device=grads[0].device)
    generator.manual_seed(int(seed))
    released = []
    noise_norm_sq = 0.0
    for grad in grads:
        clipped = grad * clip_coef
        if noise_multiplier > 0:
            noise = torch.randn(clipped.shape, generator=generator, device=clipped.device, dtype=clipped.dtype) * (noise_multiplier * clip_norm)
            clipped = clipped + noise
            noise_norm_sq += float(torch.sum(noise.detach() ** 2).cpu())
        released.append(clipped.detach())
    return released, {
        "target_grad_norm": float(grad_norm.detach().cpu()),
        "clip_norm": clip_norm,
        "clip_coef": clip_coef,
        "noise_multiplier": noise_multiplier,
        "noise_norm": float(np.sqrt(noise_norm_sq)),
    }


def image_bounds(mean, std, device):
    mean = torch.as_tensor(mean, device=device).view(1, 3, 1, 1)
    std = torch.as_tensor(std, device=device).view(1, 3, 1, 1)
    low = (0.0 - mean) / std
    high = (1.0 - mean) / std
    return low, high


def normalized_from_unit_image(image, mean, std):
    mean = torch.as_tensor(mean, device=image.device).view(1, 3, 1, 1)
    std = torch.as_tensor(std, device=image.device).view(1, 3, 1, 1)
    return (image - mean) / std


def denormalize(x, mean, std):
    mean = torch.as_tensor(mean, device=x.device).view(1, 3, 1, 1)
    std = torch.as_tensor(std, device=x.device).view(1, 3, 1, 1)
    return torch.clamp(x * std + mean, 0.0, 1.0)


def tensor_to_image(x, mean, std):
    x = denormalize(x.detach(), mean, std)[0].cpu().permute(1, 2, 0).numpy()
    return x


def mse_psnr(recon, target, mean, std):
    recon = denormalize(recon.detach(), mean, std)
    target = denormalize(target.detach(), mean, std)
    mse = torch.mean((recon - target) ** 2).item()
    psnr = float("inf") if mse == 0 else float(-10.0 * np.log10(mse))
    return mse, psnr


def total_variation_loss(x, mean, std):
    image = denormalize(x, mean, std)
    loss_h = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    loss_w = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    return loss_h + loss_w


def save_recon_snapshot(path, x, mean, std):
    plt.imsave(path, tensor_to_image(x, mean, std))


def normalize_state_dict(state, model):
    model_keys = model.state_dict().keys()
    if state and all(key.startswith("module.module.") for key in state.keys()) and not any(key.startswith("module.module.") for key in model_keys):
        return {key.replace("module.module.", "module.", 1): value for key, value in state.items()}
    return state


def load_checkpoint_model(mod, fulltrain_dataset, checkpoint_path, use_key_inr, key_strength, device, workers, key_dim=16, key_hidden=8, key_init_std=0.02, key_controlled_classifier=False, key_only_classifier_bias=False, model_name="densenet121"):
    args_model = SimpleNamespace(
        use_key_inr=use_key_inr,
        key_inr_coord_seed=20260508,
        key_inr_coord_mode="uniform",
        key_inr_coord_points=128,
        key_inr_coord_dim=2,
        key_inr_coord_constant=1.0,
        key_inr_key_dim=key_dim,
        key_inr_key_hidden=key_hidden,
        key_inr_strength=key_strength,
        key_inr_init_std=key_init_std,
        key_inr_inject_scope="all_linear",
        key_inr_controlled_classifier=int(key_controlled_classifier),
        key_inr_key_only_classifier_bias=int(key_only_classifier_bias),
        fixed_test_patients=FIXED_TEST_PATIENTS,
        test=0.25,
        window=224,
        gene_filter="tumor",
        downsample=1,
        norm=None,
        gene_transform="log",
        model=model_name,
        batch=1,
        workers=workers,
        gpu=torch.cuda.is_available(),
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
    globals_ref = mod["get_model"].__globals__
    globals_ref["args"] = args_model
    model = mod["get_model"](fulltrain_dataset).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = normalize_state_dict(checkpoint["model"], model)
    model.load_state_dict(state)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(True)
    return model, args_model


def attack_one(
    target_model,
    attack_model,
    x,
    gene,
    target_variant,
    attack_variant,
    mean,
    std,
    iterations,
    lr,
    scope,
    tv_weight=0.0,
    l2_weight=0.0,
    restarts=1,
    save_every=0,
    snapshot_prefix=None,
    grad_loss_mode="mse",
    sigmoid_param=False,
    lr_schedule="none",
    release_target_gradients_enabled=False,
    release_clip_norm=1.0,
    release_noise_multiplier=0.0,
    release_seed=0,
):
    target_params = selected_named_parameters(target_model, scope)
    attack_params = selected_named_parameters(attack_model, scope)
    target_names = [name for name, _ in target_params]
    attack_map = {name: param for name, param in attack_params}
    shared_params = [(name, attack_map[name]) for name in target_names if name in attack_map]
    target_params = [(name, param) for name, param in target_params if name in attack_map]
    if not target_params:
        raise RuntimeError("target and attack models have no shared selected parameter names")

    target_model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        target_pred = model_forward(target_model, x, target_variant)
        target_loss = torch.mean((target_pred - gene) ** 2)
        target_grads = [grad.detach() for grad in gradients(target_loss, target_params, create_graph=False)]
    release_stats = None
    if release_target_gradients_enabled:
        target_grads, release_stats = release_target_gradients(
            target_grads,
            clip_norm=release_clip_norm,
            noise_multiplier=release_noise_multiplier,
            seed=release_seed,
        )

    low, high = image_bounds(mean, std, x.device)
    best_recon = None
    best_history = None
    best_grad_loss = float("inf")
    for restart in range(restarts):
        if sigmoid_param:
            init_image = torch.rand_like(x).clamp(1e-6, 1.0 - 1e-6)
            attack_var = torch.log(init_image / (1.0 - init_image)).detach().requires_grad_(True)
            optimizer = torch.optim.Adam([attack_var], lr=lr)
        else:
            dummy = low + torch.rand_like(x) * (high - low)
            attack_var = dummy.detach().requires_grad_(True)
            optimizer = torch.optim.Adam([attack_var], lr=lr)
        scheduler = None
        if lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, iterations), eta_min=lr * 0.02)
        elif lr_schedule != "none":
            raise ValueError(f"unknown lr schedule: {lr_schedule}")

        history = []
        for step in range(iterations + 1):
            if sigmoid_param:
                dummy = normalized_from_unit_image(torch.sigmoid(attack_var), mean, std)
            else:
                dummy = attack_var
            optimizer.zero_grad(set_to_none=True)
            attack_model.zero_grad(set_to_none=True)
            pred = model_forward(attack_model, dummy, attack_variant)
            dummy_loss = torch.mean((pred - gene) ** 2)
            dummy_grads = gradients(dummy_loss, shared_params, create_graph=True)
            grad_match_loss = gradient_match_loss(dummy_grads, target_grads, grad_loss_mode)
            loss = grad_match_loss
            if tv_weight:
                loss = loss + tv_weight * total_variation_loss(dummy, mean, std)
            if l2_weight:
                image = denormalize(dummy, mean, std)
                loss = loss + l2_weight * torch.mean((image - 0.5) ** 2)
            loss.backward()
            optimizer.step()
            if scheduler is not None and step < iterations:
                scheduler.step()
            if not sigmoid_param:
                with torch.no_grad():
                    attack_var.clamp_(low, high)
            with torch.no_grad():
                current_dummy = normalized_from_unit_image(torch.sigmoid(attack_var), mean, std) if sigmoid_param else attack_var
            should_save = save_every > 0 and (step % save_every == 0 or step == iterations)
            should_record = step % 100 == 0 or step == iterations or should_save
            if should_record:
                image_mse, psnr = mse_psnr(current_dummy, x, mean, std)
                history.append({
                    "restart": restart,
                    "step": step,
                    "grad_loss": float(grad_match_loss.detach().cpu()),
                    "total_loss": float(loss.detach().cpu()),
                    "image_mse": image_mse,
                    "psnr": psnr,
                })
            if should_save and snapshot_prefix is not None:
                save_recon_snapshot(f"{snapshot_prefix}_r{restart}_iter{step:04d}.png", current_dummy, mean, std)
        final_grad_loss = history[-1]["grad_loss"]
        if final_grad_loss < best_grad_loss:
            best_grad_loss = final_grad_loss
            best_recon = current_dummy.detach().clone()
            best_history = history
    return best_recon, best_history, release_stats


def save_montage(path, rows):
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.8, 2.6 * len(rows)), constrained_layout=True)
    if len(rows) == 1:
        axes = np.asarray([axes])
    for row_idx, row in enumerate(rows):
        original = row["original"]
        recon = row["recon"]
        diff = np.clip(np.abs(original - recon) * 4.0, 0.0, 1.0)
        for col_idx, (image, title) in enumerate([(original, "original"), (recon, "recon"), (diff, "abs diff x4")]):
            ax = axes[row_idx, col_idx]
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(f"{row['method']} {title}")
    fig.savefig(path, dpi=260)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="evaluation_example_fixedsplit/checkpoint_gia_s50")
    parser.add_argument("--samples", default="0,50,100")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--key-inr-strength", type=float, default=50.0)
    parser.add_argument("--key-inr-key-dim", type=int, default=16)
    parser.add_argument("--key-inr-key-hidden", type=int, default=8)
    parser.add_argument("--key-inr-init-std", type=float, default=0.02)
    parser.add_argument("--key-inr-controlled-classifier", action="store_true")
    parser.add_argument("--key-inr-key-only-classifier-bias", action="store_true")
    parser.add_argument("--model", default="densenet121", choices=["densenet121", "simplecnn"], help="checkpoint model architecture")
    parser.add_argument("--fl-checkpoint", default="example_FL_c30_e30_fixedsplit/checkpoints/30.pt")
    parser.add_argument("--inr-checkpoint", default="example_FL_INR_s50_c30_e30_fixedsplit/checkpoints/30.pt")
    parser.add_argument("--ppml-checkpoint", default="example_PPMLOmics_e10_c30_e30_fixedsplit/checkpoints/30.pt")
    parser.add_argument("--wrong-key-seed", type=int, default=20260509)
    parser.add_argument("--param-scope", choices=["classifier", "all"], default="classifier")
    parser.add_argument("--methods", default="all", help="comma-separated method keys: all,fl,inr_correct,inr_wrong,inr_without,ppml")
    parser.add_argument("--save-every", type=int, default=0, help="save intermediate recon PNGs every N iterations")
    parser.add_argument("--tv-weight", type=float, default=0.0, help="total variation prior weight")
    parser.add_argument("--l2-weight", type=float, default=0.0, help="weak pixel prior weight")
    parser.add_argument("--restarts", type=int, default=1, help="number of random dummy initializations")
    parser.add_argument("--grad-loss", choices=["mse", "cosine", "mix"], default="mse", help="gradient matching loss")
    parser.add_argument("--sigmoid-param", action="store_true", help="optimize unconstrained logits mapped through sigmoid instead of clamped pixels")
    parser.add_argument("--lr-schedule", choices=["none", "cosine"], default="none", help="learning-rate schedule for the dummy optimizer")
    parser.add_argument("--release-target-gradients", action="store_true", help="attack clipped/noised released target gradients instead of clean target gradients")
    parser.add_argument("--release-grad-clip-norm", type=float, default=1.0, help="global norm clipping threshold for released target gradients")
    parser.add_argument("--release-grad-noise-multiplier", type=float, default=0.0, help="default Gaussian noise multiplier for released target gradients")
    parser.add_argument("--ppml-release-grad-noise-multiplier", type=float, default=None, help="PPML-specific released-gradient noise multiplier; defaults to --release-grad-noise-multiplier")
    parser.add_argument("--release-grad-seed", type=int, default=20260602, help="base random seed for released-gradient noise")
    args_cli = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    outdir = task_path(args_cli.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    args_cli.fl_checkpoint = task_path(args_cli.fl_checkpoint)
    args_cli.inr_checkpoint = task_path(args_cli.inr_checkpoint)
    args_cli.ppml_checkpoint = task_path(args_cli.ppml_checkpoint)
    sample_indices = [int(item) for item in args_cli.samples.split(",") if item.strip()]

    mod = runpy.run_path(script_path("simulation_core.py"))
    setattr(sys.modules["__main__"], "IdentityDict", mod["IdentityDict"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_args = SimpleNamespace(
        use_key_inr=0,
        key_inr_coord_seed=20260508,
        key_inr_coord_mode="uniform",
        key_inr_coord_points=128,
        key_inr_coord_dim=2,
        key_inr_coord_constant=1.0,
        key_inr_key_dim=args_cli.key_inr_key_dim,
        key_inr_key_hidden=args_cli.key_inr_key_hidden,
        key_inr_strength=args_cli.key_inr_strength,
        key_inr_init_std=args_cli.key_inr_init_std,
        key_inr_inject_scope="all_linear",
        key_inr_controlled_classifier=0,
        key_inr_key_only_classifier_bias=0,
        fixed_test_patients=FIXED_TEST_PATIENTS,
        test=0.25,
        window=224,
        gene_filter="tumor",
        downsample=1,
        norm=None,
        gene_transform="log",
        model=args_cli.model,
        batch=1,
        workers=args_cli.workers,
        gpu=torch.cuda.is_available(),
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
    globals_ref["args"] = dataset_args
    globals_ref["device"] = device
    globals_ref["logger"] = logging.getLogger("checkpoint_gia")

    train_datasets, test_dataset, fulltrain_dataset = mod["getSTDataset"](30)
    globals_ref["testDataset"] = test_dataset
    globals_ref["testLoader"] = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args_cli.workers)
    globals_ref["fulltrainDataset"] = fulltrain_dataset
    globals_ref["fulltrainLoader"] = torch.utils.data.DataLoader(fulltrain_dataset, batch_size=1, shuffle=False, num_workers=args_cli.workers)
    globals_ref["outputs"] = fulltrain_dataset[0][2].shape[0]
    mean_expression, mean_expression_tumor, mean_expression_normal, median_expression = mod["compute_mean_expression"]()
    globals_ref["mean_expression"] = mean_expression
    globals_ref["mean_expression_tumor"] = mean_expression_tumor
    globals_ref["mean_expression_normal"] = mean_expression_normal
    globals_ref["median_expression"] = median_expression

    normalize = test_dataset.transform.transforms[-1]
    image_mean = normalize.mean
    image_std = normalize.std

    methods = [
        {
            "key": "fl",
            "method": "FL",
            "checkpoint": args_cli.fl_checkpoint,
            "use_key_inr": 0,
            "target_variant": "normal",
            "attack_variant": "normal",
        },
        {
            "key": "inr_correct",
            "method": "FL-INR correct key",
            "checkpoint": args_cli.inr_checkpoint,
            "use_key_inr": 1,
            "key_controlled_classifier": args_cli.key_inr_controlled_classifier,
            "key_only_classifier_bias": args_cli.key_inr_key_only_classifier_bias,
            "target_variant": "normal",
            "attack_variant": "normal",
        },
        {
            "key": "inr_wrong",
            "method": "FL-INR wrong key attacker",
            "checkpoint": args_cli.inr_checkpoint,
            "use_key_inr": 1,
            "key_controlled_classifier": args_cli.key_inr_controlled_classifier,
            "key_only_classifier_bias": args_cli.key_inr_key_only_classifier_bias,
            "target_variant": "normal",
            "attack_variant": "normal",
            "attack_wrong_key": True,
        },
        {
            "key": "inr_without",
            "method": "FL-INR without INR attacker",
            "checkpoint": args_cli.inr_checkpoint,
            "use_key_inr": 1,
            "key_controlled_classifier": args_cli.key_inr_controlled_classifier,
            "key_only_classifier_bias": args_cli.key_inr_key_only_classifier_bias,
            "target_variant": "normal",
            "attack_variant": "without_inr",
        },
        {
            "key": "ppml",
            "method": "PPML e10",
            "checkpoint": args_cli.ppml_checkpoint,
            "use_key_inr": 0,
            "target_variant": "normal",
            "attack_variant": "normal",
        },
    ]
    requested_methods = {item.strip() for item in args_cli.methods.split(",") if item.strip()}
    if requested_methods and "all" not in requested_methods:
        methods = [method for method in methods if method["key"] in requested_methods]
    if not methods:
        raise ValueError(f"no methods selected by --methods {args_cli.methods!r}")

    rows = []
    for sample_idx in sample_indices:
        x, y, gene, coord, ind, patient, section, pixel, filename = test_dataset[sample_idx]
        x = x.unsqueeze(0).to(device)
        gene = gene.unsqueeze(0).to(device)
        montage_rows = []
        original_image = tensor_to_image(x, image_mean, image_std)
        for method in methods:
            logging.info("Attacking sample %s with %s", sample_idx, method["method"])
            safe_method = method["method"].replace(" ", "_").replace("/", "_")
            target_model, target_args = load_checkpoint_model(
                mod,
                fulltrain_dataset,
                method["checkpoint"],
                method["use_key_inr"],
                args_cli.key_inr_strength,
                device,
                args_cli.workers,
                key_dim=args_cli.key_inr_key_dim,
                key_hidden=args_cli.key_inr_key_hidden,
                key_init_std=args_cli.key_inr_init_std,
                key_controlled_classifier=method.get("key_controlled_classifier", False),
                key_only_classifier_bias=method.get("key_only_classifier_bias", False),
                model_name=args_cli.model,
            )
            attack_model, attack_args = load_checkpoint_model(
                mod,
                fulltrain_dataset,
                method["checkpoint"],
                method["use_key_inr"],
                args_cli.key_inr_strength,
                device,
                args_cli.workers,
                key_dim=args_cli.key_inr_key_dim,
                key_hidden=args_cli.key_inr_key_hidden,
                key_init_std=args_cli.key_inr_init_std,
                key_controlled_classifier=method.get("key_controlled_classifier", False),
                key_only_classifier_bias=method.get("key_only_classifier_bias", False),
                model_name=args_cli.model,
            )
            if method.get("attack_wrong_key"):
                set_wrong_key(mod, attack_model, attack_args, device, args_cli.wrong_key_seed)
            release_noise_multiplier = args_cli.release_grad_noise_multiplier
            if method["key"] == "ppml" and args_cli.ppml_release_grad_noise_multiplier is not None:
                release_noise_multiplier = args_cli.ppml_release_grad_noise_multiplier
            recon, history, release_stats = attack_one(
                target_model,
                attack_model,
                x,
                gene,
                method["target_variant"],
                method["attack_variant"],
                image_mean,
                image_std,
                args_cli.iterations,
                args_cli.lr,
                args_cli.param_scope,
                tv_weight=args_cli.tv_weight,
                l2_weight=args_cli.l2_weight,
                restarts=args_cli.restarts,
                save_every=args_cli.save_every,
                snapshot_prefix=outdir / f"sample{sample_idx}_{safe_method}",
                grad_loss_mode=args_cli.grad_loss,
                sigmoid_param=args_cli.sigmoid_param,
                lr_schedule=args_cli.lr_schedule,
                release_target_gradients_enabled=args_cli.release_target_gradients,
                release_clip_norm=args_cli.release_grad_clip_norm,
                release_noise_multiplier=release_noise_multiplier,
                release_seed=args_cli.release_grad_seed + sample_idx * 101 + len(rows),
            )
            image_mse, psnr = mse_psnr(recon, x, image_mean, image_std)
            recon_image = tensor_to_image(recon, image_mean, image_std)
            plt.imsave(outdir / f"sample{sample_idx}_{safe_method}_recon.png", recon_image)
            montage_rows.append({"method": method["method"], "original": original_image, "recon": recon_image})
            rows.append({
                "sample": sample_idx,
                "patient": patient,
                "section": section,
                "method": method["method"],
                "iterations": args_cli.iterations,
                "param_scope": args_cli.param_scope,
                "image_mse": image_mse,
                "psnr": psnr,
                "final_grad_loss": history[-1]["grad_loss"],
                "release_target_gradients": args_cli.release_target_gradients,
                "release_clip_norm": release_stats["clip_norm"] if release_stats else "",
                "release_noise_multiplier": release_stats["noise_multiplier"] if release_stats else "",
                "target_grad_norm": release_stats["target_grad_norm"] if release_stats else "",
                "release_clip_coef": release_stats["clip_coef"] if release_stats else "",
                "release_noise_norm": release_stats["noise_norm"] if release_stats else "",
            })
            with open(outdir / f"sample{sample_idx}_{safe_method}_history.csv", "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
                writer.writeheader()
                writer.writerows(history)
            del target_model, attack_model, recon
            torch.cuda.empty_cache()
        save_montage(outdir / f"sample{sample_idx}_checkpoint_gia_montage.png", montage_rows)

    with open(outdir / "metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(outdir / "metrics.md", "w") as handle:
        handle.write("| sample | method | image_mse | psnr | final_grad_loss | release_noise_multiplier | target_grad_norm | release_clip_coef | release_noise_norm |\n")
        handle.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(f"| {row['sample']} | {row['method']} | {row['image_mse']} | {row['psnr']} | {row['final_grad_loss']} | {row['release_noise_multiplier']} | {row['target_grad_norm']} | {row['release_clip_coef']} | {row['release_noise_norm']} |\n")


if __name__ == "__main__":
    main()
