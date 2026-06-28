from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union


SRC_DIR = Path(__file__).resolve().parent
TASK_ROOT = SRC_DIR.parent
FIXED_TEST_PATIENTS = "BT23269,BT23277,BT23288,BT23377,BT23901,BT23944"


Command = List[str]
RunResult = Union[subprocess.CompletedProcess, Command]


def _value(value) -> str:
    return str(value)


def _append(args: Command, flag: str, value) -> None:
    if value is None:
        return
    args.extend([flag, _value(value)])


def _append_action(args: Command, flag: str, enabled: bool) -> None:
    if enabled:
        args.append(flag)


def _script_command(script_name: str, script_args: Sequence[str], python_executable: Optional[str] = None) -> Command:
    return [python_executable or sys.executable, str(SRC_DIR / script_name), *script_args]


def _run_command(
    command: Command,
    cwd: Optional[Path] = None,
    dry_run: bool = False,
    check: bool = True,
) -> RunResult:
    if dry_run:
        return command
    return subprocess.run(command, cwd=str(cwd or TASK_ROOT), check=check)


@dataclass
class TrainingConfig:
    seed: int = 0
    device: str = "auto"
    out_dir: str = "spatial_fed_e5_c30_fixedsplit"
    methods: str = "normal,dp,ppml,inr"
    epochs: int = 5
    clients: int = 30
    batch_size: int = 16
    eval_batch_size: int = 64
    num_workers: int = 4
    lr: float = 1e-5
    lr_milestones: str = "8,15"
    lr_gamma: float = 0.3
    momentum: float = 0.9
    weight_decay: float = 0.0
    epsilon: float = 10.0
    dp_epsilon: float = 2.36
    ppml_epsilon: float = 1.29
    dp_noise: Optional[float] = None
    ppml_noise: Optional[float] = None
    delta: float = 1e-3
    l2_clip: float = 1.0
    backbone: str = "densenet121"
    pretrained: int = 1
    freeze_backbone: int = 0
    val_fraction: float = 0.15
    val_seed_offset: int = 9173
    checkpoint_every: int = 0
    fixed_test_patients: str = FIXED_TEST_PATIENTS
    key_dim: int = 16
    key_hidden: int = 8
    key_strength: float = 1.0
    key_init_std: float = 0.02
    key_coord_dim: int = 2
    key_coord_points: int = 128
    key_coord_seed: int = 20260508
    key_coord_mode: str = "uniform"
    key_coord_constant: float = 1.0

    def to_args(self) -> Command:
        args: Command = []
        for flag, value in [
            ("--seed", self.seed),
            ("--device", self.device),
            ("--out-dir", self.out_dir),
            ("--methods", self.methods),
            ("--epochs", self.epochs),
            ("--clients", self.clients),
            ("--batch-size", self.batch_size),
            ("--eval-batch-size", self.eval_batch_size),
            ("--num-workers", self.num_workers),
            ("--lr", self.lr),
            ("--lr-milestones", self.lr_milestones),
            ("--lr-gamma", self.lr_gamma),
            ("--momentum", self.momentum),
            ("--weight-decay", self.weight_decay),
            ("--epsilon", self.epsilon),
            ("--dp-epsilon", self.dp_epsilon),
            ("--ppml-epsilon", self.ppml_epsilon),
            ("--delta", self.delta),
            ("--l2-clip", self.l2_clip),
            ("--backbone", self.backbone),
            ("--pretrained", self.pretrained),
            ("--freeze-backbone", self.freeze_backbone),
            ("--val-fraction", self.val_fraction),
            ("--val-seed-offset", self.val_seed_offset),
            ("--checkpoint-every", self.checkpoint_every),
            ("--fixed-test-patients", self.fixed_test_patients),
            ("--key-dim", self.key_dim),
            ("--key-hidden", self.key_hidden),
            ("--key-strength", self.key_strength),
            ("--key-init-std", self.key_init_std),
            ("--key-coord-dim", self.key_coord_dim),
            ("--key-coord-points", self.key_coord_points),
            ("--key-coord-seed", self.key_coord_seed),
            ("--key-coord-mode", self.key_coord_mode),
            ("--key-coord-constant", self.key_coord_constant),
        ]:
            _append(args, flag, value)
        _append(args, "--dp-noise", self.dp_noise)
        _append(args, "--ppml-noise", self.ppml_noise)
        return args


@dataclass
class VisualizationConfig:
    root: str = "spatial_fed_e5_c30_fixedsplit"
    outdir: str = "evaluation_spatial_fed_e5_c30_fixedsplit"
    epoch: str = "5"
    gene: str = "FASN"
    methods: str = "normal,dp,ppml,inr"
    method_paths: str = ""

    def to_args(self) -> Command:
        args: Command = []
        for flag, value in [
            ("--root", self.root),
            ("--outdir", self.outdir),
            ("--epoch", self.epoch),
            ("--gene", self.gene),
            ("--methods", self.methods),
        ]:
            _append(args, flag, value)
        _append(args, "--method-paths", self.method_paths)
        return args


@dataclass
class AttackConfig:
    expname: str = "iDLG_attack_spatial"
    device: str = "cuda:0"
    batch_size: int = 1
    lr: float = 0.1
    epsilon: float = 100.0
    delta: float = 10e-5
    mode: str = "SGD"
    client: int = 3
    nprocess: int = 100
    shuffle_model: int = 0
    iterations: int = 901
    samples: str = "0,50,100"
    methods: str = "baseline,correct,wrong,without"
    key_dim: int = 16
    key_hidden: int = 8
    key_strength: float = 1.0
    key_init_std: float = 0.02
    key_seed: int = 20260508
    wrong_key_seed: int = 20260509
    key_points: int = 128
    key_coord_dim: int = 2

    def to_args(self) -> Command:
        args: Command = []
        for flag, value in [
            ("--expname", self.expname),
            ("--device", self.device),
            ("--batch_size", self.batch_size),
            ("--lr", self.lr),
            ("--epsilon", self.epsilon),
            ("--delta", self.delta),
            ("--mode", self.mode),
            ("--client", self.client),
            ("--nprocess", self.nprocess),
            ("--shuffle_model", self.shuffle_model),
            ("--iterations", self.iterations),
            ("--samples", self.samples),
            ("--methods", self.methods),
            ("--key-dim", self.key_dim),
            ("--key-hidden", self.key_hidden),
            ("--key-strength", self.key_strength),
            ("--key-init-std", self.key_init_std),
            ("--key-seed", self.key_seed),
            ("--wrong-key-seed", self.wrong_key_seed),
            ("--key-points", self.key_points),
            ("--key-coord-dim", self.key_coord_dim),
        ]:
            _append(args, flag, value)
        return args


@dataclass
class CheckpointGIAConfig:
    outdir: str = "evaluation_example_fixedsplit/checkpoint_gia_s50"
    samples: str = "0,50,100"
    iterations: int = 300
    lr: float = 0.05
    workers: int = 8
    key_inr_strength: float = 50.0
    key_inr_key_dim: int = 16
    key_inr_key_hidden: int = 8
    key_inr_init_std: float = 0.02
    key_inr_controlled_classifier: bool = False
    key_inr_key_only_classifier_bias: bool = False
    model: str = "densenet121"
    fl_checkpoint: str = "example_FL_c30_e30_fixedsplit/checkpoints/30.pt"
    inr_checkpoint: str = "example_FL_INR_s50_c30_e30_fixedsplit/checkpoints/30.pt"
    ppml_checkpoint: str = "example_PPMLOmics_e10_c30_e30_fixedsplit/checkpoints/30.pt"
    wrong_key_seed: int = 20260509
    param_scope: str = "classifier"
    methods: str = "all"
    save_every: int = 0
    tv_weight: float = 0.0
    l2_weight: float = 0.0
    restarts: int = 1
    grad_loss: str = "mse"
    sigmoid_param: bool = False
    lr_schedule: str = "none"
    release_target_gradients: bool = False
    release_grad_clip_norm: float = 1.0
    release_grad_noise_multiplier: float = 0.0
    ppml_release_grad_noise_multiplier: Optional[float] = None
    release_grad_seed: int = 20260602

    def to_args(self) -> Command:
        args: Command = []
        for flag, value in [
            ("--outdir", self.outdir),
            ("--samples", self.samples),
            ("--iterations", self.iterations),
            ("--lr", self.lr),
            ("--workers", self.workers),
            ("--key-inr-strength", self.key_inr_strength),
            ("--key-inr-key-dim", self.key_inr_key_dim),
            ("--key-inr-key-hidden", self.key_inr_key_hidden),
            ("--key-inr-init-std", self.key_inr_init_std),
            ("--model", self.model),
            ("--fl-checkpoint", self.fl_checkpoint),
            ("--inr-checkpoint", self.inr_checkpoint),
            ("--ppml-checkpoint", self.ppml_checkpoint),
            ("--wrong-key-seed", self.wrong_key_seed),
            ("--param-scope", self.param_scope),
            ("--methods", self.methods),
            ("--save-every", self.save_every),
            ("--tv-weight", self.tv_weight),
            ("--l2-weight", self.l2_weight),
            ("--restarts", self.restarts),
            ("--grad-loss", self.grad_loss),
            ("--lr-schedule", self.lr_schedule),
            ("--release-grad-clip-norm", self.release_grad_clip_norm),
            ("--release-grad-noise-multiplier", self.release_grad_noise_multiplier),
            ("--release-grad-seed", self.release_grad_seed),
        ]:
            _append(args, flag, value)
        _append(args, "--ppml-release-grad-noise-multiplier", self.ppml_release_grad_noise_multiplier)
        _append_action(args, "--key-inr-controlled-classifier", self.key_inr_controlled_classifier)
        _append_action(args, "--key-inr-key-only-classifier-bias", self.key_inr_key_only_classifier_bias)
        _append_action(args, "--sigmoid-param", self.sigmoid_param)
        _append_action(args, "--release-target-gradients", self.release_target_gradients)
        return args


@dataclass
class AttackVisualizationConfig:
    out_dir: str = "evaluation_example_fixedsplit/gia_fed_st_demo_leaky_lenet_e5_train_sample0_key10_dpeps236_ppmleps129_i9000_progress100_fairinit_lastlinear_activation_film_inr_only"
    seed: int = 0
    device: str = "auto"
    num_workers: int = 4
    model: str = "leaky_lenet"
    key_dim: int = 16
    key_hidden: int = 8
    key_strength: float = 10.0
    key_coord_dim: int = 2
    key_coord_points: int = 128
    key_coord_seed: int = 20260508
    wrong_key_coord_seed: int = 20260509
    key_coord_mode: str = "uniform"
    key_coord_constant: float = 1.0
    fixed_test_patients: str = FIXED_TEST_PATIENTS
    train_subset: int = 0
    test_subset: int = 0
    batch_size: int = 8
    eval_batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-3
    dp_clip_norm: float = 1.0
    target_split: str = "train"
    target_index: int = 0
    iters: int = 9000
    attack_lr: float = 0.1
    tv_weight: float = 3e-4
    l2_weight: float = 1e-6
    grad_loss: str = "l2+cosine"
    param_scope: str = "last_linear"
    restarts: int = 3
    save_intermediate_every: int = 100

    def to_args(self) -> Command:
        args: Command = []
        for flag, value in [
            ("--out-dir", self.out_dir),
            ("--seed", self.seed),
            ("--device", self.device),
            ("--num-workers", self.num_workers),
            ("--model", self.model),
            ("--key-dim", self.key_dim),
            ("--key-hidden", self.key_hidden),
            ("--key-strength", self.key_strength),
            ("--key-coord-dim", self.key_coord_dim),
            ("--key-coord-points", self.key_coord_points),
            ("--key-coord-seed", self.key_coord_seed),
            ("--wrong-key-coord-seed", self.wrong_key_coord_seed),
            ("--key-coord-mode", self.key_coord_mode),
            ("--key-coord-constant", self.key_coord_constant),
            ("--fixed-test-patients", self.fixed_test_patients),
            ("--train-subset", self.train_subset),
            ("--test-subset", self.test_subset),
            ("--batch-size", self.batch_size),
            ("--eval-batch-size", self.eval_batch_size),
            ("--epochs", self.epochs),
            ("--lr", self.lr),
            ("--dp-clip-norm", self.dp_clip_norm),
            ("--target-split", self.target_split),
            ("--target-index", self.target_index),
            ("--iters", self.iters),
            ("--attack-lr", self.attack_lr),
            ("--tv-weight", self.tv_weight),
            ("--l2-weight", self.l2_weight),
            ("--grad-loss", self.grad_loss),
            ("--param-scope", self.param_scope),
            ("--restarts", self.restarts),
            ("--save-intermediate-every", self.save_intermediate_every),
        ]:
            _append(args, flag, value)
        return args


@dataclass
class SpatialPipelineConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    checkpoint_gia: CheckpointGIAConfig = field(default_factory=CheckpointGIAConfig)
    attack_visualization: AttackVisualizationConfig = field(default_factory=AttackVisualizationConfig)


def build_training_command(config: Optional[TrainingConfig] = None, python_executable: Optional[str] = None) -> Command:
    return _script_command("train.py", (config or TrainingConfig()).to_args(), python_executable)


def build_visualization_command(config: Optional[VisualizationConfig] = None, python_executable: Optional[str] = None) -> Command:
    return _script_command("visualize.py", (config or VisualizationConfig()).to_args(), python_executable)


def build_attack_command(config: Optional[AttackConfig] = None, python_executable: Optional[str] = None) -> Command:
    return _script_command("attack.py", (config or AttackConfig()).to_args(), python_executable)


def build_checkpoint_gia_command(config: Optional[CheckpointGIAConfig] = None, python_executable: Optional[str] = None) -> Command:
    return _script_command("checkpoint_gia.py", (config or CheckpointGIAConfig()).to_args(), python_executable)


def build_attack_visualization_command(config: Optional[AttackVisualizationConfig] = None, python_executable: Optional[str] = None) -> Command:
    return _script_command("inr_gia.py", (config or AttackVisualizationConfig()).to_args(), python_executable)


def run_training(config: Optional[TrainingConfig] = None, dry_run: bool = False, check: bool = True) -> RunResult:
    return _run_command(build_training_command(config), dry_run=dry_run, check=check)


def run_visualization(config: Optional[VisualizationConfig] = None, dry_run: bool = False, check: bool = True) -> RunResult:
    return _run_command(build_visualization_command(config), dry_run=dry_run, check=check)


def run_attack(config: Optional[AttackConfig] = None, dry_run: bool = False, check: bool = True) -> RunResult:
    return _run_command(build_attack_command(config), dry_run=dry_run, check=check)


def run_checkpoint_gia(config: Optional[CheckpointGIAConfig] = None, dry_run: bool = False, check: bool = True) -> RunResult:
    return _run_command(build_checkpoint_gia_command(config), dry_run=dry_run, check=check)


def run_attack_visualization(config: Optional[AttackVisualizationConfig] = None, dry_run: bool = False, check: bool = True) -> RunResult:
    return _run_command(build_attack_visualization_command(config), dry_run=dry_run, check=check)


def run_full_pipeline(
    config: Optional[SpatialPipelineConfig] = None,
    run_train: bool = True,
    run_visualize: bool = True,
    run_idlg: bool = False,
    run_gia: bool = False,
    run_attack_visualize: bool = False,
    dry_run: bool = False,
    check: bool = True,
) -> Dict[str, RunResult]:
    config = config or SpatialPipelineConfig()
    outputs: Dict[str, RunResult] = {}
    if run_train:
        outputs["training"] = run_training(config.training, dry_run=dry_run, check=check)
    if run_visualize:
        outputs["visualization"] = run_visualization(config.visualization, dry_run=dry_run, check=check)
    if run_idlg:
        outputs["attack"] = run_attack(config.attack, dry_run=dry_run, check=check)
    if run_gia:
        outputs["checkpoint_gia"] = run_checkpoint_gia(config.checkpoint_gia, dry_run=dry_run, check=check)
    if run_attack_visualize:
        outputs["attack_visualization"] = run_attack_visualization(config.attack_visualization, dry_run=dry_run, check=check)
    return outputs
