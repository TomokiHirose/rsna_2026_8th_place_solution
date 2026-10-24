import os
from pathlib import Path

import click
import pytest
from click.testing import CliRunner, Result

from rsna_knee.config import Config

REPO_ROOT = Path(__file__).resolve().parent.parent

# Tests train on the CPU; Lightning reads Trainer defaults from PL_TRAINER_*
os.environ["PL_TRAINER_ACCELERATOR"] = "cpu"


def invoke(command: click.Command, args: list[object]) -> Result:
    """Run a click command in-process; exceptions other than SystemExit propagate."""
    return CliRunner().invoke(command, [str(arg) for arg in args], catch_exceptions=False)


@pytest.fixture
def tiny_cfg() -> Config:
    """A small offline resnet18 Window model that builds in about a second."""
    return Config(
        experiment_name="pytest",
        wandb_mode="disabled",
        data={"batch_size": 2, "num_workers": 0, "pin_memory": False},
        model={"backbone": "resnet18", "pretrained": False, "img_size": 32, "window": {"k_train": 3, "k_val": 4, "k_eval": 6}},
        train={"epochs": 1, "precision": "32-true", "grad_checkpointing": False, "rank_weight": 0.0, "rank_queue": 0},
    )
