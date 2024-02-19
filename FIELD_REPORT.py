#!/usr/bin/env python3
import getpass
import logging
import sys

import hydra
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

sys.path.append("volume_assessment")
sys.path.append("data_analysis")

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run_FIELD_REPORT(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(cfg)
    whoami = getpass.getuser()

    task = cfg.general.task
    log.info(f"Starting {task} as {whoami}")

    try:
        task = get_method(f"{task}.main")
        task(cfg)

    except Exception as e:
        log.exception("Failed")
        sys.exit(1)


if __name__ == "__main__":
    run_FIELD_REPORT()
