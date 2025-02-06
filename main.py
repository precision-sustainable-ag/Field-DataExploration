#!/usr/bin/env python3
import getpass
import logging
import sys

import hydra
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

sys.path.append("src")

log = logging.getLogger(__name__)
# Get the logger for the Azure SDK
azlogger = logging.getLogger("azure")
# Set the logging level to CRITICAL to turn off regular logging
azlogger.setLevel(logging.WARN)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run_FIELD_REPORT(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(cfg)
    whoami = getpass.getuser()

    tasks = cfg.pipeline
    log.info(f"Running {' ,'.join(tasks)} as {whoami}")

    for task in tasks:
        cfg.general.task = task
        try:
            task = get_method(f"{task}.main")
            task(cfg)

        except Exception as e:
            log.exception("Failed")
            sys.exit(1)


if __name__ == "__main__":
    run_FIELD_REPORT()
