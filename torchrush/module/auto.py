import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union

import requests
from huggingface_hub import hf_hub_download

from torchrush.module.base import RUSH_CONFIG_NAME, RUSH_FILE_NAME, BaseModule
from torchrush.utils.common import load_class

logger = logging.getLogger(__name__)


class AutoRush:
    @staticmethod
    def from_pretrained(
        pretrained_model_name_or_path: str,
        force_download: bool = False,
        resume_download: bool = False,
        proxies: Optional[Dict] = None,
        token: Optional[Union[str, bool]] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        **model_kwargs,
    ):
        model_id = pretrained_model_name_or_path

        revision = None
        if len(model_id.split("@")) == 2:
            model_id, revision = model_id.split("@")

        rush_file: Optional[str] = None
        if os.path.isdir(model_id):
            if RUSH_FILE_NAME in os.listdir(model_id):
                rush_file = os.path.join(model_id, RUSH_FILE_NAME)
            else:
                raise FileNotFoundError(f"{RUSH_FILE_NAME} not found in {Path(model_id).resolve()}")

            if RUSH_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, RUSH_CONFIG_NAME)
            else:
                raise FileNotFoundError(f"{RUSH_CONFIG_NAME} not found in {Path(model_id).resolve()}")
        else:
            try:
                rush_file = hf_hub_download(
                    repo_id=model_id,
                    filename=RUSH_FILE_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except requests.exceptions.RequestException:
                raise FileNotFoundError(f"{RUSH_FILE_NAME} not found in HuggingFace Hub")

            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=RUSH_CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except requests.exceptions.RequestException:
                raise FileNotFoundError(f"{RUSH_CONFIG_NAME} not found in HuggingFace Hub")

        with open(config_file, "r", encoding="utf-8") as f:
            rush_config = json.load(f)
        class_name = rush_config["model"]

        rushmodule: BaseModule = load_class(class_name=class_name, filepath=rush_file)
        return rushmodule.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **model_kwargs,
        )
