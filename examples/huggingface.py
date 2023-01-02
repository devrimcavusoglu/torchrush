import os
from typing import Union

from torch.nn import CrossEntropyLoss
from transformers import AutoModelForVideoClassification, AutoProcessor, PreTrainedModel

from torchrush.module.base import BaseModule


class HFModelForVideoClassification(BaseModule):
    def _init_model(self, hf_model_id: str, **kwargs):
        self.model = AutoModelForVideoClassification.from_pretrained(hf_model_id, **kwargs)
        self.processor = AutoProcessor.from_pretrained(hf_model_id, **kwargs)

    def preprocess(self, inputs):
        return self.processor(inputs)

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, y_pred, y_true):
        output, references = y_pred, y_true
        loss = CrossEntropyLoss(output.logists, references)
        return loss

    def save_pretrained_hf(
        self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs
    ):
        self.model.save_pretrained(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)
        self.processor.save_pretrained(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)


if __name__ == "__main__":
    # init rush huggingface model
    hf_model_id = "MCG-NJU/videomae-base-finetuned-kinetics"
    rush_hf_model = HFModelForVideoClassification(hf_model_id=hf_model_id)

    # export in huggingface format
    save_dir = "examples/huggingface"
    rush_hf_model.save_pretrained_hf(save_dir)

    # export in rush format
    save_dir = "examples/huggingface/rush"
    rush_hf_model.save_pretrained(save_directory=save_dir)

    # load from exported rush format
    rush_hf_model = HFModelForVideoClassification.from_pretrained(save_dir)

    # load in auto rush style
    from torchrush.module.auto import AutoRush

    rushmodule = AutoRush.from_pretrained(
        save_dir,
    )
