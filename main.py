import random
import numpy as np
import torch

from models import TrainTask, model_dict


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_options():
    default_parser = TrainTask.build_default_options()
    default_opt, unknown_opt = default_parser.parse_known_args()

    model_name = default_opt.model_name.lower()
    if model_name not in model_dict:
        available = ", ".join(sorted(model_dict.keys()))
        raise ValueError(f"Unknown model_name `{default_opt.model_name}`. Available: {available}")

    model_cls = model_dict[model_name]
    private_parser = model_cls.build_options()
    opt = private_parser.parse_args(unknown_opt, namespace=default_opt)
    return opt, model_cls


if __name__ == "__main__":
    options, model_cls = parse_options()
    set_seed(options.seed)
    trainer = model_cls(options)
    trainer.fit()
