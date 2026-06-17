import math

import torch.optim.lr_scheduler as lr_scheduler


def build_scheduler(optimizer, config):
    sched_config = config["scheduler"]
    name = sched_config["name"].lower()
    params = sched_config["params"]

    if name == "reduce_on_plateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **params
        )

    elif name == "step":
        return lr_scheduler.StepLR(
            optimizer,
            **params
        )

    elif name == "cosine_annealing":
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["epochs"],
            **params
        )
    
    elif name == "cosine_warmup":

        total_epochs = config["training"]["epochs"]
        warmup_epochs = params.get("warmup_epochs", 5)
        min_lr_ratio = params.get("min_lr_ratio", 0.01)

        def lr_lambda(epoch):

            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs

            progress = (
                epoch - warmup_epochs
            ) / max(1, total_epochs - warmup_epochs)

            cosine = 0.5 * (
                1 + math.cos(math.pi * progress)
            )

            return min_lr_ratio + (1 - min_lr_ratio) * cosine

        return lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda
        )

    elif name == "none":
        return None

    else:
        raise ValueError(f"Unknown scheduler: {name}")