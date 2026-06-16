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

    elif name == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["epochs"],
            **params
        )

    elif name == "none":
        return None

    else:
        raise ValueError(f"Unknown scheduler: {name}")