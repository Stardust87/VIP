from omegaconf import OmegaConf

from configs import FAMEConfig, I2PAConfig  # noqa: I900
from configs import types as t  # noqa: I900
from src.constants import LOG_DIR, PROJECT_NAME  # noqa: I900
from src.models import LitFAME, LitI2PA  # noqa: I900


def load_checkpoint(run_id: str):
    """
    Load model and config from checkpoint based on its run id.

    Args:
        run_id: Run id of considered checkpoint.

    Returns:
        Model and its corresponding config.
    """
    checkpoint_path = LOG_DIR / PROJECT_NAME / run_id / "checkpoints" / "last.ckpt"
    config_path = LOG_DIR / PROJECT_NAME / run_id / "config.yaml"

    cfg = OmegaConf.load(config_path)

    if cfg.model.name == t.Model.FAME:
        model_cls = LitFAME
        schema = OmegaConf.structured(FAMEConfig)
    elif cfg.model.name == t.Model.I2PA:
        model_cls = LitI2PA
        schema = OmegaConf.structured(I2PAConfig)
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")

    cfg = OmegaConf.merge(schema, cfg)

    model = model_cls.load_from_checkpoint(
        checkpoint_path=checkpoint_path, cfg=cfg, strict=False
    )
    return model, cfg
