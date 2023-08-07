from pytorch_lightning.callbacks import TQDMProgressBar
from tqdm import tqdm


class LitProgressBar(TQDMProgressBar):
    """
    Progress bar with disabled tqdm for Google Colab.
    """

    def init_validation_tqdm(self):
        bar = tqdm(
            disable=True,
        )
        return bar
