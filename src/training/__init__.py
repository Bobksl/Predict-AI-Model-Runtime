"""Training components: ranking metrics, losses, CV utilities, train loop.

Score convention (pinned project-wide, see docs/TEAM.md): the model outputs a
*slowness* score — **higher score = slower predicted runtime**. Submissions sort
scores ascending (fastest first). Labels (`config_runtime`) arrive as exact int64
and are cast to float only after differencing/rank extraction.
"""
from .metrics import opa, opa_from_batch
from .losses import pairwise_hinge_loss, listmle_loss
from .cv import make_fold_files
from .train_loop import train_from_config
from .gst import gst_forward, select_gst_window

__all__ = ["opa", "opa_from_batch", "pairwise_hinge_loss", "listmle_loss",
           "make_fold_files", "train_from_config",
           "gst_forward", "select_gst_window"]
