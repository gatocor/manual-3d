
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._reader import napari_get_reader
from ._sample_data import make_sample_data
from ._writer import write_multiple, write_single_image
from ._widget import Parabola, Box, LoadData, LoadVectorfield, MakeMovie, SetUpTracking, LoadTracking, ManualTracking#ExampleQWidget, ImageThreshold, threshold_autogenerate_widget, threshold_magic_widget, activate_widget
from .predictor import *

__all__ = (
    "napari_get_reader",
    "write_single_image",
    "write_multiple",
    "make_sample_data",
    "change_voxel_size",
    "Parabola",
    "Box",
    "LoadData",
    "LoadVectorfield",
    "MakeMovie",
    "SetUpTracking",
    "LoadTracking",
    "ManualTracking",
    # "ExampleQWidget",
    # "ImageThreshold",
    # "threshold_autogenerate_widget",
    # "threshold_magic_widget",
    # "activate_widget"
)
