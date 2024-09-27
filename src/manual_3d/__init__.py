
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._reader import napari_get_reader
from ._sample_data import make_sample_data
from ._writer import write_multiple, write_single_image
from ._widget import SetUpTracking, LoadTracking, FindPeaks#ExampleQWidget, ImageThreshold, threshold_autogenerate_widget, threshold_magic_widget, activate_widget

__all__ = (
    "napari_get_reader",
    "write_single_image",
    "write_multiple",
    "make_sample_data",
    "change_voxel_size",
    "SetUpTracking",
    "LoadTracking",
    "FindPeaks",
    # "ExampleQWidget",
    # "ImageThreshold",
    # "threshold_autogenerate_widget",
    # "threshold_magic_widget",
    # "activate_widget"
)
