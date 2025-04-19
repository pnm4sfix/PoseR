try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


from ._widget import PoserWidget
from ._loader import HyperParams
from ._loader import ZebData
from .utils import Animation


#from poser._widget import PoserWidget
#from poser._loader import HyperParams
#from poser._loader import ZebData
#from poser.utils import Animation

__all__ = "PoserWidget"
