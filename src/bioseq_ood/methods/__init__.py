from .base import MethodSpec
from .cmd import CMDMethod
from .erm import ERMMethod, SupervisedMethod
from .fixmatch import FixMatchMethod
from .mean_teacher import MeanTeacherMethod
from .pseudo_labeling import PseudoLabelingMethod
from .registry import build_method_from_config
