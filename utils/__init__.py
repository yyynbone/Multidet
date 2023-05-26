from pathlib import Path
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory parents[0] is utils  parents[1] is zjdet
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

# 使用package模式的绝对路径，不要使用相对路径 eg from .calculate import *  $error:attempted relative import with no known parent package
from utils.calculate import *
from utils.label_process import *
from utils.mix_utils import *
from utils.plots import *
from utils.logger import *
from utils.check import *
from utils.lr_schedule import *
from utils.torch_utils import *
from utils.callback import *
from utils.cal_coco import *
from utils.analyze import *
from utils.tal import *
from utils.flops_counter import *