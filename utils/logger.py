import logging
import numpy as np
import pandas as pd
import yaml
import re
class Filehandler(logging.FileHandler):

    def emit(self, record):
        """
        Emit a record.

        If the stream was not opened because 'delay' was specified in the
        constructor, open it before calling the superclass's emit.
        """
        if self.stream is None:
            self.stream = self._open()

        try:
            record.msg = clean_color(record.msg)
            msg = self.format(record)
            stream = self.stream
            # issue 35046: merged two stream.writes into one.
            stream.write(msg + self.terminator)
            self.flush()
        except RecursionError:  # See issue 36272
            raise
        except Exception:
            self.handleError(record)

# def set_logging(name=None, verbose=True, filename=None,rank=-1):
#     # Sets level and returns logger
#     # rank in world for Multi-GPU trainings
#     logging.basicConfig(format='[line:%(lineno)d]-%(levelname)s: %(message)s',  # 日志格式
#                         # format='%(asctime)s - %(name)s[line:%(lineno)d] - %(levelname)s: %(message)s',  # 日志格式
#                         datefmt='%F %T ',  # 日期格式,
#                         level=logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING)
#     logger = logging.getLogger(name)
#     if filename:
#         # logging.basicConfig(
#         #                     format='%(asctime)s - %(name)s[line:%(lineno)d] - %(levelname)s: %(message)s', # 日志格式
#         #                     datefmt='%F %T ', # 日期格式,
#         #                     level=logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING,
#         #                     filemode="a")
#         # basicconfig 默认添加了streamhandler
#         # console = logging.StreamHandler()
#         # logger.addHandler(console)
#
#         file_handler = logging.FileHandler(filename=filename, mode='a', encoding='utf-8')
#         logger.addHandler(file_handler)
#
#     return logger
def set_logging(name=None, verbose=True, filename=None, rank=-1):
    # Sets level and returns logger
    # rank in world for Multi-GPU trainings
    # console_format = logging.Formatter('%(filename)s %(funcName)s [line:%(lineno)d]-%(levelname)s: %(message)s') # 日志格式
    console_format = logging.Formatter('%(message)s')  # 日志格式
    log_format = logging.Formatter('%(message)s')  # 日志格式
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING)
    logger.propagate = False #防止出现 parent 里的 basic_config的streamhandler被调用
    if rank in (-1, 0):
        console = logging.StreamHandler()
        console.setFormatter(console_format)
        logger.addHandler(console)
        if filename:
            file_handler = Filehandler(filename=filename, mode='a', encoding='utf-8')
            file_handler.setFormatter(log_format)
            logger.addHandler(file_handler)
    return logger

LOGGER = set_logging(__name__)  # define globally (used in train.py, val.py, detect.py, etc.)

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def clean_color(s):
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    for val in colors.values():
        if val in s:
            s = s.replace(val, '')
    return s

def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)

def print_args(name, opt,logger=LOGGER):
    # Print argparser arguments
    print_log(f'{colorstr(name)}: ' + ', '.join(f'{k}={v}' for k, v in vars(opt).items()), logger)

def print_log(msg, logger=LOGGER):
    """Print a log message.
    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:

            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.info(msg)
    elif logger == 'silent':
        pass
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

def print_mutation(results, hyp, save_dir):
    evolve_csv, results_csv, evolve_yaml = save_dir / 'evolve.csv', save_dir / 'results.csv', save_dir / 'hyp_evolve.yaml'
    keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
            'val/box_loss', 'val/obj_loss', 'val/cls_loss') + tuple(hyp.keys())  # [results + hyps]
    keys = tuple(x.strip() for x in keys)
    vals = results + tuple(hyp.values())
    n = len(keys)

    # Log to evolve.csv
    s = '' if evolve_csv.exists() else (('%20s,' * n % keys).rstrip(',') + '\n')  # add header
    with open(evolve_csv, 'a') as f:
        f.write(s + ('%20.5g,' * n % vals).rstrip(',') + '\n')

    # Print to screen
    print(colorstr('evolve: ') + ', '.join(f'{x.strip():>20s}' for x in keys))
    print(colorstr('evolve: ') + ', '.join(f'{x:20.5g}' for x in vals), end='\n\n\n')

    from utils.calculate import fitness
    # Save yaml
    with open(evolve_yaml, 'w') as f:
        data = pd.read_csv(evolve_csv)
        data = data.rename(columns=lambda x: x.strip())  # strip keys
        i = np.argmax(fitness(data.values[:, :7]))  #
        f.write('# ZJDET Hyperparameter Evolution Results\n' +
                f'# Best generation: {i}\n' +
                f'# Last generation: {len(data) - 1}\n' +
                '# ' + ', '.join(f'{x.strip():>20s}' for x in keys[:7]) + '\n' +
                '# ' + ', '.join(f'{x:>20.5g}' for x in data.values[i, :7]) + '\n\n')
        yaml.safe_dump(hyp, f, sort_keys=False)

