# from pympler import asizeof
# from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
# try:
#     from reprlib import repr
# except ImportError:
#     pass
import os
import platform
from subprocess import check_output
import re
from PIL import ImageFont
import cv2
import yaml
import numpy as np
import glob
import urllib
import torch
from pathlib import Path
import pkg_resources as pkg
from utils import ROOT
from utils.mix_utils import make_divisible
from utils.logger import print_log, colorstr

def try_except(func):
    # try-except function. Usage: @try_except decorator
    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(e)

    return handler

def is_writeable(dir, test=False):
    # Return True if directory has write permissions, test opening a file with write permissions if test=True
    if test:  # method 1
        file = Path(dir) / 'tmp.txt'
        try:
            with open(file, 'w'):  # open file with write permissions
                pass
            file.unlink()  # remove file
            return True
        except OSError:
            return False
    else:  # method 2
        return os.access(dir, os.R_OK)  # possible issues on Windows


def is_docker():
    # Is environment a Docker container?
    return Path('/workspace').exists()  # or Path('/.dockerenv').exists()


def is_colab():
    # Is environment a Google Colab instance?
    try:
        import google.colab
        return True
    except ImportError:
        return False


def is_pip():
    # Is file in a pip package?
    return 'site-packages' in Path(__file__).resolve().parts


def is_ascii(s=''):
    # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode('ascii', 'ignore')) == len(s)


def is_chinese(s='人工智能'):
    # Is string composed of any Chinese characters?
    return re.search('[\u4e00-\u9fff]', s)

def file_size(path):
    # Return file/dir size (MB)
    mb = 1 << 20  # bytes to MiB (1024 ** 2)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
    else:
        return 0.0



def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

            ##### Example call #####
            d = dict(a=1, b=2, c=3, d=[4,5,6,7], e='a string of chars')
            print(total_size(d, verbose=True))

            class D():
                def __init__(self):
                    self.a = 1
                    self.b = [12, 2, 5, 4]
                    self.c = 'dfafdfa'
                    self.d = {'a': 8, 'dfa': 'df'}
            print(total_size(D, verbose=True))

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        if isinstance(o, np.ndarray):
            s = o.nbytes
        else:
            s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o))#, file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        else:
            # if not hasattr(o.__class__, '__slots__'):
            #     if hasattr(o, '__dict__'):
            #         s += sizeof(
            #             o.__dict__)  # no __slots__ *usually* means a __dict__, but some special builtin classes (such as `type(None)`) have neither
            #     # else, `o` has no attributes at all, so sys.getsizeof() actually returned the correct value
            # else:
            #     s += sum(sizeof(getattr(o, x)) for x in o.__class__.__slots__ if hasattr(o, x))
            if hasattr(o, '__dict__'):
                s += sizeof(o.__dict__)
        return s

    return sizeof(o)



def var_size(var):
    # obj.__sizeof__  对象内置属性
    # sys.getsizeof(obj) sys模块
    # total_size(obj) python 内置函数
    # size = getsizeof(var)  # 变量本身占用的内存大小，不包括变量引用的其他对象所占用的内存,
    # # 只会计算容器本身占用的内存大小，而不会计算其元素或键值对所占用的内存大小
    size = total_size(var)
    # print(size)
    assert isinstance(size, int)
    if size <=1024:
        return f'use memory of {round(size/1024, 2)} KB'
    elif size <=1024**2:
        return f'use memory of {round(size/1024**2, 2)} MB'
    else:
        return f'use memory of {round(size/1024**3, 2)} GB'

def check_font(font='Arial.ttf', size=10):
    # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
    font = Path(__file__).parent/font
    cfg = {'Windows': 'AppData/Roaming', 'Linux': '.config', 'Darwin': 'Library/Application Support'}  # 3 OS dirs
    path = Path.home() / cfg.get(platform.system(), '')  # OS-specific config dir
    font = font if font.exists() else (path / font.name)
    return ImageFont.truetype(str(font) if font.exists() else font.name, size)


def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str

def check_online():
    # Check internet connectivity
    import socket
    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
        return True
    except OSError:
        return False

@try_except
def check_requirements(requirements=ROOT / 'requirements.txt', exclude=(), install=True):
    # Check installed dependencies meet requirements (pass *.txt file or list of packages)
    prefix = colorstr('red', 'bold', 'requirements:')
    check_python()  # check python version
    if isinstance(requirements, (str, Path)):  # requirements.txt file
        file = Path(requirements)
        assert file.exists(), f"{prefix} {file.resolve()} not found, check failed."
        with file.open() as f:
            requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(f) if x.name not in exclude]
    else:  # list or tuple of packages
        requirements = [x for x in requirements if x not in exclude]

    n = 0  # number of packages updates
    for r in requirements:
        try:
            pkg.require(r)
        except Exception as e:  # DistributionNotFound or VersionConflict if requirements not met
            s = f"{prefix} {r} not found and is required by YOLOv5"
            if install:
                print(f"{s}, attempting auto-update...")
                try:
                    assert check_online(), f"'pip install {r}' skipped (offline)"
                    print(check_output(f"pip install '{r}'", shell=True).decode())
                    n += 1
                except Exception as e:
                    print(f'{prefix} {e}')
            else:
                print(f'{s}. Please install and rerun your command.')

    if n:  # if packages updated
        source = file.resolve() if 'file' in locals() else requirements
        s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n" \
            f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        print(emojis(s))

def check_python(minimum='3.6.2'):
    # Check current python version vs. required python version
    check_version(platform.python_version(), minimum, name='Python ', hard=True)

def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    if hard:  # assert min requirements met
        assert result, f'{name}{minimum} required by ZJDet, but {name}{current} is currently installed'
    else:
        return result

def check_img_size(imgsz, s=32, floor=0, logger=None):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        new_size = tuple([max(make_divisible(x, int(s)), floor) for x in imgsz])
    if new_size != imgsz:
        print_log(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}', logger)
    return new_size

def check_imshow():
    # Check if environment supports image displays
    try:
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False

def check_suffix(file='zjdets16.pt', suffix=('.pt',), msg=''):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"

def check_yaml(file, suffix=('.yaml', '.yml')):
    # Search/download YAML file (if necessary) and return path, checking suffix
    return check_file(file, suffix)

def check_file(file, suffix=''):
    # Search/download file (if necessary) and return path
    check_suffix(file, suffix)  # optional
    file = str(file)  # convert to str()
    if Path(file).is_file() or file == '':  # exists
        return file
    elif file.startswith(('http:/', 'https:/')):  # download
        url = str(Path(file)).replace(':/', '://')  # Pathlib turns :// -> :/
        file = Path(urllib.parse.unquote(file).split('?')[0]).name  # '%2F' to '/', split https://url.com/file.txt?auth
        if Path(file).is_file():
            print(f'Found {url} locally at {file}')  # file already exists
        else:
            print(f'Downloading {url} to {file}...')
            torch.hub.download_url_to_file(url, file)
            assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'  # check
        return file
    else:  # search
        files = []
        files.extend(glob.glob(str(ROOT / 'configs' / '**' / file), recursive=True))  # find file
        assert len(files), f'File not found: {file}'  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file

def check_dataset(data):
    extract_dir = ''

    # Read yaml
    if isinstance(data, (str, Path)):
        with open(data, errors='ignore') as f:
            data = yaml.safe_load(f)  # dictionary

    # Parse yaml
    path = extract_dir or Path(data.get('path') or '')  # optional 'path' default to '.'
    for k in 'train', 'val', 'test':
        if data.get(k):  # prepend path
            data[k] = str(path / data[k]) if isinstance(data[k], str) else [str(path / x) for x in data[k]]

    assert 'nc' in data, "Dataset 'nc' key missing."
    if 'names' not in data:
        data['names'] = [f'class{i}' for i in range(data['nc'])]  # assign class names if missing
    train, val, test = (data.get(x) for x in ('train', 'val', 'test'))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            raise Exception('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in val if not x.exists()])
    return data  # dictionary

def load_args(opt):
    if opt.opt_file is not None:
        opt.opt_file = check_yaml(opt.opt_file)  # check YAML
        # opt is a Namespace class, opt = Namespace()
        file = getattr(opt, 'opt_file', None)

        if file is not None:
            with open(file, errors='ignore') as f:
                args = yaml.safe_load(f)  # load hyps dict
            for k, v in args.items():
                setattr(opt, k, v)


# if __name__ == '__main__':
    # d = dict(a=1, b=2, c=3, d=[4, [5,456,455,2332,'dafadf'], 6, 7], e='a string of chars', f = np.array([1., 3, 5,7]))
    # print(total_size(d, verbose=True))
    # print(d['f'].nbytes,  getsizeof(d['f']))  #32 , 144
    # print(var_size(d))
    # class A():
    #     __slots__ = ()
    #
    # class D(A):
    #     def __init__(self):
    #         self.a = 1
    #         self.b = [12, 2, 5, 4]
    #         self.c = 'dfafdfa'
    #         self.d = {'a': 8, 'dfa': 'df'}
    # a = D()
    # print(total_size(a, verbose=True))


