"""
Logger copied from OpenAI baselines to avoid extra RL-based dependencies:
https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/logger.py
"""

import os
from pathlib import Path
import sys
import os.path as osp
import time
import datetime
from collections import defaultdict
from contextlib import contextmanager
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torch


DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50


class KVWriter(object):
    def writekvs(self, kvs):
        raise NotImplementedError


class SeqWriter(object):
    def writeseq(self, seq):
        raise NotImplementedError


class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file, suffix=""):
        self.suffix = suffix
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            self.own_file = True
        else:
            assert hasattr(filename_or_file, "read"), (
                "expected file or str, got %s" % filename_or_file
            )
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = {}
        eval_key2str = {}
        for key, val in sorted(kvs.items()):
            if hasattr(val, "__float__"):
                valstr = "%-8.4g" % val
            else:
                valstr = str(val)
            if key.startswith("eval/"):
                eval_key2str[self._truncate(key)] = self._truncate(valstr)
            else:
                key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            print("WARNING: tried to write empty key-value dict")
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        if eval_key2str:
            keywidth = max(keywidth, max(map(len, eval_key2str.keys())))
            valwidth = max(valwidth, max(map(len, eval_key2str.values())))

        # Write out the data
        dash_len = keywidth + valwidth + len(self.suffix)
        dashes = "-" * (dash_len - 1)
        suffix_dashes = (
            "-" * ((dash_len - len(self.suffix)) // 2)
            + self.suffix
            + "-" * ((dash_len - len(self.suffix)) // 2)
        )
        lines = [suffix_dashes]
        for key, val in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
            lines.append(
                "| %s%s | %s%s |"
                % (key, " " * (keywidth - len(key)), val, " " * (valwidth - len(val)))
            )
        if eval_key2str:
            eval_dashes = (
                "|"
                + "*" * ((dash_len - len("EVAL")) // 2 - 1)
                + "EVAL"
                + "*" * ((dash_len - len("EVAL")) // 2 - 1)
                + "|"
            )
            lines.append(eval_dashes)
            for key, val in eval_key2str.items():
                lines.append(
                    "| %s%s | %s%s |"
                    % (
                        key,
                        " " * (keywidth - len(key)),
                        val,
                        " " * (valwidth - len(val)),
                    )
                )
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s):
        maxlen = 30
        return s[: maxlen - 3] + "..." if len(s) > maxlen else s

    def writeseq(self, seq):
        seq = list(seq)
        for i, elem in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1:  # add space unless this is the last one
                self.file.write(" ")
        self.file.write("\n")
        self.file.flush()

    def close(self):
        if self.own_file:
            self.file.close()


class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "w+t")
        self.keys = []
        self.sep = ","

    def writekvs(self, kvs):
        # Add our current row to the history
        extra_keys = list(kvs.keys() - self.keys)
        extra_keys.sort()
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for i, k in enumerate(self.keys):
                if i > 0:
                    self.file.write(",")
                self.file.write(k)
            self.file.write("\n")
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write("\n")
        for i, k in enumerate(self.keys):
            if i > 0:
                self.file.write(",")
            v = kvs.get(k)
            if v is not None:
                self.file.write(str(v))
        self.file.write("\n")
        self.file.flush()

    def close(self):
        self.file.close()


class TensorBoardOutputFormat(KVWriter):
    """
    Dumps key/value pairs into TensorBoard's numeric format.
    """

    def __init__(self, dir):
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self.step = 1

        self.writer = SummaryWriter(dir)

    def writekvs(self, kvs):
        for k, v in kvs.items():
            self.writer.add_scalar(k, v, self.step)
        self.writer.flush()
        self.step += 1

    def log_histogram(self, key, values, step=None):
        step = step or self.step
        self.writer.add_histogram(key, values, step)

    def close(self):
        if self.writer:
            self.writer.close()
            self.writer = None


def make_output_format(format, ev_dir, log_suffix=""):
    os.makedirs(ev_dir, exist_ok=True)
    if format == "stdout":
        return HumanOutputFormat(sys.stdout, suffix=log_suffix)
    elif format == "log":
        return HumanOutputFormat(
            osp.join(ev_dir, "log%s.txt" % log_suffix), suffix=log_suffix
        )
    elif format == "csv":
        return CSVOutputFormat(osp.join(ev_dir, "progress%s.csv" % log_suffix))
    elif format == "tensorboard":
        return TensorBoardOutputFormat(osp.join(ev_dir, "tb%s" % log_suffix))
    else:
        raise ValueError("Unknown format specified: %s" % (format,))


# ================================================================
# API
# ================================================================
def log_param(key, param, step=None, log_frequency=None):
    """
    Log a parameter (e.g. a neural network weight)
    """
    return get_current().log_param(key, param, step, log_frequency)


def log_histogram(key, values, step=None):
    """
    Log a histogram of values.
    """
    return get_current().log_histogram(key, values, step)


def dump(step=None):
    """
    Write all of the diagnostics from the current iteration
    """
    return get_current().dump(step)


def logkv(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.
    """
    get_current().logkv(key, val)


def logkv_mean(key, val, step=None, n=1, log_frequency=1):
    """
    The same as logkv(), but if called many times, values averaged.
    """
    current = get_current()
    if isinstance(current, SACSVGLogger):
        current.logkv_mean(key, val, step, n, log_frequency)
    elif isinstance(current, Logger):
        current.logkv_mean(key, val)


def logkvs(d):
    """
    Log a dictionary of key-value pairs
    """
    for k, v in d.items():
        logkv(k, v)


def dumpkvs():
    """
    Write all of the diagnostics from the current iteration
    """
    return get_current().dumpkvs()


def getkvs():
    return get_current().name2val


def log(*args, level=INFO):
    """
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
    get_current().log(*args, level=level)


def debug(*args):
    log(*args, level=DEBUG)


def info(*args):
    log(*args, level=INFO)


def warn(*args):
    log(*args, level=WARN)


def error(*args):
    log(*args, level=ERROR)


def set_level(level):
    """
    Set logging threshold on current logger.
    """
    get_current().set_level(level)


def set_comm(comm):
    get_current().set_comm(comm)


def get_dir():
    """
    Get directory that log files are being written to.
    will be None if there is no output directory (i.e., if you didn't call start)
    """
    return get_current().get_dir()


record_tabular = logkv
dump_tabular = dumpkvs


@contextmanager
def profile_kv(scopename):
    logkey = "wait_" + scopename
    tstart = time.time()
    try:
        yield
    finally:
        get_current().name2val[logkey] += time.time() - tstart


def profile(n):
    """
    Usage:
    @profile("my_func")
    def my_func(): code
    """

    def decorator_with_name(func):
        def func_wrapper(*args, **kwargs):
            with profile_kv(n):
                return func(*args, **kwargs)

        return func_wrapper

    return decorator_with_name


# ================================================================
# Backend
# ================================================================


def get_current():
    logger = Logger.CURRENT or SACSVGLogger.CURRENT
    if logger is None:
        _configure_default_logger()
        logger = Logger.CURRENT
    return logger


class Logger(object):
    DEFAULT = None  # A logger with no output files. (See right below class definition)
    # So that you can still log to the terminal without setting up any output files
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, dir, output_formats, comm=None):
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats
        self.comm = comm

    # Logging API, forwarded
    # ----------------------------------------
    def log_param(self, key, param, step=None, log_frequency=None):
        self.log_histogram(key + "_w", param.weight.data, step)
        if hasattr(param.weight, "grad") and param.weight.grad is not None:
            self.log_histogram(key + "_w_g", param.weight.grad.data, step)
        if hasattr(param, "bias") and hasattr(param.bias, "data"):
            self.log_histogram(key + "_b", param.bias.data, step)
            if hasattr(param.bias, "grad") and param.bias.grad is not None:
                self.log_histogram(key + "_b_g", param.bias.grad.data, step)

    def log_historgram(self, key, values, step=None):
        for fmt in self.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                fmt.log_histogram(key, values, step)

    def logkv(self, key, val):
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
        self.name2cnt[key] = cnt + 1

    def dump(self, step=None):
        step = step or self.name2val["step"]
        if step is None:
            raise ValueError("Must specify step")
        if step != self.name2val["step"]:
            self.logkv("step", step)
        self.dumpkvs()

    def dumpkvs(self):
        d = self.name2val
        out = d.copy()  # Return the dict for unit testing purposes
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(d)
        self.name2val.clear()
        self.name2cnt.clear()
        return out

    def log(self, *args, level=INFO):
        if self.level <= level:
            self._do_log(args)

    # Configuration
    # ----------------------------------------
    def set_level(self, level):
        self.level = level

    def set_comm(self, comm):
        self.comm = comm

    def get_dir(self):
        return self.dir

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.writeseq(map(str, args))


def configure(dir=None, format_strs=["log", "stdout"], comm=None, log_suffix=""):
    """
    If comm is provided, average all numerical stats across that comm
    """
    if dir is None:
        dir = os.getenv("LOGDIR")
    if dir is None:
        dir = osp.join(
            Path().resolve(),
            "log",
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"),
        )
    assert isinstance(dir, str)
    dir = os.path.expanduser(dir)
    os.makedirs(os.path.expanduser(dir), exist_ok=True)

    format_strs = filter(None, format_strs)
    output_formats = [make_output_format(f, dir, log_suffix) for f in format_strs]

    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats, comm=comm)
    if output_formats:
        log("Logging to %s" % dir)


def _configure_default_logger():
    configure()
    Logger.DEFAULT = Logger.CURRENT


def reset():
    if Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT.close()
        Logger.CURRENT = Logger.DEFAULT
        log("Reset logger")


@contextmanager
def scoped_configure(dir=None, format_strs=None, comm=None):
    prevlogger = Logger.CURRENT
    configure(dir=dir, format_strs=format_strs, comm=comm)
    try:
        yield
    finally:
        Logger.CURRENT.close()
        Logger.CURRENT = prevlogger


class SACSVGLogger(Logger):
    _instance = None

    def __init__(self, dir, output_formats, log_frequency, comm=None):
        super().__init__(dir, output_formats, comm)
        self.log_frequency = log_frequency

    @classmethod
    def get_logger(cls):
        assert cls._instance is not None, "logger not configured"
        return cls._instance

    @classmethod
    def configure(
        cls,
        dir=None,
        format_strs=["stdout", "log"],
        log_frequency=10000,
        log_suffix="",
    ):
        if dir is None:
            dir = os.getenv("LOGDIR")
        if dir is None:
            dir = osp.join(
                Path().resolve(),
                "log",
                datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"),
            )
        assert isinstance(dir, str)
        dir = os.path.expanduser(dir)
        os.makedirs(os.path.expanduser(dir), exist_ok=True)

        format_strs = filter(None, format_strs)
        output_formats = [make_output_format(f, dir, log_suffix) for f in format_strs]

        if cls._instance is None:
            cls._instance = SACSVGLogger.CURRENT = SACSVGLogger(
                dir, output_formats, log_frequency
            )
        if output_formats:
            log("Logging to %s" % dir)
        return cls._instance

    def _should_log(self, step, log_frequency):
        log_frequency = log_frequency or self.log_frequency
        return step % log_frequency == 0

    def logkv_mean(self, key, value, step, n=1, log_frequency=1):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith("train") or key.startswith("eval")
        if type(value) == torch.Tensor:
            value = value.item()
        super().logkv_mean(key, value)

    def log_histogram(self, key, values, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith("train") or key.startswith("eval")
        for fmt in self.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                fmt.log_histogram(key, values, step)

    def log_param(self, key, param, step, log_frequency=None):
        if not self._should_log(step, log_frequency):
            return
        self.log_histogram(key + "_w", param.weight.data, step)
        if hasattr(param.weight, "grad") and param.weight.grad is not None:
            self.log_histogram(key + "_w_g", param.weight.grad.data, step)
        if hasattr(param, "bias") and hasattr(param.bias, "data"):
            self.log_histogram(key + "_b", param.bias.data, step)
            if hasattr(param.bias, "grad") and param.bias.grad is not None:
                self.log_histogram(key + "_b_g", param.bias.grad.data, step)

    def dump(self, step):
        self.logkv("step", step)
        self.dumpkvs()
