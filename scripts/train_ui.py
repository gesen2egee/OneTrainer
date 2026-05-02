from util.import_util import script_imports

script_imports()

import logging
import threading
import warnings

from modules.ui.TrainUI import TrainUI


_TORCH_LOG_DEDUPE_LOCK = threading.Lock()
_TORCH_LOG_SEEN: set[tuple[str, str]] = set()
_PY_WARN_DEDUPE_LOCK = threading.Lock()
_PY_WARN_SEEN: set[str] = set()
_ORIG_SHOWWARNING = warnings.showwarning


class _TorchWarningDedupFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING:
            return True
        logger_name = record.name or ""
        if not logger_name.startswith("torch"):
            return True

        msg = record.getMessage().strip()
        key = (logger_name, msg)
        with _TORCH_LOG_DEDUPE_LOCK:
            if key in _TORCH_LOG_SEEN:
                return False
            _TORCH_LOG_SEEN.add(key)

        record.msg = f"warning new tensor warning: {msg}"
        record.args = ()
        return True


def _dedup_showwarning(message, category, filename, lineno, file=None, line=None):
    text = str(message).strip()
    is_torch_warning = "torch" in filename.lower() or "dynamo" in text.lower() or "inductor" in text.lower()
    if not is_torch_warning:
        return _ORIG_SHOWWARNING(message, category, filename, lineno, file=file, line=line)

    with _PY_WARN_DEDUPE_LOCK:
        if text in _PY_WARN_SEEN:
            return
        _PY_WARN_SEEN.add(text)

    _ORIG_SHOWWARNING(f"warning new tensor warning: {text}", category, filename, lineno, file=file, line=line)


def _install_warning_dedupe():
    warnings.showwarning = _dedup_showwarning
    warnings.filterwarnings("once", module=r"torch\..*")
    dedupe_filter = _TorchWarningDedupFilter()
    for logger_name in (
        "torch",
        "torch._dynamo",
        "torch._inductor",
        "torch.fx.experimental.symbolic_shapes",
    ):
        logging.getLogger(logger_name).addFilter(dedupe_filter)


def main():
    _install_warning_dedupe()
    ui = TrainUI()
    ui.mainloop()


if __name__ == '__main__':
    main()
