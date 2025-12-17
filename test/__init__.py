"""Unit test package for abs."""
import importlib, sys as _sys
for _name in ("test_utilities", "test_geometry"):
    try:
        _mod = importlib.import_module(f".{_name}", package=__name__)
        _sys.modules[_name] = _mod          # alias at top-level
    except ModuleNotFoundError:
        pass
