from copy import deepcopy
import importlib
import os.path as osp

from .util import get_scheduler, get_optimizer
from basic.utils.console.log import dict_to_str, get_root_logger
from basic.utils.registry import MODEL_REGISTRY
from basic.utils.path import scandir

__all__ = ['create_model', 'get_scheduler', 'get_optimizer']

logger = get_root_logger()

# =============================================================================
# Global switch:
#   - True : import only the module that defines the requested model class
#   - False: import all model modules at startup (original behavior)
# =============================================================================
DYNAMIC_MODEL_IMPORT = True

# Scan all model files (only collect paths here)
model_folder = osp.dirname(osp.abspath(__file__))

# Absolute file paths under models/ that end with _model.py
MODEL_FILE_CANDIDATES = [
    osp.join(model_folder, v)
    for v in scandir(model_folder, suffix='_model.py', recursive=True)
]

# Cache: class name -> module name (e.g., "MyModel" -> "foo.bar_my_model")
_CLASS_TO_MODULE = {}

# Cache: module name -> imported or not
_IMPORTED_MODEL_MODULES = set()


def _file_path_to_module_name(py_file_abs_path: str) -> str:
    """
    Convert an absolute file path under this models folder to a module path
    relative to `basic.models`.

    Example:
        /.../basic/models/a/b/xxx_model.py -> a.b.xxx_model
    """
    rel = osp.relpath(py_file_abs_path, model_folder)
    rel_no_ext = osp.splitext(rel)[0]
    # Support Windows path separator as well
    return rel_no_ext.replace(osp.sep, '.')


def _safe_import_module(module_name: str) -> bool:
    """
    Safely import a module under `basic.models.<module_name>`.
    """
    if module_name in _IMPORTED_MODEL_MODULES:
        return True
    try:
        importlib.import_module(f'basic.models.{module_name}')
        _IMPORTED_MODEL_MODULES.add(module_name)
        return True
    except Exception as e:
        logger.warning(f'Failed to import {module_name} because of {e}')
        return False


def _index_classes_from_file(py_file_abs_path: str):
    """
    Parse a Python file and index all top-level class names defined in it.
    """
    import ast
    try:
        with open(py_file_abs_path, 'r', encoding='utf-8') as f:
            src = f.read()
        tree = ast.parse(src, filename=py_file_abs_path)
    except Exception as e:
        logger.warning(f'Failed to parse {py_file_abs_path} because of {e}')
        return

    module_name = _file_path_to_module_name(py_file_abs_path)

    # Collect top-level class definitions
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            cls_name = node.name
            # Keep the first hit; if duplicates exist, the first discovered wins
            _CLASS_TO_MODULE.setdefault(cls_name, module_name)


def _build_class_index_once():
    """
    Build (or extend) the class -> module index.
    """
    if _CLASS_TO_MODULE:
        return
    for fpath in MODEL_FILE_CANDIDATES:
        _index_classes_from_file(fpath)


# =============================================================================
# Original behavior: import all modules at startup
# =============================================================================
if not DYNAMIC_MODEL_IMPORT:
    for fpath in MODEL_FILE_CANDIDATES:
        _safe_import_module(_file_path_to_module_name(fpath))


def _ensure_model_registered(model_type: str):
    """
    Ensure model_type is registered in MODEL_REGISTRY.

    In dynamic mode:
      - find which file defines `class <model_type>`
      - import that module only
      - if still not registered, fall back to importing all candidates
    """
    if MODEL_REGISTRY.try_get(model_type) is not None:
        return

    if not DYNAMIC_MODEL_IMPORT:
        return

    # 1) Build class index (no imports here)
    _build_class_index_once()

    # 2) Import the module that defines the class with the same name
    module_name = _CLASS_TO_MODULE.get(model_type)
    if module_name is not None:
        _safe_import_module(module_name)
        if MODEL_REGISTRY.try_get(model_type) is not None:
            return

    # 3) Fallback: import candidates one by one until it is registered
    #    (covers cases like: class exists but registered name differs, or class created dynamically)
    for fpath in MODEL_FILE_CANDIDATES:
        _safe_import_module(_file_path_to_module_name(fpath))
        if MODEL_REGISTRY.try_get(model_type) is not None:
            return


def create_model(option, verbose=True):
    """
    Build a model from options.

    Args:
        option (dict): must contain key 'model', whose value is the model class name.
    """
    option = deepcopy(option)

    # dynamic instantiation
    assert 'model' in option, f"The option is {dict_to_str(option, max_depth=1)} and it must contain the key 'model'."
    model_type = option['model']

    # Lazily import the module if needed (trigger registry decorators)
    _ensure_model_registered(model_type)

    model_class = MODEL_REGISTRY.get(model_type)
    if model_class is None:
        raise KeyError(
            f"Model '{model_type}' is not found in MODEL_REGISTRY. "
            f"Dynamic import is {'on' if DYNAMIC_MODEL_IMPORT else 'off'}."
        )

    model = model_class(option, verbose=verbose)
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model



