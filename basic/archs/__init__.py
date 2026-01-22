from copy import deepcopy
import ast
import importlib
import os.path as osp

from basic.options.options import parse_params, parse_arguments
from basic.utils.registry import ARCH_REGISTRY, MODULE_REGISTRY
from basic.utils.console.log import get_root_logger
from basic.utils.path import scandir

__all__ = ['define_network']

logger = get_root_logger()

# =============================================================================
# Global switch:
#   - True : import only the arch/module file that defines the requested class
#   - False: import all arch/module files at startup (original behavior)
# =============================================================================
DYNAMIC_ARCH_IMPORT = True

# Folder of this package (basic/archs)
arch_folder = osp.dirname(osp.abspath(__file__))

# Collect absolute file paths (no import here)
ARCH_FILE_CANDIDATES = [
    osp.join(arch_folder, v)
    for v in scandir(arch_folder, suffix='_arch.py', recursive=True)
]
MODULE_FILE_CANDIDATES = [
    osp.join(arch_folder, v)
    for v in scandir(arch_folder, suffix='_modules.py', recursive=True)
]

# Cache: class name -> module name (relative to basic.archs)
_ARCH_CLASS_TO_MODULE = {}
_MODULE_CLASS_TO_MODULE = {}

# Cache: imported module names
_IMPORTED_ARCH_MODULES = set()


def _file_path_to_module_name(py_file_abs_path: str) -> str:
    """
    Convert an absolute file path under this arch folder to a module path
    relative to `basic.archs`.

    Example:
        /.../basic/archs/a/b/xxx_arch.py -> a.b.xxx_arch
    """
    rel = osp.relpath(py_file_abs_path, arch_folder)
    rel_no_ext = osp.splitext(rel)[0]
    return rel_no_ext.replace(osp.sep, '.')


def _safe_import_arch_module(module_name: str) -> bool:
    """
    Safely import a module under `basic.archs.<module_name>`.
    """
    if module_name in _IMPORTED_ARCH_MODULES:
        return True
    try:
        importlib.import_module(f'basic.archs.{module_name}')
        _IMPORTED_ARCH_MODULES.add(module_name)
        return True
    except Exception as e:
        logger.warning(f'Failed to import {module_name} because of {e}')
        return False


def _index_classes_from_file(py_file_abs_path: str, class_to_module: dict):
    """
    Parse a Python file and index all top-level class names defined in it.

    Args:
        py_file_abs_path: Absolute path to the python file.
        class_to_module: Dict to store mapping: class name -> module name.
    """
    try:
        with open(py_file_abs_path, 'r', encoding='utf-8') as f:
            src = f.read()
        tree = ast.parse(src, filename=py_file_abs_path)
    except Exception as e:
        logger.warning(f'Failed to parse {py_file_abs_path} because of {e}')
        return

    module_name = _file_path_to_module_name(py_file_abs_path)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_to_module.setdefault(node.name, module_name)


def _build_arch_index_once():
    """
    Build the class -> module index for *_arch.py files.
    """
    if _ARCH_CLASS_TO_MODULE:
        return
    for fpath in ARCH_FILE_CANDIDATES:
        _index_classes_from_file(fpath, _ARCH_CLASS_TO_MODULE)


def _build_module_index_once():
    """
    Build the class -> module index for *_modules.py files.
    """
    if _MODULE_CLASS_TO_MODULE:
        return
    for fpath in MODULE_FILE_CANDIDATES:
        _index_classes_from_file(fpath, _MODULE_CLASS_TO_MODULE)


# =============================================================================
# Original behavior (non-dynamic import):
# import all arch/module modules at startup so all classes are registered
# =============================================================================
if not DYNAMIC_ARCH_IMPORT:
    for fpath in ARCH_FILE_CANDIDATES:
        _safe_import_arch_module(_file_path_to_module_name(fpath))
    for fpath in MODULE_FILE_CANDIDATES:
        _safe_import_arch_module(_file_path_to_module_name(fpath))


def _ensure_arch_registered(arch_type: str):
    """
    Ensure arch_type is registered in ARCH_REGISTRY.

    In dynamic mode:
      - find which *_arch.py defines `class <arch_type>`
      - import that module only
      - if still not found, fall back to importing all *_arch.py candidates
    """
    if ARCH_REGISTRY.try_get(arch_type) is not None:
        return
    if not DYNAMIC_ARCH_IMPORT:
        return

    _build_arch_index_once()

    module_name = _ARCH_CLASS_TO_MODULE.get(arch_type)
    if module_name is not None:
        _safe_import_arch_module(module_name)
        if ARCH_REGISTRY.try_get(arch_type) is not None:
            return

    for fpath in ARCH_FILE_CANDIDATES:
        _safe_import_arch_module(_file_path_to_module_name(fpath))
        if ARCH_REGISTRY.try_get(arch_type) is not None:
            return


def _ensure_module_registered(module_type: str):
    """
    Ensure module_type is registered in MODULE_REGISTRY.

    In dynamic mode:
      - find which *_modules.py defines `class <module_type>`
      - import that module only
      - if still not found, fall back to importing all *_modules.py candidates
    """
    if MODULE_REGISTRY.try_get(module_type) is not None:
        return
    if not DYNAMIC_ARCH_IMPORT:
        return

    _build_module_index_once()

    module_name = _MODULE_CLASS_TO_MODULE.get(module_type)
    if module_name is not None:
        _safe_import_arch_module(module_name)
        if MODULE_REGISTRY.try_get(module_type) is not None:
            return

    for fpath in MODULE_FILE_CANDIDATES:
        _safe_import_arch_module(_file_path_to_module_name(fpath))
        if MODULE_REGISTRY.try_get(module_type) is not None:
            return


def define_network(option):
    """
    Define a network based on the option.

    Args:
        option (dict): Configuration dict with a 'type' field.

    Returns:
        nn.Module: Instantiated network.
    """
    option = deepcopy(option)

    net_type, net_params = parse_params(option)

    # Ensure the arch class is registered (lazy import if needed)
    _ensure_arch_registered(net_type)

    net_class = ARCH_REGISTRY.get(net_type)
    if net_class is None:
        raise KeyError(
            f"Network type '{net_type}' is not found in ARCH_REGISTRY. "
            f"Dynamic import is {'on' if DYNAMIC_ARCH_IMPORT else 'off'}."
        )

    # Recursively instantiate sub-modules
    for key, value in net_params.items():
        if isinstance(value, dict) and 'type' in value:
            net_params[key] = define_module(value)

    args, kwargs = parse_arguments(net_class, net_params)
    net = net_class(*args, **kwargs)
    return net


def define_module(option):
    """
    Define a module based on the option.

    Args:
        option (dict): Configuration dict with a 'type' field.

    Returns:
        nn.Module: Instantiated module.
    """
    option = deepcopy(option)

    module_type, module_params = parse_params(option)

    # Ensure the module class is registered (lazy import if needed)
    _ensure_module_registered(module_type)

    module_class = MODULE_REGISTRY.get(module_type)
    if module_class is None:
        raise KeyError(
            f"Module type '{module_type}' is not found in MODULE_REGISTRY. "
            f"Dynamic import is {'on' if DYNAMIC_ARCH_IMPORT else 'off'}."
        )

    # Recursively instantiate sub-modules
    for key, value in module_params.items():
        if isinstance(value, dict) and 'type' in value:
            module_params[key] = define_module(value)

    args, kwargs = parse_arguments(module_class, module_params)
    module = module_class(*args, **kwargs)
    return module
