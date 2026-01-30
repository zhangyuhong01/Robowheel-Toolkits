import pkgutil
import importlib
import os
import sys
from loguru import logger

sys.dont_write_bytecode = True 

# --- Constants ---
SUB_MODULES = ["hora"] 

# --- Logger Setup ---
reg_logger = logger.bind(module="ConfigRegistry")

package_path = os.path.dirname(__file__)
base_package_name = __name__

loaded_modules = []

# --- Auto-scanning Logic ---
for sub_mod in SUB_MODULES:
    sub_path = os.path.join(package_path, sub_mod)
    
    if not os.path.exists(sub_path):
        continue

    prefix = f"{base_package_name}.{sub_mod}."
    
    for _, name, _ in pkgutil.walk_packages([sub_path], prefix=prefix):
        try:
            importlib.import_module(name)
            loaded_modules.append(name)
            reg_logger.opt(colors=True).info(f"Auto-registered config: <cyan>{name}</cyan>")
        except Exception as e:
            reg_logger.opt(colors=True).error(f"Failed to load <bold>{name}</bold>: {e}")

# --- Summary ---
if loaded_modules:
    reg_logger.opt(colors=True).success(f"Registry ready. Loaded modules: <green>{len(loaded_modules)}</green>")
else:
    reg_logger.warning("Registry initialized but NO config modules were found.")

from .registry import get_config_class