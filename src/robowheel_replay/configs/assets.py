from pathlib import Path
from loguru import logger
from typing import Dict, Optional

class AssetLibrary:
    """
    Asset Library Manager: Scans directories for USD files and builds an index.
    """
    def __init__(self, asset_root: str):
        """
        Args:
            asset_root: Absolute path to the asset root directory.
        """
        self.asset_root = Path(asset_root)
        self.asset_index: Dict[str, str] = {} # {name: usd_path}
        self.is_scanned = False
        self.logger = logger.bind(module="AssetLibrary")

    def scan(self):
        """Scan directory to build index."""
        if self.is_scanned:
            return

        self.logger.info(f"Scanning assets in: {self.asset_root}")
        
        if not self.asset_root.exists():
            self.logger.error(f"Asset directory not found: {self.asset_root}")
            return

        count = 0
        # Recursively find all USD files
        for usd_path in self.asset_root.rglob("*.usd*"):
            # Strategy: Use parent folder name as ID (e.g., assets/bottle/coke/coke.usd -> coke)
            obj_name = usd_path.parent.name 
            
            if obj_name not in self.asset_index:
                self.asset_index[obj_name] = str(usd_path)
                count += 1

        self.is_scanned = True
        self.logger.success(f"Index built. Found {count} assets.")

    def get_usd_path(self, object_name: str) -> str:
        """Get USD path, automatically scans if not done yet."""
        if not self.is_scanned:
            self.scan()
        
        path = self.asset_index.get(object_name)
        
        if not path:
            raise FileNotFoundError(f"Asset '{object_name}' not found in {self.asset_root}")
        
        return path