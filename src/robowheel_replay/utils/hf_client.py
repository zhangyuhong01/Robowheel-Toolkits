import os
from huggingface_hub import hf_hub_download, snapshot_download
from loguru import logger

class HFLoader:
    """
    Helper to download/cache files and folders from Hugging Face Hub.
    Enforces a strict local data_root structure.
    """
    
    @staticmethod
    def get_file(repo_id: str, filename: str, local_dir: str, subfolder: str = None, repo_type: str = "dataset") -> str:
        """
        Download a single file (e.g., .hdf5) to the local data_root anchor.
        
        Args:
            repo_id: HF Repo ID (e.g. "HORA-DB/HORA")
            filename: Path to file in repo (e.g. "SingleArm/.../chunk_0.hdf5")
            local_dir: The local anchor (data_root), e.g., "./data"
        """
        target_path = os.path.join(local_dir, filename)
        logger.info(f"Syncing file {repo_id}/{filename} to {target_path}...")
        
        try:
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder,
                repo_type=repo_type,
                local_dir=local_dir
            )

            logger.success(f"File ready: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to download file from HF: {e}")
            raise e

    @staticmethod
    def sync_assets(repo_id: str, remote_dir: str, local_dir: str, repo_type: str = "dataset") -> str:
        """
        Download a specific directory (assets) from the repo to the local data_root anchor.
        
        Args:
            remote_dir: Directory in repo (e.g. "assets")
            local_dir: The local anchor (data_root), e.g., "./data"
        """
        logger.info(f"Syncing folder {repo_id}/{remote_dir} to {local_dir}/{remote_dir}...")
        
        try:
            download_path = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                allow_patterns=f"{remote_dir}/*", # Only download assets folder
                local_dir=local_dir
            )
            
            full_path = os.path.join(download_path, remote_dir)
            
            # 双重检查
            if not os.path.exists(full_path):
                logger.warning(f"Sync finished but path seems missing: {full_path}")

            logger.success(f"Assets synced. Ready at: {full_path}")
            return full_path
            
        except Exception as e:
            logger.error(f"Failed to sync assets: {e}")
            raise e