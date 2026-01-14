import h5py
import shutil
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
import os
from PIL import Image
import numpy as np
import cv2

#TODO:change your task
TASK_PROMPT="Pick up the fruit and put it on the mat."

def decode_image(dataset, index):
    raw_data = dataset[index]

    if dataset.ndim == 1 or dataset.dtype.kind == "O":
        if isinstance(raw_data, (bytes, bytearray)):
            raw_data = np.frombuffer(raw_data, dtype=np.uint8)
        img_bgr = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None, "Decode Error"
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb, "JPEG Compressed"
    if dataset.ndim == 4:
        return raw_data, "Raw Array"

    return None, f"Unknown format shape {dataset.shape}"

def main(hdf5_dir: str, *, push_to_hub: bool = False):
    hdf5_dir = Path(hdf5_dir)
    #TODO:change repo name
    REPO_NAME = "YOUR_REPO_NAME"
    HF_LEROBOT_HOME = "YOUR_HF_LEROBOT_HOME"

    output_path = Path(HF_LEROBOT_HOME) / REPO_NAME

    if output_path.exists():
        shutil.rmtree(output_path)
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="UR5",
        fps=30,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )


    for hdf5_file in hdf5_dir.glob("*.hdf5"):
        with h5py.File(hdf5_file, "r") as f:


            for demo_name in f["data"].keys():
                # import pdb;pdb.set_trace()
                demo = f["data"][demo_name]

                #TODO:check keys
                images = demo['obs']['agentview_rgb']
                wrist_images = demo['obs']['eye_in_hand_rgb']
                states = demo['robot_states'][:]
                actions = demo['actions_ee'][:].astype(np.float32)
                actions[:, 6] = (actions[:, 6] >= 0.01).astype(np.float32)

                task = TASK_PROMPT  # or f["task"] #TODO

                for i in range(images.shape[0]):
                    img_arr, img_format = decode_image(images, i)
                    wrist_arr, wrist_format = decode_image(wrist_images, i)
                    if img_arr is None or wrist_arr is None:
                        raise ValueError(
                            f"Decode failed at index {i}: image={img_format}, wrist={wrist_format}"
                        )
                    img = Image.fromarray(img_arr)
                    wrist_img = Image.fromarray(wrist_arr)
                    img_resized = img.resize((256, 256), Image.BILINEAR)
                    wrist_resized = wrist_img.resize((256, 256), Image.BILINEAR)
                    img_resized_np = np.array(img_resized)
                    wrist_resized_np = np.array(wrist_resized)

                    dataset.add_frame({
                        "image": img_resized_np,
                        "wrist_image": wrist_resized_np,
                        "state": states[i],
                        "actions": actions[i],
                        "task": task if isinstance(task, str) else task.decode(),
                    })
                
                dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            repo_id=REPO_NAME, 
            #TODO:change your tags
            tags=["TAG_ONE", "UR5", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
            repo_type="dataset",
        )

if __name__ == "__main__":
    tyro.cli(main)
