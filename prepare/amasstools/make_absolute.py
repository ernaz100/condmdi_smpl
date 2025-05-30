import os
import argparse
from typing import Tuple

import numpy as np
import torch

from loop_amass import loop_amams
from smplrifke_feats import smplrifkefeats_to_smpldata
from geometry import axis_angle_to_matrix, matrix_to_euler_angles

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

def _extract_abs_info(features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract absolute root height (z), 2-D translation (x, y) and yaw angle (around Z) from a
    SMPL-RIFKE feature tensor.

    Parameters
    ----------
    features : (T, 205) torch.Tensor (double / float)           
        Original relative feature representation as returned by
        :func:`smpldata_to_smplrifkefeats`.

    Returns
    -------
    root_z : (T,) torch.Tensor
        Absolute root height (already stored as first channel in the features).
    root_xy : (T, 2) torch.Tensor
        Absolute pelvis trajectory in global X / Y axes.
    yaw : (T,) torch.Tensor
        Absolute yaw (rotation around vertical Z) in radians. Angle at frame-0 is 0.
    """
    # 1) Recover smpldata in the canonical (global) space -----------------------------
    smpl = smplrifkefeats_to_smpldata(features)  # dict with keys: poses, trans, joints

    # Global translation (absolute) --------------------------------------------------
    root_xy = smpl["trans"][:, :2]  # (T, 2)  X, Y in meters

    # Global orientation -------------------------------------------------------------
    # Root axis-angle (first 3 values of the flattened pose)
    root_aa = smpl["poses"][:, :3]
    root_rot = axis_angle_to_matrix(root_aa)  # (T, 3, 3)

    # Extract Z-YX Euler angles → the very first angle is the yaw around Z
    yaw, _, _ = torch.unbind(matrix_to_euler_angles(root_rot, "ZYX"), dim=-1)  # (T,)

    # Root height is already absolute in the first channel of *features* ---------------
    root_z = features[:, 0]  # (T,)

    return root_z, root_xy, yaw


# --------------------------------------------------------------------------------------
# Main processing routine
# --------------------------------------------------------------------------------------

def convert_relative_to_absolute(src_path: str, dst_path: str):
    """Load a relative SMPL-RIFKE feature file, convert root channels to absolute values and
    store the result.

    The layout of the 205-D feature vector is (see *smplrifke_feats.py*):
      0   : root_z                                   (absolute)
      1-2 : vel_trajectory_local (X, Y)              (relative → to be substituted)
      3   : vel_angles  (yaw velocity around Z)      (relative → to be substituted)
      4-… : remaining pose / joint channels          (left untouched)

    After the conversion we keep exactly the same tensor shape but the channels 1-3 now
    contain absolute trajectory (X, Y) and absolute yaw.
    """
    feats_np = np.load(src_path)
    if feats_np.ndim != 2 or feats_np.shape[1] != 205:
        raise ValueError(f"Unsupported feature shape {feats_np.shape} in {src_path}")

    feats = torch.from_numpy(feats_np).double()

    root_z, root_xy, yaw = _extract_abs_info(feats)

    # Replace the corresponding channels ------------------------------------------------
    feats[:, 1:3] = root_xy  # absolute X / Y
    feats[:, 3] = yaw        # absolute yaw (rad)

    # Persist (keeping the original dtype, usually float32)
    np.save(dst_path, feats.numpy().astype(feats_np.dtype))


# --------------------------------------------------------------------------------------
# Entry-point script
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert SMPL-RIFKE relative features to absolute representation.")
    parser.add_argument("--folder", type=str, default="datasets/motions", help="Root dataset folder (same as in get_smplrifke).")
    parser.add_argument("--base_name", type=str, default="AMASS_20.0_fps_nh_smplrifke", help="Directory containing the relative features (inside --folder).")
    parser.add_argument("--new_name", type=str, default="AMASS_20.0_fps_nh_smplrifke_abs", help="Name for the output directory (inside --folder).")
    parser.add_argument("--force_redo", action="store_true", help="Overwrite existing converted files.")

    args = parser.parse_args()

    base_folder = os.path.join(args.folder, args.base_name)
    new_folder = os.path.join(args.folder, args.new_name)

    if not os.path.exists(base_folder):
        raise FileNotFoundError(f"{base_folder} does not exist. Run get_smplrifke.py first.")

    os.makedirs(new_folder, exist_ok=True)

    print("Converting features to absolute representation → results stored in: ")
    print(new_folder)

    iterator = loop_amams(base_folder, new_folder, ext=".npy", newext=".npy", force_redo=args.force_redo)

    for src_path, dst_path in iterator:
        try:
            convert_relative_to_absolute(src_path, dst_path)
        except Exception as exc:
            print(f"[WARN] Failed to convert {src_path}: {exc}")


if __name__ == "__main__":
    main() 