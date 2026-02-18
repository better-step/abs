from runtime import *
import os
import shutil
import subprocess
from pathlib import Path
import h5py


def process_one_file(src_path: Path, pre_out_dir: Path, repack_out_dir: Path) -> None:

    pre_out_path = pre_out_dir / src_path.name
    repack_out_path = repack_out_dir / src_path.name

    shutil.copyfile(src_path, pre_out_path)

    with h5py.File(pre_out_path, "r+") as f:
        parts = f.get("parts", None)
        if parts is None:
            raise KeyError(f"'parts' group not found in: {pre_out_path}")

        parts.attrs["version"] = "3.0"

        for part in parts.values():
            process_meshes(part)
            process_edges(part)
            process_halfedges(part)
            process_loops(part)
            process_shells(part)
            process_solids(part)
            process_faces(part)

            data = ((False, "2dcurves"), (True, "3dcurves"))
            for has_trafo, curve_type in data:
                process_curves(part, curve_type, has_trafo)

            process_surfaces(part)

    if repack_out_path.exists():
        repack_out_path.unlink()

    cmd = ["h5repack", "-i", str(pre_out_path), "-o", str(repack_out_path)]
    subprocess.run(cmd, check=True)


def iter_hdf5_files(input_dir: Path):
    for p in sorted(input_dir.glob("*.hdf5")):
        yield p


if __name__ == "__main__":
    INPUT_DIR = Path("/media/nafiseh/5f43a9e1-ea28-46ea-ab14-42040d28983d/abs/sample_data/fusion")  # folder full of .hdf5
    PRE_OUT_DIR = Path("/media/nafiseh/5f43a9e1-ea28-46ea-ab14-42040d28983d/abs/sample_data/c1")    # output BEFORE sys call
    REPACK_OUT_DIR = Path("/media/nafiseh/5f43a9e1-ea28-46ea-ab14-42040d28983d/abs/sample_data/c2")  # output AFTER h5repack

    PRE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPACK_OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = list(iter_hdf5_files(INPUT_DIR))
    if not files:
        raise FileNotFoundError(f"No .hdf5/.h5 files found in: {INPUT_DIR}")

    failures = []
    for src in files:
        try:
            print(f"Processing: {src.name}")
            process_one_file(src, PRE_OUT_DIR, REPACK_OUT_DIR)
        except Exception as e:
            failures.append((src, str(e)))
            print(f"  ERROR on {src.name}: {e}")

    print("\nDone.")
    if failures:
        print("Some files failed:")
        for f, msg in failures:
            print(f" - {f.name}: {msg}")
