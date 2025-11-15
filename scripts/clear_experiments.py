"""
Clear Experiments Script

Clear all experiment results from the experiments directory.
"""

import argparse
import shutil
from pathlib import Path


def clear_experiments(
    experiment_dir: str = "./experiments", keep_structure: bool = False
):
    """
    Clear all experiment results.

    Args:
        experiment_dir: Directory containing experiments
        keep_structure: If True, only clear contents but keep directory structure
    """
    exp_dir = Path(experiment_dir)

    if not exp_dir.exists():
        print(f"Experiments directory does not exist: {experiment_dir}")
        return

    print(f"Clearing experiments directory: {experiment_dir}")

    if keep_structure:
        # Clear contents but keep directory structure
        for item in exp_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        print(
            f"✓ Cleared all contents from {experiment_dir} (kept directory structure)"
        )
    else:
        # Remove entire directory and recreate
        shutil.rmtree(exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Removed and recreated {experiment_dir}")

    print("Experiments directory cleared successfully!")


def main():
    parser = argparse.ArgumentParser(description="Clear experiment results")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="./experiments",
        help="Experiments directory to clear (default: ./experiments)",
    )
    parser.add_argument(
        "--keep_structure",
        action="store_true",
        help="Keep directory structure, only clear contents",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args()

    if not args.force:
        response = input(
            f"Are you sure you want to clear all experiments in '{args.experiment_dir}'? (yes/no): "
        )
        if response.lower() not in ["yes", "y"]:
            print("Cancelled.")
            return

    clear_experiments(args.experiment_dir, args.keep_structure)


if __name__ == "__main__":
    main()
