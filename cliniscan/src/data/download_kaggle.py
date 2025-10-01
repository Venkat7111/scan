import os
import shutil
import zipfile
import subprocess
from pathlib import Path
import argparse

KAGGLE_DATASET_DEFAULT = "paultimothymooney/chest-xray-pneumonia"


def ensure_kaggle_credentials() -> Path:
	"""Ensure kaggle.json is available under %USERPROFILE%\.kaggle on Windows."""
	user_home = Path(os.path.expanduser("~"))
	kaggle_dir = user_home / ".kaggle"
	kaggle_dir.mkdir(parents=True, exist_ok=True)
	cred_target = kaggle_dir / "kaggle.json"

	# If kaggle.json exists in cwd, copy it into place
	local_cred = Path("kaggle.json")
	if local_cred.exists() and not cred_target.exists():
		shutil.copy2(str(local_cred), str(cred_target))

	if not cred_target.exists():
		raise FileNotFoundError(
			"kaggle.json not found. Place it in the project root or in %USERPROFILE%/.kaggle/."
		)

	# Kaggle CLI on Windows doesn't require chmod, but keep restricted if possible
	try:
		os.chmod(cred_target, 0o600)
	except Exception:
		pass

	return cred_target


def unzip_latest_zip(output_dir: Path) -> Path:
	zips = sorted(output_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
	if not zips:
		raise RuntimeError("No zip file downloaded from Kaggle.")
	zip_path = zips[0]
	print(f"Unzipping {zip_path.name}...")
	with zipfile.ZipFile(zip_path, 'r') as zf:
		zf.extractall(output_dir)
	return output_dir


def download_dataset(output_dir: Path, dataset: str) -> Path:
	"""Download and unzip a Kaggle dataset to output_dir. Returns extracted path if found."""
	ensure_kaggle_credentials()
	output_dir.mkdir(parents=True, exist_ok=True)

	print("Checking Kaggle CLI installation...")
	subprocess.run(["kaggle", "--version"], check=True)

	print(f"Downloading dataset {dataset} from Kaggle...")
	subprocess.run([
		"kaggle", "datasets", "download",
		"-d", dataset,
		"-p", str(output_dir)
	], check=True)

	unzip_latest_zip(output_dir)

	# Common folder name for chest xray dataset
	extracted = output_dir / "chest_xray"
	if extracted.exists():
		print(f"Dataset ready at: {extracted}")
		return extracted
	print(f"Files downloaded to: {output_dir}")
	return output_dir


def download_competition(output_dir: Path, competition: str) -> Path:
	"""Download and unzip a Kaggle competition dataset to output_dir."""
	ensure_kaggle_credentials()
	output_dir.mkdir(parents=True, exist_ok=True)

	print("Checking Kaggle CLI installation...")
	subprocess.run(["kaggle", "--version"], check=True)

	print(f"Downloading competition {competition} from Kaggle...")
	subprocess.run([
		"kaggle", "competitions", "download",
		"-c", competition,
		"-p", str(output_dir)
	], check=True)

	unzip_latest_zip(output_dir)
	print(f"Competition files downloaded to: {output_dir}")
	return output_dir


def main():
	parser = argparse.ArgumentParser(description="Kaggle downloader for datasets and competitions")
	parser.add_argument("--type", choices=["dataset", "competition"], default="dataset")
	parser.add_argument("--ref", type=str, default=KAGGLE_DATASET_DEFAULT, help="dataset slug (owner/name) or competition name")
	parser.add_argument("--out", type=str, default="data/raw")
	args = parser.parse_args()

	out_dir = Path(args.out)
	if args.type == "dataset":
		path = download_dataset(out_dir, args.ref)
		print(f"Done. Path: {path}")
	else:
		path = download_competition(out_dir, args.ref)
		print(f"Done. Path: {path}")


if __name__ == "__main__":
	main()
