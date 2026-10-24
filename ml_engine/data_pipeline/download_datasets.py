"""Automated Kaggle Dataset Downloader for Car Diagnostics & Engine Knock Audio."""
import subprocess

DATASETS = [
    ("malakragaie/car-diagnostics-dataset", "raw_downloads/car_diagnostics"),
    ("arindaemmanuel/revix-engine-knock-complete-dataset", "raw_downloads/revix_knock")
]

if __name__ == "__main__":
    for slug, dest in DATASETS:
        subprocess.run(["kaggle", "datasets", "download", "-d", slug, "--unzip", "-p", dest], check=False)
