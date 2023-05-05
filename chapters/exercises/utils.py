import subprocess
import sys

from pkg_resources import (
    get_distribution,
    DistributionNotFound,
    RequirementParseError
)


def ensure_packages(requirements_file: str):
    with open(requirements_file, 'r') as f:
        packages = [line.strip() for line in f.readlines()]

    for pkg in packages:
        try:
            get_distribution(pkg.split("==")[0])
        except (DistributionNotFound, RequirementParseError):
            print(f"{pkg} not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
