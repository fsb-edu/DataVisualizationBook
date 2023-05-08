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

    to_install = ""
    for pkg in packages:
        try:
            get_distribution(pkg.split("==")[0])
        except (DistributionNotFound, RequirementParseError):
            if "jupyter" not in pkg:
                print(f"{pkg} not found. Will be installed...")
                to_install += f"{pkg} "

    try:
        print(f"Installing missing dependencies: {to_install}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", to_install]
        )
    except Exception as e:
        print("Installation failed with error:", e)
