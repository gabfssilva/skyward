"""Disk Size - Verify unified disk_gb parameter.

Provisions an AWS instance with a custom disk size and checks
the actual disk space available on the remote machine.
"""

import skyward as sky


@sky.function
def check_disk() -> str:
    """Return disk info from the remote instance."""
    import subprocess

    result = subprocess.run(
        ["df", "-h", "/"],
        capture_output=True, text=True,
    )
    return result.stdout


if __name__ == "__main__":
    with sky.Compute(
        provider=sky.AWS(),
        disk_gb=450,
    ) as pool:
        print(check_disk() >> pool)
