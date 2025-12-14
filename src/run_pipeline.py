import subprocess, sys

steps = [
    ["python", "-m", "src.01-data-preprocessing"],
    ["python", "-m", "src.02-training"],
    ["python", "-m", "src.03-evaluation"],
    ["python", "-m", "src.04-inference"],
]

for cmd in steps:
    print("\n==>", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)

print("\nPipeline finished successfully.")
