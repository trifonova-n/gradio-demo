import os
import subprocess
import sys


def main():
    subprocess.check_call(
        f"python /workspace/model/demo.py --config /workspace/model/config/vox-256.yaml --checkpoint /workspace/checkpoints/vox.pth.tar --source_image '/workspace/image.png' --driving_video /workspace/smile.mp4 --result_video '/workspace/result.mp4'", shell=True)
    return '/workspace/result.mp4'


if __name__ == "__main__":
    print("Running demo")
    res = main()
    print(f"Result saved in {res}")
