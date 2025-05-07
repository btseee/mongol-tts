from setuptools import setup, find_packages
import subprocess
import os
import sys

subprocess.run(["git", "submodule", "update", "--init", "--recursive"], check=True)

def install_submodule_requirements(path):
    req_file = os.path.join(path, "requirements.txt")
    if os.path.exists(req_file):
        print(f"Installing requirements for submodule: {path}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])

install_submodule_requirements("coqui-ai-TTS")
install_submodule_requirements("utils/num2words")

if sys.platform.startswith("linux"):
    if not sys.stdout.encoding or "UTF" not in sys.stdout.encoding.upper():
        os.environ["PYTHONIOENCODING"] = "utf-8"
        print("Set PYTHONIOENCODING=utf-8 for UTF-8 support")

setup(
    name="mongol-tts",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "torch",
    ],
    python_requires=">=3.7",
)
