import os
import sys
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install as _install
from setuptools.command.develop import develop as _develop

# ----- Python version check -----
MIN_PYTHON = (3, 8)
if sys.version_info < MIN_PYTHON:
    sys.exit(f"ERROR: mongol-tts requires Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or later.")

HERE = os.path.abspath(os.path.dirname(__file__))

def update_submodules():
    subprocess.check_call(
        ['git', 'submodule', 'update', '--init', '--recursive'],
        cwd=HERE
    )

def install_submodules():
    # coqui-ai-TTS
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '-e', os.path.join(HERE, 'coqui-ai-TTS')],
        cwd=HERE
    )
    # num2words
    num2words_path = os.path.join(HERE, 'utils', 'num2words')
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', num2words_path],
        cwd=HERE
    )

class PostInstall(_install):
    def run(self):
        update_submodules()
        install_submodules()
        super().run()

class PostDevelop(_develop):
    def run(self):
        update_submodules()
        install_submodules()
        super().run()

setup(
    name='mongol-tts',
    version='1.0',
    packages=find_packages(where='.', exclude=('tests',)),
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.8',
    install_requires=[
        'torch',
    ],
    cmdclass={
        'install': PostInstall,
        'develop': PostDevelop,
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
