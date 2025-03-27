import runpod
from train import run_training

def handler(job):
    result = run_training()
    return result

runpod.serverless.start({"handler": handler})