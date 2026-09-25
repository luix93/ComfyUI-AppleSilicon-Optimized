"""Keep native binaries and their torch library links specific to one runtime."""

import hashlib
import os

import torch


def build_root():
    identity = f"{torch.__version__}:{torch.version.git_version}:{os.path.realpath(torch.__file__)}"
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return os.path.join("/tmp/asfp8_build", suffix)
