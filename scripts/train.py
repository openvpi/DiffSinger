import importlib
import os
# import pathlib
import sys
from pathlib import Path
# import torch



root_dir = Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))

os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Prevent unacceptable slowdowns when using 16 precision
# from utils.training_utils import get_latest_checkpoint_path
from utils.hparams import set_hparams, hparams

set_hparams()
if hparams['ddp_backend'] == 'nccl_no_p2p':
    print("Disabling NCCL P2P")
    os.environ['NCCL_P2P_DISABLE'] = '1'






def run_task():
    assert hparams['task_cls'] != ''
    pkg = ".".join(hparams["task_cls"].split(".")[:-1])
    cls_name = hparams["task_cls"].split(".")[-1]
    task_cls = getattr(importlib.import_module(pkg), cls_name)


    task_cls.start()


if __name__ == '__main__':
    run_task()
