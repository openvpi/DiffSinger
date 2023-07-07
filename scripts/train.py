import importlib
import os
import pathlib
import sys
from pathlib import Path
import torch

from utils.training_utils import get_latest_checkpoint_path

root_dir = Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))

os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Prevent unacceptable slowdowns when using 16 precision

from utils.hparams import set_hparams, hparams

set_hparams()
if hparams['ddp_backend'] == 'nccl_no_p2p':
    print("Disabling NCCL P2P")
    os.environ['NCCL_P2P_DISABLE'] = '1'
def load_pre_train_model(model,type:str='part'):
    r""" load pre model

        Args:
            model :
            type (str): part mean loading wavenet  full mean loading fs2 and wavenet

        """
    pre_train_ckpt_path=hparams.get('pre_train_path')
    if pre_train_ckpt_path is not None:
        ckpt=torch.load(pre_train_ckpt_path)
        if type == "part":
            state_dict = {}
            for i in ckpt['state_dict']:
                if 'diffusion' in i:
                    print(i)
                    state_dict[i] = ckpt['state_dict'][i]
            model.load_state_dict(state_dict,strict=False)
        elif type == "full":
            ...


def load_warp(model):
    if get_latest_checkpoint_path(pathlib.Path(hparams['work_dir'])) is not None:
        pass
        return None
    type=hparams.get('pre_train_load_type')
    if type is None:
        type="part"
    load_pre_train_model(model=model,type=type)





def run_task():
    assert hparams['task_cls'] != ''
    pkg = ".".join(hparams["task_cls"].split(".")[:-1])
    cls_name = hparams["task_cls"].split(".")[-1]
    task_cls = getattr(importlib.import_module(pkg), cls_name)
    load_warp(task_cls)

    task_cls.start()


if __name__ == '__main__':
    run_task()
