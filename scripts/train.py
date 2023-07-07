import importlib
import os
import pathlib
import sys
from pathlib import Path
import torch



root_dir = Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))

os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Prevent unacceptable slowdowns when using 16 precision
from utils.training_utils import get_latest_checkpoint_path,remove_map
from utils.hparams import set_hparams, hparams

set_hparams()
if hparams['ddp_backend'] == 'nccl_no_p2p':
    print("Disabling NCCL P2P")
    os.environ['NCCL_P2P_DISABLE'] = '1'
def load_pre_train_model(type='wavenet'):
    r""" load pre model

        Args:
            model :
            type (str): wavenet mean loading wavenet  full mean loading fs2 and wavenet

        """
    pre_train_ckpt_path=hparams.get('pre_train_path')
    if pre_train_ckpt_path is not None:
        ckpt=torch.load(pre_train_ckpt_path)
        if ckpt.get('category') is None:
            raise RuntimeError("")
        if isinstance(type, str):
            if type == "wavenet":
                state_dict = {}
                for i in ckpt['state_dict']:
                    if i in remove_map['base']:
                        continue
                    if 'diffusion' in i:

                        state_dict[i] = ckpt['state_dict'][i]
                return state_dict
                # model.load_state_dict(state_dict=state_dict,strict=False)
            elif type == "full":
                ...
        elif isinstance(type, list):
            ...
        else:
            raise RuntimeError("")
    else:
        return None


def load_warp():
    if get_latest_checkpoint_path(pathlib.Path(hparams['work_dir'])) is not None:
        pass
        return None
    type=hparams.get('pre_train_load_type')
    if type is None:
        type="wavenet"
    return load_pre_train_model(type=type)





def run_task():
    assert hparams['task_cls'] != ''
    pkg = ".".join(hparams["task_cls"].split(".")[:-1])
    cls_name = hparams["task_cls"].split(".")[-1]
    task_cls = getattr(importlib.import_module(pkg), cls_name)
    ckpt =load_warp()

    task_cls.start(ckpt)


if __name__ == '__main__':
    run_task()
