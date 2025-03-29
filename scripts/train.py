import argparse
import importlib
import os
import shutil

import sys
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.utilities.rank_zero import rank_zero_only

root_dir = Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Prevent unacceptable slowdowns when using 16 precision
sys.path.insert(0, str(root_dir))

from utils.config_utils import read_full_config, print_config, dump_config
from utils.training_utils import (
    DsModelCheckpoint, DsTQDMProgressBar, DsTensorBoardLogger,
    get_latest_checkpoint_path, get_strategy
)

parser = argparse.ArgumentParser(description='Training a model.')
parser.add_argument('--config', type=Path, required=False, default=None, metavar='CONFIG',
                    help='Path to the configuration file.')
parser.add_argument('--ckpt_folder', type=Path, required=False, default=None, metavar='CKPT',
                    help='Folder to the checkpoint.')
parser.add_argument('--exp_name', type=str, required=False, default='', metavar='EXP',
                    help='Experiment name.')
parser.add_argument('--reset', action='store_true', help='Overwrite cached configurations.')
parser.add_argument('--disable_nccl_p2p', action='store_true', help='Disable NCCL P2P.')
args = parser.parse_args()

if not args.disable_nccl_p2p:
    print("Disabling NCCL P2P")
    os.environ['NCCL_P2P_DISABLE'] = '1'

def run_task():
    config, config_chain = read_full_config(args.config, args.exp_name, args.ckpt_folder, reset=args.reset)
    print_config(config, config_chain)
    assert config['task_cls'] != ''
    dump_config(config)
    pkg = ".".join(config["task_cls"].split(".")[:-1])
    cls_name = config["task_cls"].split(".")[-1]
    task_cls = getattr(importlib.import_module(pkg), cls_name)
    task = task_cls(config)
    work_dir = Path(config['work_dir'])
    trainer = pl.Trainer(
        accelerator=config['pl_trainer_accelerator'],
        devices=config['pl_trainer_devices'],
        num_nodes=config['pl_trainer_num_nodes'],
        strategy=get_strategy(
            config['pl_trainer_devices'],
            config['pl_trainer_num_nodes'],
            config['pl_trainer_accelerator'],
            config['pl_trainer_strategy'],
            config['pl_trainer_precision'],
        ),
        precision=config['pl_trainer_precision'],
        callbacks=[
            DsModelCheckpoint(
                dirpath=work_dir,
                filename='model_ckpt_steps_{step}',
                auto_insert_metric_name=False,
                monitor='step',
                mode='max',
                save_last=False,
                # every_n_train_steps=config['val_check_interval'],
                save_top_k=config['num_ckpt_keep'],
                permanent_ckpt_start=config['permanent_ckpt_start'],
                permanent_ckpt_interval=config['permanent_ckpt_interval'],
                verbose=True
            ),
            # LearningRateMonitor(logging_interval='step'),
            DsTQDMProgressBar(),
        ],
        logger=DsTensorBoardLogger(
            save_dir=str(work_dir),
            name='lightning_logs',
            version='latest'
        ),
        gradient_clip_val=config['clip_grad_norm'],
        val_check_interval=config['val_check_interval'] * config['accumulate_grad_batches'],
        # so this is global_steps
        check_val_every_n_epoch=None,
        log_every_n_steps=1,
        max_steps=config['max_updates'],
        use_distributed_sampler=False,
        num_sanity_val_steps=config['num_sanity_val_steps'],
        accumulate_grad_batches=config['accumulate_grad_batches']
    )
    if not config['infer']:  # train
        @rank_zero_only
        def train_payload_copy():
            # Copy files to work_dir
            binary_dir = Path(config['binary_data_dir'])
            spk_map_dst = work_dir / 'spk_map.json'
            spk_map_src = binary_dir / 'spk_map.json'
            shutil.copy(spk_map_src, spk_map_dst)
            print(f'| Copied spk map to {spk_map_dst}.')
            lang_map_dst = work_dir / 'lang_map.json'
            lang_map_src = binary_dir / 'lang_map.json'
            shutil.copy(lang_map_src, lang_map_dst)
            print(f'| Copied lang map to {lang_map_dst}.')
            for lang in config['dictionaries'].keys():
                dict_dst = work_dir / f'dictionary-{lang}.txt'
                dict_src = binary_dir / f'dictionary-{lang}.txt'
                shutil.copy(dict_src, dict_dst)
                print(f'| Copied dictionary for language \'{lang}\' to {dict_dst}.')

        train_payload_copy()
        trainer.fit(task, ckpt_path=get_latest_checkpoint_path(work_dir))
    else:
        trainer.test(task)


if __name__ == '__main__':
    run_task()
