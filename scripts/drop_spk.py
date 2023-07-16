import torch
import argparse
import pathlib
import re


def modify_spk_embed(spk_embed):
    num_spk, hidden_size = spk_embed.shape
    all_ids = set(range(num_spk))
    if args.drop is not None:
        drop_ids = set([int(i) for i in args.drop.split(',') if i != '']).intersection(all_ids)
    else:
        drop_ids = all_ids - set([int(i) for i in args.retain.split(',') if i != ''])

    fill_list = None
    if args.fill == 'zeros':
        fill_list = [0. for _ in drop_ids]
    elif args.fill == 'random':
        fill_list = [torch.randn(1, hidden_size, dtype=torch.float32, device='cpu') for _ in drop_ids]
    elif args.fill == 'mean':
        mean = torch.mean(spk_embed, dim=0, keepdim=True)
        fill_list = [mean for _ in drop_ids]
    elif args.fill == 'cyclic':
        retain_ids = sorted(all_ids - drop_ids)
        num_retain = len(retain_ids)
        fill_list = [spk_embed[retain_ids[i % num_retain], :] for i, _ in enumerate(drop_ids)]

    for spk_id, fill in zip(sorted(drop_ids), fill_list):
        spk_embed[spk_id, :] = fill


parser = argparse.ArgumentParser(description='Drop or edit spk_embed in a checkpoint.')
parser.add_argument('input', type=str, help='Path to the input file')
parser.add_argument('output', type=str, help='Path to the output file')
drop_retain_group = parser.add_mutually_exclusive_group()
drop_retain_group.add_argument('--drop', type=str, required=False, metavar='ID,ID,...',
                               help='Drop specific speaker IDs.')
drop_retain_group.add_argument('--retain', type=str, required=False, metavar='ID,ID,...',
                               help='Retain specific speaker IDs and drop all the others.')
parser.add_argument('--fill', type=str, required=False, default='zeros', metavar='METHOD',
                    choices=['zeros', 'random', 'mean', 'cyclic'],
                    help='Specify a filling method for the dropped embedding. '
                         'Available methods: zeros, random, mean, cyclic')
parser.add_argument('--overwrite', required=False, default=False,
                    action='store_true', help='Overwrite if the output file exists.')
args = parser.parse_args()
assert args.drop is not None or args.retain is not None, 'Either --drop or --retain should be specified.'
if args.drop and not re.fullmatch(r'(\d+)?(,\d+)*,?', args.drop):
    print(f'Invalid format for --drop: \'{args.drop}\'')
    exit(-1)
if args.retain and not re.fullmatch(r'(\d+)?(,\d+)*,?', args.retain):
    print(f'Invalid format for --retain: \'{args.retain}\'')
    exit(-1)

import torch
input_ckpt = pathlib.Path(args.input).resolve()
output_ckpt = pathlib.Path(args.output).resolve()
assert input_ckpt.exists(), 'The input file does not exist.'
assert args.overwrite or not output_ckpt.exists(), \
    'The output file already exists or is the same as the input file.\n' \
    'This is not recommended because spk_embed dropping scripts may not be stable, ' \
    'and you may be at risk of losing your model.\n' \
    'If you are sure to OVERWRITE the existing file, please re-run this script with the \'--overwrite\' argument.'

ckpt_loaded = torch.load(input_ckpt, map_location='cpu')
state_dict = ckpt_loaded['state_dict']
if 'model.fs2.spk_embed.weight' in state_dict:
    modify_spk_embed(state_dict['model.fs2.spk_embed.weight'])
if 'model.spk_embed.weight' in state_dict:
    modify_spk_embed(state_dict['model.spk_embed.weight'])

torch.save(ckpt_loaded, output_ckpt)
