"""Dispatch model recipes from this directory's config.json."""
from pathlib import Path
from _common import settings as s
from _common.runner import call

def train_models(names):
    config = s.current()
    shared = config['training']
    for name in names:
        output = s.output() / 'training' / name
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f'Choose a fresh output_dir in config.json: {output}')
        if name in {'ours', 'without_dice', 'without_focal', 'without_expected_l1'}:
            from seg_discrete_overlap.train_eval import build_parser
            args = build_parser().parse_args(['train'])
            for key, value in config['models'][name]['training'].items():
                setattr(args, key, value)
            args.annotations = s.data_path('annotations')
            args.images = s.data_path('images')
            args.split_file = s.data_path('split')
            args.output = output
            args.cache_dir = s.output() / 'cache'
            args.device = config['device']
            args.seed = config['seed']
            args.handler(args)
        elif name == 'independent_sigmoid':
            from independent_sigmoid.train import main
            call(main, ['--annotations', s.data_path('annotations'), '--images', s.data_path('images'),
                        '--split', s.data_path('split'), '--output', output, '--cache-dir', s.output() / 'cache',
                        '--epochs', shared['epochs'], '--batch-size', shared['batch_size'],
                        '--num-workers', shared['num_workers'], '--image-size', shared['image_size'],
                        '--device', config['device'], '--seed', config['seed']])
        elif name == 'single_label_unetplusplus':
            from train_single_label import main
            call(main, ['--output', output, '--epochs', shared['epochs'],
                        '--batch-size', shared['batch_size'], '--device', config['device']])
        else:
            from train_fcn import main
            mode = {'fcn_single_label': 'single_label', 'fcn_lookup_multilabel': 'lookup_multilabel'}[name]
            call(main, [mode, '--output', output, '--epochs', shared['epochs'],
                        '--batch-size', shared['batch_size'], '--num-workers', shared['num_workers'],
                        '--device', config['device']])
        print(f'Trained {name}: {output}; set models.{name}.checkpoint to the new best.pt before evaluation.')
