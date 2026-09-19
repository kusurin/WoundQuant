"""Repository-relative configuration and paths shared by result groups."""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / 'exp'
_active = None
_active_group = None

def path(value):
    value = Path(value).expanduser()
    return value if value.is_absolute() else REPO / value

def config(group=None):
    if group is None or group == _active_group:
        if _active is None:
            raise RuntimeError('Start this experiment using its run.py entry point.')
        return _active
    configured = (_active or {}).get('dependencies', {}).get(group)
    file = path(configured) if configured else EXP / group / 'config.json'
    return json.loads(file.read_text(encoding='utf-8'))

def current():
    return config()

def output(group=None):
    return path(config(group)['output_dir'])

def data_path(key, group=None):
    overrides = {'images': 'WOUNDQUANT_IMAGES', 'exclusive': 'WOUNDQUANT_EXCLUSIVE'}
    value = os.environ.get(overrides.get(key, '')) or config(group)['data'][key]
    return path(value)

def parse(entry, actions, *, models=False):
    global _active, _active_group
    folder = Path(entry).resolve().parent
    parser = argparse.ArgumentParser(description=f'{folder.name}: ' + ', '.join(actions))
    parser.add_argument('action', choices=actions, nargs='?', default='all')
    parser.add_argument('--config', type=Path, default=folder / 'config.json')
    if models:
        parser.add_argument('--models', nargs='+', help='Model names from config.json; default: all')
    args = parser.parse_args()
    _active = json.loads(args.config.read_text(encoding='utf-8'))
    _active_group = folder.name
    # Existing local image locations are optional, never required by a public clone.
    local = REPO / '.local/paths.json'
    if local.is_file():
        for key, value in json.loads(local.read_text(encoding='utf-8')).items():
            os.environ.setdefault(key, value)
    sys.path.insert(0, str(EXP / '_common'))
    os.chdir(REPO)
    return args
