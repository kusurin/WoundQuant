"""Train, evaluate and summarize model comparisons and loss ablations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import settings as s
from _common.runner import call

def main():
    args = s.parse(__file__, ['train', 'evaluate', 'summarize', 'all'], models=True)
    models = args.models or list(s.current()['models'])
    unknown = set(models) - set(s.current()['models'])
    if unknown:
        raise ValueError('Unknown models: ' + ', '.join(sorted(unknown)))
    if args.action == 'train':
        from train import train_models
        train_models(models)
        return
    if args.action in {'evaluate', 'all'}:
        from _common.export import test_ids, ruler
        ruler(test_ids())
        import evaluate
        cli_names = {'single_label_unetplusplus': 'baseline'}
        call(evaluate.main, [*[cli_names.get(name, name) for name in models], '--device', s.current()['device']])
    if args.action == 'summarize':
        import evaluate
        evaluate.combine_area_outputs()

if __name__ == '__main__':
    main()

