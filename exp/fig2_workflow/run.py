"""Run segmentation, ruler estimation and area measurement on an example image."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import settings as s

def main():
    args = s.parse(__file__, ['prepare', 'analyze', 'all'])
    if args.action in {'prepare', 'all'}:
        from _common.export import example_ids, segmentation, ruler
        ids = example_ids()
        segmentation(ids)
        ruler(ids)
    if args.action in {'analyze', 'all'}:
        import analyze
        analyze.main()

if __name__ == '__main__':
    main()

