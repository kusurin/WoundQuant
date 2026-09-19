"""Compute paired annotation agreement, ICC and mixed-model statistics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import settings as s
from _common.runner import call

def main():
    s.parse(__file__, ['analyze', 'all'])
    import analyze
    cfg = s.current()['analysis']
    call(analyze.main, ['--icc-dir', s.data_path('icc'), '--result-dir', s.output() / 'statistics',
                       '--bootstrap', cfg['bootstrap'], '--seed', cfg['seed'], '--test-tail', cfg['test_tail']])

if __name__ == '__main__':
    main()

