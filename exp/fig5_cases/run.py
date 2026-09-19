"""Select representative cases from the performance analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import settings as s

def main():
    s.parse(__file__, ['analyze', 'all'])
    import analyze
    analyze.main()

if __name__ == '__main__':
    main()

