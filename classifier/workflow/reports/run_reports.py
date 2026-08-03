try:
    from ._bootstrap import PROJECT_ROOT  # noqa: F401
except ImportError:
    from _bootstrap import PROJECT_ROOT  # noqa: F401

from pathlib import Path
import argparse
import subprocess
import sys


REPORTS_DIR = Path(__file__).resolve().parent


def run_command(cmd):
    print('\n' + '=' * 80)
    print('RUNNING:')
    print(' '.join(map(str, cmd)))
    print('=' * 80 + '\n')

    process = subprocess.Popen(cmd)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f'Command failed with code {process.returncode}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate paper-level quantitative report figures and tables.'
    )
    parser.add_argument(
        '--summary',
        choices=['median', 'mean'],
        default='median',
        help='Summary statistic for metric and trade-off reports.',
    )
    parser.add_argument(
        '--error-bar',
        choices=['minmax', 'std', 'sem', 'iqr', 'none'],
        default='minmax',
        help='Error-bar style for metric and trade-off reports.',
    )
    parser.add_argument(
        '--result-table-percent',
        action='store_true',
        help='Export the result table in percent units in the metrics report.',
    )
    parser.add_argument('--skip-confmat', action='store_true')
    parser.add_argument('--skip-metrics', action='store_true')
    parser.add_argument('--skip-tradeoff', action='store_true')
    parser.add_argument('--skip-performance-statistics', action='store_true')
    parser.add_argument(
        '--performance-bootstrap-reps',
        type=int,
        default=10_000,
        help='CD-HIT cluster-bootstrap replicates for model-performance comparisons.',
    )
    parser.add_argument(
        '--performance-permutation-reps',
        type=int,
        default=100_000,
        help='Cluster sign-flip permutation replicates for model-performance comparisons.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.skip_confmat:
        run_command([sys.executable, str(REPORTS_DIR / 'make_confmat.py')])

    if not args.skip_metrics:
        metrics_cmd = [
            sys.executable,
            str(REPORTS_DIR / 'make_metrics.py'),
            '--summary', args.summary,
            '--error-bar', args.error_bar,
        ]
        if args.result_table_percent:
            metrics_cmd.append('--result-table-percent')
        run_command(metrics_cmd)

    if not args.skip_tradeoff:
        run_command([
            sys.executable,
            str(REPORTS_DIR / 'make_model_tradeoff.py'),
            '--summary', args.summary,
            '--error-bar', args.error_bar,
        ])

    if not args.skip_performance_statistics:
        run_command([
            sys.executable,
            str(REPORTS_DIR / 'make_performance_statistics.py'),
            '--bootstrap-reps', str(args.performance_bootstrap_reps),
            '--permutation-reps', str(args.performance_permutation_reps),
        ])


if __name__ == '__main__':
    main()
