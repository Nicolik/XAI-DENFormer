import sys
from classifier.workflow.attention_stat_panel_strategy_consensus import main as _main


def _append_default(flag, value):
    if flag not in sys.argv[1:]:
        sys.argv.extend([flag, value])


def main():
    _append_default('--input-kind', 'gxi_box')
    _append_default('--output-kind', 'gxi_stat_panel')
    _append_default('--value-col', 'gxi')
    _main()


if __name__ == '__main__':
    main()
