try:
    from ._bootstrap import PROJECT_ROOT  # noqa: F401
except ImportError:
    from _bootstrap import PROJECT_ROOT  # noqa: F401
from classifier.workflow.attention_stat_panel_strategy_consensus import main

if __name__ == '__main__':
    main()
