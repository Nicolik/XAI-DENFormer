try:
    from ._bootstrap import PROJECT_ROOT  # noqa: F401
except ImportError:
    from _bootstrap import PROJECT_ROOT  # noqa: F401
from classifier.workflow.gxi_aggregate import main

if __name__ == '__main__':
    main()
