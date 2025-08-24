import os
import sys
import warnings
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def pytest_configure(config):
    # Suppress pandas FutureWarnings seen in the test runs that are
    # benign and originate from pandas internals or test patterns.
    # `message` must be a string (regex pattern), not a compiled pattern.
    warnings.filterwarnings(
        "ignore",
        message=r"Downcasting object dtype arrays on .fillna",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The default fill_method='pad' in DataFrame.pct_change",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Series.__getitem__ treating keys as positions is deprecated",
        category=FutureWarning,
    )
