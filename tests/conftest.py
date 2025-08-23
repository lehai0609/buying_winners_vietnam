import warnings

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
