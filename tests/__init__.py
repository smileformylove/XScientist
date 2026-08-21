"""Local test package.

The marker is intentional: some Python installations ship a top-level
``tests`` package.  Making this directory a regular package ensures imports
such as ``from tests.test_ara_diff import ...`` resolve to this checkout when
the test suite is launched through the ``pytest`` console script.
"""
