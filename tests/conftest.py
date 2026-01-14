"""Pytest configuration for tests of the notebook plugin.

This file registers the plugin under test so that the tests in this
package can exercise it.  Without this file pytest will not know
about the plugin when run inside this package.
"""

pytest_plugins = ["pytester"]

collect_ignore = ["fixture_testing"]
