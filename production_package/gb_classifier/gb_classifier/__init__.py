import logging

from gb_classifier.config.core import config, PACKAGE_ROOT
from gb_classifier.version import VERSION as __version__

# https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
logging.getLogger(config.app_config.package_name).addHandler(logging.NullHandler())

# with open(PACKAGE_ROOT / "VERSION") as version_file:
#     __version__ = version_file.read().strip()
