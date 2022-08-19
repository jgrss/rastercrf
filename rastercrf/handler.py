import logging


logger = logging.getLogger(__name__)

_FORMAT = '%(asctime)s:%(levelname)s:%(lineno)s:%(module)s.%(funcName)s:%(message)s'

_handler = logging.StreamHandler()
_formatter = logging.Formatter(_FORMAT, '%H:%M:%S')
_handler.setFormatter(_formatter)
# logging.basicConfig(level=logging.DEBUG)
logger.addHandler(_handler)
logger.setLevel(logging.INFO)
