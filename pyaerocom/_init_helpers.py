def _init_logger():
    import logging
    ### LOGGING
    # Note: configuration will be propagated to all child modules of
    # pyaerocom, for details see 
    # http://eric.themoritzfamily.com/learning-python-logging.html
    logger = logging.getLogger('pyaerocom')
    
    default_formatter = logging.Formatter(\
       "%(asctime)s:%(levelname)s:\n%(message)s")
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(default_formatter)
    
    logger.addHandler(console_handler)
    
    logger.setLevel(logging.CRITICAL)
    
    print_log = logging.getLogger('pyaerocom_print')
    
    print_handler = logging.StreamHandler()
    print_handler.setFormatter(logging.Formatter("%(message)s"))
    
    print_log.addHandler(print_handler)
    
    print_log.setLevel(logging.INFO)
    return (logger, print_log)

def change_verbosity(new_level='debug', log=None):
    if log is None:
        from pyaerocom import logger
        log = logger
    if isinstance(new_level, str):
        if not new_level in LOGLEVELS:
            raise ValueError("Invalid input for loglevel, choose "
                             "from {}".format(LOGLEVELS.keys()))
        new_level = LOGLEVELS[new_level]
    log.setLevel(new_level)
    
### Functions for package initialisation
def _init_supplemental():
    from pkg_resources import get_distribution
    from os.path import abspath, dirname
    return (get_distribution('pyaerocom').version, abspath(dirname(__file__)))

def check_requirements(logger):
    GEONUM_AVAILABLE = True
    BASEMAP_AVAILABLE = True
    try:
        import geonum
    except:
        GEONUM_AVAILABLE = False
        logger.warning('geonum library is not installed. Some features will not '
                    'be available (e.g. conversion of pressure to altitude')
    try:
        from mpl_toolkits.basemap import Basemap
    except:
        BASEMAP_AVAILABLE = False
        logger.warning('basemap extension library is not installed (or cannot be '
                    'imported. Some features will not be available')
    return (GEONUM_AVAILABLE, BASEMAP_AVAILABLE)

LOGLEVELS = {'debug': 10,
             'info': 20,
             'warning': 30,
             'error': 40,
             'critical': 50}