import logging
import urllib


def init_logger(path, logger_name=None):
    '''
    Initialize logger, using either a FileHandler or a StreamHandler.
    '''
    assert isinstance(path, str) or isinstance(path, OutStream)

    if(logger_name):
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()
    
    # Assign a handler only if the object doesn't already have one
    if(not logger.handlers):
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(u'%(asctime)s, %(levelname)s %(message)s',
                                      datefmt='%H:%M:%S')
        if(isinstance(path, str)):
            handler = logging.FileHandler(path, mode='w', encoding='utf-8')
            open(path, 'a').close()
        else:
            handler = logging.StreamHandler(path)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger



# Adapted from https://github.com/epfl-dlab/WikiPDA/blob/master/PaperAndCode/TopicsExtractionPipeline/GenerateDataframes.py
def normalize_title(title, dumps=True):
    """ Replace _ with space, remove anchor and namespace prefix, capitalize """
    title = urllib.parse.unquote(title)
    if(dumps):
        try:
            title = title.split(':', 1)[1]
        # Currently happens only for broken cross-namespace redirects
        except IndexError:
            return ''
    title = title.strip()
    if len(title) > 0:
        title = title[0].upper() + title[1:]
    n_title = title.replace("_", " ")
    if '#' in n_title:
        n_title = n_title.split('#')[0]
    return n_title
