import logging
logger = logging.getLogger('base')

def create_model(opt,tb_logger):
    model = opt['model']
    
    if model == 'base':
        pass
    elif model == 'condition':
        from .Generation_condition import GenerationModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))

    m = M(opt,tb_logger)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
