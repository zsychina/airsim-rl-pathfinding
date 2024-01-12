import torch
import logging
import airsim

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('blocks')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f'Using {device}')


