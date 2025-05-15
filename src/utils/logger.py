import logging
import sys

# Configuration minimale mais complète
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        # logging.FileHandler('C:/Users/Abdessamad/Desktop/MLOpsClassificationTexteV2/app.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('app')

def logging(niveau,Msg,logger=logger)->None:
    if niveau=="debug":
        logger.debug(Msg)

    if niveau=="info":
        logger.info(Msg)

    if niveau=="warning":
        logger.warning(Msg)

    if niveau=="error":
        logger.error(Msg)

    if niveau=="critical":
        logger.critical(Msg)

# logging.shutdown()