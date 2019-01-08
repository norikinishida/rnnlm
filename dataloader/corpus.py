import utils

def read_corpus(path):
    """
    :type path: str
    :rtype: DataPool
    """
    datapool = utils.DataPool(paths=[path], processes=[lambda l: l.split()])
    utils.writelog("dataloader.read_corpus", "# of instances=%d" % len(datapool))
    return datapool
