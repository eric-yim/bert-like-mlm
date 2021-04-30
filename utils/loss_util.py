class WeightedCriterion:
    """
    A class for calculating a loss with per sample weights
    """
    def __init__(self,criterion):
        """
        Initialize with nn.SomeLoss
        i.e. torch.nn.NLLLoss
        """
        self.criterion = criterion(reduction='none')
    def __call__(self,pred,target,weights):
        loss = self.criterion(pred,target)
        loss = loss * weights
        return loss.sum() / weights.sum()
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count