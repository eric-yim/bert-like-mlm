from omegaconf import OmegaConf
from utils.models import TransformerModel
from utils import data_load as dl
from utils.dataloader_util import BertMaskDataset
from utils.loss_util import WeightedCriterion, AverageMeter
import torch
import torch.nn as nn
import numpy as np
from scipy.special import softmax
def prepare_for_inference(data):
    """
    Transposes to match transformer shape (Seq x Batch x Embed)
    Put on GPU

    Input a dictionary
    """
    for k,v in data.items():
        data[k] = data[k].transpose_(0,1).cuda()
def to_numpy(data):
    """
    Transposes to original shape (Batch x Embed)
    To numpy

    Input a dictionary
    """
    for k,v in data.items():
        data[k] = data[k].transpose_(0,1).cpu().detach().numpy()
def train(model,optimizer,criterion,dataloader):
    losses = AverageMeter()
    model.train()
    model.cuda()
    for data in dataloader:
        b = data['masked'].size(0)
        prepare_for_inference(data)
        optimizer.zero_grad()
        
        out = model(data['masked'])

        loss = criterion(
            out.view(-1,out.size(-1)),
            torch.reshape(data['original'],(-1,)),
            torch.reshape(data['loss_weights'],(-1,)))
        loss.backward()
        optimizer.step()
        losses.update(loss.item(),b)
    return losses.avg
def create_samples(model,dataloader):
    model.eval()
    model.cuda()
    with torch.no_grad():
        for data in dataloader:

            prepare_for_inference(data)
            out = model(data['masked'])
            data = {'masked':data['masked'],'loss_weights':data['loss_weights'],'out':out}
            to_numpy(data)
            result = samples(data['masked'],data['out'],data['loss_weights'])
            for res,m in zip(result,data['masked']):
                print("-"*40)
                print(m)
                print(res)
                
            break
        

def samples(masked,predictions,loss_weights):
    """
    Inputs are numpy ndarrays 
    Masked shape: B x Seq
    Predictions shape: B x Seq x F
    Loss Weights: B x Seq
    """
    result = masked.copy()
    loss_weights = loss_weights > 0.5
    for i in range(masked.shape[0]):
        for j in range(masked.shape[1]):
            if loss_weights[i,j]:
                pred = predictions[i,j]
                p = softmax(pred,axis=-1)
                print(p)
                result[i,j] = np.random.choice(len(pred),p=p)
    return result

        
def main(args):
    # Load Data
    data = dl.load(args.data.path)
    dataset = BertMaskDataset(data,**args.data.dataset)
    sampler=torch.utils.data.sampler.RandomSampler(list(range(len(dataset))),**args.train.sampler)
    dataloader=torch.utils.data.DataLoader(
            dataset, shuffle=False,sampler=sampler,**args.train.dataloader)
    # Init Model, Optimizer, Criterion
    model = TransformerModel(**args.model.kwargs)
    optimizer = torch.optim.Adam(model.parameters(), **args.train.optimizer)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **args.train.scheduler)
    criterion = WeightedCriterion(nn.CrossEntropyLoss)

    best_loss = float('inf')
    for epoch in range(args.train.start_epoch,args.train.epochs):
        loss=train(model,optimizer,criterion,dataloader)
        scheduler.step()
        print("Epoch: {:3d}\t"
            "Loss: {:.3f}\t"
            "LR: {:0.2e}\t".format(epoch,loss,scheduler.get_last_lr()[0])
        )
        if loss < best_loss:
            best_loss = loss
            torch.save({'model_state_dict': model.state_dict()},args.model.save)
            print(f"Saved to {args.model.save}")
    create_samples(model,dataloader)
    #import IPython ; IPython.embed() ; exit(1)
if __name__=="__main__":
    args = OmegaConf.load('args.yaml')
    main(args)