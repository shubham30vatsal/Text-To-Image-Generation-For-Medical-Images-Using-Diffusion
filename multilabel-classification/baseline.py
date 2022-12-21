import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
from tqdm import tqdm
from asl import AsymmetricLoss

def build_model(pretrained=True, freeze_fe=False, nr_concepts=100):
    weights = None
    if pretrained:
        weights = DenseNet121_Weights.DEFAULT
    model = densenet121(weights=weights)

    if freeze_fe:
        for param in model.parameters():
            param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(1024, 256), 
        nn.Linear(256, nr_concepts) 
    )
    # make the classification layer learnable
    # model.classifier = nn.Linear(1024, nr_concepts)
    
    return model


# training function
def do_epoch(model, dataloader, criterion, device, optimizer=None, weights=None, validation=False):
    if validation == True: model.eval()
    else: model.train()
    counter = 0
    running_loss = 0.0
    with torch.set_grad_enabled(validation == False):
        for data in tqdm(dataloader):
            counter += 1
            
            image, target = data['image'].to(device), data['label'].to(device)
            torch.set_printoptions(linewidth=1000)
#             print("Image Name   : ", data['image_name'][0])
#             print("Image Labels : ", target[0])
            
            if validation == False: optimizer.zero_grad()
            # forward pass
            outputs = model(image)
            if not isinstance(criterion, AsymmetricLoss) and not isinstance(criterion, nn.BCEWithLogitsLoss):
                # apply sigmoid activation to get all the outputs between 0 and 1
                outputs = torch.sigmoid(outputs)
            # compute loss
            loss = criterion(outputs, target)
            if isinstance(criterion, nn.BCELoss) and weights is not None:
                loss = (loss * weights).mean()
            # compute gradients
            if validation == False: 
                loss.backward()
                optimizer.step()
            running_loss += loss.item()

        return running_loss / counter