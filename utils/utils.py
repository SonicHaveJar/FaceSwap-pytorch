import torch
import os

def compare(arg1, arg2, opt):
    if opt == 'smaller':
        return arg1 if arg1 < arg2 else arg2
    elif opt == 'bigger':
        return arg1 if arg1 > arg2 else arg2

#Forward propagation with the autoencoders combined
def forward(x, A, B, direction='A2B'):
    if direction == 'A2B':
        out = A.encoder(x)
        out = B.decoder(out)
        return out
    else:        
        out = B.encoder(x)
        out = A.decoder(out)
        return out

#https://medium.com/analytics-vidhya/saving-and-loading-your-model-to-resume-training-in-pytorch-cb687352fa61
def save_ckp(state, ckp_path, name):
    if not os.path.exists(ckp_path):
        os.makedirs(ckp_path)
    save_path = ckp_path+name+'.pth'
    torch.save(state, save_path)

#https://medium.com/analytics-vidhya/saving-and-loading-your-model-to-resume-training-in-pytorch-cb687352fa61
def load_ckp(file_path, model, optimizer):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']