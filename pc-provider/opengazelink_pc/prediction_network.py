"""Independent temporal encoder with fixation, pursuit and saccade experts."""
import torch
from torch import nn


class PredictionNetwork(nn.Module):
    def __init__(self,mean,scale):
        super().__init__()
        self.register_buffer("mean",torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer("scale",torch.as_tensor(scale,dtype=torch.float32))
        self.encoder=nn.Sequential(nn.Linear(411,48),nn.SiLU())
        self.temporal=nn.GRU(48,64,batch_first=True)
        self.phase=nn.Linear(70,3)
        self.landing=nn.Linear(70,3)
        self.future=nn.Sequential(nn.Linear(71,64),nn.SiLU(),nn.Linear(64,8))
        nn.init.zeros_(self.future[-1].weight)
        nn.init.zeros_(self.future[-1].bias)

    def forward(self,sequence,state,horizon):
        mask=sequence[:,:,409:410]
        x=((sequence-self.mean)/self.scale).clamp(-8,8)*mask
        encoded=self.encoder(x)*mask
        temporal,_=self.temporal(encoded)
        z=torch.cat((temporal[:,-1],state),dim=1)
        logits=self.phase(z)
        probabilities=logits.softmax(-1)
        landing_raw=self.landing(z)
        landing=.6*landing_raw[:,:2].tanh()
        remaining=.008+.142*landing_raw[:,2:3].sigmoid()
        result=self.future(torch.cat((z,horizon),dim=1))
        # Acceleration is damped over long horizons to avoid quadratic runaway.
        motion=state[:,:2]*horizon+.5*state[:,2:4]*horizon.square()*torch.exp(-horizon/.06)
        fixation=.05*result[:,:2].tanh()
        pursuit=motion+.1*result[:,2:4].tanh()
        jump=landing*(horizon/remaining).clamp(0,1)+.05*result[:,4:6].tanh()
        delta=probabilities[:,0:1]*fixation+probabilities[:,1:2]*pursuit+probabilities[:,2:3]*jump
        log_scale=(-4+result[:,6:8]).clamp(-7,-1)
        return delta,log_scale,logits,landing,remaining
