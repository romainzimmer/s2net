import torch
import numpy as np

class SNN(torch.nn.Module):
    
    def __init__(self, layers):
        
        super(SNN, self).__init__()
        self.layers = torch.nn.ModuleList(layers)
        
    def forward(self, x):
        loss_seq = []
        
        for l in self.layers:
            x, loss = l(x)
            loss_seq.append(loss)
            
        return x, loss_seq
    
    
    def clamp(self):
        
        for l in self.layers:
            l.clamp()  
            
            
            
    def reset_parameters(self):
        
        for l in self.layers:
            l.reset_parameters()
            
class SpikingDenseLayer(torch.nn.Module):
    
    def __init__(self, input_shape, output_shape, spike_fn, w_init_mean, w_init_std,
                 recurrent=False, lateral_connections=True, eps=1e-8):
        
        super(SpikingDenseLayer, self).__init__()
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.spike_fn = spike_fn
        self.recurrent = recurrent
        self.eps = eps
        self.lateral_connections = lateral_connections

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std
        
        self.w = torch.nn.Parameter(torch.empty((input_shape, output_shape)), requires_grad=True)
        if recurrent:
            self.v = torch.nn.Parameter(torch.empty((output_shape, output_shape)), requires_grad=True)
            
        self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(output_shape), requires_grad=True)
        
        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None
        
        self.training = True
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        
        h = torch.einsum("abc,cd->abd", x, self.w)
        nb_steps = h.shape[1]
        
        # membrane potential 
        mem = torch.zeros((batch_size, self.output_shape),  dtype=x.dtype, device=x.device)
        # output spikes
        spk = torch.zeros((batch_size, self.output_shape),  dtype=x.dtype, device=x.device)
        
        # output spikes recording
        spk_rec = torch.zeros((batch_size, nb_steps, self.output_shape),  dtype=x.dtype, device=x.device)
        
        if self.lateral_connections:
            d = torch.einsum("ab, ac -> bc", self.w, self.w)
            
        norm = (self.w**2).sum(0)
        
        
        for t in range(nb_steps):
            
            # reset term
            if self.lateral_connections:
                rst = torch.einsum("ab,bc ->ac", spk, d)
            else:
                rst = spk*self.b*norm
            
            input_ = h[:,t,:]
            if self.recurrent:
                input_ = input_ + torch.einsum("ab,bc->ac", spk, self.v)
       
            # membrane potential update
            mem = (mem-rst)*self.beta + input_*(1.-self.beta)
            mthr = torch.einsum("ab,b->ab",mem, 1./(norm+self.eps))-self.b 
                
            spk = self.spike_fn(mthr)
            
            spk_rec[:,t,:] = spk   
            
        # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()       
        
        loss = 0.5*(spk_rec**2).mean()
        
        return spk_rec, loss
   
    def reset_parameters(self):
        
        torch.nn.init.normal_(self.w,  mean=self.w_init_mean, std=self.w_init_std*np.sqrt(1./self.input_shape))
        if self.recurrent:
            torch.nn.init.normal_(self.v,  mean=self.w_init_mean, std=self.w_init_std*np.sqrt(1./self.output_shape))
        torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)
    
    def clamp(self):
        
        self.beta.data.clamp_(0.,1.)
        self.b.data.clamp_(min=0.)

class SpikingConv1DLayer(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 spike_fn, w_init_mean, w_init_std, recurrent=False,
                 lateral_connections=True,
                 eps=1e-8, stride=1,flatten_output=False):
        
        super(SpikingConv1DLayer, self).__init__()
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.spike_fn = spike_fn
        self.recurrent = recurrent
        self.lateral_connections = lateral_connections
        self.eps = eps
        
        self.flatten_output = flatten_output
        
        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std
        
        self.w = torch.nn.Parameter(torch.empty((out_channels, in_channels, kernel_size)), requires_grad=True)
        if recurrent:
            self.v = torch.nn.Parameter(torch.empty((out_channels, out_channels)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(out_channels), requires_grad=True)
        
        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None
        
        self.training = True
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        
        conv_x = torch.nn.functional.conv1d(x, self.w, padding=(np.ceil(((self.kernel_size-1)*self.dilation)/2).astype(int),),
                                      dilation=(self.dilation,),
                                      stride=(self.stride,))
        nb_steps = conv_x.shape[2]
        
        # membrane potential 
        mem = torch.zeros((batch_size, self.out_channels),  dtype=x.dtype, device=x.device)
        # output spikes
        spk = torch.zeros((batch_size, self.out_channels),  dtype=x.dtype, device=x.device)
        
        # output spikes recording
        spk_rec = torch.zeros((batch_size, self.out_channels, nb_steps),  dtype=x.dtype, device=x.device)
        
        
        if self.lateral_connections:
            d = torch.einsum("abc, ebc -> ae", self.w, self.w)     
        b = self.b
            
        norm = (self.w**2).sum((1,2))
        
        
        for t in range(nb_steps):
            
            # reset term
            if self.lateral_connections:
                rst = torch.einsum("ab,bd ->ad", spk, d)
            else:
                rst = torch.einsum("ab,b,b->ab", spk, self.b, norm)
            
            input_ = conv_x[:,:,t]
            if self.recurrent:
                input_ = input_ + torch.einsum("ab,bd->ad", spk, self.v)
            
            # membrane potential update
            mem = (mem-rst)*self.beta + input_*(1.-self.beta)
            mthr = torch.einsum("ab,b->ab",mem, 1./(norm+self.eps))-b 
                
            spk = self.spike_fn(mthr)
            
            spk_rec[:,:,t] = spk   
            
        # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()       
        
        loss = 0.5*(spk_rec**2).mean()
        
        if self.flatten_output:
            
            output = torch.transpose(spk_rec, 1, 2).contiguous()            
        else:
            output = spk_rec
        
        return output, loss
   
    def reset_parameters(self):
        
        torch.nn.init.normal_(self.w,  mean=self.w_init_mean, std=self.w_init_std*np.sqrt(1./(self.in_channels*np.prod(self.kernel_size))))
        if self.recurrent:
            torch.nn.init.normal_(self.v,  mean=self.w_init_mean, std=self.w_init_std*np.sqrt(1./self.out_channels))
        torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)
    
    def clamp(self):
        
        self.beta.data.clamp_(0.,1.)
        self.b.data.clamp_(min=0.)
    
class SpikingConv2DLayer(torch.nn.Module):
    
    def __init__(self, input_shape, output_shape,
                 in_channels, out_channels, kernel_size, dilation,
                 spike_fn, w_init_mean, w_init_std, recurrent=False,
                 lateral_connections=True,
                 eps=1e-8, stride=(1,1),flatten_output=False):
        
        super(SpikingConv2DLayer, self).__init__()
        
        self.kernel_size = np.array(kernel_size)
        self.dilation = np.array(dilation)
        self.stride = np.array(stride)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.spike_fn = spike_fn
        self.recurrent = recurrent
        self.lateral_connections = lateral_connections
        self.eps = eps
        
        self.flatten_output = flatten_output
        
        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std
        
        self.w = torch.nn.Parameter(torch.empty((out_channels, in_channels, *kernel_size)), requires_grad=True)
        if recurrent:
            self.v = torch.nn.Parameter(torch.empty((out_channels, out_channels)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(out_channels), requires_grad=True)
        
        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None
        
        self.training = True
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        
        conv_x = torch.nn.functional.conv2d(x, self.w, padding=tuple(np.ceil(((self.kernel_size-1)*self.dilation)/2).astype(int)),
                                      dilation=tuple(self.dilation),
                                      stride=tuple(self.stride))
        conv_x = conv_x[:,:,:,:self.output_shape]
        nb_steps = conv_x.shape[2]
        
        # membrane potential 
        mem = torch.zeros((batch_size, self.out_channels, self.output_shape),  dtype=x.dtype, device=x.device)
        # output spikes
        spk = torch.zeros((batch_size, self.out_channels, self.output_shape),  dtype=x.dtype, device=x.device)
        
        # output spikes recording
        spk_rec = torch.zeros((batch_size, self.out_channels, nb_steps, self.output_shape),  dtype=x.dtype, device=x.device)
        
        
        if self.lateral_connections:
            d = torch.einsum("abcd, ebcd -> ae", self.w, self.w)     
        b = self.b.unsqueeze(1).repeat((1,self.output_shape))
            
        norm = (self.w**2).sum((1,2,3))
        
        
        for t in range(nb_steps):
            
            # reset term
            if self.lateral_connections:
                rst = torch.einsum("abc,bd ->adc", spk, d)
            else:
                rst = torch.einsum("abc,b,b->abc", spk, self.b, norm)
            
            input_ = conv_x[:,:,t,:]
            if self.recurrent:
                input_ = input_ + torch.einsum("abc,bd->adc", spk, self.v)
            
            # membrane potential update
            mem = (mem-rst)*self.beta + input_*(1.-self.beta)
            mthr = torch.einsum("abc,b->abc",mem, 1./(norm+self.eps))-b 
                
            spk = self.spike_fn(mthr)
            
            spk_rec[:,:,t,:] = spk   
            
        # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()       
        
        loss = 0.5*(spk_rec**2).mean()
        
        if self.flatten_output:
            
            output = torch.transpose(spk_rec, 1, 2).contiguous()
            output = output.view(batch_size, nb_steps, self.out_channels*self.output_shape)
            
        else:
            
            output = spk_rec
        
        return output, loss
   
    def reset_parameters(self):
        
        torch.nn.init.normal_(self.w,  mean=self.w_init_mean, std=self.w_init_std*np.sqrt(1./(self.in_channels*np.prod(self.kernel_size))))
        if self.recurrent:
            torch.nn.init.normal_(self.v,  mean=self.w_init_mean, std=self.w_init_std*np.sqrt(1./self.out_channels))
        torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)
    
    def clamp(self):
        
        self.beta.data.clamp_(0.,1.)
        self.b.data.clamp_(min=0.)
        
        
class SpikingConv3DLayer(torch.nn.Module):
    
    def __init__(self, input_shape, output_shape,
                 in_channels, out_channels, kernel_size, dilation,
                 spike_fn, w_init_mean, w_init_std, recurrent=False,
                 lateral_connections=True,
                 eps=1e-8, stride=(1,1,1),flatten_output=False):
        
        super(SpikingConv3DLayer, self).__init__()
        
        self.kernel_size = np.array(kernel_size)
        self.dilation = np.array(dilation)
        self.stride = np.array(stride)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.spike_fn = spike_fn
        self.recurrent = recurrent
        self.lateral_connections = lateral_connections
        self.eps = eps
        
        self.flatten_output = flatten_output
        
        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std
        
        self.w = torch.nn.Parameter(torch.empty((out_channels, in_channels, *kernel_size)), requires_grad=True)
        if recurrent:
            self.v = torch.nn.Parameter(torch.empty((out_channels, out_channels)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(out_channels), requires_grad=True)
        
        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None
        
        self.training = True
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        
        conv_x = torch.nn.functional.conv3d(x, self.w, padding=tuple(np.ceil(((self.kernel_size-1)*self.dilation)/2).astype(int)),
                                      dilation=tuple(self.dilation),
                                      stride=tuple(self.stride))
        conv_x = conv_x[:,:,:,:self.output_shape[0],:self.output_shape[1]]
        nb_steps = conv_x.shape[2]
        
        # membrane potential 
        mem = torch.zeros((batch_size, self.out_channels, *self.output_shape),  dtype=x.dtype, device=x.device)
        # output spikes
        spk = torch.zeros((batch_size, self.out_channels, *self.output_shape),  dtype=x.dtype, device=x.device)
        
        # output spikes recording
        spk_rec = torch.zeros((batch_size, self.out_channels, nb_steps, *self.output_shape),  dtype=x.dtype, device=x.device)
        
        
        if self.lateral_connections:
            d = torch.einsum("abcde, fbcde -> af", self.w, self.w)     
        b = self.b.unsqueeze(1).unsqueeze(1).repeat((1,*self.output_shape))
            
        norm = (self.w**2).sum((1,2,3,4))
        
        
        for t in range(nb_steps):
            
            # reset term
            if self.lateral_connections:
                rst = torch.einsum("abcd,be ->aecd", spk, d)
            else:
                rst = torch.einsum("abcd,b,b->abcd", spk, self.b, norm)
            
            input_ = conv_x[:,:,t,:,:]
            if self.recurrent:
                input_ = input_ + torch.einsum("abcd,be->aecd", spk, self.v)
            
            # membrane potential update
            mem = (mem-rst)*self.beta + input_*(1.-self.beta)
            mthr = torch.einsum("abcd,b->abcd",mem, 1./(norm+self.eps))-b 
                
            spk = self.spike_fn(mthr)
            
            spk_rec[:,:,t,:,:] = spk   
            
        # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()       
        
        loss = 0.5*(spk_rec**2).mean()
        
        if self.flatten_output:
            
            output = torch.transpose(spk_rec, 1, 2).contiguous()
            output = output.view(batch_size, nb_steps, self.out_channels*np.prod(self.output_shape))
            
        else:
            
            output = spk_rec
        
        return output, loss
   
    def reset_parameters(self):
        
        torch.nn.init.normal_(self.w,  mean=self.w_init_mean, std=self.w_init_std*np.sqrt(1./(self.in_channels*np.prod(self.kernel_size))))
        if self.recurrent:
            torch.nn.init.normal_(self.v,  mean=self.w_init_mean, std=self.w_init_std*np.sqrt(1./self.out_channels))
        torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)
    
    def clamp(self):
        
        self.beta.data.clamp_(0.,1.)
        self.b.data.clamp_(min=0.)
        
        
        
class ReadoutLayer(torch.nn.Module):
    
    "Fully connected readout"
    
    def __init__(self,  input_shape, output_shape, w_init_mean, w_init_std, eps=1e-8, time_reduction="mean"):
        
        
        assert time_reduction in ["mean", "max"], 'time_reduction should be "mean" or "max"'
        
        super(ReadoutLayer, self).__init__()
        

        self.input_shape = input_shape
        self.output_shape = output_shape
        
        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std
        
        
        self.eps = eps
        self.time_reduction = time_reduction
        
        
        self.w = torch.nn.Parameter(torch.empty((input_shape, output_shape)), requires_grad=True)
        if time_reduction == "max":
            self.beta = torch.nn.Parameter(torch.tensor(0.7*np.ones((1))), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(output_shape), requires_grad=True)
        
        self.reset_parameters()
        self.clamp()
        
        self.mem_rec_hist = None
        
    def forward(self, x):
        
        batch_size = x.shape[0]
        
       
        h = torch.einsum("abc,cd->abd", x, self.w)
        
        norm = (self.w**2).sum(0)
        
        if self.time_reduction == "max":
            nb_steps = x.shape[1]
            # membrane potential 
            mem = torch.zeros((batch_size, self.output_shape),  dtype=x.dtype, device=x.device)

            # memrane potential recording
            mem_rec = torch.zeros((batch_size, nb_steps, self.output_shape),  dtype=x.dtype, device=x.device)

            for t in range(nb_steps):

                # membrane potential update
                mem = mem*self.beta + (1-self.beta)*h[:,t,:]
                mem_rec[:,t,:] = mem
                
            output = torch.max(mem_rec, 1)[0]/(norm+1e-8) - self.b
            
        elif self.time_reduction == "mean":
            
            mem_rec = h
            output = torch.mean(mem_rec, 1)/(norm+1e-8) - self.b
        
        # save mem_rec for plotting
        self.mem_rec_hist = mem_rec.detach().cpu().numpy()
        
        loss = None
        
        return output, loss
    
    def reset_parameters(self):
        torch.nn.init.normal_(self.w,  mean=self.w_init_mean,
                              std=self.w_init_std*np.sqrt(1./(self.input_shape)))
        
        if self.time_reduction == "max":
            torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
            
        torch.nn.init.normal_(self.b, mean=1., std=0.01)
    
    def clamp(self):
        
        if self.time_reduction == "max":
            self.beta.data.clamp_(0.,1.)
        
        
class SurrogateHeaviside(torch.autograd.Function):
    
    # Activation function with surrogate gradient
    sigma = 10.0

    @staticmethod 
    def forward(ctx, input):
        
        output = torch.zeros_like(input)
        output[input > 0] = 1.0
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # approximation of the gradient using sigmoid function
        grad = grad_input*torch.sigmoid(SurrogateHeaviside.sigma*input)*torch.sigmoid(-SurrogateHeaviside.sigma*input)
        return grad
    
