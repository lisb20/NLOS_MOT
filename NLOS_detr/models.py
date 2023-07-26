import time

import torch
from torch import nn, Tensor
from torchvision import models as tvmodels
import pdb
from einops import rearrange
from scipy.optimize import linear_sum_assignment


class PAC_Cell(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True,
                 rnn_hdim: int = 128):
        super(PAC_Cell, self).__init__()

        assert model_name in ['PAC_Net', 'P_Net', 'C_Net', 'baseline']
        self.rnn_hdim = rnn_hdim

        self.backbone_builder = {
            'PAC_Net': tvmodels.resnet18,
            'P_Net': tvmodels.resnet34,
            'C_Net': tvmodels.resnet34,
            'baseline': tvmodels.resnet50,
        }[model_name]
        self.backbone_weight = {
            'PAC_Net': tvmodels.ResNet18_Weights.DEFAULT,
            'P_Net': tvmodels.ResNet34_Weights.DEFAULT,
            'C_Net': tvmodels.ResNet34_Weights.DEFAULT,
            'baseline': tvmodels.ResNet50_Weights.DEFAULT,
        }[model_name] if pretrained else None

        self.rnn_cell = nn.GRUCell

        self.p_cell = self.cell_builder()
        self.c_cell = self.cell_builder()

    def forward(self, h: Tensor, frames: tuple[Tensor, Tensor]):
        diff_frame, frame = frames
        h = self.propagate(h, diff_frame)
        h = self.calibrate(h, frame)
        return h

    def propagate(self, h: Tensor, diff_frame: Tensor):
        feature = self.p_cell['feature_extractor'][diff_frame]
        return self.p_cell['rnn_cell'](input=feature, hx=h)

    def calibrate(self, h: Tensor, frame: Tensor):
        feature = self.c_cell['feature_extractor'][frame]
        return self.c_cell['rnn_cell'](input=feature, hx=h)

    def cell_builder(self):
        backbone = self.backbone_builder(weights=self.weight, progress=True)
        # backbone = self.backbone_builder(progress=True)
        backbone.fc = nn.Linear(backbone.fc.in_features, self.rnn_hdim)
        return nn.ModuleDict({
            'feature_extractor': backbone,
            'rnn_cell': self.rnn_cell(input_size=self.rnn_hdim, hidden_size=self.rnn_hdim)
        })


class PAC_Net_Base(nn.Module):
    def __init__(self, model_name: str, pretrained: bool,
                 rnn_type: str = 'gru', rnn_hdim: int = 128,
                 route_len: int = 128, max_peo: int = 3,
                 v_loss: bool = True, **kwargs):
        super(PAC_Net_Base, self).__init__()

        self.T = route_len
        T = route_len
        self.rnn_hdim = rnn_hdim
        self.v_loss = v_loss
        self.max_peo = max_peo

        assert model_name in ['PAC_Net', 'P_Net', 'C_Net', 'baseline']
        # CNN
        self.backbone_builder = {
            'PAC_Net': tvmodels.resnet18,
            'P_Net': tvmodels.resnet34,
            'C_Net': tvmodels.resnet34,
            'baseline': tvmodels.resnet50,
        }[model_name]
        self.backbone_weight = {
            'PAC_Net': tvmodels.ResNet18_Weights.DEFAULT,
            'P_Net': tvmodels.ResNet34_Weights.DEFAULT,
            'C_Net': tvmodels.ResNet34_Weights.DEFAULT,
            'baseline': tvmodels.ResNet50_Weights.DEFAULT,
        }[model_name] if pretrained else None

        # RNN
        rnn_dict = {
            'rnn': nn.RNNCell,
            'gru': nn.GRUCell,
        } if model_name == 'PAC_Net' else {
            'rnn': nn.RNN,
            'gru': nn.GRU,
            'lstm': nn.LSTM}
        self.rnn_builder = rnn_dict[rnn_type]

        # MLP
        mlp_dim = rnn_hdim // 2
        act = nn.Tanh if model_name == 'P_Net' else nn.Sigmoid
        # self.decoder = nn.Sequential(
        #     nn.Linear(rnn_hdim, mlp_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(mlp_dim, 2),
        #     act()
        # )
        self.pos_encoding = nn.Embedding(T, rnn_hdim)

        self.transformer = nn.Transformer(
            d_model=rnn_hdim, 
            nhead=4, 
            num_encoder_layers=2, 
            num_decoder_layers=2,
            batch_first=True,
            dim_feedforward=128,
        )
        self.query = nn.Embedding(self.max_peo, rnn_hdim)
        
        self.x_decoder = nn.Sequential(
            nn.Linear(rnn_hdim, T * 2),
            act()
        )

        assert rnn_hdim == 128
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hdim, mlp_dim),
            act(),
            nn.Linear(mlp_dim, self.max_peo + 1),
            nn.Softmax(dim=2),
        )
        self.criterion = nn.MSELoss(reduction='mean')
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, Ix: tuple):
        raise NotImplementedError

    def vis_forward(self, Ix: tuple, **kwargs):
        return self.forward(Ix)[1].detach().cpu()

    def compute_v_loss(self, x_pred: Tensor, x_gt: Tensor):
        # x_pred T*2
        v_pred = torch.sub(x_pred[1:,:], x_pred[:-1,:])
        v_gt = torch.sub(x_gt[1:,:], x_gt[1:,:])

        return self.criterion(v_pred, v_gt)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


class PAC_Net(PAC_Net_Base):
    def __init__(self, pretrained: bool = True,
                 rnn_type: str = 'gru', rnn_hdim: int = 128,
                 v_loss: bool = True, warm_up: int = 32, max_peo: int = 3, x_alpha: float = 1.0,
                 v_alpha: float = 1.0, m_alpha: float = 1.0, **kwargs):
        super(PAC_Net, self).__init__(model_name='PAC_Net', pretrained=pretrained,
                                      rnn_type=rnn_type, rnn_hdim=rnn_hdim, v_loss=v_loss, max_peo=max_peo)
        self.warmup_frames = warm_up

        self.x_alpha = x_alpha
        self.v_alpha = v_alpha
        self.m_alpha = m_alpha


        # predict module
        self.c_encoder, self.c_cell = self._make_model(self.backbone_builder, self.backbone_weight,
                                                       self.rnn_builder, rnn_hdim)
        self.p_encoder, self.p_cell = self._make_model(self.backbone_builder, self.backbone_weight,
                                                       self.rnn_builder, rnn_hdim)
        init_modules = [self.c_encoder.fc, self.p_encoder.fc]

        if self.warmup_frames > 0:
            # warmup module
            self.warmup_c_encoder, self.warmup_c_cell = self._make_model(self.backbone_builder, self.backbone_weight,
                                                                         self.rnn_builder, rnn_hdim)
            self.warmup_p_encoder, self.warmup_p_cell = self._make_model(self.backbone_builder, self.backbone_weight,
                                                                         self.rnn_builder, rnn_hdim)
            init_modules.extend([self.warmup_c_encoder.fc, self.warmup_p_encoder.fc])

        for m in init_modules:
            m.apply(self._init_weights)

    def forward(self, Ix: tuple):
        I, x_gt = Ix  # (B, C, T, H, W)
        delta_I = torch.sub(I[:, :, 1:], I[:, :, :-1]).float()
        B, T = x_gt.shape[:2] ## (b,T,6)
        if self.warmup_frames > 0:
            T -= self.warmup_frames
            warmup_I, I = I[:, :, :self.warmup_frames], I[:, :, self.warmup_frames:]
            warmup_delta_I, delta_I = delta_I[:, :, :self.warmup_frames], delta_I[:, :, self.warmup_frames:]
            hv_t = self.warm_up(warmup_I, warmup_delta_I)
        else:
            hv_t = None

        fx = self.c_encoder(rearrange(I.float(), 'b c t h w -> (b t) c h w'))
        fx = rearrange(fx, '(b t) d -> t b d', b=B)

        fv = self.p_encoder(rearrange(delta_I, 'b c t h w -> (b t) c h w'))
        fv = rearrange(fv, '(b t) d -> t b d', b=B)

        Hx = torch.zeros(B, T, self.rnn_hdim, device=x_gt.device)
        for t in range(T):
            hx_t = self.c_cell(input=fx[t], hx=hv_t)
            Hx[:, t] = hx_t
            if t < T - 1:
                hv_t = self.p_cell(input=fv[t], hx=hx_t)
        
        # print(Hx.shape) ## H;  b*T*hid
        # pdb.set_trace()
        pos_encoding = self.pos_encoding(torch.arange(T, device=Hx.device))
        # print(pos_encoding.shape)
        # pdb.set_trace()
        pos_encoding = pos_encoding.unsqueeze(0).repeat(B, 1, 1)
        Hx = Hx + pos_encoding

        tgt = self.query.weight.unsqueeze(0).repeat(B, 1, 1)
        npeo = tgt.shape[1]

        transformer_out = self.transformer(Hx, tgt)

        class_pred = self.classifier(transformer_out) ## b*maxpeo*maxpeo+1

        x_pred = self.x_decoder(transformer_out) ## b*maxpeo*2T
        ## b*5*2T -> b*T*10    2T:[x0,y0,x1,y1] -> [[x0,x1 ....],[y0,y1,.....]]
        x_pred = rearrange(x_pred, 'b n (t1 t2) -> b t1 (n t2)', t1=T, t2=2)   
        
        ## Todo: 匹配npeo&gt 
        cost_matrix = torch.zeros((npeo,npeo), device=x_gt.device, requires_grad=False)

        pred_matched = torch.zeros((B,T,npeo * 2), device=x_gt.device, requires_grad=False)
        x_loss_tot = 0
        v_loss_tot = 0
        m_loss_tot = 0      

        class_gt = torch.zeros((B, npeo, self.max_peo + 1), device=x_gt.device) ## class: idx==0 -> empty 只分为有无人
        valid_peo = []
        pred_peo = []
        for b in range(B):
            for n in range(npeo):
                trace = x_gt[b,:,2*n:2*n+2]
                if torch.sum(trace) == trace.shape[0]:  ## 空的填充为0.5， trace: T*2 sum为t
                    class_gt[b,n,0] = 1
                else:
                    class_gt[b,n,n+1] = 1
            with torch.no_grad():
                valid_peo.append(npeo - torch.sum(class_gt[b,:,0]))

        with torch.no_grad():
            match = []
            for b in range(B):
                p = torch.argmax(class_pred[b,:,:],dim=1)
                pred_peo.append(npeo - torch.sum(p==0))
                for i in range(npeo):
                    for j in range(npeo):
                        x_loss = self.criterion(x_pred[b,:,2*i:2*i+2], x_gt[b,self.warmup_frames:,2*j:2*j+2])
                        v_loss = self.compute_v_loss(x_pred[b,:,2*i:2*i+2], x_gt[b,self.warmup_frames:,2*j:2*j+2])
                        # m_loss = self.criterion(class_pred[b,i,:], class_gt[b,j,:])
                        m_loss = self.cross_entropy(class_pred[b,i,:], class_gt[b,j,:])
                        cost_matrix[i,j] = x_loss * self.x_alpha + v_loss * self.v_alpha + m_loss * self.m_alpha
            
                match_res = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
                match.append(match_res[1])

        cnt = 0
        for b in range(B):
            m = match[b]
            for i in range(npeo):
                m_loss_tot += self.cross_entropy(class_pred[b,i,:], class_gt[b,m[i],:])
                pred_matched[b,:,2*i:2*i+2] = x_pred[b,:,2*m[i]:2*m[i]+2]
                x_loss_tot += self.criterion(x_pred[b,:,2*i:2*i+2], x_gt[b,self.warmup_frames:,2*m[i]:2*m[i]+2])
                v_loss_tot += self.compute_v_loss(x_pred[b,:,2*i:2*i+2], x_gt[b,self.warmup_frames:,2*m[i]:2*m[i]+2])
            cnt += 1 if valid_peo[b] == pred_peo[b] else 0

        return (x_loss_tot,v_loss_tot,m_loss_tot), (pred_matched, cnt/B)


        # # x_pred = self.decoder(transformer_out)  # * self.factor
        # # loss_match = 
        # loss_x = self.criterion(x_pred, x_gt[:, self.warmup_frames:])
        # loss_v = self.compute_v_loss(x_pred, x_gt[:, self.warmup_frames:]) if self.v_loss else torch.tensor(0)

        # return (loss_x, loss_v), x_pred

    def warm_up(self, I: Tensor, delta_I: Tensor):
        B, T = I.size(0), I.size(2)
        fx = self.warmup_c_encoder(rearrange(I.float(), 'b c t h w -> (b t) c h w'))
        fx = rearrange(fx, '(b t) d -> t b d', b=B)

        fv = self.warmup_p_encoder(rearrange(delta_I, 'b c t h w -> (b t) c h w'))
        fv = rearrange(fv, '(b t) d -> t b d', b=B)

        hv_t = torch.zeros(B, self.rnn_hdim, device=I.device)
        for t in range(T):
            hx_t = self.warmup_c_cell(input=fx[t], hx=hv_t)
            hv_t = self.warmup_p_cell(input=fv[t], hx=hx_t)

        return hv_t

    @staticmethod
    def _make_model(backbone_builder, weight, rnn_cell, rnn_hdim):
        encoder = backbone_builder(weights=weight, progress=True)
        # encoder = backbone_builder(progress=True)
        encoder.fc = nn.Linear(encoder.fc.in_features, rnn_hdim)

        return encoder, rnn_cell(input_size=rnn_hdim, hidden_size=rnn_hdim)

    def vis_forward(self, Ix: tuple, phase: str = 'x'):
        I, x_gt = Ix  # (B, C, T, H, W)
        delta_I = torch.sub(I[:, :, 1:], I[:, :, :-1]).float()
        B, T = x_gt.shape[:2]
        H = torch.zeros(B, T, self.rnn_hdim, device=I.device)

        # warm up
        tic = time.time()
        if self.warmup_frames > 0:
            T -= self.warmup_frames
            warmup_I, I = I[:, :, :self.warmup_frames], I[:, :, self.warmup_frames:]
            warmup_delta_I, delta_I = delta_I[:, :, :self.warmup_frames], delta_I[:, :, self.warmup_frames:]
            warmup_fx = self.warmup_c_encoder(rearrange(I.float(), 'b c t h w -> (b t) c h w'))
            warmup_fx = rearrange(warmup_fx, '(b t) d -> t b d', b=B)

            warmup_fv = self.warmup_p_encoder(rearrange(warmup_delta_I, 'b c t h w -> (b t) c h w'))
            warmup_fv = rearrange(warmup_fv, '(b t) d -> t b d', b=B)

            hv_t = torch.zeros(B, self.rnn_hdim, device=I.device)
            for t in range(self.warmup_frames):
                hx_t = self.warmup_c_cell(input=warmup_fx[t], hx=hv_t)
                hv_t = self.warmup_p_cell(input=warmup_fv[t], hx=hx_t)
                H[:, t] = hx_t if phase == 'x' else hv_t
        else:
            hv_t = None

        # tracking
        fx = self.c_encoder(rearrange(I.float(), 'b c t h w -> (b t) c h w'))
        fx = rearrange(fx, '(b t) d -> t b d', b=B)

        fv = self.p_encoder(rearrange(delta_I, 'b c t h w -> (b t) c h w'))
        fv = rearrange(fv, '(b t) d -> t b d', b=B)

        Hx = torch.zeros(B, T, self.rnn_hdim, device=x_gt.device)
        for t in range(T):
            hx_t = self.c_cell(input=fx[t], hx=hv_t)
            Hx[:, t] = hx_t
            if t < T - 1:
                hv_t = self.p_cell(input=fv[t], hx=hx_t)
            H[:, t + self.warmup_frames] = hx_t if phase == 'x' else hv_t

        pos_encoding = self.pos_encoding(torch.arange(T, device=Hx.device))
        pos_encoding = pos_encoding.unsqueeze(0).repeat(B, 1, 1)
        Hx = Hx + pos_encoding

        tgt = torch.rand(B, self.max_peo, self.rnn_hdim, device=Hx.device)
        npeo = tgt.shape[1]

        transformer_out = self.transformer(Hx, tgt)
        ## b*3*hdim
        class_pred = self.classifier(transformer_out) ## b*3*4

        x_pred = self.x_decoder(transformer_out) ## b*3*2T
        ## b*3*2T -> b*T*6    2T:[x0,y0,x1,y1] -> [[x0,x1 ....],[y0,y1,.....]]
        x_pred = rearrange(x_pred, 'b n (t1 t2) -> b t1 (n t2)', t1=T, t2=2) 


        toc = time.time()
        fps = 320/(toc-tic)

        return x_pred.detach().cpu()  # , fps


class P_Net(PAC_Net_Base):
    def __init__(self, pretrained: bool = True,
                 rnn_type: str = 'gru', rnn_hdim: int = 128, rnn_layer: int = 2,
                 v_loss: bool = True, **kwargs):
        super(P_Net, self).__init__(model_name='P_Net', pretrained=pretrained,
                                    rnn_type=rnn_type, rnn_hdim=rnn_hdim, v_loss=v_loss)

        self.encoder = self.backbone_builder(weights=self.backbone_weight, progress=True)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, rnn_hdim)

        self.rnn = self.rnn_builder(input_size=rnn_hdim, hidden_size=rnn_hdim, num_layers=rnn_layer, batch_first=True)

        for m in [self.encoder.fc, self.decoder]:
            m.apply(self._init_weights)

    def forward(self, Ix: tuple):
        ori_I, x_gt = Ix  # (B, C, T, H, W)
        delta_I = torch.sub(ori_I[:, :, 1:], ori_I[:, :, :-1]).float()
        B, T = x_gt.shape[:2]

        delta_I = rearrange(delta_I, 'b c t h w -> (b t) c h w')
        fv = self.encoder(delta_I)
        fv = rearrange(fv, '(b t) d -> b t d', b=B)

        Hv = self.rnn(fv)[0]
        v_pred = self.decoder(Hv)  # * self.factor  # (B, T, 2)
        x_pred = torch.zeros_like(x_gt)
        x_pred[:, 0] = x_gt[:, 0]
        for t in range(T - 1):
            x_pred[:, t + 1] = x_pred[:, t] + v_pred[:, t]

        loss_x = self.criterion(x_pred, x_gt)
        loss_v = self.compute_v_loss(v_pred, x_gt) if self.v_loss else torch.tensor(0)

        return (loss_x, loss_v), x_pred

    def compute_v_loss(self, v_pred: Tensor, x_gt: Tensor):
        v_gt = torch.sub(x_gt[:, 1:], x_gt[:, :-1])
        return self.criterion(v_pred, v_gt)


class C_Net(PAC_Net_Base):
    def __init__(self, pretrained: bool = False,
                 rnn_type: str = 'gru', rnn_hdim: int = 128, rnn_layer: int = 2,
                 v_loss: bool = True, warm_up: int = 32):
        super(C_Net, self).__init__(model_name='C_Net', pretrained=pretrained,
                                    rnn_type=rnn_type, rnn_hdim=rnn_hdim,
                                    v_loss=v_loss)
        self.warmup_frames = warm_up

        self.encoder = self.backbone_builder(weights=self.backbone_weight, progress=True)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, rnn_hdim)
        self.rnn = self.rnn_builder(input_size=rnn_hdim, hidden_size=rnn_hdim, num_layers=rnn_layer,
                                    batch_first=True)
        init_modules = [self.encoder.fc, self.decoder]

        if self.warmup_frames > 0:
            self.warmup_encoder = self.backbone_builder(weights=self.backbone_weight, progress=True)
            self.warmup_encoder.fc = nn.Linear(self.warmup_encoder.fc.in_features, rnn_hdim)
            self.warmup_rnn = self.rnn_builder(input_size=rnn_hdim, hidden_size=rnn_hdim, num_layers=rnn_layer,
                                               batch_first=True)
            init_modules.append(self.warmup_encoder.fc)

        for m in init_modules:
            m.apply(self._init_weights)

    def forward(self, Ix):
        I, x_gt = Ix
        B, T = x_gt.shape[:2]
        if self.warmup_frames > 0:
            T -= self.warmup_frames
            warmup_I, I = I[:, :, :self.warmup_frames], I[:, :, self.warmup_frames:]
            hx = self.warm_up(warmup_I.float())
        else:
            hx = None

        fx = self.encoder(rearrange(I.float(), 'b c t h w -> (b t) c h w'))
        fx = rearrange(fx, '(b t) d -> b t d', b=B)
        Hx = self.rnn(fx, hx)[0]  # (B, T, D)
        x_pred = self.decoder(Hx)  # * self.factor  # (B, T, 2)

        loss_x = self.criterion(x_pred, x_gt[:, self.warmup_frames:])
        loss_v = self.compute_v_loss(x_pred, x_gt[:, self.warmup_frames:]) if self.v_loss else torch.tensor(0)

        return (loss_x, loss_v), x_pred

    def vis_forward(self, Ix: tuple, **kwargs):
        I, x_gt = Ix
        B, T = x_gt.shape[:2]
        H = torch.zeros(B, T, self.rnn_hdim, device=x_gt.device)

        # warmup
        if self.warmup_frames > 0:
            T -= self.warmup_frames
            warmup_I, I = I[:, :, :self.warmup_frames], I[:, :, self.warmup_frames:]
            warmup_fx = self.warmup_encoder(rearrange(warmup_I.float(), 'b c t h w -> (b t) c h w'))
            warmup_fx = rearrange(warmup_fx, '(b t) d -> b t d', b=B)
            Hx, hx = self.warmup_rnn(warmup_fx)
            H[:, :self.warmup_frames] = Hx
        else:
            hx = None

        # tracking
        fx = self.encoder(rearrange(I.float(), 'b c t h w -> (b t) c h w'))
        fx = rearrange(fx, '(b t) d -> b t d', b=B)
        Hx = self.rnn(fx, hx)[0]  # (B, T, D)
        H[:, self.warmup_frames:] = Hx
        x_pred = self.decoder(H) * self.factor  # (B, T, 2)

        return x_pred.detach().cpu()

    def warm_up(self, I: Tensor):
        B = I.size(0)
        fx = self.warmup_encoder(rearrange(I, 'b c t h w -> (b t) c h w'))
        fx = rearrange(fx, '(b t) d -> b t d', b=B)

        hx = self.warmup_rnn(fx)[1]  # (2, B, D)

        return hx


class NLOS_baseline(PAC_Net_Base):
    def __init__(self, pretrained: bool = False, rnn_hdim=128,
                 v_loss: bool = True, **kwargs):
        super(NLOS_baseline, self).__init__(model_name='C_Net', pretrained=pretrained, rnn_hdim=rnn_hdim,
                                            v_loss=v_loss)

        self.encoder = self.backbone_builder(weights=self.backbone_weight, progress=True)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, rnn_hdim)

        init_modules = [self.encoder.fc, self.decoder]
        for m in init_modules:
            m.apply(self._init_weights)

    def forward(self, Ix):
        I, x_gt = Ix
        B, T = x_gt.shape[:2]

        fx = self.encoder(rearrange(I.float(), 'b c t h w -> (b t) c h w'))
        fx = rearrange(fx, '(b t) d -> b t d', b=B)
        x_pred = self.decoder(fx)  # * self.factor  # (B, T, 2)

        loss_x = self.criterion(x_pred, x_gt)
        loss_v = self.compute_v_loss(x_pred, x_gt) if self.v_loss else torch.tensor(0)

        return (loss_x, loss_v), x_pred
