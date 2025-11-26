import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Any, Dict, Optional
from einops import rearrange
from modal_2d.classifier import VitBlock, PatchEmbedding2D
from utils_2d.warp import Warper2d, warp2D
from modal_2d.dvss import VSSBlock_new,VSSLayer
from modal_2d.LASA import LRSA


image_warp = warp2D()
def project(x, image_size):
    """将 torch.Size([1, 512, 768]) 转换成图像形状[channel, W/p, H/p]"""
    W, H = image_size[0], image_size[1]
    x = rearrange(x, 'b (w h) hidden -> b w h hidden', w=W // 16, h=H // 16)
    x = x.permute(0, 3, 1, 2)
    return x
def img_warp(flow, I):

    return Warper2d()(flow, I)
def flow_integration_ir(flow1, flow2, flow3, flow4, flow5, flow6, flow7, flow8, flow9, flow10):
    up1 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
    up2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
    up3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    flow1, flow2 = up1(flow1)*16, up1(flow2)*16
    flow3, flow4 = up2(flow3)*8, up2(flow4)*8
    flow5, flow6 = up3(flow5)*4, up3(flow6)*4
    flow7, flow8 = up4(flow7)*2, up4(flow8)*2
    flow_neg = flow1 + flow3 + flow5 + flow7 + flow9
    flow_pos = flow2 + flow4 + flow6 + flow8 + flow10
    flow = flow_pos - flow_neg
    return flow, flow_neg, flow_pos

def reg(flow, feature):
    feature = Warper2d()(flow, feature)
    return feature


#----
def flow_integration_ir_adaptive(flows, use_learned_weights=False, weights=None):
    """根据已有层数自适应融合光流，支持学习权重。
    flows 顺序为 [flow1, flow2, ..., flowK]，奇数为负向，偶数为正向。
    不同层的放大因子对应原固定5层的 [16, 8, 4, 2, 1]。
    """
    
    if len(flows) == 10:
        return flow_integration_ir(*flows)

    # 生成放大因子表
    scale_factors = [16, 8, 4, 2, 1]
    # 预计算所有需要的上采样操作，避免重复创建
    upsamplers = {}
    for scale in scale_factors:
        if scale == 1:
            upsamplers[scale] = lambda x: x
        else:
            upsamplers[scale] = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True).to(device)

    device = flows[0].device
    flow_neg = None
    flow_pos = None

    # 遍历已存在的成对流
    num_pairs = len(flows) // 2
    for i in range(num_pairs):
        neg_flow = flows[2 * i]
        pos_flow = flows[2 * i + 1]
        scale = scale_factors[i]

           # 应用权重（如果提供）
        if use_learned_weights and weights is not None:
            weight = weights[i] if i < len(weights) else 1.0
            neg_flow = neg_flow * weight
            pos_flow = pos_flow * weight
        
        # 上采样和缩放
        neg_scaled = upsamplers[scale](neg_flow) * scale
        pos_scaled = upsamplers[scale](pos_flow) * scale
        
        # 累积流场
        if flow_neg is None:
            flow_neg = neg_scaled
            flow_pos = pos_scaled
        else:
            flow_neg = flow_neg + neg_scaled
            flow_pos = flow_pos + pos_scaled

    flow = flow_pos - flow_neg
    return flow, flow_neg, flow_pos

def flow_integration_ir_optimized(flows, target_size=None):
    """优化的流场集成，支持批处理和内存优化"""
    if len(flows) == 10:
        return flow_integration_ir(*flows)
    
    device = flows[0].device
    batch_size = flows[0].shape[0]
    
    # 如果指定了目标尺寸，直接上采样到该尺寸
    if target_size is not None:
        target_h, target_w = target_size
        scale_factors = [16, 8, 4, 2, 1]
        
        # 批处理所有流场的上采样
        scaled_flows = []
        for i in range(0, len(flows), 2):
            neg_flow, pos_flow = flows[i], flows[i+1]
            scale = scale_factors[i//2]
            
            # 计算目标尺寸
            target_h_scale = target_h // scale
            target_w_scale = target_w // scale
            
            # 上采样
            neg_scaled = F.interpolate(neg_flow, size=(target_h_scale, target_w_scale), 
                                     mode='bilinear', align_corners=True) * scale
            pos_scaled = F.interpolate(pos_flow, size=(target_h_scale, target_w_scale), 
                                     mode='bilinear', align_corners=True) * scale
            
            scaled_flows.extend([neg_scaled, pos_scaled])
        
        # 求和
        flow_neg = torch.stack([scaled_flows[i] for i in range(0, len(scaled_flows), 2)], dim=0).sum(dim=0)
        flow_pos = torch.stack([scaled_flows[i] for i in range(1, len(scaled_flows), 2)], dim=0).sum(dim=0)
        
        flow = flow_pos - flow_neg
        return flow, flow_neg, flow_pos
    else:
        return flow_integration_ir_adaptive(flows)

class model_classifer_lite(nn.Module):
    def __init__(self, in_c, num_heads, num_vit_blk, img_size, patch_size):
        super(model_classifer_lite, self).__init__()
        self.hidden_size = 256
        self.embedding = PatchEmbedding2D(in_c = in_c, embedding_dim=256, patch_size=patch_size)

        self.vit_blks = nn.Sequential()  #添加n个vit块
        for i in range(num_vit_blk):
            self.vit_blks.add_module(
                name=f'vit{i}',
                module=VitBlock(hidden_size=256,
                         num_heads=num_heads,
                         vit_drop=0.1, qkv_bias=False,
                         mlp_dim=256, mlp_drop=0.0)
            )
        self.norm = nn.LayerNorm(256)
        self.head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                  nn.GELU(),
                                  nn.Linear(self.hidden_size, 2))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

    def forward(self, x):
        class_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.vit_blks(x)
        class_token = x[:, 0]
        predict = self.head(class_token)
        return predict, class_token, x[:, 1:]

class Classifier_lite(nn.Module):
    def __init__(self, in_c, num_heads, num_vit_blk, img_size, patch_size):
        super(Classifier_lite, self).__init__()
        self.hidden_size = 256
        self.embedding = PatchEmbedding2D(in_c = in_c, embedding_dim=256, patch_size=patch_size)

        self.vit_blks = nn.Sequential()  #添加n个vit块
        for i in range(num_vit_blk):
            self.vit_blks.add_module(
                name=f'vit{i}',
                module=VitBlock(hidden_size=256,
                         num_heads=num_heads,
                         vit_drop=0.1, qkv_bias=False,
                         mlp_dim=256, mlp_drop=0.0)
            )
        self.norm = nn.LayerNorm(256)
        self.head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                  nn.GELU(),
                                  nn.Linear(self.hidden_size, 2))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

    def forward(self, x):
        x = self.embedding(x)  # image_embedding
        class_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.vit_blks(x)
        class_token = x[:, 0]
        predict = self.head(class_token)
        return predict, class_token, x[:, 1:]

class Transfer(nn.Module):
    def __init__(self, num_vit, num_heads, img_size, patch_size):
        super(Transfer, self).__init__()
        self.num_vit = num_vit
        self.num_heads = num_heads
        self.hidden_dim = 256
        self.img_size = img_size
        self.patch_size = patch_size
        self.cls1 = nn.Parameter(torch.zeros(1, 1, 256))
        self.cls2 = nn.Parameter(torch.zeros(1, 1, 256))
        self.VitBLK1 = nn.Sequential()
        for i in range(self.num_vit):
            self.VitBLK1.add_module(name=f'vit{i}',
                                    module=VitBlock(hidden_size=self.hidden_dim,
                                                    num_heads=self.num_heads,
                                                    vit_drop=0.0,
                                                    qkv_bias=False,
                                                    mlp_dim=256,
                                                    mlp_drop=0.0))
        self.VitBLK2 = nn.Sequential()
        for i in range(self.num_vit):
            self.VitBLK2.add_module(name=f'vit{i}',
                                    module=VitBlock(hidden_size=self.hidden_dim,
                                                    num_heads=self.num_heads,
                                                    vit_drop=0.0,
                                                    qkv_bias=False,
                                                    mlp_dim=256,
                                                    mlp_drop=0.0))
        
        # 添加LASA模块用于特征转换过程中的局部注意力增强
        self.lasa1 = LRSA(dim=256, qk_dim=64, mlp_dim=512, heads=4)
        self.lasa2 = LRSA(dim=256, qk_dim=64, mlp_dim=512, heads=4)
        
    def forward(self, x1, x2, cls1, cls2):
        cls1, cls2 = cls1.unsqueeze(dim=1), cls2.unsqueeze(dim=1)
        cls1 = cls1.expand(-1, x1.shape[1], -1)
        cls2 = cls2.expand(-1, x1.shape[1], -1)
        x1, x2 = x1+cls2, x2 + cls1
        class_token1 = self.cls1.expand(x1.shape[0], -1, -1)
        class_token2 = self.cls2.expand(x1.shape[0], -1, -1)
        # x1, x2 = self.MLP1(x1), self.MLP2(x2)
        x1 = torch.cat((x1, class_token1), dim=1)
        x2 = torch.cat((x2, class_token2), dim=1)
        x1 = self.VitBLK1(x1)
        x2 = self.VitBLK2(x2)
        
        # 使用LASA模块增强转换后的特征
        x1_patches = x1[:, 1:]  # 移除class token
        x2_patches = x2[:, 1:]  # 移除class token
        
        # 转换为图像格式进行LASA处理
        x1_img = project(x1_patches, self.img_size)
        x2_img = project(x2_patches, self.img_size)
        
        # 应用LASA增强
        x1_enhanced = self.lasa1(x1_img, self.patch_size)
        x2_enhanced = self.lasa2(x2_img, self.patch_size)
        
        # 转换回序列格式
        x1_enhanced = x1_enhanced.flatten(2).transpose(1, 2)
        x2_enhanced = x2_enhanced.flatten(2).transpose(1, 2)
        
        class_token1 = x1[:, 0, :]
        class_token2 = x2[:, 0, :]
        return  x1_enhanced, x2_enhanced, class_token1, class_token2

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 使用 dvss 的 VSSLayer 替换原有 Restormer
        # 先用 1x1 卷积将通道映射到 8，再经过 VSSLayer；第二段继续用 VSSLayer 后再映射到 3 通道
        self.conv_in = nn.Conv2d(1, 8, kernel_size=1, stride=1, padding=0)
        self.vss1 = VSSLayer(dim=8, depth=1)
        self.vss2 = VSSLayer(dim=8, depth=1)
        self.conv_out = nn.Conv2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, img):
        # BCHW -> 1x1 conv to 8ch -> BHWC for VSSLayer -> back to BCHW
        x = self.conv_in(img)
        f = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        f = self.vss1(f)
        f = f.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        f_ = f.permute(0, 2, 3, 1)
        f_ = self.vss2(f_)
        f_ = f_.permute(0, 3, 1, 2).contiguous()
        f_ = self.conv_out(f_)
        return f, f_


class ModelTransfer_lite(nn.Module):
    def __init__(self, num_vit, num_heads, img_size):
        super(ModelTransfer_lite, self).__init__()
        self.img_size = img_size
        self.patch_size = 16  # 设置patch size
        self.transfer = Transfer(num_vit=num_vit, num_heads=num_heads, img_size=self.img_size, patch_size=self.patch_size)
        self.classifier = Classifier_lite(in_c=3, num_heads=4, num_vit_blk=2, img_size=self.img_size, patch_size=self.patch_size)
        self.modal_dis = model_classifer_lite(in_c=3, num_heads=4, num_vit_blk=2, img_size=self.img_size, patch_size=self.patch_size)

    def forward(self, img1, img2):

        pre1, cls1, x1_ = self.classifier(img1)
        pre2, cls2, x2_ = self.classifier(img2)
        x1, x2, new_cls1, new_cls2 = self.transfer(x1_, x2_, cls1, cls2)
        feature_pred1, _, _ = self.modal_dis(x1)
        feature_pred2, _, _ = self.modal_dis(x2)
        return  pre1, pre2, feature_pred1, feature_pred2, x1, x2, x1_, x2_  # 分类器预测结果，特征转换器分类结果


class CrossAttention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CrossAttention, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1),
                                   # nn.InstanceNorm2d(in_channel),
                                   nn.ReLU(),
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1),
                                   # nn.InstanceNorm2d(in_channel),
                                   nn.ReLU(),
                                   )
    def forward(self, f1, f2):
        f1_hat = f1
        f1 = self.conv1(f1)
        f2 = self.conv2(f2)
        att_map = f1 * f2
        att_shape = att_map.shape
        att_map = torch.reshape(att_map, [att_shape[0], att_shape[1], -1])
        att_map = F.softmax(att_map, dim=2)
        att_map = torch.reshape(att_map, att_shape)
        f1 = f1 * att_map
        f1 = f1 + f1_hat
        return f1

class ResBlk(nn.Module):
    def __init__(self, in_channel):
        super(ResBlk, self).__init__()
        self.feature_output = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )
    def forward(self ,x):
        return x + self.feature_output(x)

class FusionRegBlk_lite(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FusionRegBlk_lite, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(kernel_size=1, stride=1, padding=0, in_channels=in_channel * 2, out_channels=in_channel),
            nn.LeakyReLU())

        self.crossAtt1 = CrossAttention(in_channel, out_channel)
        # self.crossAtt2 = CrossAttention(in_channel, out_channel)
        self.feature_output = nn.Sequential(
            ResBlk(in_channel),
        )

        self.flow_output = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=2, kernel_size=3, padding=1, stride=1),
            nn.Tanh(),
            # nn.Conv2d(2, 2, 1, 1, 0),
        )
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, padding=0, stride=1),
                                 nn.LeakyReLU(),)

    def forward(self, f1, f2): # f2是cat后的特征

        f2 = self.conv1x1(f2)
        f1 = self.crossAtt1(f1, f2) + self.crossAtt1(f2, f1)
        f1 = self.feature_output(f1)
        f2 = self.flow_output(f1)  # 从此开始f2是flow
        f1 = self.up1(f1)
        return f1, f2


class UpBlk(nn.Module):
    def __init__(self, in_c, out_c):
        super(UpBlk, self).__init__()
        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=1),
        )
        self.conv1 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1, stride=1)
        self.in1 = nn.InstanceNorm2d(num_features=out_c)


    def forward(self, x):
        x = self.up_sample(x)
        x = self.conv1(x)
        x = self.in1(x)
        return F.leaky_relu(x)

class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.channels = channels

        self.up1 = UpBlk(self.channels[0], self.channels[1])
        self.up2 = UpBlk(self.channels[1], self.channels[2])
        self.up3 = UpBlk(self.channels[2], self.channels[3])
        self.up4 = UpBlk(self.channels[3], self.channels[4])

class RegNet_lite(nn.Module):
    def __init__(self, adaptive=True, min_steps=2, stop_threshold=0.05, stop_metric='max', 
                 channels=[256, 64, 32, 16, 8, 1], max_steps=5):
        super(RegNet_lite, self).__init__()
        self.adaptive = adaptive
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.stop_threshold = stop_threshold
        self.stop_metric = stop_metric
        self.channels = channels
        
        # 使用nn.ModuleList动态创建模块，减少重复代码
        self.f1_fusion_blocks = nn.ModuleList()
        self.f2_fusion_blocks = nn.ModuleList()
        
        # 动态创建融合配准模块
        for i in range(max_steps):
            self.f1_fusion_blocks.append(
                FusionRegBlk_lite(in_channel=self.channels[i], out_channel=self.channels[i+1])
            )
            self.f2_fusion_blocks.append(
                FusionRegBlk_lite(in_channel=self.channels[i], out_channel=self.channels[i+1])
            )
        
        self.D = Decoder(self.channels)
        
        # 添加流场权重学习模块，用于自适应融合
        self.flow_weights = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(2, 8, 1),
                nn.ReLU(),
                nn.Conv2d(8, 1, 1),
                nn.Sigmoid()
            ) for _ in range(max_steps)
        ])

    def _should_stop(self, flow_tensor):
        """检查流场是否收敛，支持向量化计算"""
        mag = torch.norm(flow_tensor, dim=1)
        if self.stop_metric == 'mean':
            val = mag.mean()
        else:
            val = mag.amax()
        return val.item() < self.stop_threshold

    def _compute_weighted_flows(self, flows):
        """计算加权融合的流场"""
        weighted_flows = []
        for i, (flow1, flow2) in enumerate(zip(flows[::2], flows[1::2])):
            # 使用学习到的权重进行自适应融合
            weight = self.flow_weights[i](flow1)
            weighted_flow = weight * flow1 + (1 - weight) * flow2
            weighted_flows.extend([flow1, weighted_flow])
        return weighted_flows

    def forward(self, f1, f2):
        flows = []
        current_step = 0
        
        # 获取上采样层
        upsamplers = [self.D.up1, self.D.up2, self.D.up3, self.D.up4]
        
        # 循环处理每个步骤，支持早期停止
        for step in range(self.max_steps):
            # 特征拼接
            f_cat = torch.cat((f1, f2), dim=1)
            
            # 使用动态索引访问融合模块
            f1_, flow1 = self.f1_fusion_blocks[step](f1, f_cat)
            f2_, flow2 = self.f2_fusion_blocks[step](f2, f_cat)
            flows.extend([flow1, flow2])
            
            # 特征配准和上采样
            f1 = reg(flow1, f1)
            f2 = reg(flow2, f2)
            
            # 上采样到下一层
            if step < len(upsamplers):
                f1 = upsamplers[step](f1)
                f2 = upsamplers[step](f2)
            
            current_step = step + 1
            
            # 自适应停止检查
            if self.adaptive:
                if (self._should_stop(flow1) and self._should_stop(flow2) and 
                    current_step >= self.min_steps):
                    break
        
        # 计算加权流场融合
        if self.adaptive and len(flows) > 2:
            weighted_flows = self._compute_weighted_flows(flows)
            flow, flow_neg, flow_pos = flow_integration_ir_adaptive(weighted_flows)
        else:
            flow, flow_neg, flow_pos = flow_integration_ir_adaptive(flows)

        return f1, f2, flows, flow, flow_neg, flow_pos



class UpSampler_V2(nn.Module):
    def __init__(self, in_c, out_c):
        super(UpSampler_V2, self).__init__()
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.up3 =  nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, AU_F, BU_F, feature):
        AU_F = self.up1(AU_F)
        BU_F = self.up1(BU_F)
        feature = self.up3(feature)
        return AU_F, BU_F, feature

class VSSBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.pre = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.pre = nn.Identity()
        self.block = VSSBlock_new(hidden_dim=out_channels, **kwargs)
        self.post = nn.Identity()

    def forward(self, x):
        x = self.pre(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.block(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.post(x)
        return x

class FusionNet_lite(nn.Module):
    def __init__(self):
        super(FusionNet_lite, self).__init__()
        self.cn = [256, 64, 32, 16, 12, 8]

        
        self.F1 = VSSBlock2D(in_channels=self.cn[0]*2, out_channels=self.cn[1])
        self.up_sample1 = UpSampler_V2(in_c=self.cn[0], out_c=self.cn[1])

        
        self.F2 = VSSBlock2D(in_channels=self.cn[1]*3, out_channels=self.cn[2])
        self.up_sample2 = UpSampler_V2(in_c=self.cn[1], out_c=self.cn[2])

       
        self.F3 = VSSBlock2D(in_channels=self.cn[2]*3, out_channels=self.cn[3])
        self.up_sample3 = UpSampler_V2(in_c=self.cn[2], out_c=self.cn[3])

       
        self.F4 = VSSBlock2D(in_channels=self.cn[3]*3, out_channels=self.cn[4])
        self.up_sample4 =nn.Upsample(scale_factor=2, mode='bilinear')
      
        self.outLayer = nn.Sequential(VSSBlock2D(in_channels=self.cn[4] + 16, out_channels=self.cn[4]),
                                      VSSBlock2D(in_channels=self.cn[4], out_channels=1),
                                      nn.Sigmoid())

    def forward(self, AS_F, BS_F, AU_F, BU_F, flow):
        """
            AU_F, BU_F是原图同尺度
        """

        flow_d = F.upsample(flow, BU_F.size()[2:4], mode='bilinear') / 16
        BU_F_w = img_warp(flow_d, BU_F)
        feature = self.F1(torch.cat((AU_F, BU_F_w), dim=1))  # 通道数降低了

        AU_F, BU_F, feature = self.up_sample1(AU_F, BU_F, feature)
        flow_d = F.upsample(flow, BU_F.size()[2:4], mode='bilinear') / 8
        BU_F_w = img_warp(flow_d, BU_F)
        feature = self.F2(torch.cat([torch.cat([AU_F, BU_F_w], dim=1), feature], dim=1))

        AU_F, BU_F, feature = self.up_sample2(AU_F, BU_F, feature)
        flow_d = F.upsample(flow, BU_F.size()[2:4], mode='bilinear') / 4
        BU_F_w = img_warp(flow_d, BU_F)
        feature = self.F3(torch.cat([torch.cat([AU_F, BU_F_w], dim=1), feature], dim=1))

        AU_F, BU_F, feature = self.up_sample3(AU_F, BU_F, feature)
        flow_d = F.upsample(flow, BU_F.size()[2:4], mode='bilinear') / 2
        BU_F_w = img_warp(flow_d, BU_F)
        feature = self.F4(torch.cat([torch.cat([AU_F, BU_F_w], dim=1), feature], dim=1))

        feature = self.up_sample4(feature)
        BS_F_w = img_warp(flow, BS_F)
        S_F = torch.cat([AS_F, BS_F_w], dim=1)
        feature = self.outLayer(torch.cat([feature, S_F], dim=1))
        return feature

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    print(param_count)



# 测试代码
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = Encoder().to(device)
    count_param(encoder)
    transfer = ModelTransfer_lite(2,4, [256,256]).to(device)
    count_param(transfer)
    reg_net = RegNet_lite().to(device)
    count_param(reg_net)
    fusion_net = FusionNet_lite().to(device)
    count_param(fusion_net)

    img1 = torch.rand(1, 1, 256, 256).to(device)
    img2 = torch.rand(1, 1, 256, 256).to(device)
    AS_F, feature1 = encoder(img1)
    BS_F, feature2 = encoder(img2)

    pre1, pre2, feature_pred1, feature_pred2, feature1, feature2, AU_F, BU_F = transfer(feature1, feature2) # 分类器， 模态鉴别器， 转换后特征， 转换前特征

    feature1 = project(feature1, [256, 256]).to(device)
    feature2 = project(feature2, [256, 256]).to(device)
    AU_F = project(AU_F, [256, 256]).to(device)
    BU_F = project(BU_F, [256, 256]).to(device)

    _, _, flows, flow, _, _ = reg_net(feature1, feature2)
    fusion_img = fusion_net(AS_F, BS_F, AU_F, BU_F, flow)
    print(fusion_img.shape)

    warped_img2 = image_warp(img2, flow)

