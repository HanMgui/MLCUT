from packaging import version
import torch
from torch import nn

import copy #hmg添加

"""
Add dckw(in 2110.06848) to patchnce.
"""

SMALL_NUM = torch.log(torch.tensor(1e-45))#np.log(1e-45)


class PatchNCELoss2_all(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.fw=self.forward_o
        self.weight_fn=self.wf
        print('self.weight_fn=self.wf')
        if self.opt.usehmgmodification and self.opt.K_num_patches>0:
            self.fw=self.forward_K #覆盖掉self.fw=self.forward_o
            if self.opt.nce_includes_all_negatives_from_minibatch:
                self.rangeq=torch.tensor(list(range(0,opt.num_patches*opt.batch_size)))
                rangek=[]
                for b in range(opt.batch_size):
                    rangek+=list(range(b*opt.K_num_patches*opt.num_patches,(b*opt.K_num_patches+1)*opt.num_patches))
                self.rangek=torch.tensor(rangek)
            else:
                self.rangeq=torch.tensor(list(range(0,opt.num_patches)))#(q0,k0)(q1,k1)(q2,k2)是l_neg里正样本的位置，这些位置需要置非常小的值
                self.rangek=torch.tensor(list(range(0,opt.num_patches)))
            if len(opt.gpu_ids)>0:
                self.rangeq.cuda(opt.gpu_ids[0])
                self.rangek.cuda(opt.gpu_ids[0])

    def wf(self,feat_q,feat_k):
        a=2 - feat_q.size(0) * torch.nn.functional.softmax((feat_q * feat_k).sum(dim=1) / 0.5, dim=0).squeeze()
        return a

    def forward_o(self, feat_q, feat_k,nce_T_L_i=None):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()
        if self.opt.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size
        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        # npatches = feat_q.size(1)

        cross_view_distance = torch.bmm(feat_q, feat_k.transpose(2, 1))/self.opt.nce_T  #直接在这里除tao吧，反正正负都要除
        positive_loss=torch.tensor([], device=feat_q.device)
        for i in range(batch_dim_for_bmm):
            positive_loss= torch.cat((positive_loss,(-torch.diag(cross_view_distance[i])).unsqueeze(dim=0)),dim=0)
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(feat_q, feat_k)
        neg_similarity = cross_view_distance
        neg_mask = torch.eye(feat_q.size(1), device=feat_q.device)   
        negative_loss = torch.logsumexp(neg_similarity + neg_mask * SMALL_NUM, dim=1, keepdim=False)
        return (positive_loss + negative_loss).mean()

    def forward_K(self, feat_q, feat_k,nce_T_L_i=0.07):
        #记得在dcl_model.py里改一下引用的代码
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()
        if self.opt.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size
        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)

        cross_view_distance = torch.bmm(feat_q, feat_k.transpose(2, 1))/nce_T_L_i
        positive_loss=torch.tensor([], device=feat_q.device)
        for b in range(batch_dim_for_bmm):
            positive_loss=torch.cat((positive_loss,cross_view_distance[b][self.rangeq,self.rangek].unsqueeze(dim=0)),dim=0)  #copy.deepcopy(l_all[0][self.rangeq,self.rangek_b])好像不需要copy
            cross_view_distance[b][self.rangeq,self.rangek]+=SMALL_NUM #把q*k+都置负很大
        sortnum=cross_view_distance.topk(feat_q.shape[1],dim=2,sorted=False)[0] #与其排序后去掉小的，不如直接用取得的结果，反正算损失的时候也不按顺序算
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(feat_q, sortnum)         
        neg_similarity=sortnum.view(batchSize,-1)
        negative_loss = torch.logsumexp(neg_similarity, dim=1, keepdim=False).view(batch_dim_for_bmm,-1) 
        return (positive_loss + negative_loss).mean()


    def forward(self, feat_q, feat_k,nce_T_L_i=0.07):
        return self.fw(feat_q, feat_k,nce_T_L_i=nce_T_L_i)