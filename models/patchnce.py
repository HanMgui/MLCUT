from packaging import version
import torch
from torch import nn



"""
Note this is different from the vanilla CUT. I use cosine similarity here,
but both implementations are identical (after the L2 normalization).
You may choose class PatchNCEloss2 (used in vanilla CUT, no cosine similarity), by replacing line 5 in ./dcl_model.py
or ./simdcl_model.py with "from .patchnce import PatchNCELoss2".
"""

'''
class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.similarity_function = self._get_similarity_function()
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    def _get_similarity_function(self):

        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        return self._cosine_simililarity

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, M, C)
        # v shape: (N, M)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        feat_k = feat_k.detach()
        l_pos = self.cos(feat_q,feat_k)
        l_pos = l_pos.view(batchSize, 1)
        l_neg_curbatch = self.similarity_function(feat_q.view(batchSize,1,-1),feat_k.view(1,batchSize,-1))
        l_neg_curbatch = l_neg_curbatch.view(1,batchSize,-1)
        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(batchSize, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, batchSize)
        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T  # 这是把同一个batch里的所有点都当负样本了啊
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        return loss


# Used in vanilla CUT
class PatchNCELoss2(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
               
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1)) 
        l_pos = l_pos.view(batchSize, 1)  

        # neg logit
        if self.opt.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T#(256*b,1)(256*b,256)   (256*b,1)(256*b,256*b)

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        return loss

        # # neg logit

        # # Should the negatives from the other samples of a minibatch be utilized?
        # # In CUT and FastCUT, we found that it's best to only include negatives
        # # from the same image. Therefore, we set
        # # --nce_includes_all_negatives_from_minibatch as False
        # # However, for single-image translation, the minibatch consists of
        # # crops from the "same" high-resolution image.
        # # Therefore, we will include the negatives from the entire minibatch.
        # if self.opt.nce_includes_all_negatives_from_minibatch:
        #     # reshape features as if they are all negatives of minibatch of size 1.
        #     batch_dim_for_bmm = 1
        # else:
        #     batch_dim_for_bmm = self.opt.batch_size

class PatchNCELoss2_K(nn.Module): #hmg添加,使用这个函数说明满足opt.usehmgmodification and opt.K_num_patches>0
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
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

    def forward(self, feat_q, feat_k,nce_T_L_i=0.07):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        if self.opt.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        # npatches = feat_q.size(1)
        l_all = torch.bmm(feat_q, feat_k.transpose(2, 1))
        l_pos=torch.tensor([]).to(feat_q.device)
        for b in range(batch_dim_for_bmm):
            l_pos=torch.cat((l_pos,l_all[b][self.rangeq,self.rangek].unsqueeze(dim=0)),dim=0)  #copy.deepcopy(l_all[0][self.rangeq,self.rangek_b])好像不需要copy
            l_all[b][self.rangeq,self.rangek]=-100 #把q*k+都置负很大
        l_pos = l_pos.view(batchSize, 1) 
        sortnum=l_all.topk(feat_q.shape[1],dim=2,sorted=False)[0] #与其排序后去掉小的，不如直接用取得的结果，反正算损失的时候也不按顺序算
        l_neg=sortnum.view(batchSize,-1)

        out = torch.cat((l_pos, l_neg), dim=1) / nce_T_L_i
        # a=torch.cat((l_pos, l_neg), dim=1)
        # num,ind=a.sort()
        # (num[:,-1]-num[:,-2]).sort()[0]

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        return loss
'''
import numpy #hmg添加
from collections import Counter
class PatchNCELoss2_all(nn.Module):#hmg添加
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.fw=self.forward_o
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

    def forward_o(self, feat_q, feat_k,nce_T_L_i=None,saveneg=False):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
               
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1)) 
        l_pos = l_pos.view(batchSize, 1)  

        # neg logit
        if self.opt.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        if saveneg:#hmg测试，保存负样本的相似度            
            b=torch.floor(l_neg*100)/100
            b=b[0].to('cpu').detach().numpy()
            r=Counter(b)
            f=open('/media/cvlab/data/Projects/hmg/DCLGAN-MAIN/compare_im/Negative_hist/'+self.opt.pretrained_name+'.txt','a')
            for rk in r.keys():
                if rk>-9:
                    f.write('%.2f:%d'%(rk,r[rk]))
                    f.write('\n')
            f.close()

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T#(256*b,1)(256*b,256)   (256*b,1)(256*b,256*b)

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        return loss

    def forward_K(self, feat_q, feat_k,nce_T_L_i=0.07,saveneg=False):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        if self.opt.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        # npatches = feat_q.size(1)
        l_all = torch.bmm(feat_q, feat_k.transpose(2, 1))
        l_pos=torch.tensor([]).to(feat_q.device)
        for b in range(batch_dim_for_bmm):
            l_pos=torch.cat((l_pos,l_all[b][self.rangeq,self.rangek].unsqueeze(dim=0)),dim=0)  #copy.deepcopy(l_all[0][self.rangeq,self.rangek_b])好像不需要copy
            l_all[b][self.rangeq,self.rangek]=-100 #把q*k+都置负很大
        l_pos = l_pos.view(batchSize, 1) 
        sortnum=l_all.topk(feat_q.shape[1],dim=2,sorted=False)[0] #与其排序后去掉小的，不如直接用取得的结果，反正算损失的时候也不按顺序算
        l_neg=sortnum.view(batchSize,-1)

        if saveneg:#hmg测试，保存负样本的相似度            
            
            b=torch.floor(l_neg[128]/l_pos[128]*100)/100
            b=b.to('cpu').detach().numpy()
            r=Counter(b)
            f=open('/media/cvlab/data/Projects/hmg/DCLGAN-MAIN/compare_im/Negative_hist/'+self.opt.pretrained_name+'.txt','a')
            for rk in r.keys():
                if rk>-9:
                    f.write('%.2f:%d'%(rk,r[rk]))
                    f.write('\n')
            f.close()


        out = torch.cat((l_pos, l_neg), dim=1) / nce_T_L_i
        # a=torch.cat((l_pos, l_neg), dim=1)
        # num,ind=a.sort()
        # (num[:,-1]-num[:,-2]).sort()[0]

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        return loss

    def forward(self, feat_q, feat_k,nce_T_L_i=0.07,saveneg=False):
        return self.fw(feat_q, feat_k,nce_T_L_i=nce_T_L_i,saveneg=saveneg)