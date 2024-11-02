#from DeepPurpose import DTI as models
from DeepPurpose.encoders import transformer, MPNN, MLP, DGL_GCN
from DeepPurpose.utils import *
# from DeepPurpose.utils import *
from DeepPurpose.dataset import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
# from alphafold2_pytorch.alphafold2 import *
from DeepPurpose.model_helper import *
import scipy.sparse as sp
from torch_geometric.nn import GCNConv, GATConv, GINConv, Sequential
from torch_geometric.nn import APPNP, global_add_pool, global_mean_pool, global_max_pool     

class Model(nn.Module):
    def __init__(self, config, device):
        super(Model, self).__init__() 
        self.config = config
        self.device = device
        self.drop_path_prob = 0.0
        #encoder
        if self.config.drug_1d_encoder:
            self.Transformer_drug = transformer('drug', **vars(self.config))
        if self.config.drug_2d_encoder:
            self.DGL_GCN = DGL_GCN(74, [self.config.gnn_hid_dim_drug] * self.config.gnn_num_layer, [self.config.gnn_activation] * self.config.gnn_num_layer, self.config.hidden_dim_drug)

        self.emb = Embeddings_score(self.config.input_dim_protein, self.config.msa_dim, self.config.max_p, self.config.transformer_dropout_rate)
        #self.emb = Embeddings_score(self.config.input_dim_protein, 64, self.config.max_p, self.config.transformer_dropout_rate)
        if self.config.score_encoder:
            self.Transformer_target_score =   transformer_score(**vars(self.config)) #change
        self.Transformer_target_local =  transformer('protein', **vars(self.config))
        if self.config.target_global_1d:
            self.Transformer_target_global =  transformer('protein', **vars(self.config))
        if self.config.target_2d_encoder:
            self.target_2d_GCN_model = GCNnet(self.config.msa_dim, self.config.target_2d_dim, 'mean')

        #pred
        if self.config.interaction:
            # self.MLP_Inter = MLP(self.config.transformer_emb_size_target + self.config.transformer_emb_size_drug + self.config.hidden_dim_drug + self.config.max_msa + self.config.max_p_2d+3*self.config.MLP_out_size_O, 1, self.config.MLP_hidden_size_list_O)
            #
            # self.MLP_Inter = MLP(self.config.transformer_emb_size_target + self.config.target_2d_dim + self.config.transformer_emb_size_drug + self.config.hidden_dim_drug + 2*self.config.MLP_out_size_O, 1, self.config.MLP_hidden_size_list_O)
            self.MLP_Inter = MLP(3*self.config.MLP_out_size_O, 1, self.config.MLP_hidden_size_list_O)
            #self.MLP_Inter = MLP(2*self.config.MLP_out_size_O, 1, self.config.MLP_hidden_size_list_O)
            #interaction
            self.Inter_1d = MLP(self.config.transformer_emb_size_target + self.config.transformer_emb_size_drug, self.config.MLP_out_size_O, self.config.MLP_hidden_size_list_N)
            # self.Inter_2d = MLP(self.config.hidden_dim_druge + self.config.max_msa + self.config.max_p_2d, self.config.MLP_out_size_O, self.config.MLP_hidden_size_list_N)
            self.Inter_2d = MLP(self.config.hidden_dim_drug + self.config.target_2d_dim, self.config.MLP_out_size_O, self.config.MLP_hidden_size_list_N)
        else:

            self.MLP_NoInter = MLP(self.config.transformer_emb_size_target  + self.config.hidden_dim_drug, 1, self.config.MLP_hidden_size_list_N)


        
        self.Inter_mix = MLP(self.config.transformer_emb_size_target + self.config.transformer_emb_size_drug + self.config.hidden_dim_drug + self.config.target_2d_dim, self.config.MLP_out_size_O, self.config.MLP_hidden_size_list_N)


        #Mix
        if self.config.use_mix:
            self.MLP_pro_1d_drug = nn.Sequential(  #3层
                nn.Linear(self.config.transformer_emb_size_drug, self.config.Mix_hidden_size),
                nn.BatchNorm1d(self.config.Mix_hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.Mix_hidden_size, self.config.Mix_hidden_size),
                nn.BatchNorm1d(self.config.Mix_hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.Mix_hidden_size, self.config.Mix_out_size)
            )
            self.MLP_pro_2d_drug = nn.Sequential(  #3层
                nn.Linear(self.config.hidden_dim_drug , self.config.Mix_hidden_size),
                nn.BatchNorm1d(self.config.Mix_hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.Mix_hidden_size, self.config.Mix_hidden_size),
                nn.BatchNorm1d(self.config.Mix_hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.Mix_hidden_size, self.config.Mix_out_size)
            )
            self.MLP_pro_1d_target = nn.Sequential(  #3层
                nn.Linear(self.config.transformer_emb_size_target, self.config.Mix_hidden_size),
                nn.BatchNorm1d(self.config.Mix_hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.Mix_hidden_size, self.config.Mix_hidden_size),
                nn.BatchNorm1d(self.config.Mix_hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.Mix_hidden_size, self.config.Mix_out_size)
            )
            self.MLP_pro_2d_target = nn.Sequential(  #3层
                # nn.Linear(self.config.max_p_2d + self.config.max_msa, self.config.Mix_hidden_size),
                nn.Linear(self.config.target_2d_dim, self.config.Mix_hidden_size),
                nn.BatchNorm1d(self.config.Mix_hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.Mix_hidden_size, self.config.Mix_hidden_size),
                nn.BatchNorm1d(self.config.Mix_hidden_size),
                nn.ReLU(),
                nn.Linear(self.config.Mix_hidden_size, self.config.Mix_out_size)
            )

        self.target_2d_GCN_model = GCNnet(64, self.config.target_2d_dim, 'mean').to(self.device)

    def forward(self, data):
        loss = 0.0
        # msa_emb = None
        if self.config.interaction:
            if self.config.use_score:
                drug_2d, drug_1d, target_1d, target_2d, MSA, score, label = data
                #target encoder
                e = target_1d[0].to(self.device)
                msa = MSA[0].to(self.device)
                emb, msa_emb = self.emb(e, msa)
                # print("msa_emb.shape")
                # print(msa_emb.shape)
                if self.config.score_encoder:
                    e_mask =  target_1d[1].to(self.device)
                    mask_score_pre =  self.Transformer_target_score(emb, e_mask)
                    mask_score = torch.where(mask_score_pre < self.config.score_threshold, torch.zeros_like(mask_score_pre), torch.ones_like(mask_score_pre)) #筛选
                    target_mask_1d =  (target_1d[0], mask_score.to('cpu') * target_1d[1])
                    # loss_fn_score = torch.nn.MSELoss(reduction='mean')
                    loss_fn_score = torch.nn.MSELoss()
                    score = score.to(self.device)
                    score_loss = loss_fn_score(mask_score_pre, score)
                    loss += score_loss
                else:
                    mask_score = torch.where(score < self.config.score_threshold, torch.zeros_like(score), torch.ones_like(score)) #筛选 
                    target_mask_1d = (target_1d[0], target_1d[1]* mask_score)
                target_enc_1d = self.Transformer_target_local(target_mask_1d)
                if self.config.target_global_1d:
                    target_enc_1d_global = self.Transformer_target_global(target_1d)
                    target_enc_1d =  target_enc_1d.add(target_enc_1d_global)/2  #加权平均处理
            else:
                drug_2d, drug_1d, target_1d, target_2d, MSA, label = data
                #target encoder
                target_enc_1d =  self.Transformer_target_global(target_1d)
            
  
            #target_enc_2d = self.AlphaFold_emb(target_2d[0], MSA[0], target_2d[1], MSA[1]) #2d也是向量
            # target_enc_2d = self.target_2d_GCN(target_2d[0], msa_emb)
            target_enc_2d = self.target_2d_GCN(target_2d[0], msa_emb, self.target_2d_GCN_model.to(self.device))
            #print('0--target_enc_2d: ', target_enc_2d.mean())  
     


            #矩阵转向量
 
            # if self.config.use_mix:
            #     target_dual_loss = self.Mix_CL(target_enc_1d, target_enc_2d, 1.0, 'target')
            #     # print(loss, target_dual_loss)
            #     loss +=  target_dual_loss

            target_enc = torch.cat([target_enc_1d, target_enc_2d], dim=-1)
            #print('target_enc: ', target_enc.mean())  
            #drug encoder
            drug_enc_1d = self.Transformer_drug(drug_1d)
            #epodrug_2d = drug_2d[:-1] + [torch.squeeze(drug_2d[4])]
            
            #print(drug_2d.ndata['feat'])
            drug_enc_2d = self.DGL_GCN(drug_2d) 
    
            if self.config.use_mix:
                drug_dual_loss = self.Mix_CL(drug_enc_1d, drug_enc_2d, 1.0, 'drug') 
                loss +=  drug_dual_loss
            drug_enc = torch.cat([drug_enc_1d, drug_enc_2d], dim=-1)

                
            # interaction
            inter_1d = self.Inter_1d(torch.cat([drug_enc_1d, target_enc_1d], dim=-1))
            inter_2d = self.Inter_2d(torch.cat([drug_enc_2d, target_enc_2d], dim=-1))
            #print('--inter_2d: ', inter_2d.mean())
            inter_mix = self.Inter_mix(torch.cat([drug_enc, target_enc], dim=-1))
            #out_put = torch.cat([drug_enc, target_enc], dim=-1)
            # out_put = torch.cat([drug_enc, inter_1d, inter_2d, target_enc], dim=-1)
            #out_put = torch.cat([inter_1d, inter_2d], dim=-1)
            out_put = torch.cat([inter_1d, inter_2d, inter_mix], dim=-1)

            # print(out_put.shape)
            pred = self.MLP_Inter(out_put)
            #print('1--pred: ', pred.mean()) 
            label = label.float().to(self.device)
            if self.config.binary:
                loss_fct = torch.nn.BCELoss()
                m = torch.nn.Sigmoid()
                n = torch.squeeze(m(pred), 1)
                pred_loss = loss_fct(n, label)
            else:
                loss_fct = torch.nn.MSELoss()
                n = torch.squeeze(pred, 1)
                pred_loss = loss_fct(n, label) 
            loss += pred_loss
        else:
            if self.config.target_2d_encoder:
                if self.config.use_score:
                    drug_2d, drug_1d, target_1d, target_2d, MSA, score, label = data
                    #target encoder
                    e = target_1d[0].to(self.device)
                    msa = MSA[0].to(self.device)
                    emb, msa_emb = self.emb(e, msa)
                    if self.config.score_encoder:
                        e_mask =  target_1d[1].to(self.device)
                        mask_score_pre =  self.Transformer_target_score(emb, e_mask)
                        mask_score = torch.where(mask_score_pre < self.config.score_threshold, torch.zeros_like(mask_score_pre), torch.ones_like(mask_score_pre)) #筛选
                        target_mask_1d =  (target_1d[0], mask_score.to('cpu') * target_1d[1])
                        # loss_fn_score = torch.nn.MSELoss(reduction='mean')
                        loss_fn_score = torch.nn.MSELoss()
                        score = score.to(self.device)
                        score_loss = loss_fn_score(mask_score_pre, score)
                        loss += score_loss
                    else:
                        mask_score = torch.where(score < self.config.score_threshold, torch.zeros_like(score), score) #筛选 
                        target_mask_1d = target_1d[0] * mask_score
                        target_mask_1d = (target_mask_1d, target_1d[1])
                    target_enc_1d = self.Transformer_target_local(target_mask_1d)
                    if self.config.target_global_1d:
                        target_enc_1d_global = self.Transformer_target_global(target_1d)
                        target_enc_1d =  target_enc_1d.add(target_enc_1d_global)/2  #加权平均处理
                else:
                    drug_2d, drug_1d, target_1d, target_2d, MSA, label = data
                    #target encoder
                    target_enc_1d =  self.Transformer_target_global(target_1d)


                
                #矩阵转向量

                target_enc = target_enc_1d

                #drug encoder
                drug_enc_1d = self.Transformer_drug(drug_1d)
                #drug_2d = drug_2d[:-1] + [torch.squeeze(drug_2d[4])]
                drug_enc_2d = self.DGL_GCN(drug_2d)   

                if self.config.use_mix:
                    drug_dual_loss = self.Mix_CL(drug_enc_1d, drug_enc_2d, 1.0, 'drug') 
                    loss +=  drug_dual_loss
                drug_enc = torch.cat([drug_enc_1d, drug_enc_2d], dim=-1)

                out_put = torch.cat([drug_enc, target_enc], dim=-1)
                pred = self.MLP_Inter(out_put)
                label = label.float().to(self.device)
                if self.config.binary:
                    loss_fct = torch.nn.BCELoss()
                    m = torch.nn.Sigmoid()
                    n = torch.squeeze(m(pred), 1)
                    pred_loss = loss_fct(n, label)
                else:
                    loss_fct = torch.nn.MSELoss()
                    n = torch.squeeze(pred, 1)
                    pred_loss = loss_fct(n, label)   
                loss += pred_loss
            else:
                if self.config.use_score:
                    if self.config.drug_1d_encoder:
                        drug_1d, target_1d, MSA, score, label = data
                        drug_enc = self.Transformer_drug(drug_1d)
                    if self.config.drug_2d_encoder:
                        drug_2d, target_1d, MSA, score, label = data
                        drug_enc = self.DGL_GCN(drug_2d) 

                    #target encoder
                    if self.config.score_encoder:
                        e = target_1d[0].to(self.device)
                        msa = MSA[0].to(self.device)
                        emb, msa_emb = self.emb(e, msa)
                        e_mask =  target_1d[1].to(self.device)
                        mask_score_pre =  self.Transformer_target_score(emb, e_mask)
                        mask_score = torch.where(mask_score_pre < self.config.score_threshold, torch.zeros_like(mask_score_pre), torch.ones_like(mask_score_pre)) #筛选
                        target_mask_1d =  (target_1d[0], mask_score.to('cpu') * target_1d[1])
                        # loss_fn_score = torch.nn.MSELoss(reduction='mean')
                        loss_fn_score = torch.nn.MSELoss()
                        score = score.to(self.device)
                        score_loss = loss_fn_score(mask_score_pre, score)
                        loss += score_loss
                    else:
                        mask_score = torch.where(score < self.config.score_threshold, torch.zeros_like(score), torch.ones_like(score)) #筛选 
                        
                        target_mask_1d = (target_1d[0], target_1d[1]* mask_score)
             
                    
                    target_enc = self.Transformer_target_local(target_mask_1d)
                    if self.config.target_global_1d:
                        target_enc_1d_global = self.Transformer_target_global(target_1d)
                        target_enc =  target_enc.add(target_enc_1d_global)/2 #加权平均处理
                    
                else:
                    if self.config.drug_1d_encoder:
                        drug_1d, target_1d, label = data
                        drug_enc = self.Transformer_drug(drug_1d)
                    if self.config.drug_2d_encoder:
                        drug_2d, target_1d, label = data
                        #drug_2d = drug_2d[:-1] + [torch.squeeze(drug_2d[4])]
                        #drug_enc = self.MPNN(drug_2d) 
                        drug_enc =  self.DGL_GCN(drug_2d) 

                    #target encoder
                    target_enc =  self.Transformer_target_global(target_1d)
                out_put = torch.cat([drug_enc, target_enc], dim=-1)
                pred = self.MLP_NoInter(out_put)
                label = label.float().to(self.device)
                #label = Variable(torch.from_numpy(np.array(label).cpu()).float()).to(self.device)
                if self.config.binary:
                    loss_fct = torch.nn.BCELoss()
                    m = torch.nn.Sigmoid()
                    n = torch.squeeze(m(pred), 1)
                    pred_loss = loss_fct(n, label)
                else:
                    loss_fct = torch.nn.MSELoss()
                    n = torch.squeeze(pred, 1)
                    pred_loss = loss_fct(n, label)
                loss += pred_loss
        return pred, loss
  
    def Mix_CL(self, data_1d, data_2d, temperature, encoding=None):
        '''
        Trans_out = self.Transformer_1d(data_1d)
        GNN_out = self.GNN_2d(data_2d)  #最后的 pool 是 mean+max
        GNN_out_1 = self.GNN_Pool_mean(GNN_out)
        GNN_out_2 = self.GNN_Pool_max(GNN_out)
        GNN_Pool_out = torch.cat(GNN_out_1, GNN_out_2)
        '''
        # print(encoding, data_1d.shape, data_2d.shape)  # [bs, dim], [bs, dim]
        if encoding == 'drug':
            project_1d = self.MLP_pro_1d_drug(data_1d)
            project_2d = self.MLP_pro_2d_drug(data_2d)
            # with torch.no_grad():
            #     project_1d_SG = self.MLP_pro_1d_drug(data_1d)
            #     project_2d_SG = self.MLP_pro_2d_drug(data_2d)
        elif encoding == 'target':
            project_1d = self.MLP_pro_1d_target(data_1d)
            project_2d = self.MLP_pro_2d_target(data_2d)
            # with torch.no_grad():
                # project_1d_SG = self.MLP_pro_1d_target(data_1d)
                # project_2d_SG = self.MLP_pro_2d_target(data_2d)
        # pre_1d = self.MLP_1d(project_1d)
        # pre_2d = self.MLP_2d(project_2d)
        # pos = torch.einsum("nc,nc->n", [project_1d,project_2d]).unsqueeze(-1)

        # print(project_1d.shape, project_2d.shape)
        features = torch.cat([project_1d.unsqueeze(1),project_2d.unsqueeze(1)], dim=1) # [batchsize, 2, feature_dim] 

        b, n, dim = features.size()
        assert (n == 2)
        mask = torch.eye(b, dtype=torch.float32).to(self.device)

        # contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_features = torch.cat([project_1d, project_2d], dim=0)
        anchor = project_1d

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / temperature  # 两两之间的相似度

        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2)

        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).to(self.device), 0)
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask # 分母的底

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Mean log-likelihood for positive
        CL_loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()
        # logits = torch.einsum("nc,ck->nk", [project_1d,project_2d.t()]) / temperature
        # # print(logits.shape)
        # labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        # # loss_fn = F.cosine_similarity
        # loss_fn = nn.CrossEntropyLoss()
        # CL_loss = loss_fn(logits, labels)
        # dual_loss = -loss_fn(pre_1d, project_2d_SG) - loss_fn(pre_2d, project_1d_SG)
        return 0.1 * CL_loss



    def target_2d_GCN(self, x, m, model):
        '''
        x: distance_map, [bs, max_len, max_len], 
        m: msa embd, [bs, max_len, dim_in], 
        model: GCNnet(dim_in, dim_out)
        '''
        x = x.to(self.device)
        m = m.to(self.device)
        bs = x.shape[0]
        x_input = m.reshape(-1, m.shape[2])
        # print(x_input.shape)

        # target_enc_2d_list = []
        edge_index_list = []
        batch_i = []
        for i in range(bs):
            # x_input = m[i]
            dis_flag = (x[i] >= 0.5).detach().cpu().numpy().astype(int)   # adj matrix
            # dis_flag[np.eye(x[i].shape[0], dtype=bool)]=0
            s = sp.coo_matrix(dis_flag)
            edge_index = torch.LongTensor(np.vstack((s.row,s.col)) + x.shape[2] * i).to(self.device)
            # print(edge_index)
            # out = model(x_input, edge_index)
            # print(out.shape)   # [max_len, dim_out]
            # target_enc_2d_list.append(out.mean(0).unsqueeze(0))  # [1, dim_out]
            edge_index_list.append(edge_index)
            batch_i.append(torch.LongTensor([i]).repeat(x.shape[1]))
        # target_enc_2d = torch.cat(target_enc_2d_list, dim=0)
        edge_index_total = torch.cat(edge_index_list, dim=1)
        # print(x_input.shape, edge_index_total.shape, edge_index_total)
        batch_total = torch.cat(batch_i, dim=0).to(self.device)
        # print(batch_total)
        target_enc_2d = model(x_input, edge_index_total, batch_total)
        target_enc_2d = target_enc_2d.reshape(bs, -1, m.shape[2]).mean(1)
        # print(target_enc_2d.shape)

        return target_enc_2d   # [bs, dim_out]




class transformer_score(nn.Sequential):
    def __init__(self, **config):
        super().__init__()
        
        self.encoder = Encoder_MultipleLayers(config['transformer_n_layer_target'], 
                                                    config['msa_dim'],
                                                    config['transformer_intermediate_size_target'], 
                                                    32,
                                                    config['transformer_attention_probs_dropout'],
                                                    config['transformer_hidden_dropout_rate'])
        self.lin = nn.Linear(config['msa_dim'],  1)
    ### parameter v (tuple of length 2) is from utils.drug2emb_encoder 
    def forward(self, emb, e_mask):
        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0

        encoded_layers = self.encoder(emb.float(), ex_e_mask.float())
        m = torch.nn.Sigmoid()
        encoded_layers = m(self.lin(encoded_layers))   # [bs, max_len, 1]
        # print(encoded_layers.shape)
        # return encoded_layers[:,0]
        return encoded_layers.squeeze(-1)   # [bs, max_len]


class Embeddings_score(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)
        self.hidden_size = hidden_size
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, target_ids, input_ids):
        seq_length = target_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=target_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(target_ids)

        words_embeddings = self.word_embeddings(target_ids)
        msa_embeddings = self.word_embeddings(input_ids.reshape([-1, self.hidden_size]))
        position_embeddings = self.position_embeddings(position_ids)
        msa_embeddings = msa_embeddings.reshape([target_ids.shape[0], -1, target_ids.shape[1] ,self.hidden_size]).mean(1, keepdim=False)
        
        # print(words_embeddings.shape, position_embeddings.shape, msa_embeddings.shape)
        embeddings = words_embeddings + position_embeddings + msa_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings,  msa_embeddings


class GCNnet(nn.Module):
    def __init__(self, dim_in, dim_out, pooling):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_in)
        self.gcn2 = GCNConv(dim_in, dim_in)
        self.lin = nn.Linear(dim_in, dim_out)
        self.dropout = nn.Dropout(0.1)
        self.pooling = pooling

    def reset_parameters(self):
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self,  x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        # print(x.shape)
        x = self.dropout(F.relu(self.gcn1(x, edge_index)))
        x = self.gcn2(x, edge_index)
        # print(x.shape)
        if self.pooling == 'mean':
            out = global_mean_pool(x, batch)
        elif self.pooling == 'add':
            out = global_add_pool(x, batch)
        elif self.pooling == 'max':
            out = global_max_pool(x, batch)
        out = self.dropout(F.relu(self.lin(out)))
        # print(out.shape)
        # return F.log_softmax(out, dim=-1)
        return out