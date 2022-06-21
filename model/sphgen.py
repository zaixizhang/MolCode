import torch
import torch.nn as nn
import torch.nn.functional as F
from .spherenet import SphereNet
from .rgcn import RGCN
from .net_utils import *
from .geometric_computing import *
from .att import MH_ATT
from  rdkit import Chem


class SphGen(nn.Module):
    def __init__(self, cutoff, num_node_types, num_edge_types, num_layers,
        hidden_channels, int_emb_size, basis_emb_size, out_emb_channels,
        num_spherical, num_radial, num_flow_layers, deq_coeff=0.9, use_gpu=True, n_att_heads=4):

        super(SphGen, self).__init__()
        self.use_gpu = use_gpu
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.emb_size = hidden_channels
        
        self.feat_net3d = SphereNet(cutoff, num_node_types, num_layers, 
            hidden_channels, int_emb_size, basis_emb_size, out_emb_channels,
            num_spherical, num_radial)

        self.feat_net2d = RGCN(num_node_types, nhid=hidden_channels, nout=hidden_channels, edge_dim=3,
                         num_layers=3, dropout=0., normalization=False)
        self.node_masks, self.adj_masks, self.link_prediction_index, self.flow_core_edge_masks = self.initialize_masks(max_node_unroll=35, max_edge_unroll=12)
        self.repeat_num = self.node_masks.size(0)
        self.graph_size = 35

        self.mask_node = nn.Parameter(self.node_masks.view(1, self.repeat_num, self.graph_size, 1), requires_grad=False)  # (1, repeat_num, n, 1)
        self.mask_edge = nn.Parameter(self.adj_masks.view(1, self.repeat_num, 1, self.graph_size, self.graph_size), requires_grad=False)  # (1, repeat_num, 1, n, n)
        self.index_select_edge = nn.Parameter(self.link_prediction_index, requires_grad=False)  # (edge_step_length, 2)
        
        node_feat_dim, dist_feat_dim, angle_feat_dim, torsion_feat_dim = hidden_channels * 2, hidden_channels * 2, hidden_channels * 3, hidden_channels * 4

        self.node_flow_layers = nn.ModuleList([ST_Net_Exp(node_feat_dim, num_node_types, hid_dim=hidden_channels, bias=True) for _ in range(num_flow_layers)])
        self.dist_flow_layers = nn.ModuleList([ST_Net_Exp(dist_feat_dim, 1, hid_dim=hidden_channels, bias=True) for _ in range(num_flow_layers)])
        self.angle_flow_layers = nn.ModuleList([ST_Net_Exp(angle_feat_dim, 1, hid_dim=hidden_channels, bias=True) for _ in range(num_flow_layers)])
        self.torsion_flow_layers = nn.ModuleList([ST_Net_Exp(torsion_feat_dim, 1, hid_dim=hidden_channels, bias=True) for _ in range(num_flow_layers)])
        self.focus_mlp = MLP(hidden_channels)
        self.deq_coeff = deq_coeff
        self.edge_flow_layers = nn.ModuleList([ST_Net_Exp(hidden_channels * 3, self.num_edge_types, hid_dim=hidden_channels, bias=True) for _ in range(num_flow_layers)])
        
        self.node_att = MH_ATT(n_att_heads, q_dim=hidden_channels, k_dim=hidden_channels, v_dim=hidden_channels, out_dim=hidden_channels)
        self.dist_att = MH_ATT(n_att_heads, q_dim=hidden_channels, k_dim=hidden_channels, v_dim=hidden_channels, out_dim=hidden_channels)
        self.angle_att = MH_ATT(n_att_heads, q_dim=2*hidden_channels, k_dim=hidden_channels, v_dim=hidden_channels, out_dim=hidden_channels)
        self.torsion_att = MH_ATT(n_att_heads, q_dim=3*hidden_channels, k_dim=hidden_channels, v_dim=hidden_channels, out_dim=hidden_channels)
        '''
        if use_gpu:
            self.node_att, self.dist_att, self.angle_att, self.torsion_att = self.node_att.to('cuda'), self.dist_att.to('cuda'), self.angle_att.to('cuda'), self.torsion_att.to('cuda')
            self.feat_net3d = self.feat_net3d.to('cuda')
            self.node_flow_layers = self.node_flow_layers.to('cuda')
            self.dist_flow_layers = self.dist_flow_layers.to('cuda')
            self.angle_flow_layers = self.angle_flow_layers.to('cuda')
            self.torsion_flow_layers = self.torsion_flow_layers.to('cuda')
            self.focus_mlp = self.focus_mlp.to('cuda')
        '''


    def forward(self, data_batch):
        '''
        # 2D embeddings
        x = data_batch['atom_array']  # (B, N, node_dim)
        adj = data_batch['adj_array']  # (B, 4, N, N)
        adj_cont = adj[:, :, self.flow_core_edge_masks.type(torch.bool)].clone()  # (B, 4, edge_num)
        adj_cont = adj_cont.permute(0, 2, 1).contiguous()  # (B, edge_num, 4)
        adj_cont += self.deq_coeff * torch.rand(adj_cont.size(), device=adj_cont.device)  # (B, edge_num, 4)
        batch_size = x.size(0)
        adj = adj[:, :3]
        x = torch.where(self.mask_node, x.unsqueeze(1).repeat(1, self.repeat_num, 1, 1), torch.zeros([1], device=x.device)).view(-1, self.graph_size, self.num_node_types)  # (batch*repeat_num, N, 9)
        adj = torch.where(self.mask_edge, adj.unsqueeze(1).repeat(1, self.repeat_num, 1, 1, 1), torch.zeros([1], device=x.device)).view(-1, self.num_edge_types - 1, self.graph_size, self.graph_size)# (batch*repeat_num, 3, N, N)
        x_node_pred = data_batch['x']  # (B, N, node_dim)
        adj_node_pred = data_batch['adj']

        node_emb = self.feat_net2d(torch.cat([x_node_pred,x], dim=0), torch.cat([adj_node_pred, adj], dim=0)) # (batch*repeat_num, N, d)
        if hasattr(self, 'batchNorm'):
            node_emb = self.batchNorm(node_emb.transpose(1, 2)).transpose(1, 2) # (batch, N, d)

        graph_emb_node2d = torch.sum(node_emb[:x_node_pred.shape[0]], dim=1, keepdim=False)
        node_emb_edge = node_emb[x_node_pred.shape[0]:].contiguous().view(batch_size, self.repeat_num, self.graph_size, -1)  # (batch, repeat_num, N, d)
        graph_emb_edge = torch.sum(node_emb_edge, dim=2, keepdim=False) # (batch, repeat_num, d)

        # input for st_net_edge
        graph_emb_edge = graph_emb_edge.unsqueeze(2)  # (batch, repeat_num, 1, d)

        all_node_emb_edge = node_emb_edge # (batch, repeat_num, N, d)
        index = self.index_select_edge.view(1, -1, 2, 1).repeat(batch_size, 1, 1, self.emb_size)  # (batch_size, repeat_num, 2, d)
        graph_node_emb_edge = torch.cat((torch.gather(all_node_emb_edge, dim=2, index=index), graph_emb_edge),dim=2)  # (batch_size, repeat_num, 3, d)
        graph_node_emb_edge = graph_node_emb_edge.view(batch_size * self.repeat_num, -1)  # (batch_size * (repeat_num), 3*d)
        adj_deq = adj_cont.view(-1, self.num_edge_types)  # (batch*(repeat_num-N), 4)
        edge_latent, edge_log_jacob = flow_forward(self.edge_flow_layers, adj_deq, graph_node_emb_edge)
        '''
        #3d embeddings
        z, pos, batch = data_batch['atom_type'], data_batch['position'], data_batch['batch']
        node_feat = self.feat_net3d(z, pos, batch)
        focus_score = self.focus_mlp(node_feat)
        new_atom_type, focus = data_batch['new_atom_type'], data_batch['focus']
        x_z = F.one_hot(new_atom_type, num_classes=self.num_node_types).float()
        x_z += self.deq_coeff * torch.rand(x_z.size(), device=x_z.device)

        local_node_type_feat, query_batch = node_feat[focus[:,0]], batch[focus[:,0]]
        global_node_type_feat = self.node_att(local_node_type_feat, node_feat, node_feat, query_batch, batch)
        node_type_feat = torch.cat((local_node_type_feat, global_node_type_feat), dim=-1)
        node_latent, node_log_jacob = flow_forward(self.node_flow_layers, x_z, node_type_feat)
        node_type_emb_block = self.feat_net3d.init_e.emb
        node_type_emb = node_type_emb_block(new_atom_type)[batch]
        node_emb = node_feat * node_type_emb

        #predict bond type based on 3D embeddings
        new_bond_type, edge_index, focus_index = data_batch['new_bond_type'], data_batch['edge_index'], data_batch['focus_index']
        e_z = F.one_hot(new_bond_type, num_classes=self.num_edge_types).float()
        e_z += self.deq_coeff * torch.rand(e_z.size(), device=e_z.device)
        edge_type_feat = torch.cat((global_node_type_feat[edge_index], node_feat[focus_index], node_type_emb_block(new_atom_type)[edge_index]), dim=-1)
        edge_latent, edge_log_jacob = flow_forward(self.edge_flow_layers, e_z, edge_type_feat)

        #position prediction
        c1_focus, c2_c1_focus = data_batch['c1_focus'], data_batch['c2_c1_focus']
        dist, angle, torsion = data_batch['new_dist'], data_batch['new_angle'], data_batch['new_torsion']

        local_dist_feat, dist_query_batch = node_emb[focus[:,0]], batch[focus[:,0]]
        #local_dist_feat1 = node_type_emb[focus[:,0]]
        global_dist_feat = self.dist_att(local_dist_feat, node_emb, node_emb, dist_query_batch, batch, data_batch['new_atom_bond'])
        local_angle_feat, angle_query_batch = torch.cat((node_emb[c1_focus[:,1]], node_emb[c1_focus[:,0]]), dim=1), batch[c1_focus[:,0]]
        #local_angle_feat1 = node_type_emb[c1_focus[:, 0]]
        global_angle_feat = self.angle_att(local_angle_feat, node_emb, node_emb, angle_query_batch, batch, data_batch['new_atom_bond1'])
        local_torsion_feat, torsion_query_batch = torch.cat((node_emb[c2_c1_focus[:,2]], node_emb[c2_c1_focus[:,1]], node_emb[c2_c1_focus[:,0]]), dim=1), batch[c2_c1_focus[:,0]]
        #local_torsion_feat1 = node_type_emb[c2_c1_focus[:,0]]
        global_torsion_feat = self.torsion_att(local_torsion_feat, node_emb, node_emb, torsion_query_batch, batch, data_batch['new_atom_bond2'])

        dist_feat = torch.cat((local_dist_feat, global_dist_feat), dim=-1)
        angle_feat = torch.cat((local_angle_feat, global_angle_feat), dim=-1)
        torsion_feat = torch.cat((local_torsion_feat, global_torsion_feat), dim=-1)
        
        dist_latent, dist_log_jacob = flow_forward(self.dist_flow_layers, dist, dist_feat)
        angle_latent, angle_log_jacob = flow_forward(self.angle_flow_layers, angle, angle_feat)
        torsion_latent, torsion_log_jacob = flow_forward(self.torsion_flow_layers, torsion, torsion_feat)

        return (node_latent, node_log_jacob), (edge_latent, edge_log_jacob), focus_score, (dist_latent, dist_log_jacob), (angle_latent, angle_log_jacob), (torsion_latent, torsion_log_jacob)

    #random/targeted generation
    def generate_3d(self, type_to_atomic_number, num_gen=100, temperature=[1.0, 1.0, 1.0, 1.0, 1.0], min_atoms=2, max_atoms=35, focus_th=0.5, add_final=False):
        with torch.no_grad():
            if self.use_gpu:
                prior_node = torch.distributions.normal.Normal(torch.zeros([self.num_node_types]).cuda(), temperature[0] * torch.ones([self.num_node_types]).cuda())
                prior_dist = torch.distributions.normal.Normal(torch.zeros([1]).cuda(), temperature[1] * torch.ones([1]).cuda())
                prior_angle = torch.distributions.normal.Normal(torch.zeros([1]).cuda(), temperature[2] * torch.ones([1]).cuda())
                prior_torsion = torch.distributions.normal.Normal(torch.zeros([1]).cuda(), temperature[3] * torch.ones([1]).cuda())
                prior_edge = torch.distributions.normal.Normal(torch.zeros([self.num_edge_types]).cuda(), temperature[4] * torch.ones([self.num_edge_types]).cuda())
            else:
                prior_node = torch.distributions.normal.Normal(torch.zeros([self.num_node_types]), temperature[0] * torch.ones([self.num_node_types]))
                prior_dist = torch.distributions.normal.Normal(torch.zeros([1]), temperature[1] * torch.ones([1]))
                prior_angle = torch.distributions.normal.Normal(torch.zeros([1]), temperature[2] * torch.ones([1]))
                prior_torsion = torch.distributions.normal.Normal(torch.zeros([1]), temperature[3] * torch.ones([1]))
                prior_edge = torch.distributions.normal.Normal(torch.zeros([self.num_edge_types]),temperature[4] * torch.ones([self.num_edge_types]))

            node_type_emb_block = self.feat_net3d.init_e.emb
            z = torch.ones([num_gen, 1], dtype=int)
            pos = torch.zeros([num_gen, 1, 3], dtype=torch.float32)
            con_mat = torch.zeros([num_gen, max_atoms, max_atoms], dtype=torch.long)
            focuses = torch.zeros([num_gen, 0], dtype=int)
            if self.use_gpu:
                z, pos, focuses, con_mat = z.cuda(), pos.cuda(), focuses.cuda(), con_mat.cuda()
            out_dict = {}
            out_dict1 = {}

            mask_index = lambda mask, p: p[mask].view(num_gen, -1, 3)
            feat_index = lambda node_id, f: f[torch.arange(num_gen), node_id]
            pos_index = lambda node_id, p: p[torch.arange(num_gen), node_id].view(num_gen, 1, 3)

            for i in range(max_atoms-1):
                batch = torch.arange(num_gen, device=z.device).view(num_gen, 1).repeat(1, i + 1)
                if i == 0:
                    node_feat = node_type_emb_block(z.view(-1))
                elif i == 1:
                    node_feat = self.feat_net3d.dist_only_forward(z.view(-1), pos.view(-1, 3), batch.view(-1))
                else:
                    node_feat = self.feat_net3d(z.view(-1), pos.view(-1, 3), batch.view(-1))

                focus_score = self.focus_mlp(node_feat).view(num_gen, i + 1)
                can_focus = torch.logical_and(focus_score < focus_th, z > 0)
                complete_mask = (can_focus.sum(dim=-1) == 0)

                if i > max(0, min_atoms - 2) and torch.sum(complete_mask) > 0:
                    out_dict[i + 1] = {}
                    out_node_types = z[complete_mask].view(-1, i + 1).cpu().numpy()
                    out_dict[i + 1]['_atomic_numbers'] = type_to_atomic_number[out_node_types]
                    out_dict[i + 1]['_positions'] = pos[complete_mask].view(-1, i + 1, 3).cpu().numpy()
                    out_dict[i + 1]['_focus'] = focuses[complete_mask].view(-1, i).cpu().numpy()

                continue_mask = torch.logical_not(complete_mask)
                dirty_mask = torch.nonzero(torch.isnan(focus_score).sum(dim=-1))[:, 0]
                if len(dirty_mask) > 0:
                    continue_mask[dirty_mask] = False
                dirty_mask = torch.nonzero(torch.isinf(focus_score).sum(dim=-1))[:, 0]
                if len(dirty_mask) > 0:
                    continue_mask[dirty_mask] = False

                if torch.sum(continue_mask) == 0:
                    break

                node_feat = node_feat.view(num_gen, i + 1, -1)
                num_gen = torch.sum(continue_mask).cpu().item()
                z, pos, can_focus, focuses = z[continue_mask], pos[continue_mask], can_focus[continue_mask], focuses[continue_mask]
                con_mat = con_mat[continue_mask]
                focus_node_id = torch.multinomial(can_focus.float(), 1).view(num_gen)
                node_feat = node_feat[continue_mask]

                latent_node = prior_node.sample([num_gen])

                local_node_type_feat, query_batch = feat_index(focus_node_id, node_feat), torch.arange(num_gen, device=node_feat.device)
                key_value_batch = torch.arange(num_gen, device=node_feat.device).view(num_gen, 1).repeat(1, i + 1).view(-1)
                global_node_type_feat = self.node_att(local_node_type_feat, node_feat.view(num_gen * (i + 1), -1),node_feat.view(num_gen * (i + 1), -1), query_batch, key_value_batch, bond=None)

                node_type_feat = torch.cat((local_node_type_feat, global_node_type_feat), dim=-1)

                latent_node = flow_reverse(self.node_flow_layers, latent_node, node_type_feat)
                node_type_id = torch.argmax(latent_node, dim=1)
                node_type_emb = node_type_emb_block(node_type_id)
                node_emb = node_feat * node_type_emb.view(num_gen, 1, -1)

                #predict edge type
                latent_edge = prior_edge.sample([num_gen])
                edge_feat = torch.cat((global_node_type_feat, feat_index(focus_node_id, node_feat), node_type_emb.view(num_gen, -1)), dim=-1)
                latent_edge = flow_reverse(self.edge_flow_layers, latent_edge, edge_feat)
                edge = torch.argmax(latent_edge, dim=1)
                con_mat[torch.arange(num_gen), focus_node_id, i+1] = edge
                con_mat[torch.arange(num_gen), i+1, focus_node_id] = edge

                latent_dist = prior_dist.sample([num_gen])
                local_dist_feat = feat_index(focus_node_id, node_emb)
                global_dist_feat = self.dist_att(local_dist_feat, node_emb.view(num_gen * (i + 1), -1),node_emb.view(num_gen * (i + 1), -1), query_batch, key_value_batch, con_mat[:, i + 1, :i + 1].reshape(-1))
                dist_feat = torch.cat((local_dist_feat, global_dist_feat), dim=-1)
                dist = flow_reverse(self.dist_flow_layers, latent_dist, dist_feat)
                # dist = dist.abs()
                if i == 0:
                    new_pos = torch.cat((dist, torch.zeros_like(dist, device=dist.device), torch.zeros_like(dist, device=dist.device)),dim=-1)
                if i > 0:
                    mask = torch.ones([num_gen, i + 1], dtype=torch.bool)
                    mask[torch.arange(num_gen), focus_node_id] = False
                    c1_dists = torch.sum(torch.square(mask_index(mask, pos) - pos_index(focus_node_id, pos)), dim=-1)
                    c1_node_id = torch.argmin(c1_dists, dim=-1)
                    c1_node_id[c1_node_id >= focus_node_id] += 1

                    # predict edge type
                    latent_edge = prior_edge.sample([num_gen])
                    edge_feat = torch.cat((global_node_type_feat, feat_index(c1_node_id, node_feat), node_type_emb.view(num_gen, -1)), dim=-1)
                    latent_edge = flow_reverse(self.edge_flow_layers, latent_edge, edge_feat)
                    edge = torch.argmax(latent_edge, dim=1)
                    con_mat[torch.arange(num_gen), c1_node_id, i + 1] = edge
                    con_mat[torch.arange(num_gen), i + 1, c1_node_id] = edge

                    latent_angle = prior_angle.sample([num_gen])
                    local_angle_feat = torch.cat(
                        (feat_index(focus_node_id, node_emb), feat_index(c1_node_id, node_emb)), dim=1)
                    global_angle_feat = self.angle_att(local_angle_feat, node_emb.view(num_gen * (i + 1), -1),
                                                       node_emb.view(num_gen * (i + 1), -1), query_batch,
                                                       key_value_batch, con_mat[:, i + 1, :i + 1].reshape(-1))
                    angle_feat = torch.cat((local_angle_feat, global_angle_feat), dim=-1)

                    angle = flow_reverse(self.angle_flow_layers, latent_angle, angle_feat)
                    # angle = angle.abs()
                    if i == 1:
                        fc1 = feat_index(c1_node_id, pos) - feat_index(focus_node_id, pos)
                        new_pos_x = torch.cos(angle) * torch.sign(fc1[:, 0:1]) * dist
                        new_pos_y = torch.sin(angle) * torch.sign(fc1[:, 0:1]) * dist
                        new_pos = torch.cat((new_pos_x, new_pos_y, torch.zeros_like(dist, device=dist.device)), dim=-1)
                        new_pos += feat_index(focus_node_id, pos)
                    else:
                        mask[torch.arange(num_gen), c1_node_id] = False
                        c2_dists = torch.sum(torch.square(mask_index(mask, pos) - pos_index(c1_node_id, pos)), dim=-1)
                        c2_node_id = torch.argmin(c2_dists, dim=-1)
                        c2_node_id[c2_node_id >= torch.min(focus_node_id, c1_node_id)] += 1
                        c2_node_id[c2_node_id >= torch.max(focus_node_id, c1_node_id)] += 1

                        # predict edge type
                        latent_edge = prior_edge.sample([num_gen])
                        edge_feat = torch.cat((global_node_type_feat, feat_index(c2_node_id, node_feat),node_type_emb.view(num_gen, -1)), dim=-1)
                        latent_edge = flow_reverse(self.edge_flow_layers, latent_edge, edge_feat)
                        edge = torch.argmax(latent_edge, dim=1)
                        con_mat[torch.arange(num_gen), c2_node_id, i + 1] = edge
                        con_mat[torch.arange(num_gen), i + 1, c2_node_id] = edge
                        valid_mask = self.check_valency1(con_mat, z, node_type_id)

                        latent_torsion = prior_torsion.sample([num_gen])

                        local_torsion_feat = torch.cat((feat_index(focus_node_id, node_emb),
                                                        feat_index(c1_node_id, node_emb),
                                                        feat_index(c2_node_id, node_emb)), dim=1)
                        global_torsion_feat = self.torsion_att(local_torsion_feat, node_emb.view(num_gen * (i + 1), -1),
                                                               node_emb.view(num_gen * (i + 1), -1), query_batch,
                                                               key_value_batch, con_mat[:, i + 1, :i + 1].reshape(-1))
                        torsion_feat = torch.cat((local_torsion_feat, global_torsion_feat), dim=-1)

                        torsion = flow_reverse(self.torsion_flow_layers, latent_torsion, torsion_feat)
                        new_pos = dattoxyz(pos_index(focus_node_id, pos), pos_index(c1_node_id, pos),
                                           pos_index(c2_node_id, pos), dist, angle, torsion)

                        complete_mask = torch.logical_not(valid_mask)
                        if i > max(0, min_atoms - 2) and torch.sum(complete_mask) > 0:
                            out_dict1[i + 1] = {}
                            out_node_types = z[complete_mask].view(-1, i + 1).cpu().numpy()
                            out_dict1[i + 1]['_atomic_numbers'] = type_to_atomic_number[out_node_types]
                            out_dict1[i + 1]['_positions'] = pos[complete_mask].view(-1, i + 1, 3).cpu().numpy()
                            out_dict1[i + 1]['_focus'] = focuses[complete_mask].view(-1, i).cpu().numpy()

                z = torch.cat((z, node_type_id[:, None]), dim=1)
                pos = torch.cat((pos, new_pos.view(num_gen, 1, 3)), dim=1)
                focuses = torch.cat((focuses, focus_node_id[:, None]), dim=1)
                '''
                if i > 1:
                    if torch.sum(valid_mask) == 0:
                        break
                    num_gen = torch.sum(valid_mask).cpu().item()
                    z, pos, can_focus, focuses = z[valid_mask], pos[valid_mask], can_focus[valid_mask], focuses[valid_mask]
                    con_mat = con_mat[valid_mask]
                '''

            if add_final and torch.sum(continue_mask) > 0:
                out_dict[i + 2] = {}
                out_node_types = z.view(-1, i + 2).cpu().numpy()
                out_dict[i + 2]['_atomic_numbers'] = type_to_atomic_number[out_node_types]
                out_dict[i + 2]['_positions'] = pos.view(-1, i + 2, 3).cpu().numpy()
                out_dict[i + 2]['_focus'] = focuses.view(-1, i + 1).cpu().numpy()
            '''
            for num_atom in out_dict:
                if num_atom in out_dict1.keys():
                    out_dict[num_atom]['_atomic_numbers'] = np.concatenate((out_dict[num_atom]['_atomic_numbers'], out_dict1[num_atom]['_atomic_numbers']), axis=0)
                    out_dict[num_atom]['_positions'] = np.concatenate((out_dict[num_atom]['_positions'], out_dict1[num_atom]['_positions']), axis=0)
                    out_dict[num_atom]['_focus'] = np.concatenate((out_dict[num_atom]['_focus'], out_dict1[num_atom]['_focus']), axis=0)
            '''
            return out_dict

    #conformation generation with given 2d molecular graphs
    def generate_conf(self, type_to_atomic_number, num_gen=100, temperature=[1.0, 1.0, 1.0, 1.0, 1.0], node_type=[], con_mat=[], focuses=[]):
        with torch.no_grad():
            if self.use_gpu:
                prior_dist = torch.distributions.normal.Normal(torch.zeros([1]).cuda(), temperature[1] * torch.ones([1]).cuda())
                prior_angle = torch.distributions.normal.Normal(torch.zeros([1]).cuda(), temperature[2] * torch.ones([1]).cuda())
                prior_torsion = torch.distributions.normal.Normal(torch.zeros([1]).cuda(), temperature[3] * torch.ones([1]).cuda())
            else:
                prior_dist = torch.distributions.normal.Normal(torch.zeros([1]), temperature[1] * torch.ones([1]))
                prior_angle = torch.distributions.normal.Normal(torch.zeros([1]), temperature[2] * torch.ones([1]))
                prior_torsion = torch.distributions.normal.Normal(torch.zeros([1]), temperature[3] * torch.ones([1]))

            node_type_emb_block = self.feat_net3d.init_e.emb
            z = node_type.repeat(num_gen,1)
            pos = torch.zeros([num_gen, 1, 3], dtype=torch.float32)
            num_node = len(node_type)
            con_mat = con_mat.repeat(num_gen,1,1)
            focuses = focuses.repeat(num_gen, 1)
            if self.use_gpu:
                z, pos, focuses, con_mat = z.cuda(), pos.cuda(), focuses.cuda(), con_mat.cuda()

            mask_index = lambda mask, p: p[mask].view(num_gen, -1, 3)
            feat_index = lambda node_id, f: f[torch.arange(num_gen), node_id]
            pos_index = lambda node_id, p: p[torch.arange(num_gen), node_id].view(num_gen, 1, 3)

            for i in range(num_node-1):
                batch = torch.arange(num_gen, device=z.device).view(num_gen, 1).repeat(1, i + 1)
                if i == 0:
                    node_feat = node_type_emb_block(z[:,0].reshape(-1))
                elif i == 1:
                    node_feat = self.feat_net3d.dist_only_forward(z[:,:2].reshape(-1), pos.view(-1, 3), batch.view(-1))
                else:
                    node_feat = self.feat_net3d(z[:,:i+1].reshape(-1), pos.view(-1, 3), batch.view(-1))

                node_feat = node_feat.view(num_gen, i + 1, -1)
                focus_node_id = focuses[:, i].reshape(num_gen)

                local_node_type_feat, query_batch = feat_index(focus_node_id, node_feat), torch.arange(num_gen, device=node_feat.device)
                key_value_batch = torch.arange(num_gen, device=node_feat.device).view(num_gen, 1).repeat(1, i + 1).view(-1)
                global_node_type_feat = self.node_att(local_node_type_feat, node_feat.view(num_gen * (i + 1), -1),node_feat.view(num_gen * (i + 1), -1), query_batch, key_value_batch, bond=None)

                node_type_id = z[:, i+1]
                node_type_emb = node_type_emb_block(node_type_id)
                node_emb = node_feat * node_type_emb.view(num_gen, 1, -1)

                latent_dist = prior_dist.sample([num_gen])
                local_dist_feat = feat_index(focus_node_id, node_emb)
                global_dist_feat = self.dist_att(local_dist_feat, node_emb.view(num_gen * (i + 1), -1),node_emb.view(num_gen * (i + 1), -1), query_batch, key_value_batch, con_mat[:, i + 1, :i + 1].reshape(-1))
                dist_feat = torch.cat((local_dist_feat, global_dist_feat), dim=-1)
                dist = flow_reverse(self.dist_flow_layers, latent_dist, dist_feat)
                # dist = dist.abs()
                if i == 0:
                    new_pos = torch.cat((dist, torch.zeros_like(dist, device=dist.device), torch.zeros_like(dist, device=dist.device)),dim=-1)
                if i > 0:
                    mask = torch.ones([num_gen, i + 1], dtype=torch.bool)
                    mask[torch.arange(num_gen), focus_node_id] = False
                    c1_dists = torch.sum(torch.square(mask_index(mask, pos) - pos_index(focus_node_id, pos)), dim=-1)
                    c1_node_id = torch.argmin(c1_dists, dim=-1)
                    c1_node_id[c1_node_id >= focus_node_id] += 1

                    latent_angle = prior_angle.sample([num_gen])
                    local_angle_feat = torch.cat(
                        (feat_index(focus_node_id, node_emb), feat_index(c1_node_id, node_emb)), dim=1)
                    global_angle_feat = self.angle_att(local_angle_feat, node_emb.view(num_gen * (i + 1), -1),
                                                       node_emb.view(num_gen * (i + 1), -1), query_batch,
                                                       key_value_batch, con_mat[:, i + 1, :i + 1].reshape(-1))
                    angle_feat = torch.cat((local_angle_feat, global_angle_feat), dim=-1)
                    angle = flow_reverse(self.angle_flow_layers, latent_angle, angle_feat)
                    # angle = angle.abs()
                    if i == 1:
                        fc1 = feat_index(c1_node_id, pos) - feat_index(focus_node_id, pos)
                        new_pos_x = torch.cos(angle) * torch.sign(fc1[:, 0:1]) * dist
                        new_pos_y = torch.sin(angle) * torch.sign(fc1[:, 0:1]) * dist
                        new_pos = torch.cat((new_pos_x, new_pos_y, torch.zeros_like(dist, device=dist.device)), dim=-1)
                        new_pos += feat_index(focus_node_id, pos)
                    else:
                        mask[torch.arange(num_gen), c1_node_id] = False
                        c2_dists = torch.sum(torch.square(mask_index(mask, pos) - pos_index(c1_node_id, pos)), dim=-1)
                        c2_node_id = torch.argmin(c2_dists, dim=-1)
                        c2_node_id[c2_node_id >= torch.min(focus_node_id, c1_node_id)] += 1
                        c2_node_id[c2_node_id >= torch.max(focus_node_id, c1_node_id)] += 1

                        latent_torsion = prior_torsion.sample([num_gen])

                        local_torsion_feat = torch.cat((feat_index(focus_node_id, node_emb),
                                                        feat_index(c1_node_id, node_emb),
                                                        feat_index(c2_node_id, node_emb)), dim=1)
                        global_torsion_feat = self.torsion_att(local_torsion_feat, node_emb.view(num_gen * (i + 1), -1),
                                                               node_emb.view(num_gen * (i + 1), -1), query_batch,
                                                               key_value_batch, con_mat[:, i + 1, :i + 1].reshape(-1))
                        torsion_feat = torch.cat((local_torsion_feat, global_torsion_feat), dim=-1)

                        torsion = flow_reverse(self.torsion_flow_layers, latent_torsion, torsion_feat)
                        new_pos = dattoxyz(pos_index(focus_node_id, pos), pos_index(c1_node_id, pos),
                                           pos_index(c2_node_id, pos), dist, angle, torsion)

                pos = torch.cat((pos, new_pos.view(num_gen, 1, 3)), dim=1)

            pos_out = pos.view(-1, num_node, 3).cpu()
            return pos_out


    def initialize_masks(self, max_node_unroll=30, max_edge_unroll=12):
        """
        Args:
            max node unroll: maximal number of nodes in molecules to be generated (default: 38)
            max edge unroll: maximal number of edges to predict for each generated nodes (default: 12, calculated from zink250K data)
        Returns:
            node_masks: node mask for each step
            adj_masks: adjacency mask for each step
            is_node_update_mask: 1 indicate this step is for updating node features
            flow_core_edge_mask: get the distributions we want to model in adjacency matrix
        """
        num_masks = int(max_node_unroll + (max_edge_unroll - 1) * max_edge_unroll / 2 + (max_node_unroll - max_edge_unroll) * (max_edge_unroll))
        # total generation steps
        num_mask_edge = int(num_masks - max_node_unroll)
        # total edge prediction
        '''
        node_masks1 = torch.zeros([max_node_unroll, max_node_unroll]).bool()
        adj_masks1 = torch.zeros([max_node_unroll, max_node_unroll, max_node_unroll]).bool()
        node_masks2 = torch.zeros([num_mask_edge, max_node_unroll]).bool()
        adj_masks2 = torch.zeros([num_mask_edge, max_node_unroll, max_node_unroll]).bool()
        link_prediction_index = torch.zeros([num_mask_edge, 2]).long()
        flow_core_edge_masks = torch.zeros([max_node_unroll, max_node_unroll]).bool()
        '''
        node_masks1 = torch.zeros([max_node_unroll, max_node_unroll]).type(torch.ByteTensor)
        adj_masks1 = torch.zeros([max_node_unroll, max_node_unroll, max_node_unroll]).type(torch.ByteTensor)
        node_masks2 = torch.zeros([num_mask_edge, max_node_unroll]).type(torch.ByteTensor)
        adj_masks2 = torch.zeros([num_mask_edge, max_node_unroll, max_node_unroll]).type(torch.ByteTensor)
        link_prediction_index = torch.zeros([num_mask_edge, 2]).type(torch.long)
        flow_core_edge_masks = torch.zeros([max_node_unroll, max_node_unroll]).type(torch.ByteTensor)

        cnt = 0
        cnt_node = 0
        cnt_edge = 0
        for i in range(max_node_unroll):
            node_masks1[cnt_node][:i] = 1
            adj_masks1[cnt_node][:i, :i] = 1
            cnt += 1
            cnt_node += 1

            edge_total = 0
            if i < max_edge_unroll:
                start = 0
                edge_total = i
            else:
                start = i - max_edge_unroll
                edge_total = max_edge_unroll
            for j in range(edge_total):
                if j == 0:
                    node_masks2[cnt_edge][:i+1] = 1
                    adj_masks2[cnt_edge] = adj_masks1[cnt_node-1].clone()
                    adj_masks2[cnt_edge][i,i] = 1
                else:
                    node_masks2[cnt_edge][:i+1] = 1
                    adj_masks2[cnt_edge] = adj_masks2[cnt_edge-1].clone()
                    adj_masks2[cnt_edge][i, start + j -1] = 1
                    adj_masks2[cnt_edge][start + j -1, i] = 1
                cnt += 1
                cnt_edge += 1
        assert cnt == num_masks, 'masks cnt wrong'
        assert cnt_node == max_node_unroll, 'node masks cnt wrong'
        assert cnt_edge == num_mask_edge, 'edge masks cnt wrong'

        cnt = 0
        for i in range(max_node_unroll):
            if i < max_edge_unroll:
                start = 0
                edge_total = i
            else:
                start = i - max_edge_unroll
                edge_total = max_edge_unroll

            for j in range(edge_total):
                link_prediction_index[cnt][0] = start + j
                link_prediction_index[cnt][1] = i
                cnt += 1
        assert cnt == num_mask_edge, 'edge mask initialize fail'

        for i in range(max_node_unroll):
            if i == 0:
                continue
            if i < max_edge_unroll:
                start = 0
                end = i
            else:
                start = i - max_edge_unroll
                end = i
            flow_core_edge_masks[i][start:end] = 1

        #node_masks = torch.cat((node_masks1, node_masks2), dim=0)
        #adj_masks = torch.cat((adj_masks1, adj_masks2), dim=0)

        node_masks = nn.Parameter(node_masks2, requires_grad=False)
        adj_masks = nn.Parameter(adj_masks2, requires_grad=False)
        link_prediction_index = nn.Parameter(link_prediction_index, requires_grad=False)
        flow_core_edge_masks = nn.Parameter(flow_core_edge_masks, requires_grad=False)

        return node_masks, adj_masks, link_prediction_index, flow_core_edge_masks

    def _get_embs_edge(self, x, adj, index):
        """
        Args:
            x: current node feature matrix with shape (batch, N, 9)
            adj: current adjacency feature matrix with shape (batch, 4, N, N)
            index: link prediction index with shape (batch, 2)
        Returns:
            Embedding(concatenate graph embedding, edge start node embedding and edge end node embedding)
                for updating edge features with shape (batch, 3d)
        """
        batch_size = x.size(0)

        adj = adj[:, :3] # (batch, 3, N, N)

        node_emb = self.feat_net2d(x, adj) # (batch, N, d)
        if hasattr(self, 'batchNorm'):
            node_emb = self.batchNorm(node_emb.transpose(1, 2)).transpose(1, 2) # (batch, N, d)

        graph_emb = torch.sum(node_emb, dim = 1, keepdim=False).contiguous().view(batch_size, 1, -1) # (batch, 1, d)

        index = index.repeat(batch_size, 1, self.emb_size) # (batch, 2, d)
        graph_node_emb = torch.cat((torch.gather(node_emb, dim=1, index=index), graph_emb),dim=1)  # (batch_size, 3, d)
        graph_node_emb = graph_node_emb.view(batch_size, -1) # (batch_size, 3d)
        return graph_node_emb

    def check_valency(self, mol):
        try:
            s = Chem.MolToSmiles(mol, isomericSmiles=True)
            m = Chem.MolFromSmiles(s)
            Chem.SanitizeMol(m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            return True
        except:
            return False

    def check_valency1(self, conmat, z, node_type_id):
        a = torch.cat((z, node_type_id[:, None]), dim=1)
        nnode = a.shape[1]
        valence = torch.tensor([1,4,3,2,1]).cuda()
        check =  valence[a] - torch.sum(conmat, dim=2)[:, :nnode]
        valid_mask = (torch.sum(check>=0, dim=1) == nnode)
        valid_mask = torch.logical_and(torch.sum(conmat[:,:,nnode-1], dim=1)>0, valid_mask)
        return valid_mask
