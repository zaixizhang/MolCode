import torch
from torch.utils.data import DataLoader, dataset
import os
import numpy as np
from model import SphGen
from dataset import QM9Gen, collate_mols
import torch.optim as optim
from torch_scatter import scatter
import torch.nn as nn

class Runner():
    def __init__(self, conf, root_path, atomic_num_to_type={1:0, 6:1, 7:2, 8:3, 9:4}, out_path=None):
        self.conf = conf
        self.root_path = root_path
        self.atomic_num_to_type = atomic_num_to_type
        self.model = SphGen(**conf['model'])
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), **conf['optim'])
        self.focus_ce = torch.nn.BCELoss()
        self.out_path = out_path

        '''
        if self.conf['model']['use_gpu']:
            self.model = self.model.to(self.conf['my_device'])
            self.model = nn.DataParallel(self.model, device_ids=[0,1,2,3], output_device=self.conf['my_device'])
            #self.optimizer = nn.DataParallel(self.optimizer, device_ids=[0, 1, 2, 3])

        '''

    def _train_epoch(self, loader):
        self.model.train()
        total_ll_node, total_ll_edge, total_ll_dist, total_ll_angle, total_ll_torsion, total_focus_ce = 0, 0, 0, 0, 0, 0

        for iter_num, data_batch in enumerate(loader):
            for key in data_batch:
                data_batch[key] = data_batch[key].cuda()
            node_out, edge_out, focus_score, dist_out, angle_out, torsion_out = self.model(data_batch)
            cannot_focus = data_batch['cannot_focus']

            ll_node = torch.mean(1/2 * (node_out[0] ** 2) - node_out[1])
            ll_edge = torch.mean(1/2 * (edge_out[0] ** 2) - edge_out[1])
            ll_dist = torch.mean(1/2 * (dist_out[0] ** 2) - dist_out[1])
            ll_angle = torch.mean(1/2 * (angle_out[0] ** 2) - angle_out[1])
            ll_torsion = torch.mean(1/2 * (torsion_out[0] ** 2) - torsion_out[1])
            focus_ce = self.focus_ce(focus_score, cannot_focus)
            
            loss = ll_node + ll_dist + ll_angle + ll_torsion + focus_ce + ll_edge

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_ll_node += ll_node.to('cpu').item()
            total_ll_edge += ll_edge.to('cpu').item()
            total_ll_dist += ll_dist.to('cpu').item()
            total_ll_angle += ll_angle.to('cpu').item()
            total_ll_torsion += ll_torsion.to('cpu').item()
            total_focus_ce += focus_ce.to('cpu').item()

            if iter_num % self.conf['verbose'] == 0:
                print('Training iteration {} | loss node {:.4f} edge {:.4f} dist {:.4f} angle {:.4f} torsion {:.4f} focus {:.4f}'.format(iter_num, ll_node.to('cpu').item(),
                    ll_edge.to('cpu').item(), ll_dist.to('cpu').item(), ll_angle.to('cpu').item(), ll_torsion.to('cpu').item(), focus_ce.to('cpu').item()))
        
        iter_num += 1   
        return total_ll_node / iter_num, total_ll_edge / iter_num, total_ll_dist / iter_num, total_ll_angle / iter_num, total_ll_torsion / iter_num, total_focus_ce / iter_num


    def train(self, split_path=None):
        '''
        self.model.load_state_dict(torch.load('/apdcephfs/private_zaixizhang/exp_gen/7/best_valid1.pth'))
        print('Checkpoint loaded!')
        '''
        print(os.environ.get('CUDA_VISIBLE_DEVICES', ""))
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        ngpus_per_node = torch.cuda.device_count()
        print("ngpus_per_node: %s" % ngpus_per_node)

        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        print("local_rank: %s" % local_rank)
        device = torch.device("cuda", local_rank)
        print(device)
        torch.cuda.set_device(local_rank)
        self.model = self.model.to(device)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], find_unused_parameters=True)

        idxs = np.load(split_path)
        subset_idxs = idxs['train_idx'].tolist()
        dataset = QM9Gen(self.conf['model']['cutoff'], self.root_path, subset_idxs, self.atomic_num_to_type)
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=ngpus_per_node, rank=0)
        loader = DataLoader(dataset, batch_size=self.conf['batch_size'], shuffle=False, collate_fn=collate_mols, sampler=train_sampler)
        valid_idxs = idxs['val_idx'].tolist()
        valid_dataset = QM9Gen(self.conf['model']['cutoff'], self.root_path, valid_idxs, self.atomic_num_to_type)
        valid_loader = DataLoader(valid_dataset, batch_size=self.conf['batch_size'], shuffle=False, collate_fn=collate_mols,
                                  sampler=torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas=ngpus_per_node, rank=0))


        epochs = self.conf['epochs']
        best_loss = 10000.
        for epoch in range(epochs):
            train_sampler.set_epoch(epoch)
            avg_ll_node, avg_ll_edge, avg_ll_dist, avg_ll_angle, avg_ll_torsion, avg_focus_ce = self._train_epoch(loader)
            print('Training {:.0f} | Average loss node {:.4f} edge {:.4f} dist {:.4f} angle {:.4f} torsion {:.4f} focus {:.4f}'.format(epoch, avg_ll_node, avg_ll_edge, avg_ll_dist, avg_ll_angle, avg_ll_torsion, avg_focus_ce))
            if self.out_path is not None:
                file_obj = open(os.path.join(self.out_path, 'record.txt'), 'a')
                file_obj.write('Training {:.0f} | Average loss node {:.4f} edge {:.4f} dist {:.4f} angle {:.4f} torsion {:.4f} focus {:.4f}\n'.format(epoch, avg_ll_node, avg_ll_edge, avg_ll_dist, avg_ll_angle, avg_ll_torsion, avg_focus_ce))
                file_obj.close()
            val_loss = self.valid(valid_loader)
            if val_loss <= best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.out_path, 'best_valid.pth'))
                if self.out_path is not None:
                    file_obj = open(os.path.join(self.out_path, 'record.txt'), 'a')
                    file_obj.write('Best Valid model saved | Validation loss {:.4f}\n'.format(val_loss))
                    file_obj.close()
        torch.save(self.model.state_dict(), os.path.join(self.out_path, 'last.pth'))
    

    def valid(self, loader):
        self.model.eval()
        with torch.no_grad():
            total_ll_node, total_ll_edge, total_ll_dist, total_ll_angle, total_ll_torsion, total_focus_ce = 0, 0, 0, 0, 0, 0
            for iter_num, data_batch in enumerate(loader):
                for key in data_batch:
                    data_batch[key] = data_batch[key].cuda()
                node_out, edge_out, focus_score, dist_out, angle_out, torsion_out = self.model(data_batch)
                cannot_focus = data_batch['cannot_focus']

                ll_node = torch.mean(1/2 * (node_out[0] ** 2) - node_out[1])
                ll_edge = torch.mean(1/2 * (edge_out[0] ** 2) - edge_out[1])
                ll_dist = torch.mean(1/2 * (dist_out[0] ** 2) - dist_out[1])
                ll_angle = torch.mean(1/2 * (angle_out[0] ** 2) - angle_out[1])
                ll_torsion = torch.mean(1/2 * (torsion_out[0] ** 2) - torsion_out[1])
                focus_ce = self.focus_ce(focus_score, cannot_focus)

                total_ll_node += ll_node.to('cpu').item()
                total_ll_edge += ll_edge.to('cpu').item()
                total_ll_dist += ll_dist.to('cpu').item()
                total_ll_angle += ll_angle.to('cpu').item()
                total_ll_torsion += ll_torsion.to('cpu').item()
                total_focus_ce += focus_ce.to('cpu').item()

            iter_num += 1
            loss = (total_ll_node +total_ll_edge +total_ll_dist +total_ll_angle +total_ll_torsion +total_focus_ce)/iter_num

        return loss

    #random/targeted generation
    def generate(self, num_gen, temperature=[1.0, 1.0, 1.0, 1.0, 1.0], min_atoms=2, max_atoms=35, focus_th=0.5, add_final=False):
        num_remain = num_gen
        one_time_gen = self.conf['chunk_size']
        type_to_atomic_number_dict = {self.atomic_num_to_type[k]:k for k in self.atomic_num_to_type}
        type_to_atomic_number = np.zeros([max(type_to_atomic_number_dict.keys())+1], dtype=int)
        for k in type_to_atomic_number_dict:
            type_to_atomic_number[k] = type_to_atomic_number_dict[k]
        mol_dicts = {}

        self.model = self.model.to(self.conf['my_device'])
        self.model.eval()
        while num_remain > 0:
            if num_remain > one_time_gen:
                mols = self.model.generate_3d(type_to_atomic_number, one_time_gen, temperature, min_atoms, max_atoms, focus_th, add_final)
            else:
                mols = self.model.generate_3d(type_to_atomic_number, num_remain, temperature, min_atoms, max_atoms, focus_th, add_final)
            
            for num_atom in mols:
                if not num_atom in mol_dicts.keys():
                    mol_dicts[num_atom] = mols[num_atom]
                else:
                    mol_dicts[num_atom]['_atomic_numbers'] = np.concatenate((mol_dicts[num_atom]['_atomic_numbers'], mols[num_atom]['_atomic_numbers']), axis=0)
                    mol_dicts[num_atom]['_positions'] = np.concatenate((mol_dicts[num_atom]['_positions'], mols[num_atom]['_positions']), axis=0)
                    mol_dicts[num_atom]['_focus'] = np.concatenate((mol_dicts[num_atom]['_focus'], mols[num_atom]['_focus']), axis=0)
                num_mol = len(mols[num_atom]['_atomic_numbers'])
                num_remain -= num_mol
            
            print('{} molecules are generated!'.format(num_gen-num_remain))
            file_obj = open(os.path.join('/apdcephfs/private_zaixizhang/exp_gen/', 'generation1.txt'), 'a')
            file_obj.write('{} molecules are generated!\n'.format(num_gen-num_remain))
            file_obj.close()
        
        return mol_dicts

    #conformation generation given 2D molecular graphs
    def generate_conf(self, num_gen, temperature=[1.0, 1.0, 1.0, 1.0, 1.0], node_type=[], con_mat=[], focuses=[]):
        num_remain = num_gen
        one_time_gen = self.conf['chunk_size']
        type_to_atomic_number_dict = {self.atomic_num_to_type[k]: k for k in self.atomic_num_to_type}
        type_to_atomic_number = np.zeros([max(type_to_atomic_number_dict.keys()) + 1], dtype=int)
        for k in type_to_atomic_number_dict:
            type_to_atomic_number[k] = type_to_atomic_number_dict[k]
        pos_gen = torch.empty([0,3], dtype=float)

        self.model = self.model.to(self.conf['my_device'])
        self.model.eval()
        while num_remain > 0:
            if num_remain > one_time_gen:
                mols = self.model.generate_conf(type_to_atomic_number, one_time_gen, temperature, node_type, con_mat, focuses)
            else:
                mols = self.model.generate_conf(type_to_atomic_number, num_remain, temperature, node_type, con_mat, focuses)

            num_mol = mols.shape[0]
            pos_gen = torch.cat([pos_gen, mols.reshape(-1,3)], dim=0)

            num_remain -= num_mol

            print('{} molecules are generated!'.format(num_gen - num_remain))

            file_obj = open(os.path.join('/apdcephfs/private_zaixizhang/exp_gen/', 'generation.txt'), 'a')
            file_obj.write('{} molecules are generated!\n'.format(num_gen - num_remain))
            file_obj.close()

        return pos_gen
            
            
