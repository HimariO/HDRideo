"""
Two-stage model (2-exposure)
The first stage is flownet + weightnet
The second stage is the refinenet
"""
import os
import numpy as np
import torch
from .base_model import BaseModel
from utils import eval_utils as eutils
from utils import image_utils as iutils
from models import model_utils as mutils
from models import network_utils as nutils
from collections import OrderedDict
from models.hdr2E_flow_model import hdr2E_flow_model
np.random.seed(0)


class PreprocessMixin:

    def prepare_inputs(self, data):
        hdr2E_flow_model.prepare_inputs(self, data)
        
        # for Stage 2
        expms2 = []
        ldrs = data['ldrs']
        expos = data['expos']

        for i in range(len(ldrs)):
            neighb_idx = 1 if i == 0 else i - 1
            cur_h_idx = (expos[i] > expos[neighb_idx]).view(-1).long()
            # two = torch.cat([ldrs[neighb_idx], ldrs[i]], dim=0)
            expm2 = self.expos2_mask(ldrs[i], cur_h_idx, h_thr=self.opt['hthr'], l_thr=self.opt['lthr'])
            expms2.append(expm2)

        stage1_out_mask = []
        for i in range(1, len(ldrs)-1):
            if self.nexps == 2:
                cur_h_idx = (expos[i] > expos[i-1]).view(-1).long()
                two = torch.cat([ldrs[i - 1], ldrs[i]], dim=0)
                mask = 1.0 - self.out_mask(ldrs[i], cur_h_idx, h_thr=self.opt['o_hthr'], l_thr=0.4)
                stage1_out_mask.append(mask)

        data.update({'expms2': expms2, 'stage1_out_mask': stage1_out_mask})

    @property
    def expos2_mask(self):
        get_expos_mask_method = mutils.pt_get_in_exposure_mask
        return get_expos_mask_method
    
    def get_high_expo_idxs(self, expos, frame_idx):
        i = frame_idx
        if i == 0:
            h_idx = (expos[i] > expos[i+1]).view(-1)
        else: # 0 < i < len(expos)-1
            h_idx = (expos[i] > expos[i-1]).view(-1)
        return h_idx
    
    def prepare_inputs_direct2(self, data, pred, idxs):
        # Problematic: data and pred does not share the same index
        hdr_mid, ldr_mid = self.hdr_mid, self.ldr_mid
        pi, ci, ni = idxs
        merge_input = {}
        pred_p_hdr, pred_c_hdr, pred_n_hdr = pred[pi]['hdr'], pred[ci]['hdr'], pred[ni]['hdr']

        if self.opt['s2_inldr']:
            pred_p_ldr = mutils.pt_hdr_to_ldr_clamp(pred[pi]['hdr'].detach(),expo=data['expos'][ldr_mid-1])
            pred_c_ldr = mutils.pt_hdr_to_ldr_clamp(pred[ci]['hdr'].detach(),expo=data['expos'][ldr_mid])
            pred_n_ldr = mutils.pt_hdr_to_ldr_clamp(pred[ni]['hdr'].detach(),expo=data['expos'][ldr_mid+1])
            merge_input.update({'cur': pred_c_ldr, 'prev': pred_p_ldr, 'nxt': pred_n_ldr})

        if self.opt['s2_inexpm']:
            p_expm, c_expm, n_expm = data['expms2'][ldr_mid-1], data['expms2'][ldr_mid], data['expms2'][ldr_mid+1],
            merge_input.update({'p_expm': p_expm, 'c_expm': c_expm, 'n_expm': n_expm})

        inputs = []
        if self.opt['s2_inldr']:
            inputs.append(torch.cat([pred_p_ldr, pred_p_hdr], 1))
            inputs.append(torch.cat([pred_c_ldr, pred_c_hdr], 1))
            inputs.append(torch.cat([pred_n_ldr, pred_n_hdr], 1))
        elif self.opt['s2_inexpm']:
            inputs.append(torch.cat([p_expm, pred_p_hdr], 1))
            inputs.append(torch.cat([c_expm, pred_c_hdr], 1))
            inputs.append(torch.cat([n_expm, pred_n_hdr], 1))
        else:
            inputs = [pred_p_hdr, pred_c_hdr, pred_n_hdr]

        inputs = torch.stack(inputs, 1)

        merge_input.update({'x': inputs})
        merge_input.update({'cur_hdr': pred_c_hdr, 'prev_hdr': pred_p_hdr, 'nxt_hdr': pred_n_hdr})
        merge_input.update({'p_e': data['expos'][self.ldr_mid-1], 'c_e': data['expos'][self.ldr_mid], 'n_e': data['expos'][self.ldr_mid+1]})

        if 'gt_ref_ws' in data:
            merge_input.update({'gt_ref_w': data['gt_ref_ws'][self.hdr_mid]})
        return merge_input
    
    def prepare_aligned_mnet2_inputs(self, data, pred, s1_align_ldrs, idxs):
        pred_p_to_c = nutils.affine_warp(pred[0]['hdr'], data['matches'][self.ldr_mid-1][:,1].view(-1,2,3))
        pred_n_to_c = nutils.affine_warp(pred[2]['hdr'], data['matches'][self.ldr_mid+1][:,0].view(-1,2,3))

        self.aligned_data2 = [eutils.pt_mulog_transform(pred_p_to_c), eutils.pt_mulog_transform(pred_n_to_c)]

        new_pred = [{'hdr': pred_p_to_c}, {'hdr': pred[1]['hdr']}, {'hdr': pred_n_to_c}]
        new_data = {}
        
        ldrs = [None, s1_align_ldrs[0], None, s1_align_ldrs[1], None]

        """ 
        This part is slightly different from the paper.
        The released stage 2 model takes expms2 as input. However, the expms2 can be removed
        when training a new model. The result should be similar
        """  
        new_data['expms2'] = [None]
        cur_h_idx = self.get_high_expo_idxs(data['expos'], self.ldr_mid-1)
        new_data['expms2'].append(self.expos2_mask(ldrs[1], cur_h_idx, h_thr=self.opt['hthr'], l_thr=self.opt['lthr']))

        new_data['expms2'].append(data['expms2'][self.ldr_mid]) # Okay

        cur_h_idx = self.get_high_expo_idxs(data['expos'], self.ldr_mid+1)
        new_data['expms2'].append(self.expos2_mask(ldrs[3], cur_h_idx, h_thr=self.opt['hthr'], l_thr=self.opt['lthr']))

        new_data['expos'] = data['expos']
        new_data['gt_ref_ws'] = data['gt_ref_ws']

        return self.prepare_inputs_direct2(new_data, new_pred, idxs=[0,1,2])
    

class EvalMixin:

    def prepare_records(self):
        data = self.data
        records, iter_res = OrderedDict(), []

        # stage2
        gt = {
            'hdr': data['hdrs'][self.hdr_mid], 
            'log_hdr': data['log_hdrs'][self.hdr_mid], 
            'c_expo': data['expos'][self.ldr_mid], 
            'p_expo': data['expos'][self.ldr_mid-1]
        }
        pred = {
            'hdr': self.pred2['hdr'], 
            'log_hdr': self.pred2['log_hdr']
        }

        records_sub, iter_res_sub = self._prepare_records(gt, pred)
        records.update(records_sub)
        iter_res += iter_res_sub

        # stage 1
        for i in range(self.hdr_mid, self.hdr_mid+1):
            gt = {
                'hdr': data['hdrs'][i],
                'log_hdr': data['log_hdrs'][i],
                'c_expo': data['expos'][i+1],
                'p_expo': data['expos'][i]
            }
            pred = {
                'hdr': self.preds[i]['hdr'], 
                'log_hdr': self.preds[i]['log_hdr']
            }
            records_sub, iter_res_sub = self._prepare_records(gt, pred, key='%d'%(i+1))
            records.update(records_sub)
            iter_res += iter_res_sub

        if hasattr(self, 'fpreds'):
            self.prepare_flow_info(records, flow=self.fpreds[1]['flow1'])
        return records, iter_res

    def prepare_visual(self):
        data, preds = self.data, self.preds
        pred2 = self.pred2

        visuals = []
        visuals += [
            data['log_hdrs'][self.hdr_mid], 
            pred2['log_hdr'], 
            preds[self.hdr_mid]['log_hdr']
        ]
        diff = eutils.pt_cal_diff_map(pred2['log_hdr'].detach(), data['log_hdrs'][self.hdr_mid])
        visuals.append(eutils.pt_colormap(diff))
        diff = eutils.pt_cal_diff_map(preds[self.hdr_mid]['log_hdr'].detach(), data['log_hdrs'][self.hdr_mid])
        visuals.append(eutils.pt_colormap(diff))

        for i, pred in enumerate(preds):
            if i != self.hdr_mid:
                visuals += [pred['log_hdr']]

        if self.opt['mask_o']:
            visuals += [self.data['gt_ref_ws'][self.hdr_mid]]

        if self.opt['s2_inexpm'] and 'expms2' in data:
            for i in range(self.ldr_mid-1, self.ldr_mid+2):
                visuals += [data['expms2'][i]]

        for i, ldr in enumerate(data['ldrs']):
            visuals += [ldr]
        for i, ldr_adj in enumerate(data['ldr_adjs']):
            visuals += [ldr_adj]

        visuals.append(eutils.pt_blend_images(data['ldrs']))

        if hasattr(self, 'fpreds') and 'flow1' in self.fpreds[self.hdr_mid]:
            flow1_color = eutils.pt_flow_to_color(self.fpreds[self.hdr_mid]['flow1'].detach())
            flow2_color = eutils.pt_flow_to_color(self.fpreds[self.hdr_mid]['flow2'].detach())
            visuals += [flow1_color, flow2_color]

        for key in ['attention1', 'attention2', 'attention3']:
            if key in pred2:
                atts = pred2[key]
                if atts.dim() == 3: 
                    atts = eutils.pt_colormap(atts, thres=1)
                visuals.append(atts)

        if 'weights' in pred2:
            visuals += pred2['weights']

        if self.split not in ['train', 'val'] and self.opt['origin_hw']:
            new_visuals = eutils.crop_list_of_tensors(visuals, data['hw'])
            return new_visuals
        return visuals
    
    def save_visual_details(self, log, split, epoch, i):
        save_dir = log.config_save_detail_dir(split, epoch)
        data, pred = self.data, self.pred2
        hdr = pred['hdr']
        if self.opt['origin_hw']: 
            h, w = data['hw']
            hdr = eutils.crop_tensor(hdr, h, w)
        hdr_numpy = hdr[0].cpu().numpy().transpose(1, 2, 0)
        hdr_name = os.path.join(save_dir, '%04d_%s_%s.hdr' % (i, data['scene'][0], data['img_name'][0]))
        iutils.save_hdr(hdr_name, hdr_numpy)



class hdr2E_flow2s_model(PreprocessMixin, EvalMixin, hdr2E_flow_model):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--fnet_name', default='spynet_triple') # for 2 exposures
        parser.add_argument('--fnet_checkp', default='')
        parser.add_argument('--fnet_init', default='xavier')
        parser.add_argument('--mnet_name', default='weight_net')
        parser.add_argument('--mnet_afunc', default='LReLU')
        parser.add_argument('--mnet_checkp', default='')
        parser.add_argument('--cmid', default=256, type=int)
        parser.add_argument('--mnet2_name', default='DAHDRnet', help='deformable alignmnet HDRnet')
        parser.add_argument('--mnet2_checkp', default='')
        parser.add_argument('--test_root', default='mnet2_checkp')
        parser.add_argument('--nframes', type=int, default=5)
        parser.add_argument('--nexps', type=int, default=2)

        parser.add_argument('--PCD', default='', help='pyramid cascaded deformable alignment module')
        parser.add_argument('--TAM', default='TA2', help='temporal attention module')
        parser.add_argument('--up_s1', default=False, action='store_true', help='update stage1')
        parser.add_argument('--m1_mo', default=True, action='store_false')
        parser.add_argument('--s2_inexpm', default=True, action='store_false') # TODO
        parser.add_argument('--s2_inldr', default=False, action='store_true') # TODO
        parser.add_argument('--tone_low', default=True, action='store_false')
        parser.add_argument('--tone_ref', default=False, action='store_true')

        if is_train:
            parser.set_defaults(init_lr=0.0001, init_type='normal', batch=8, vgg_l=True)
            parser.set_defaults(factor_of_k=16, train_save=400)

        parser.set_defaults(mask_o=True) # TODO

        str_keys = ['mnet_name','mnet2_name', 'TAM']
        val_keys = ['hdr_w', 'cmid']
        bool_keys = ['up_s1', 's2_inexpm','s2_inldr']
        bool_val_dicts = {}
        bool_to_val_dicts = {}
        return parser, str_keys, val_keys, bool_keys, bool_val_dicts, bool_to_val_dicts

    def __init__(self, args, log):
        opt = vars(args)
        BaseModel.__init__(self, opt)

        # Stage 1
        c_in_fnet, c_out_fnet = 9, 4
        other_opt = {}
        self.fnet = self.import_network(args.fnet_name, backup_module='flow_networks')(
                c_in=c_in_fnet, c_out=c_out_fnet, requires_grad=self.opt['up_s1'], other=other_opt)
        self.fnet = mutils.init_net(self.fnet, init_type=opt['fnet_init'], gpu_ids=args.gpu_ids)

        other_opt = {}
        c_in_mnet, c_out_mnet, c_mid = 30, 15, opt['cmid']
        self.mnet = self.import_network(args.mnet_name)(c_in_mnet, c_out_mnet, c_mid=c_mid, use_bn=opt['use_bn'], afunc=opt['mnet_afunc'])
        self.mnet = mutils.init_net(self.mnet, init_type=opt['init_type'], gpu_ids=args.gpu_ids)

        if not self.opt['up_s1']: # if update stage 1, for end-to-end finetune
            self.fixed_net_names += ['fnet', 'mnet']
            self.set_requires_grad([self.fnet, self.mnet], False)
        else:
            self.net_names += ['fnet', 'mnet'] 
        
        self.net_names += ['mnet2']

        # Stage 2
        c_in, c_out = self.get_stage2_io_ch_nums(opt)
        other_opt = {'PCD': opt['PCD'], 'TAM': opt['TAM']}

        self.mnet2 = self.import_network(args.mnet2_name)(c_in=c_in, c_out=3, nf=64, nframes=3, groups=8,
                front_RBs=5, back_RBs=10, other=other_opt)

        self.mnet2 = mutils.init_net(self.mnet2, init_type=opt['init_type'], gpu_ids=args.gpu_ids)

        if self.is_train: # Criterion
            self.config_optimizers(opt, log)
        self.config_criterions(opt, log)

        self.load_checkpoint(log)
        self.backward_grid = {}
        torch.manual_seed(0)

    def get_stage2_io_ch_nums(self, opt):
        assert(not (opt['s2_inldr'] and opt['s2_inexpm']))
        c_out = c_in = 3
        if opt['s2_inldr'] or opt['s2_inexpm']: # or self.opt['s2_expm2']: 
            c_in *= 2
        return c_in, c_out

    def config_optimizers(self, opt, log):
        params = []
        params.append({'params': list(self.mnet2.parameters())})
        if self.opt['up_s1']:
            params.append({'params': list(self.mnet.parameters()), 'lr': opt['init_lr']*0.1})
            params.append({'params': list(self.fnet.parameters()), 'lr': opt['init_lr']*0.1})

        self.optimizer = torch.optim.Adam(params, lr=opt['init_lr'], 
                            betas=(opt['beta_1'], opt['beta_2']))
        self.optimizers.append(self.optimizer)
        self.setup_lr_scheduler() # defined in base model
    
    def forward(self, split='train'):
        self.split = split
        self.prepare_inputs(self.data)
        if self.is_train:
            self.forward_train()
        else:
            self.forward_run_model()
            self.cached_data = self.data

        self.loss_terms = None
        return self.pred2

    def forward_run_model(self):
        data = self.data
        cache_num = self.nframes - 2 if self.nexps == 2 else self.nframes - 4

        if not hasattr(self, 'preds') or not data['reuse_cached_data']: # Initialize cache
            self.preds, self.fpreds = [], []

            if self.opt['align']:
                self.aligned_ldrs = []

        if len(self.preds) == cache_num: # remove the oldest cache
            self.fpreds.pop(0)
            self.preds.pop(0)

            if self.opt['align']:
                self.aligned_ldrs.pop(0)

        for i in range(len(self.preds), cache_num): # [2]
            # print('forward: %d' % i)
            idxs = range(i, i + 3) # [2, 3, 4]
            pi, ci, ni = idxs
        
            if not self.is_train and self.opt['align']:
                p_align, n_align = self.global_align_nbr_ldrs(data, idxs)
                fnet_in = [p_align, data['ldr_adjs'][ci], n_align]
                fpred = self.fnet(fnet_in)
                aligned_data = [p_align, n_align]
                mnet1_in, merge_hdrs, warped_imgs = self.prepare_aligned_mnet_inputs(data, fpred, idxs, aligned_data)

                self.aligned_ldrs.append(aligned_data)
            else:
                fnet_in = [data['ldrs'][pi], data['ldr_adjs'][ci], data['ldrs'][ni]]
                fpred = self.fnet(fnet_in)

                mnet1_in, merge_hdrs, warped_imgs = self.prepare_mnet_inputs(self.opt, data, fpred, idxs)
            mpred = self.mnet(mnet1_in, merge_hdrs)

            if self.opt['m1_mo']:
                mask = data['stage1_out_mask'][ci-1]
                mpred['hdr'] = data['l2hdrs'][ci] * mask + mpred['hdr'] * (1 - mask)

            mu = 5000 if self.split not in ['train', 'val'] else self.mu
            mpred['log_hdr'] = eutils.pt_mulog_transform(mpred['hdr'], mu)
            self.fpreds.append(fpred)
            self.preds.append(mpred)
        
        self.forward_stage2()

    def forward_train(self):
        self.forward_stage1()
        self.forward_stage2()

    def forward_stage1(self):
        data = self.data

        self.fpreds, self.preds = [], []
        for i in range(self.opt['nframes'] - 2):
            idxs = range(i, i + 3) # [0,1,2], [1,2,3], [2,3,4]
            pi, ci, ni = idxs
            fnet_in = [data['ldrs'][pi], data['ldr_adjs'][ci], data['ldrs'][ni]]
            fpred = self.fnet(fnet_in)

            mnet1_in, merge_hdrs, warped_imgs = self.prepare_mnet_inputs(self.opt, data, fpred, idxs)
            mpred = self.mnet(mnet1_in, merge_hdrs)

            if self.opt['m1_mo']:
                mask = data['stage1_out_mask'][ci-1]
                mpred['hdr'] = data['l2hdrs'][ci] * mask + mpred['hdr'] * (1 - mask)

            mu = 5000 if self.split not in ['train', 'val'] else self.mu
            mpred['log_hdr'] = eutils.pt_mulog_transform(mpred['hdr'], mu)
            self.fpreds.append(fpred)
            self.preds.append(mpred)

    def forward_stage2(self):
        data = self.data

        if not self.is_train and self.opt['align']:
            mnet2_in = self.prepare_aligned_mnet2_inputs(self.data, self.preds, self.aligned_ldrs[1], [0, 1, 2])
        else:
            mnet2_in = self.prepare_inputs_direct2(self.data, self.preds, [0,1,2])

        self.pred2 = self.mnet2(mnet2_in)
        if self.opt['mask_o']:
            mask = self.data['gt_ref_ws'][self.hdr_mid]
            self.pred2['hdr'] = self.preds[1]['hdr'] * mask + self.pred2['hdr'] * (1 - mask)

        mu = 5000 if self.split not in ['train', 'val'] else self.mu
        self.pred2['log_hdr'] = eutils.pt_mulog_transform(self.pred2['hdr'].clamp(0, 1), mu)

    def compute_log_hdr_loss(self, pred_log_hdr, gt_log_hdr, weight=1, vgg=True):
        # Attention: Only for middle hdr
        loss = 0
        loss_terms = {}
        hdr_loss = weight * self.opt['hdr_w'] * self.hdr_crit(pred_log_hdr, gt_log_hdr)
        loss_terms['hdr_loss'] = hdr_loss.item()

        if self.opt['mask_o']:
            ratio = (1 - self.data['gt_ref_ws'][self.hdr_mid]).mean().item()
            mhdr_loss = hdr_loss / (ratio + 1e-8)
            loss += mhdr_loss
            loss_terms['mhdr_loss'] = mhdr_loss.item()
        else:
            loss += hdr_loss

        if self.opt['vgg_l'] and vgg:
            vgg_l, vgg_l_term = self.vgg_crit(pred_log_hdr, gt_log_hdr)
            loss += weight * self.opt['vgg_w'] * vgg_l
            for k in vgg_l_term: 
                loss_terms[k] = vgg_l_term[k]
        return loss, loss_terms


    def optimize_weights(self):
        self.loss = 0
        self.loss_terms = OrderedDict()
        
        def prepare_hdrs_for_loss_comp():
            gt_log_hdr = self.data['log_hdrs'][self.hdr_mid]
            pred_log_hdr = self.pred2['log_hdr']
            return gt_log_hdr, pred_log_hdr
        
        gt_log_hdr, pred_log_hdr = prepare_hdrs_for_loss_comp()

        loss, loss_terms = self.compute_log_hdr_loss(gt_log_hdr, pred_log_hdr, vgg=True)
        self.loss += loss
        self.loss_terms.update(loss_terms)
            
        if self.opt['up_s1']:
            for i in range(len(self.preds)):
                hdr_loss = self.hdr_crit(self.preds[i]['log_hdr'], self.data['log_hdrs'][i])
                self.loss += self.opt['hdr_w'] * hdr_loss
                self.loss_terms['hdr%d_err'%i] = hdr_loss.item()
        else:
            with torch.no_grad():
                for i in range(len(self.preds)):
                    hdr_loss = self.hdr_crit(self.preds[i]['log_hdr'], self.data['log_hdrs'][i])
                    self.loss_terms['hdr%d_err'%i] = hdr_loss.item()

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def prepare_predict(self):
        prediction = [self.pred2['log_hdr'].detach().cpu(), self.preds[self.hdr_mid]['log_hdr'].detach().cpu()]
        if self.opt['origin_hw']: 
            prediction = eutils.crop_list_of_tensors(prediction, self.data['hw'])
        return prediction 
    
    def global_align_nbr_ldrs(self, data, idxs):
        pi, ci, ni = idxs
        match_p = data['matches'][pi][:,1].view(-1, 2, 3)
        match_n = data['matches'][ni][:,0].view(-1, 2, 3)
        p_to_c = nutils.affine_warp(data['ldrs'][pi], match_p)
        n_to_c = nutils.affine_warp(data['ldrs'][ni], match_n)
        return p_to_c, n_to_c

    def prepare_aligned_mnet_inputs(self, data, fpred, idxs, align_data):
        pi, ci, ni = idxs
        new_data = {}
        p_align, n_align = align_data
        new_data['ldrs'] = [p_align, data['ldrs'][ci], n_align]
        new_data['l2hdrs'] = [mutils.pt_ldr_to_hdr(p_align, data['expos'][pi]),
                       data['l2hdrs'][ci], mutils.pt_ldr_to_hdr(n_align, data['expos'][ni])]
        new_data['expos'] = data['expos'][pi:pi+3]

        return self.prepare_mnet_inputs(self.opt, new_data, fpred, idxs=[0, 1, 2]) # reuse idx

