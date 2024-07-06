import pytorch_lightning as pl
from .models import *
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys
sys.path.append('./src/')
from evaluation.evaluation_model import behavior
from evaluation.saliency_metric import saliency_map_metric, nw_matching, compare_multi_gazes

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerModel_Mixed(pl.LightningModule):
    def __init__(self, args, max_len):
        super().__init__()
        self.args = args
        self.max_len = max_len
        torch.manual_seed(0)
        TGT_VOCAB_SIZE = 0 #self.args.package_size+ 2#4
        #self.TGT_VOCAB_SIZE = TGT_VOCAB_SIZE
        #self.TGT_IDX = self.args.package_size
        self.PAD_IDX = self.args.package_size+1
        self.BOS_IDX = self.args.package_size+2
        self.EOS_IDX = self.args.package_size #+3

        EMB_SIZE = 512
        NHEAD = 4
        FFN_HID_DIM = 512
        NUM_ENCODER_LAYERS = 4
        NUM_DECODER_LAYERS = 4

        inputDim = 3
        self.model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                         NHEAD, TGT_VOCAB_SIZE, inputDim, FFN_HID_DIM,
                                        args.functionChoice, args.alpha, args.changeX, args.CA_version,
                                        args.CA_head, args.CA_dk, args.spp, args.PE_matrix).to(DEVICE).float()
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.loss_fn = [torch.nn.CrossEntropyLoss(ignore_index=self.PAD_IDX[0]), torch.nn.CrossEntropyLoss(ignore_index=self.PAD_IDX[1])]
        self.norm = torch.nn.Softmax(dim=1)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
        self.loggerS= SummaryWriter(f'./lightning_logs/{args.log_name}')
        self.total_step = 0

    def log_gradients_in_model(self, step):
        for tag, value in self.model.named_parameters():
            #print('-'*10)
            if value.grad is not None:
                #print(tag, value.grad.cpu())
                self.loggerS.add_histogram(tag + "/grad", value.grad.cpu(), step)
            #print('-' * 10)

    def train_one_dataset(self, batch, type, return_logits=False):
        src_pos, src_img, tgt_pos, tgt_img = batch
        # src_pos(28, b), src_img(b, 28, w, h, 3), tgt_pos(max_len, b), tgt_img(b, max_len, w, h, 3)
        src_pos = src_pos.to(DEVICE)
        src_img = src_img.to(DEVICE)
        tgt_pos = tgt_pos.to(DEVICE)
        tgt_img = tgt_img.to(DEVICE)

        src_pos_2d, tgt_input_2d, src_img, tgt_img, src_mask, tgt_mask, \
        src_padding_mask, tgt_padding_mask, src_padding_mask = self.processData3d(src_pos, src_img, tgt_pos,
                                                                                  tgt_img, type)

        logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                            src_img, tgt_img,
                            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask, type)
        tgt_out = tgt_pos[1:, :]
        loss = self.loss_fn[type](logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        if return_logits:
            logits_new = F.softmax(logits[:, 0, :], dim=0)
            return logits_new
        return loss

    def training_step(self, batch, batch_idx):
        data1, data2 = batch
        if len(data1) != 0:
            loss1 = self.train_one_dataset(data1, 0)
        else:
            loss1 = 0
        if len(data2) != 0:
            loss2 = self.train_one_dataset(data2, 1)
        else:
            loss2 = 0
        loss = loss1 + loss2
        #self.log_gradients_in_model(self.total_step)
        self.total_step += 1
        self.log('training_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, }

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        self.log('training_loss_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def processData3d(self, src_pos, src_img, tgt_pos, tgt_img, type):
        tgt_input = tgt_pos[:-1, :]
        tgt_img = tgt_img[:, :-1, :, :, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input, self.PAD_IDX[type])

        src_pos_2d, tgt_input_2d = self.generate3DInput(tgt_input, src_pos, type)

        return src_pos_2d, tgt_input_2d,  src_img, tgt_img, src_mask, tgt_mask, \
               src_padding_mask, tgt_padding_mask, src_padding_mask


    def generate3DInput(self, tgt_input, src_pos, type):
        tgt_input_2d = torch.zeros((tgt_input.size()[0], tgt_input.size()[1], 3)).to(DEVICE).float()

        tgt_input_2d[:, :, 0] = tgt_input // self.args.shelf_col[type]
        tgt_input_2d[:, :, 1] = torch.remainder(tgt_input, self.args.shelf_col[type])
        tgt_input_2d[0, :, 0] = float(self.args.shelf_row[type]) / 2
        tgt_input_2d[0, :, 1] = float(self.args.shelf_col[type]) / 2

        src_pos_2d = torch.zeros((src_pos.size()[0], src_pos.size()[1], 3)).to(DEVICE).float()
        src_pos_2d[:, :, 0] = src_pos // self.args.shelf_col[type]
        src_pos_2d[:, :, 1] = torch.remainder(src_pos, self.args.shelf_col[type])

        # changed to three dimension
        batch = tgt_input.size()[1]
        src_pos_2d[-1, :, 2] = 1 # the last one is target
        for i in range(batch):
            Index = src_pos[-1, i]
            tgt1 = torch.where(tgt_input[:, i] == Index)[0]
            tgt_input_2d[tgt1, i, 2] = 1
        return src_pos_2d, tgt_input_2d

    def valid_one_dataset(self, batch, type):
        src_pos, src_img, tgt_pos, tgt_img = batch
        # src_pos(28, b), src_img(b, 28, w, h, 3), tgt_pos(max_len, b), tgt_img(b, max_len, w, h, 3)
        src_pos = src_pos.to(DEVICE)
        src_img = src_img.to(DEVICE)
        tgt_pos = tgt_pos.to(DEVICE)
        tgt_img = tgt_img.to(DEVICE)
        loss, LOSS, GAZE = self.test_max(src_pos, src_img, tgt_pos, tgt_img, type)
        return loss, GAZE

    def validation_step(self, batch, batch_idx):
        data1, data2 = batch
        if len(data1)==0:
            logits = self.train_one_dataset(data2, 1, True)
            loss, GAZE = self.valid_one_dataset(data2, 1)
            gt = data2[2][1:,:][:-1]
            target = data2[0][-1]
            sim = saliency_map_metric(logits, data2[2][1:,0])
            ss = nw_matching(gt[:,0].detach().cpu().numpy(), GAZE[:,0].detach().cpu().numpy())
        else:
            logits = self.train_one_dataset(data1, 0, True)
            loss, GAZE = self.valid_one_dataset(data1, 0)
            gt = data1[2][1:,:][:-1]
            target = data1[0][-1]
            sim = saliency_map_metric(logits, data1[2][1:,0])
            ss = nw_matching(gt[:,0].detach().cpu().numpy(), GAZE[:,0].detach().cpu().numpy())

        self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss': loss, 'GAZE': GAZE,  'GAZE_gt': gt,  'target': target, 'sim': sim, 'ss': ss}

    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in validation_step_outputs]).mean()
        avg_sim = np.stack([x['sim'] for x in validation_step_outputs]).mean()
        avg_ss = np.stack([x['ss'] for x in validation_step_outputs]).mean()
        res_gt, res_max = torch.zeros(6).to(DEVICE), torch.zeros(6).to(DEVICE)
        i = 0
        for output in validation_step_outputs:
            gaze = output['GAZE'].cpu().detach().numpy().T
            gaze_gt = output['GAZE_gt'].cpu().detach().numpy().T
            target = output['target'].cpu().detach().numpy()
            behavior(res_gt, target, gaze_gt)
            behavior(res_max, target, gaze)
            i += 1
        res_gt = res_gt / i
        res_max = res_max / i
        res_max[5] = torch.mean(torch.abs(res_max[:5] - res_gt[:5]) / res_gt[:5])
        delta = res_max[5]
        #print('delta: ', delta)
        #print('sim: ', avg_sim)
        self.log('validation_loss_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_delta_each_epoch', delta, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_sim_each_epoch', avg_sim, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_ss_each_epoch', avg_ss, on_epoch=True, prog_bar=True, sync_dist=True)

    def generate_one_scanpath(self, tgt_pos, tgt_img, src_pos, src_img, new_src_img, getMaxProb, type):
        length = tgt_pos.size(0)
        loss = 0
        LOSS = torch.zeros((length - 1, 1)) - 1
        GAZE = torch.zeros((self.max_len, 1)) - 1

        for i in range(1, self.max_len + 1):
            if i == 1:
                tgt_input = tgt_pos[:i, :]
                tgt_img_input = tgt_img[:, :i, :, :, :]
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input, self.PAD_IDX[type])
                src_pos_2d, tgt_input_2d = self.generate3DInput(tgt_input, src_pos, type)

                logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                                    src_img, tgt_img_input,
                                    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask, type)
                # the first token cannot be end token
                logits = logits[:, :, :-2]  # discard padding prob
                if getMaxProb:
                    _, predicted = torch.max(logits[-1, :, :], 1)
                else:
                    logits_new = F.softmax(logits[-1, :, :].view(-1), dim=0)
                    predicted = torch.multinomial(logits_new, 1, replacement=True)
                if i < length:
                    tgt_out = tgt_pos[i, :]
                    LOSS[i - 1][0] = self.loss_fn[type](logits[-1, :, :].reshape(-1, logits[-1, :, :].shape[-1]),
                                                  tgt_out.reshape(-1).long())
                    loss += self.loss_fn[type](logits[-1, :, :].reshape(-1, logits[-1, :, :].shape[-1]),
                                         tgt_out.reshape(-1).long())

                GAZE[i - 1][0] = predicted
                # LOGITS[i-1,:] = self.norm(logits[-1,:,:]).reshape(1,-1)

                next_tgt_img_input = torch.cat((tgt_img_input, new_src_img[:, predicted, :, :, :]), dim=1)
                next_tgt_input = torch.cat((tgt_input, predicted.view(-1, 1)), dim=0)
            else:
                tgt_input = next_tgt_input
                tgt_img_input = next_tgt_img_input
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input, self.PAD_IDX[type])
                src_pos_2d, tgt_input_2d = self.generate3DInput(tgt_input, src_pos, type)
                logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                                    src_img, tgt_img_input,
                                    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask, type)
                logits = logits[:, :, :-1]  # discard padding prob
                if getMaxProb:
                    _, predicted = torch.max(logits[-1, :, :], 1)
                else:
                    logits_new = F.softmax(logits[-1, :, :].view(-1), dim=0)
                    predicted = torch.multinomial(logits_new, 1, replacement=True)
                if i < length:
                    tgt_out = tgt_pos[i, :]
                    LOSS[i - 1][0] = self.loss_fn[type](logits[-1, :, :].reshape(-1, logits[-1, :, :].shape[-1]),
                                                  tgt_out.reshape(-1).long())
                    loss += self.loss_fn[type](logits[-1, :, :].reshape(-1, logits[-1, :, :].shape[-1]),
                                         tgt_out.reshape(-1).long())
                GAZE[i - 1][0] = predicted
                if self.EOS_IDX[type] in GAZE[:, 0] and i >= length:
                    break
                # LOGITS[i-1,:] = self.norm(logits[-1,:,:]).reshape(1,-1)
                next_tgt_img_input = torch.cat((next_tgt_img_input, new_src_img[:, predicted, :, :, :]), dim=1)
                next_tgt_input = torch.cat((next_tgt_input, predicted.view(-1, 1)), dim=0)
        loss = loss / (length - 1)
        return loss, LOSS, GAZE  # ,LOGITS

    def test_max(self, src_pos, src_img, tgt_pos, tgt_img, type):
        #tgt_input = tgt_pos[:-1, :]
        tgt_img = tgt_img[:, :-1, :, :, :]
        blank = torch.zeros((1, 4, src_img.size()[2], src_img.size()[3], 3)).to(DEVICE)
        new_src_img = torch.cat((src_img[:,:-1,:,:], blank), dim=1) #31,300,186,3
        loss, LOSS, GAZE = self.generate_one_scanpath(tgt_pos, tgt_img, src_pos, src_img, new_src_img, True, type)
        if self.EOS_IDX[type] in GAZE:
            endIndex = torch.where(GAZE == self.EOS_IDX[type])[0][0]
            GAZE = GAZE[:endIndex]
            # LOGITS = LOGITS[:endIndex]
        return loss, LOSS, GAZE

    def test_expect(self,src_pos, src_img, tgt_pos, tgt_img, type):
        #tgt_input = tgt_pos[:-1, :]
        tgt_img = tgt_img[:, :-1, :, :, :]
        length = tgt_pos.size(0)
        loss = 0
        blank = torch.zeros((1, 4, src_img.size()[2], src_img.size()[3], 3)).to(DEVICE)
        new_src_img = torch.cat((src_img[:,:-1,:,:], blank), dim=1) #31,300,186,3
        iter = self.args.stochastic_iteration
        GAZE = torch.zeros((self.max_len, iter))-1
        for n in range(iter):
            loss_per, _, GAZE_per = self.generate_one_scanpath(tgt_pos, tgt_img, src_pos, src_img, new_src_img, False, type)
            GAZE[:, n:(n+1)] = GAZE_per
            loss += loss_per / (length-1)
        loss= loss / iter
        GAZE_ALL = []
        for i in range(iter):
            if self.EOS_IDX[type] in GAZE[:,i]:
                j = torch.where(GAZE[:,i]==self.EOS_IDX[type])[0][0]
                GAZE_ALL.append(GAZE[:j, i])
            else:
                GAZE_ALL.append(GAZE[:,i])
        return loss,GAZE_ALL

    def test_gt(self,src_pos, src_img, tgt_pos, tgt_img, type):
        tgt_input = tgt_pos[:-1, :]
        tgt_img_input = tgt_img[:, :-1, :, :, :]
        length = tgt_pos.size(0) - 1
        soft = torch.nn.Softmax(dim=2)
        # src: 15, b; tgt_input: 14, b; src_msk: 15, 15; tgt_msk: 13, 13; tgt_padding_msk: 2, 13; src_padding_msk: 2, 15
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_pos, tgt_input, self.PAD_IDX[type])
        src_pos_2d, tgt_input_2d = self.generate3DInput(tgt_input, src_pos, type)

        logits = self.model(src_pos_2d.float(), tgt_input_2d.float(),  # src_pos, tgt_input,
                            src_img, tgt_img_input,
                            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask, type)
        tgt_out = tgt_pos[1:, :]
        loss = self.loss_fn[type](logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        _, predicted = torch.max(logits, 2)
        LOGITS_tf = soft(logits).squeeze(1)
        print(predicted.view(-1))
        return loss, predicted[:-1], tgt_out[:-1], LOGITS_tf[:-1]

    def test_one_dataset(self, batch, type):
        src_pos, src_img, tgt_pos, tgt_img = batch
        src_pos = src_pos.to(DEVICE)
        src_img = src_img.to(DEVICE)
        tgt_pos = tgt_pos.to(DEVICE)
        tgt_img = tgt_img.to(DEVICE)
        loss_gt, GAZE_tf, GAZE_gt, LOGITS_tf = self.test_gt(src_pos, src_img, tgt_pos, tgt_img, type)
        sim = saliency_map_metric(LOGITS_tf, GAZE_gt[:, 0])
        loss_max, LOSS, GAZE = self.test_max(src_pos, src_img, tgt_pos, tgt_img, type)
        ss_max = compare_multi_gazes(GAZE_gt, [GAZE[:, 0]])
        loss_expect, GAZE_expect = self.test_expect(src_pos, src_img, tgt_pos, tgt_img, type)
        ss_exp = compare_multi_gazes(GAZE_gt, GAZE_expect)
        return loss_max, loss_expect, loss_gt, GAZE, GAZE_expect, GAZE_gt, sim, ss_max, ss_exp

    def test_step(self, batch, batch_idx):
        data1, data2 = batch
        if len(data1) == 0:
            loss_max, loss_expect, loss_gt, GAZE, GAZE_expect, GAZE_gt, sim, ss_max, ss_exp \
                = self.test_one_dataset(data2, 1)
        else:
            loss_max, loss_expect, loss_gt, GAZE, GAZE_expect, GAZE_gt, sim, ss_max, ss_exp \
                = self.test_one_dataset(data1, 0)

        self.log('testing_loss', loss_max, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'loss_max': loss_max, 'loss_expect': loss_expect, 'loss_gt': loss_gt,
                'GAZE': GAZE, 'GAZE_gt': GAZE_gt, 'GAZE_expect': GAZE_expect, 'sim':sim,
                'ss_max': ss_max, 'ss_exp': ss_exp}

    def test_epoch_end(self, test_step_outputs):
        all_loss, all_gaze, all_gaze_gt, all_gaze_expect = \
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for output in test_step_outputs:
            gazes = output['GAZE'].cpu().detach().numpy().T
            all_gaze = pd.concat([all_gaze, pd.DataFrame(gazes)],axis=0)
            gazes_gt = output['GAZE_gt'].cpu().detach().numpy().T
            all_gaze_gt = pd.concat([all_gaze_gt, pd.DataFrame(gazes_gt)],axis=0)
            for i in range(self.args.stochastic_iteration):
                gazes_expect = output['GAZE_expect'][i].cpu().detach().view(1, -1).numpy()
                all_gaze_expect = pd.concat([all_gaze_expect, pd.DataFrame(gazes_expect)],axis=0)

        all_gaze.reset_index().drop(['index'],axis=1)
        all_gaze_gt.reset_index().drop(['index'],axis=1)
        all_gaze_expect.reset_index().drop(['index'],axis=1)
        all_gaze.to_csv(self.args.output_path + '/gaze_max' + self.args.output_postfix + '.csv', index=False)
        all_gaze_gt.to_csv(self.args.output_path + '/gaze_gt' + self.args.output_postfix + '.csv', index=False)
        all_gaze_expect.to_csv(self.args.output_path + '/gaze_expect' + self.args.output_postfix + '.csv', index=False)
        avg_loss = torch.stack([x['loss_max'].cpu().detach() for x in test_step_outputs]).mean()
        self.log('test_loss_max_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        avg_loss = torch.stack([x['loss_expect'].cpu().detach() for x in test_step_outputs]).mean()
        self.log('test_loss_expect_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        avg_loss = torch.stack([x['loss_gt'].cpu().detach() for x in test_step_outputs]).mean()
        self.log('test_loss_gt_each_epoch', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        avg_sim = np.stack([x['sim'] for x in test_step_outputs]).mean()
        self.log('test_sim', avg_sim, on_epoch=True, prog_bar=True, sync_dist=True)
        avg_ss_max = np.stack([x['ss_max'] for x in test_step_outputs]).mean()
        self.log('test_ss_max', avg_ss_max, on_epoch=True, prog_bar=True, sync_dist=True)
        avg_ss_exp = np.stack([x['ss_exp'] for x in test_step_outputs]).mean()
        self.log('test_ss_exp', avg_ss_exp, on_epoch=True, prog_bar=True, sync_dist=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler_lambda1, gamma=self.args.scheduler_lambda2)
        return [optimizer], [scheduler]

