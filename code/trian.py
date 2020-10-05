# -*- coding: utf-8 -*-

import os
import math
import argparse
import random
import numpy
import torch
import torch.nn as nn
from bucket_iterator import BucketIterator
from sklearn import metrics
from data_utils import ABSADatesetReader
from  models.pos_gat import PoS_GAT
import torch.nn.functional as F
from torchsummary import summary
import  numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import  time

class Instructor:
    def __init__(self, opt):
        self.opt = opt

        absa_dataset = ABSADatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim, use_bert=opt.use_bert, max_len=70)

        self.train_data_loader = BucketIterator(data=absa_dataset.train_data, batch_size=opt.batch_size, shuffle=True,
                                                max_len=70)
        self.test_data_loader = BucketIterator(data=absa_dataset.test_data, batch_size=opt.batch_size, shuffle=False,sort=False,
                                               max_len=70)

        self.model = opt.model_class(absa_dataset.embedding_matrix, opt).to(opt.device)
        # self.model = opt.model_class(absa_dataset.embedding_matrix, opt)
        # self.model = nn.DataParallel(self.model).to(opt.device)
        self._print_args()
        self.global_f1 = 0.

        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def label_smoothing(self, inputs, epsilon=0.1):
        '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
        inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
        epsilon: Smoothing rate.

        For example,

        ```
        import tensorflow as tf
        inputs = tf.convert_to_tensor([[[0, 0, 1],
           [0, 1, 0],
           [1, 0, 0]],
          [[1, 0, 0],
           [1, 0, 0],
           [0, 1, 0]]], tf.float32)

        outputs = label_smoothing(inputs)

        with tf.Session() as sess:
            print(sess.run([outputs]))

        >>
        [array([[[ 0.03333334,  0.03333334,  0.93333334],
            [ 0.03333334,  0.93333334,  0.03333334],
            [ 0.93333334,  0.03333334,  0.03333334]],
           [[ 0.93333334,  0.03333334,  0.03333334],
            [ 0.93333334,  0.03333334,  0.03333334],
            [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
        ```
        '''
        V = inputs.get_shape().as_list()[-1]  # number of channels
        return ((1 - epsilon) * inputs) + (epsilon / V)

    def _train(self, criterion, optimizer):
        max_test_acc = 0
        max_test_f1 = 0
        global_step = 0
        continue_not_increase = 0
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            # for name, param in self.model.named_parameters():
            #     print(name, ' ', param.size())
            n_correct, n_total = 0, 0
            increase_flag = False
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                # switch model to training mode, clear gradient accumulators'
                self.model.train()
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                inputs = [inputs,True]
                outputs = self.model(inputs)
                ## ce loss#########################################
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                # predict the best model
                # self.model.load_state_dict(torch.load('state_dict/' + self.opt.model_name + '_' + self.opt.dataset + '.pkl'))

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    test_acc, test_f1, test_loss,t_targets_all, t_outputs_all = self._evaluate_acc_f1(criterion)
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                    if test_f1 > max_test_f1:
                        increase_flag = True
                        max_test_f1 = test_f1

                        if self.opt.save and test_f1 > self.global_f1:
                            self.global_f1 = test_f1
                            np.save("./attention_numpy/ture_outputs_all.npy", t_targets_all)
                            np.save("./attention_numpy/pre_outputs_all.npy", t_outputs_all)
                            torch.save(self.model.state_dict(),
                                       'state_dict/' + self.opt.model_name + '_' + self.opt.dataset + '.pkl')
                            print('>>> this best model saved.this f1 is {:.4f}'.format(max_test_f1))
                    print('>>> this repeat f1 is {:.4f}'.format(max_test_f1 ))
                    print('loss: {:.4f}, acc: {:.4f}, test_loss_all{:.4f}，test_acc: {:.4f}, test_f1: {:.4f}'.format(loss.item(), train_acc,
                                                                                                                    test_loss,test_acc, test_f1))
            if increase_flag == False:
                continue_not_increase += 1
                if continue_not_increase >= (((opt.batch_size)/30)*20):
                    print('early stop.')
                    break
            else:
                continue_not_increase = 0
        return max_test_acc, max_test_f1

    def _evaluate_acc_f1(self,criterion):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total,test_loss_all = 0, 0,0
        t_targets_all, t_outputs_all, = None, None,

        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col].to(opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(opt.device)
                t_inputs = [t_inputs,False]
                t_outputs = self.model(t_inputs)

                test_loss = criterion(t_outputs,t_targets)
                test_loss_all += test_loss.item()

                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        test_loss_all = test_loss_all/n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
        return test_acc, f1,test_loss_all,t_targets_all.cpu(),torch.argmax(t_outputs_all, -1).cpu()

    def run(self, repeats=3):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        # criterion = F.multi_margin_loss(prob, t_label)
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        time_str = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
        f_out = open('log/' + self.opt.model_name + '_' + self.opt.dataset +time_str +'_val.txt', 'w', encoding='utf-8')

        max_test_acc_avg = 0
        max_test_f1_avg = 0
        write_list = []
        for i in range(repeats):
            print('repeat: ', (i + 1))
            self._reset_params()
            max_test_acc, max_test_f1 = self._train(criterion, optimizer)
            print("----------------------------")
            print('max_test_acc: {0}     max_test_f1: {1}'.format(max_test_acc, max_test_f1))
            max_test_acc_avg += max_test_acc
            max_test_f1_avg += max_test_f1
            print('#' * 100)
            f_out.write('max_test_acc: {0}, max_test_f1: {1}\n'.format(max_test_acc, max_test_f1))
        print("max_test_acc_avg:", max_test_acc_avg / repeats)
        print("max_test_f1_avg:", max_test_f1_avg / repeats)
        f_out.write('max_test_acc_avg: {0}, max_test_f1_avg: {1}'.format(max_test_acc_avg / repeats, max_test_f1_avg / repeats))
        f_out.close()


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', default='asgcn_new', type=str)
    parser.add_argument('--model_name', default='pos_gat', type=str)  # GCapNet,pos_gat
    parser.add_argument('--dataset', default='rest14', type=str, help='lap14,twitter, rest14, rest15, rest16')
    parser.add_argument('--optimizer', default='rmsprop', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--l2reg', default=0.0001, type=float)
    parser.add_argument('--num_epoch', default=200, type=int)
    parser.add_argument('--batch_size', default=60, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--save', default=True, type=bool)
    parser.add_argument('--seed', default=776, type=int)  # 776
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--use_lstm_attention', default=True, type=bool)
    parser.add_argument('--use_bert', default=False, type=bool)
    parser.add_argument('--use_speech_weight', default=True, type=bool)
    opt = parser.parse_args()

    model_classes = {

        'pos_gat': PoS_GAT,
        'GCapNet':GCapNet,
    }
    if opt.use_speech_weight ==True:
        input_colses = {
            'pos_gat': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph','speech_list'],
            'GCapNet': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph', 'speech_list'],

        }
    else:
        input_colses = {
            'pos_gat': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
            'GCapNet': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],

        }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    # summary(opt.model_class,input_size=(32,32,300))
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
    ins = Instructor(opt)
    ins.run()
