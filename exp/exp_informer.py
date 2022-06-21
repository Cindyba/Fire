from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from xgboost import XGBRegressor
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack, Informer_Dnn, Informer_, DynamicWeighting, Fusion, DynamicWeighting_Embedding_Xgboost, Informer_embeddingX, Res_Embedding_Xgboost, DynamicWeighting_Only
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

import os
import time
import pickle

import warnings
warnings.filterwarnings('ignore')

f = open (r'log_embedding.txt','w')


class Data_Xgboost(Dataset):
    def __init__(self, new_dataset, y):
        self.input_i = new_dataset
        self.input_x = y

    def __getitem__(self, item):
        batch_x, batch_y, batch_x_mark, batch_y_mark = self.input_i[item]
        return batch_x, batch_y, batch_x_mark, batch_y_mark, self.input_x[item]

    def __len__(self):
        return len(self.input_i)


class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
            'informer_dnn':Informer_Dnn,
            'informer_embedding': Informer_,
            'DynamicWeighting' : DynamicWeighting,
            'Fusion':Fusion,
            'DWEX': DynamicWeighting_Embedding_Xgboost,
            'IEX': Informer_embeddingX,
            'ResEX': Res_Embedding_Xgboost,
            'A_embedding': DynamicWeighting_Only,
        }
        if self.args.model=='informer' or self.args.model=='informerstack' or self.args.model=='informer_dnn' \
                or self.args.model== 'informer_embedding' \
                or self.args.model == 'DynamicWeighting' or self.args.model =='Fusion' \
                or self.args.model == 'DWEX' or self.args.model == 'IEX' or self.args.model == 'ResEX'\
                or self.args.model == 'A_embedding':
            #e_layers = self.args.e_layers if self.args.model=='informer' or self.args.model== 'informer_dnn' or self.args.model== 'informer_embedding' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                self.args.e_layers, #e_layers
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
            'Fire':Dataset_Custom,
            'Fire_17t_18v':Dataset_Custom,
            'img_fire':Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        elif flag=='val':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size; freq=args.freq
        else:
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size; freq=args.freq
        
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq
        )

        assert flag in ['train', 'test', 'val', 'pred']
        num_type = {'train': 60, 'val': 42, 'test': 42, 'pred':42}  # 12m-- train': 54, 'val': 24, 'test': 24, 'pred':24
        num = int(num_type[flag])
        new_dataset = []
        for i in range(len(data_set)):
            if i % num < (num - 36):
                new_dataset.append(data_set[i])

        print(flag, len(data_set), len(new_dataset))
        '''data_loader = DataLoader(
            new_dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)'''

        return new_dataset, data_set  #, data_loader

    def _get_data_Informer(self, flag):
        args = self.args

        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
            'Fire_embedding': Dataset_Custom,
            'Fire_17t_18v': Dataset_Custom,
            'img_fire': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq = args.detail_freq
            Data = Dataset_Pred
        elif flag == 'val':
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq = args.freq
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq = args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq
        )

        assert flag in ['train', 'test', 'val', 'pred']
        num_type = {'train': 60, 'val': 42, 'test': 42,
                    'pred': 42}  # 12m-- train': 54, 'val': 24, 'test': 24, 'pred':24
        num = int(num_type[flag])
        new_dataset = []
        for i in range(len(data_set)):
            if i % num < (num - 36):
                new_dataset.append(data_set[i])

        print(flag, len(data_set), len(new_dataset))
        data_loader = DataLoader(
            new_dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _get_loader(self, new_dataset, y, flag):
        args = self.args
        data = Data_Xgboost(new_dataset, y)
        if flag == 'test':
            shuffle_flag = False;drop_last = True; batch_size = args.batch_size
        elif flag == 'pred':
            shuffle_flag = False;drop_last = False;batch_size = 1
        elif flag == 'val':
            shuffle_flag = True;drop_last = True;batch_size = args.batch_size
        else:
            shuffle_flag = True;drop_last = True;batch_size = args.batch_size

        data_loader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay = 0.001)
        model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, weight_decay = 0.001, momentum=0.9)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()  # 回归
        #criterion = nn.BCELoss()   # 分类
        return criterion

    def xgboost_input(self, xgboost_train):
        x_train = []
        y_train = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(xgboost_train):
            #36month
            #data = batch_x.reshape(batch_x.shape[0]*batch_x.shape[1])
            '''x = np.concatenate((batch_x[:, 1:4], batch_x[:,17:19], batch_x[:,-1:]), axis=1)
            x = x.reshape(x.shape[0]*x.shape[1])
            static = np.concatenate((batch_x[0, -5:-1], batch_x[0, 4:17], batch_x[0, :1]))
            data = np.append(x, static)'''
            #1month
            data = batch_x[-1,:]
            x_train.append(data)
            y_train.append(batch_y[-1][-1])
        x = np.array(x_train)
        #x = np.array(x_train).reshape((-1, 36 * 24))
        y = np.array(y_train)
        return x, y

    def xgboost_load(self, flag):
        model_input, all_data = self._get_data(flag)
        x, y = self.xgboost_input(model_input)
        return x, y, model_input, all_data

    def Xgboost_exp(self):
        x_train, y_train, train_model_input, train_data = self.xgboost_load(flag='train')
        x_vali, y_vali, vali_model_input, vali_data = self.xgboost_load(flag='val')
        x_test, y_test, test_model_input, test_data = self.xgboost_load(flag='test')

        print(x_train.shape, y_train.shape)
        #XG_model = XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=1500, reg_alpha=0.005, subsample=0.8,
                                #gamma=0, colsample_bylevel=0.8, objective='reg:squarederror')
        print('start...')
        #xgb_model = XG_model.fit(x_train, y_train)
        XG_model = pickle.load(open("pima_new.pickle.dat", "rb"))
        print('seccess')
        print(x_test.shape)
        train_results = XG_model.predict(x_train)
        test_results = XG_model.predict(x_test)
        val_results = XG_model.predict(x_vali)

        MSE_XG = mean_squared_error(y_test, test_results)
        MAE_XG = mean_absolute_error(y_test, test_results)
        RS_XG = r2_score(y_test, test_results)
        print("XG'sMSEis", MSE_XG)
        print("XG'sMAEis", MAE_XG)
        print("XG'sRSquaredis", RS_XG)

        train_loader = self._get_loader(train_model_input, train_results, flag='train')
        test_loader = self._get_loader(test_model_input, test_results, flag='test')
        vali_loader = self._get_loader(vali_model_input, val_results, flag='val')

        return train_loader, test_loader, vali_loader, train_data, test_data, vali_data

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval() #不启用 BatchNormalization 和 Dropout，将BatchNormalization和Dropout置为False
        total_loss = []
        #for i, (batch_x,batch_y,batch_x_mark,batch_y_mark, xgboost_output) in enumerate(vali_loader):
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

            pred, true = self._process_one_batch(vali_data, batch_x.type(torch.FloatTensor),
                                                 batch_y.type(torch.FloatTensor), batch_x_mark.type(torch.FloatTensor),
                                                 batch_y_mark.type(torch.FloatTensor))
            #weight, pred, true, y_eui, y_informer, y_poi, y_img, model_weight = self._process_one_batch(
            #    vali_data, batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor), batch_x_mark.type(torch.FloatTensor), batch_y_mark.type(torch.FloatTensor), xgboost_output.type(torch.FloatTensor))

            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()  #启用BatchNormalization和 Dropout，将BatchNormalization和Dropout置为True
        return total_loss

    def valiXgboost(self, vali_data, vali_loader, criterion):
        self.model.eval() #不启用 BatchNormalization 和 Dropout，将BatchNormalization和Dropout置为False
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark, xgboost_output) in enumerate(vali_loader):
            out, pred, true, weight = self.Xgboost_process_one_batch(vali_data, batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor), batch_x_mark.type(torch.FloatTensor), batch_y_mark.type(torch.FloatTensor), xgboost_output.type(torch.FloatTensor))
            loss = criterion(out.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()  #启用BatchNormalization和 Dropout，将BatchNormalization和Dropout置为True
        return total_loss

    def valiXgboost_informer(self, vali_data, vali_loader, criterion):
        self.model.eval() #不启用 BatchNormalization 和 Dropout，将BatchNormalization和Dropout置为False
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark, xgboost_output) in enumerate(vali_loader):
            pred, true = self.Xgboost_informer_process_one_batch(vali_data, batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor), batch_x_mark.type(torch.FloatTensor), batch_y_mark.type(torch.FloatTensor), xgboost_output.type(torch.FloatTensor))
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()  #启用BatchNormalization和 Dropout，将BatchNormalization和Dropout置为True
        return total_loss

    def valiPre(self, model, vali_data, vali_loader, criterion):
        model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark, xgboost_output) in enumerate(vali_loader):
            pred, true = self._process_one_batch(vali_data, batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor), batch_x_mark.type(torch.FloatTensor), batch_y_mark.type(torch.FloatTensor))
            out = model(xgboost_output.to(self.device), pred.to(self.device))
            loss = criterion(out.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        model.train()  #启用BatchNormalization和 Dropout，将BatchNormalization和Dropout置为True
        return total_loss

    def train(self, setting):
        # 仅informer 无xgboost数据输入
        train_data, train_loader = self._get_data_Informer(flag = 'train')
        vali_data, vali_loader = self._get_data_Informer(flag = 'val')
        test_data, test_loader = self._get_data_Informer(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optim, milestones=[5, 15, 30, 40], gamma=0.1)  # ReduceLROnPlateau(model_optim, 'max', verbose=True, patience=3, factor=0.5)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()


        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            #informer+xgboost
            #for i, (batch_x,batch_y,batch_x_mark,batch_y_mark, xgboost_output) in enumerate(train_loader):
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                iter_count += 1

                model_optim.zero_grad()
                #, y_eui, y_informer, y_poi, y_img
                pred, true = self._process_one_batch(train_data, batch_x.type(torch.FloatTensor),
                                                                                      batch_y.type(torch.FloatTensor),batch_x_mark.type(torch.FloatTensor), batch_y_mark.type(torch.FloatTensor))
                #informer+xgboost
                #weight, pred, true, y_eui, y_informer, y_poi, y_img, model_weight = self._process_one_batch(
                 #   train_data, batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor), batch_x_mark.type(torch.FloatTensor), batch_y_mark.type(torch.FloatTensor), xgboost_output.type(torch.FloatTensor))

                loss = criterion(pred, true)
                # + criterion(y_eui, true) + criterion(y_informer, true) + criterion(y_poi,true) + criterion(y_img, true)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()), file=f)
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time), file = f)
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print('lr:', model_optim.param_groups[0]['lr'])
            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time), file = f)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss), file = f)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping", file = f)
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            #scheduler.step() #vali_loss
            
        best_model_path = path +'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model, test_loader, test_data

    def trainXgboost(self, setting):
        # 输入包括informer的时序数据和Xgboost的静态数据
        # 训练调用 Xgboost_process_one_batch
        # model输入包括Xgboost输出 输出为[加权预测结果以及informer单独的结果]
        train_loader, test_loader, vali_loader, train_data, test_data, vali_data = self.Xgboost_exp()

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optim, milestones=[5, 15, 30, 40],
                                                         gamma=0.1)  # ReduceLROnPlateau(model_optim, 'max', verbose=True, patience=3, factor=0.5)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, xgboost_output) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                out, pred, true, weight = self.Xgboost_process_one_batch(train_data, batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor), batch_x_mark.type(torch.FloatTensor), batch_y_mark.type(torch.FloatTensor), xgboost_output.type(torch.FloatTensor))

                loss = criterion(pred, true) + criterion(out, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()), file=f)
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time), file=f)
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print('lr:', model_optim.param_groups[0]['lr'])
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time), file=f)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.valiXgboost(vali_data, vali_loader, criterion)
            test_loss = self.valiXgboost(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss), file=f)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping", file=f)
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            # scheduler.step() #vali_loss

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model, test_loader, test_data

    def trainXgboost_informer(self, setting):
        # 输入包括informer的时序数据和Xgboost的静态数据
        # 训练调用 Xgboost_informer_process_one_batch
        # model输入包括Xgboost输出 输出为预测结果
        train_loader, test_loader, vali_loader, train_data, test_data, vali_data = self.Xgboost_exp()

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optim, milestones=[5, 15, 30, 40],
                                                         gamma=0.1)  # ReduceLROnPlateau(model_optim, 'max', verbose=True, patience=3, factor=0.5)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, xgboost_output) in enumerate(train_loader):

                iter_count += 1

                model_optim.zero_grad()
                pred, true = self.Xgboost_informer_process_one_batch(train_data, batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor), batch_x_mark.type(torch.FloatTensor), batch_y_mark.type(torch.FloatTensor), xgboost_output.type(torch.FloatTensor))

                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()), file=f)
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time), file=f)
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print('lr:', model_optim.param_groups[0]['lr'])
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time), file=f)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.valiXgboost_informer(vali_data, vali_loader, criterion)
            test_loss = self.valiXgboost_informer(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss), file=f)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping", file=f)
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            # scheduler.step() #vali_loss

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model, test_loader, test_data

    def trainPre_training(self, setting):
        # 输入包括informer的时序数据和Xgboost的静态数据
        # 加载预训练模型 只训练最后一层融合层
        # 训练调用 _process_one_batch （model输入中没有Xgboost的输出）
        train_loader, test_loader, vali_loader, train_data, test_data, vali_data = self.Xgboost_exp()

        path = './checkpoints/informer_embedding_img_fire_0.707150749221614'
        best_model_path = path + '/' + 'checkpoint.pth'
        print(best_model_path)
        self.model.load_state_dict(torch.load(best_model_path), False)
        self.model.eval()

        model = Fusion().to(self.device)

        save_path = setting + 'fusion'
        new_path = os.path.join(self.args.checkpoints, save_path)
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = optim.SGD(model.parameters(), lr=self.args.learning_rate, weight_decay=0.001, momentum=0.9)

        criterion = self._select_criterion()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optim, milestones=[15, 50, 120],
                                                         gamma=0.1)  # ReduceLROnPlateau(model_optim, 'max', verbose=True, patience=3, factor=0.5)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, xgboost_output) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, true = self._process_one_batch(train_data, batch_x.type(torch.FloatTensor),
                                                     batch_y.type(torch.FloatTensor),
                                                     batch_x_mark.type(torch.FloatTensor),
                                                     batch_y_mark.type(torch.FloatTensor))

                out = model(xgboost_output.to(self.device), pred.to(self.device))
                loss = criterion(out, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()), file=f)
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time), file=f)
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            print('lr:', model_optim.param_groups[0]['lr'])
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time), file=f)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.valiPre(model, vali_data, vali_loader, criterion)
            test_loss = self.valiPre(model, test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss), file=f)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, model, new_path)
            if early_stopping.early_stop:
                print("Early stopping", file=f)
                break

            #adjust_learning_rate(model_optim, epoch + 1, self.args)
            scheduler.step() #vali_loss

        best_new_model_path = new_path + '/' + 'checkpoint.pth'
        model.load_state_dict(torch.load(best_new_model_path))

        return model, test_loader, test_data

    def test(self, setting, test_loader, test_data):
        
        self.model.eval()
        
        preds = []
        trues = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(test_data,batch_x.type(torch.FloatTensor),
                                                 batch_y.type(torch.FloatTensor),
                                                 batch_x_mark.type(torch.FloatTensor),
                                                batch_y_mark.type(torch.FloatTensor))

            #preds.append(np.multiply(weight.detach().cpu().numpy(), xgboost_output.detach().cpu().numpy()) + np.multiply(1-weight.detach().cpu().numpy(), pred.detach().cpu().numpy()))

            #xgboost pred
            #preds.append(weight.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
        '''
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor), batch_x_mark.type(torch.FloatTensor), batch_y_mark.type(torch.FloatTensor))
            #test_data.inverse_transform(pred)
            #test_data.inverse_transform(true)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
        '''

        print('test_loader len:', len(test_loader))
        preds = np.array(preds)
        trues = np.array(trues)
        #informer = np.array(informer)
        #print('test shape:', preds.shape, trues.shape, file = f)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        #informer = informer.reshape(-1, informer.shape[-2], informer.shape[-1])
        print('preds shape:', preds.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, r2 = metric(preds, trues)
        #acc, precisionT, precisionN, recallT, recallN, auc = metric(preds, trues)
        print('mae:{} mse:{} rmse:{} mape:{} mspe:{} r2:{}'.format(mae, mse, rmse, mape, mspe, r2), file = f)
        print('mae:{} mse:{} rmse:{} mape:{} mspe:{} r2:{}'.format(mae, mse, rmse, mape, mspe, r2))
        '''mae1, mse1, rmse1, mape1, mspe1, r21 = metric(informer, trues)
        print('mae1：{} mse1：{} rmse1：{} mape1：{} mspe1：{} r21:{}'.format(mae1, mse1, rmse1, mape1, mspe1, r21))'''
        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe, r2]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        return mae, mse, r2

    def testXgboost(self, setting, test_loader, test_data):

        self.model.eval()

        preds = []
        trues = []
        informer = []
        weights = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark, xgboost_output) in enumerate(test_loader):
            out, pred, true, weight = self.Xgboost_process_one_batch(
                test_data, batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor), batch_x_mark.type(torch.FloatTensor), batch_y_mark.type(torch.FloatTensor), xgboost_output.type(torch.FloatTensor))

            preds.append(out.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            informer.append(pred.detach().cpu().numpy())
            weights.append(weight.detach().cpu().numpy())
        av_weights = np.average(weights)
        print(av_weights)
        print('test_loader len:', len(test_loader))
        preds = np.array(preds)
        trues = np.array(trues)
        informer = np.array(informer)
        # print('test shape:', preds.shape, trues.shape, file = f)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        informer = informer.reshape(-1, informer.shape[-2], informer.shape[-1])
        # print('preds shape:', preds.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, r2 = metric(preds, trues)
        print('mae:{} mse:{} rmse:{} mape:{} mspe:{} r2:{}'.format(mae, mse, rmse, mape, mspe, r2), file=f)
        print('mae:{} mse:{} rmse:{} mape:{} mspe:{} r2:{}'.format(mae, mse, rmse, mape, mspe, r2))
        mae1, mse1, rmse1, mape1, mspe1, r21 = metric(informer, trues)
        print('mae1：{} mse1：{} rmse1：{} mape1：{} mspe1：{} r21:{}'.format(mae1, mse1, rmse1, mape1, mspe1, r21))
        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe, r2]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return mae, mse, r2, av_weights, r21

    def testXgboost_informer(self, setting, test_loader, test_data):
        self.model.eval()

        preds = []
        trues = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark, xgboost_output) in enumerate(test_loader):
            pred, true = self.Xgboost_informer_process_one_batch(
                test_data, batch_x.type(torch.FloatTensor),
                batch_y.type(torch.FloatTensor), batch_x_mark.type(torch.FloatTensor),
                batch_y_mark.type(torch.FloatTensor), xgboost_output.type(torch.FloatTensor))

            trues.append(true.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())

        print('test_loader len:', len(test_loader))
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('preds shape:', preds.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, r2 = metric(preds, trues)
        print('mae:{} mse:{} rmse:{} mape:{} mspe:{} r2:{}'.format(mae, mse, rmse, mape, mspe, r2), file=f)
        print('mae:{} mse:{} rmse:{} mape:{} mspe:{} r2:{}'.format(mae, mse, rmse, mape, mspe, r2))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, r2]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return mae, mse, r2

    def testPre(self, model, test_loader, test_data):
        model.eval()

        preds = []
        trues = []
        informer = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark, xgboost_output) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor), batch_x_mark.type(torch.FloatTensor), batch_y_mark.type(torch.FloatTensor))
            out = model(xgboost_output.to(self.device), pred.to(self.device))
            preds.append(out.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            informer.append(pred.detach().cpu().numpy())

        print('test_loader len:', len(test_loader))
        preds = np.array(preds)
        trues = np.array(trues)
        informer = np.array(informer)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, 1, 1)
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        informer = informer.reshape(-1, informer.shape[-2], informer.shape[-1])

        # result save
        folder_path = './results/' + 'fusion/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, r2 = metric(preds, trues)
        print('mae:{} mse:{} rmse:{} mape:{} mspe:{} r2:{}'.format(mae, mse, rmse, mape, mspe, r2), file=f)
        print('mae:{} mse:{} rmse:{} mape:{} mspe:{} r2:{}'.format(mae, mse, rmse, mape, mspe, r2))
        mae1, mse1, rmse1, mape1, mspe1, r21 = metric(informer, trues)
        print('mae1：{} mse1：{} rmse1：{} mape1：{} mspe1：{} r21:{}'.format(mae1, mse1, rmse1, mape1, mspe1, r21))
        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe, r2]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return mae, mse, r2

    def predict_(self, setting, load=False):
        vali_data, vali_loader = self._get_data_Informer(flag='val')
        test_data, test_loader = self._get_data_Informer(flag='test')

        if load:
            path = './checkpoints/AAAI_0.7271867572858424'
            best_model_path = path + '/' + 'checkpoint.pth'
            print(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path),False)

        self.model.eval()

        preds = []
        trues = []
        print('pre len:', len(vali_loader))

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor), batch_x_mark.type(torch.FloatTensor), batch_y_mark.type(torch.FloatTensor))
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor), batch_x_mark.type(torch.FloatTensor), batch_y_mark.type(torch.FloatTensor))
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        print(preds.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = np.array(trues)
        print(trues.shape)
        trues = trues.reshape(-1, preds.shape[-2], preds.shape[-1])

        # print(preds)
        # print(trues)

        # result save
        folder_path = './results/AAAI/'
        print(folder_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        np.save(folder_path + 'real_trues.npy', trues)

        return


    def predict(self, setting, load=False):
        train_loader, pred_loader, vali_loader, train_data, pred_data, vali_data = self.Xgboost_exp()
        
        if load:
            path = './checkpoints/DynamicWeighting_img_fire_0.721499969115844'
            best_model_path = path+'/'+'checkpoint.pth'
            print(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path),False)

        self.model.eval()
        
        preds = []
        trues = []
        print('pre len:', len(pred_loader))
        informer = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, xgboost_output) in enumerate(pred_loader):
            pred, informer_out, true, weight = self.Xgboost_process_one_batch(
                pred_data, batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor),
                batch_x_mark.type(torch.FloatTensor), batch_y_mark.type(torch.FloatTensor),
                xgboost_output.type(torch.FloatTensor))
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            informer.append(informer_out.detach().cpu().numpy())

        print('vali len:', len(vali_loader))
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, xgboost_output) in enumerate(vali_loader):
            pred, informer_out, true, weight = self.Xgboost_process_one_batch(
                vali_data, batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor),
                batch_x_mark.type(torch.FloatTensor), batch_y_mark.type(torch.FloatTensor),
                xgboost_output.type(torch.FloatTensor))
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            informer.append(informer_out.detach().cpu().numpy())

        preds = np.array(preds)
        print(preds.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = np.array(trues)
        print(trues.shape)
        trues = trues.reshape(-1, preds.shape[-2], preds.shape[-1])

        #print(preds)
        #print(trues)
        
        # result save
        folder_path = './results/' + setting +'/'
        print(folder_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        np.save(folder_path + 'real_trues.npy', trues)
        
        return

    def predict_DWEX(self, setting, load=False):
        train_loader, pred_loader, vali_loader, train_data, pred_data, vali_data = self.Xgboost_exp()

        if load:
            path = './checkpoints/DWEX_img_fire_ftMS_sl36_ll20_pl1_dm512_nh8_el1_dl1_df2048_atprob_fc4_0.72563293_batch_size_128'
            best_model_path = path + '/' + 'checkpoint.pth'
            print(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path), False)

        self.model.eval()

        preds = []
        trues = []
        print('pre len:', len(pred_loader))
        informer = []
        xgboost_out = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, xgboost_output) in enumerate(pred_loader):
            pred, informer_out, true, weight = self.Xgboost_process_one_batch(
                pred_data, batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor),
                batch_x_mark.type(torch.FloatTensor), batch_y_mark.type(torch.FloatTensor),
                xgboost_output.type(torch.FloatTensor))
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            informer.append(informer_out.detach().cpu().numpy())
            # xgboost_out.append(xgboost_output.unsqueeze(dim=1).unsqueeze(dim=1).detach().cpu().numpy())
        print('vali len:', len(vali_loader))
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, xgboost_output) in enumerate(vali_loader):
            pred, informer_out, true, weight = self.Xgboost_process_one_batch(
                vali_data, batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor),
                batch_x_mark.type(torch.FloatTensor), batch_y_mark.type(torch.FloatTensor),
                xgboost_output.type(torch.FloatTensor))
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            informer.append(informer_out.detach().cpu().numpy())
            # xgboost_out.append(xgboost_output.unsqueeze(dim=1).unsqueeze(dim=1).detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        informers = np.array(informer)
        informers = informers.reshape(-1, informers.shape[-2], informers.shape[-1])
        trues = np.array(trues)
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # xgboost_outs = np.array(xgboost_out)
        # xgboost_outs = xgboost_outs.reshape(-1, xgboost_outs.shape[-2], xgboost_outs.shape[-1])
        # print(preds.shape)
        # print(trues.shape)
        # print(xgboost_outs.shape)

        # print(float(np.sum(np.true_divide(preds-informers,xgboost_outs-informers))/xgboost_outs.shape[0]))

        # result save
        folder_path = './results/' + setting + '/'
        print(folder_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        np.save(folder_path + 'real_trues.npy', trues)

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs, informer_out, y_eui, y_informer, y_poi, y_img, weight = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs, informer_out, y_eui, y_informer, y_poi, y_img, weight = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs, informer_out, y_eui, y_informer, y_poi, y_img, weight = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                #, y_eui, y_informer, y_poi, y_img
                outputs = self.model(batch_x.to(self.device), batch_x_mark.to(self.device), dec_inp.to(self.device), batch_y_mark.to(self.device))
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        return outputs, batch_y

    def Xgboost_process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, xgboost_output):
        # model输入包括Xgboost输出 输出为[加权预测结果以及informer单独的结果,权重值]
        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs, informer_out, weight = self.model(batch_x.to(self.device), batch_x_mark.to(self.device), dec_inp.to(self.device), batch_y_mark.to(self.device), xgboost_output.to(self.device))[0]
                else:
                    outputs, informer_out, weight = self.model(batch_x.to(self.device), batch_x_mark.to(self.device), dec_inp.to(self.device), batch_y_mark.to(self.device), xgboost_output.to(self.device))
        else:
            if self.args.output_attention:
                outputs, informer_out, weight = self.model(batch_x.to(self.device), batch_x_mark.to(self.device), dec_inp.to(self.device), batch_y_mark.to(self.device), xgboost_output.to(self.device))[0]
            else:
                outputs, informer_out, weight = self.model(batch_x.to(self.device), batch_x_mark.to(self.device), dec_inp.to(self.device), batch_y_mark.to(self.device), xgboost_output.to(self.device))
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        return outputs, informer_out, batch_y, weight

    def Xgboost_informer_process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, xgboost_output):
        # model输入包括Xgboost输出 输出为预测结果
        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x.to(self.device), batch_x_mark.to(self.device), dec_inp.to(self.device), batch_y_mark.to(self.device), xgboost_output.to(self.device))[0]
                else:
                    outputs = self.model(batch_x.to(self.device), batch_x_mark.to(self.device), dec_inp.to(self.device), batch_y_mark.to(self.device), xgboost_output.to(self.device))
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x.to(self.device), batch_x_mark.to(self.device), dec_inp.to(self.device), batch_y_mark.to(self.device), xgboost_output.to(self.device))[0]
            else:
                outputs = self.model(batch_x.to(self.device), batch_x_mark.to(self.device), dec_inp.to(self.device), batch_y_mark.to(self.device), xgboost_output.to(self.device))
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        return outputs, batch_y


