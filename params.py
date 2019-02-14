import os
import wordvector as wv
from datetime import datetime


class args():
    def __init__(self):
        self.BATCH_SIZE = 128
        print('[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '] Start loading word embedding...')
#         self.embed, self.embeding_num, self.embeding_dim = wv.get_embedding(vec_file='./dataset/wordvector/vec_50.txt', dim=50)
        self.embed, self.embeding_dim = wv.get_embedding_300()
        print('[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '] Word embedding loaded')
        self.output_num = 3
        self.dropout = 0.5
        self.static = False
        self.lr = 0.001
        self.epochs = 10
        self.log_interval = 20
        self.test_interval = 400
        self.save_interval = 10
        self.loss_log_interval = 50
        self.version = 'v10'
        self.train_filename = 'dataset/' + self.version + '/train_align_all_balanced_def.tsv'
        self.valid_filename = 'dataset/' + self.version + '/valid_align_all_balanced_def.tsv'
        self.test_filename = 'dataset/' + self.version + '/test_align_all_balanced_def.tsv'
        self.gold_filename = 'dataset/test_align_gold_def_v2.tsv'
        self.save_dir = 'models/data_' + self.version  # model save path
        
        # Mode
        self.mode = 'defent'
        self.model = 'pcnn'
        self.predict_dir = 'predict/data_' + self.version + '/' + self.model + '_' + self.mode
        self.gold_dir = 'predict/data_' + self.version + '/gold/' + self.model + '_' + self.mode
#         self.model_filename = 'pcnn_final_defent_no_static_4'
        self.model_filename = self.model + '_final_' + self.mode
        if self.static == False:
            self.model_filename = self.model_filename + '_no_static'
        
        # Kernel size configuration
        if self.mode == 'nodef' and self.model == 'cnn':
            self.kernel_num = 100
            self.kernel_sizes = [1, 2]
        elif self.mode == 'nodef' and self.model == 'pcnn':
            self.kernel_num = 100
            self.kernel_sizes = [1]
        elif self.mode == 'def' and self.model == 'cnn':
            self.kernel_num = 100
            self.kernel_sizes = [1, 2]
        elif self.mode == 'def' and self.model == 'pcnn':
            self.kernel_num = 100
            self.kernel_sizes = [1]
        else:
            self.kernel_num = 100
            self.kernel_sizes = [1, 2]
        
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
            
        if not os.path.exists(self.predict_dir):
            os.makedirs(self.predict_dir)
            
        if not os.path.exists(self.gold_dir):
            os.makedirs(self.gold_dir)
