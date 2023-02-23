import numpy as np
import tensorflow as tf
import os

from tensorflow.keras import callbacks

from keras import layers
from keras import models
from keras import optimizers

from buffer import MemoryBuffer

class DQNagent():
    def __init__(self,environment,Name,b_size):
        self.name = Name
        self.state_collector = environment.state_collector
        self.epoch = 1
        
        # 動作數量
        self.action_size = 2
        # 經驗回放池大小
        self.with_per = True
        self.buffer_size = b_size
        self.exp_buffer = MemoryBuffer(self.buffer_size, self.with_per)
        self.total_episode = 500
        self.discount_factor = 0.99   # 折扣因子
        self.update_steps = 100
        self.train_steps = 4
        self.pretrain_steps = 2000
        
        self.learning_rate = 0.0004
        self.optimizer = optimizers.Adam(lr=self.learning_rate)
        self.batch_size = 32
     
        # 貪婪策略設定
        self.epsilon = 1.0
        # epsilon最小值
        self.eps_min = 0.1
        # epsilon最大值
        self.eps_max = 1.0
        # epsilon衰退率(從一開始最大到200回合變最小，到哪個回合隨你設)
        self.eps_decay_steps = 200
        # 建立實際Q網路
        self.Q = self.build_model('Q')
        # 建立目標Q網路(跟上面那個相同架構)
        self.target_Q = self.build_model('target_Q')
        # 複製實際Q網路得權重到目標Q網路
        self.update_target_network()

    def list_attribute(self):
        print("Agent Name   : ",self.name)
        print("Action Size  : ",self.action_size)
        print("PER          : ",self.with_per)
        print("Discount Rate: ",self.action_size)
        print("Batch Size   : ",self.batch_size)
        print("Learning Rate: ",self.learning_rate)
        print("Buffer Size  : ",self.exp_buffer.buffer_size)
        print("Target Q update steps : ",self.update_steps)
        print("Q training steps      : ",self.train_steps)
        print("Pre-trained steps     : ",self.pretrain_steps)

    def build_model(self,T):
        input_data = []
        output = []

        # downstream state
        input_img = layers.Input(shape=self.state_collector['downstream'],dtype='float32', name = 'downstream')
        x = layers.Conv2D(16, kernel_size=(5, 3), strides = (1, 1), activation = "relu", name ='downstream_cnn_1')(input_img)
        x = layers.MaxPooling2D(pool_size=(2, 2), name = 'downstream_maxpool_1')(x)
        x = layers.Conv2D(32, kernel_size=(1, 2), strides = (1, 1), activation = "relu", name ='downstream_cnn_2')(x)
        x = layers.MaxPooling2D(pool_size=(1, 2), name = 'downstream_maxpool_2')(x)
        x = layers.Flatten(name = 'downstream_flat')(x)
        input_data.append(input_img)
        output.append(x)

        # upstream state
        input_img = layers.Input(shape=self.state_collector['upstream'],dtype='float32', name = 'upstream')
        x = layers.Conv2D(16, kernel_size=(5, 3), strides = (1, 1), activation = "relu", name ='upstream_cnn_1')(input_img)
        x = layers.MaxPooling2D(pool_size=(2, 2), name = 'upstream_maxpool_1')(x)
        x = layers.Conv2D(32, kernel_size=(1, 2), strides = (1, 1), activation = "relu", name ='upstream_cnn_2')(x)
        x = layers.MaxPooling2D(pool_size=(1, 2), name = 'upstream_maxpool_2')(x)
        x = layers.Flatten(name = 'upstream_flat')(x)
        input_data.append(input_img)
        output.append(x)

        # bus lane state
        input_img = layers.Input(shape=self.state_collector['buslane'],dtype='float32', name = 'buslane')
        x = layers.Conv2D(16, kernel_size=(5, 3), strides = (1, 1), activation = "relu", name ='buslane_cnn_1')(input_img)
        x = layers.MaxPooling2D(pool_size=(2, 2), name = 'buslane_maxpool_1')(x)
        x = layers.Conv2D(32, kernel_size=(1, 2), strides = (1, 1), activation = "relu", name ='buslane_cnn_2')(x)
        x = layers.MaxPooling2D(pool_size=(1, 2), name = 'buslane_maxpool_2')(x)
        x = layers.Flatten(name = 'buslane_flat')(x)
        input_data.append(input_img)
        output.append(x)

        # concatenate upstream and dwon stream
        fmodel = layers.concatenate(name = 'concat')(output)
        fmodel = layers.Dense(288, activation='relu',  name = 'dense_1')(fmodel)
        fmodel = layers.Dense(144, activation='relu',  name = 'dense_2')(fmodel)
        fmodel = layers.Dense(72,  activation='relu',  name = 'dense_3')(fmodel)
        fmodel = layers.Dense(36,  activation='relu',  name = 'dense_4')(fmodel)
        fmodel = layers.Dense(18,  activation='relu',  name = 'dense_5')(fmodel)
        fmodel = layers.Dense(10,  activation='relu', name = 'dense_6')(fmodel)
        fmodel = layers.Dense(self.action_size, activation='linear', name = 'action_output')(fmodel)
        model  = models.Model(inputs=input_data, outputs=fmodel, name = '%s_%s'%(self.name,T))
        model.compile(loss='mse', optimizer = self.optimizer)

        return model

    # 權重複製
    def update_target_network(self):
        self.target_Q.set_weights(self.Q.get_weights())
    
    # 經驗存取，經驗也就是<s,a,r,s'>過程，最後那個error是給優先經驗回放用的，你可能看懂優先經驗回放後才懂XD
    # 在存經驗前，需要從其他AGENT取得fingerprints，並放入state
    def remember(self, state, action, reward, next_state):
        self.exp_buffer.memorize(state, action, reward, next_state)

    def epsilon_decay(self):
        self.epsilon = max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * (self.epoch - 1)/self.eps_decay_steps)

    def choose_action(self, state):
        """ Apply espilon-greedy policy to pick action
        """
        # state = {'downstream': ,'upstream': ,'FP_input'...}
        # epsilon greedy decay in every episode
        self.epsilon = max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * (self.epoch - 1)/self.eps_decay_steps)
        #
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            # reshape state as (1, minutes, vd count, features)
            for k,v in state.items():
                Shape = [i for i in v.shape]
                Shape.insert(0,-1)
                state[k] = v.reshape(Shape)
            action = self.Q.predict(state)
            action = np.argmax(action, axis=-1)

        return int(action)

    def training(self):

        # Return a batch of experience
        # batch = [(trajectory,idx)] refer to buffer.py line 71
        batch, idx = self.exp_buffer.sample_batch(self.batch_size)
        
        # decomposition batch data into <s,a,r,s',FP>
        all_state = {} # s
        all_next_state = {} # s'
        for pos in self.state_collector:
            all_state[pos] = np.array([i[0][pos] for i in batch])
            all_next_state[pos] = np.array([i[3][pos] for i in batch])

        action = np.array([i[1] for i in batch])
        reward = np.array([i[2] for i in batch])
        
        y = self.Q.predict(all_state)
        next_Q = self.target_Q.predict(all_next_state)
        
        # double dqn strategy using main q network to find best action
        next_act = self.Q.predict(all_next_state)
        max_act = np.argmax(next_act, axis=1)
        
        discount_Q = [self.discount_factor * next_Q[index,max_act[index]] for index in range(self.batch_size)]
        target_y = reward + discount_Q

        if(self.exp_buffer.with_per):
            for i in range(len(action)):
                # calculate TD-error
                error = np.abs(target_y[i] - y[i][action[i]])
                self.exp_buffer.update(idx[i], error)
    
        for index in range(len(action)):
            y[index][int(action[index])] = target_y[index]

        # training Q network
        history = self.Q.fit(all_state,{'action_output': y}, epochs=1, verbose=0)
        # record loss (unnecessary)
        loss = history.history['loss']
        return loss
    
    # load model parameter
    def load(self,name1,name2):
        self.Q = tf.keras.models.load_model(name1)
        self.target_Q = tf.keras.models.load_model(name2)
    # save model parameter
    def save(self):
        self.Q.save('epoch_'+str(self.epoch)+'_epsilon_'+str(self.epsilon)+'_Q.h5')
        self.target_Q.save('epoch_'+str(self.epoch)+'_epsilon_'+str(self.epsilon)+'_targetQ.h5')

class IDQNagent(DQNagent):
    def __init__(self,environment,Name,Kind,action_s,b_size,fingerprint=True,Move=False,deep=False):
        self.env = environment
        self.FP = fingerprint
        self.FP_size = 2
        self.move = Move
        self.kind = Kind # 'mainline' or 'ramp'
        if self.kind == 'm':
            self.state_collector = {'downstream':environment.state_collector['downstream'],'buslane':environment.state_collector['buslane'],'metering':environment.state_collector['metering']}
        elif self.kind == 'r':
            self.state_collector = {'upstream':environment.state_collector['upstream'],'buslane':environment.state_collector['buslane'],'metering':environment.state_collector['metering']}
        else: print('Incorrect agent kind. Please set kind = "m" or "r" .')

        if self.FP:
            self.state_collector['fp'] = (self.FP_size,)
        if not self.move:
            self.state_collector.pop('metering')
        self.name = Name
        self.epoch = 1
        
        # 動作數量
        self.action_size = action_s
        # 經驗回放池大小
        self.with_per = True
        self.buffer_size = b_size
        self.exp_buffer = MemoryBuffer(self.buffer_size, self.with_per)
        self.total_episode = 500
        self.discount_factor = 0.99   # 折扣因子
        self.update_steps = 100
        self.train_steps = 4
        self.pretrain_steps = 2500
        
        self.learning_rate = 0.0004
        self.optimizer = optimizers.Adam(lr=self.learning_rate)
        self.batch_size = 32
     
        # 貪婪策略設定
        self.epsilon = 1.0
        # epsilon最小值
        self.eps_min = 0.1
        # epsilon最大值
        self.eps_max = 1.0
        # epsilon衰退率(從一開始最大到200回合變最小，到哪個回合隨你設)
        self.eps_decay_steps = 200
        # 建立實際Q網路
        self.Q = self.build_model('Q',deep)
        # 建立目標Q網路(跟上面那個相同架構)
        self.target_Q = self.build_model('target_Q',deep)
        # 複製實際Q網路得權重到目標Q網路
        self.update_target_network()

    def list_attribute(self):
        print("Agent Name   : ",self.name)
        print("Action Size  : ",self.action_size)
        print("PER          : ",self.with_per)
        print("Discount Rate: ",self.discount_factor)
        print("Decay Steps  : ",self.eps_decay_steps)
        print("Batch Size   : ",self.batch_size)
        print("Learning Rate: ",self.learning_rate)
        print("Buffer Size  : ",self.exp_buffer.buffer_size)
        print("Target Q update steps : ",self.update_steps)
        print("Q training steps      : ",self.train_steps)
        print("Pre-trained steps     : ",self.pretrain_steps)

    def build_model(self,T,deep):
        print('IDQN')
        input_data = []
        concat_layer = []
        if deep:
            if self.kind == 'm':
                # downstream state
                input_img1 = layers.Input(shape=self.state_collector['downstream'],dtype='float32', name = 'downstream')
                input_data.append(input_img1)
                x1 = layers.Conv2D(16, kernel_size=(5, 3), strides = (1, 1), activation = "relu", name ='downstream_cnn_1')(input_img1)
                x1 = layers.MaxPooling2D(pool_size=(2, 2), name = 'downstream_maxpool_1')(x1)
                x1 = layers.Conv2D(32, kernel_size=(1, 2), strides = (1, 1), activation = "relu", name ='downstream_cnn_2')(x1)
                x1 = layers.MaxPooling2D(pool_size=(1, 2), name = 'downstream_maxpool_2')(x1)
                x1 = layers.Flatten(name = 'downstream_flat')(x1)
                concat_layer.append(x1)

                # bus lane state
                input_img2 = layers.Input(shape=self.state_collector['buslane'],dtype='float32', name = 'buslane')
                input_data.append(input_img2)
                x2 = layers.Conv2D(16, kernel_size=(5, 3), strides = (1, 1), activation = "relu", name ='buslane_cnn_1')(input_img2)
                x2 = layers.MaxPooling2D(pool_size=(2, 2), name = 'buslane_maxpool_1')(x2)
                x2 = layers.Conv2D(32, kernel_size=(1, 2), strides = (1, 1), activation = "relu", name ='buslane_cnn_2')(x2)
                x2 = layers.MaxPooling2D(pool_size=(1, 2), name = 'buslane_maxpool_2')(x2)
                x2 = layers.Flatten(name = 'buslane_flat')(x2)
                concat_layer.append(x2)

                fmodel = layers.concatenate(inputs = concat_layer,name = 'concat')
                fmodel = layers.Dense(256, activation='relu',  name = 'dense_1')(fmodel)
                fmodel = layers.Dense(128, activation='relu',  name = 'dense_2')(fmodel)
                fmodel = layers.Dense(56,  activation='relu',  name = 'dense_3')(fmodel)
                fmodel = layers.Dense(28,  activation='relu',  name = 'dense_4')(fmodel)

            if self.kind == 'r':
                # upstream state
                input_img1 = layers.Input(shape=self.state_collector['upstream'],dtype='float32', name = 'upstream')
                input_data.append(input_img1)
                x1 = layers.Conv2D(16, kernel_size=(5, 3), strides = (1, 1), activation = "relu", name ='upstream_cnn_1')(input_img1)
                x1 = layers.MaxPooling2D(pool_size=(2, 2), name = 'upstream_maxpool_1')(x1)
                x1 = layers.Conv2D(32, kernel_size=(1, 2), strides = (1, 1), activation = "relu", name ='upstream_cnn_2')(x1)
                x1 = layers.MaxPooling2D(pool_size=(1, 2), name = 'upstream_maxpool_2')(x1)
                x1 = layers.Flatten(name = 'upstream_flat')(x1)
                concat_layer.append(x1)

                # bus lane state
                input_img2 = layers.Input(shape=self.state_collector['buslane'],dtype='float32', name = 'buslane')
                input_data.append(input_img2)
                x2 = layers.Conv2D(16, kernel_size=(5, 3), strides = (1, 1), activation = "relu", name ='buslane_cnn_1')(input_img2)
                x2 = layers.MaxPooling2D(pool_size=(2, 2), name = 'buslane_maxpool_1')(x2)
                x2 = layers.Conv2D(32, kernel_size=(1, 2), strides = (1, 1), activation = "relu", name ='buslane_cnn_2')(x2)
                x2 = layers.MaxPooling2D(pool_size=(1, 2), name = 'buslane_maxpool_2')(x2)
                x2 = layers.Flatten(name = 'buslane_flat')(x2)
                concat_layer.append(x2)

                fmodel = layers.concatenate(inputs = concat_layer,name = 'concat')
                fmodel = layers.Dense(144, activation='relu',  name = 'dense_1')(fmodel)
                fmodel = layers.Dense(72, activation='relu',  name = 'dense_2')(fmodel)
                fmodel = layers.Dense(36,  activation='relu',  name = 'dense_3')(fmodel)
            
            if self.FP == True:
                # include fingerprints
                fp_input = layers.Input(shape=self.state_collector['fp'],dtype='float32', name='fp')
                input_data.append(fp_input)
                fmodel = layers.concatenate(inputs = [fmodel,fp_input], name = 'concat_FP')
            
            if self.move == True:
                # include other agent's action
                move_input = layers.Input(shape=self.state_collector['metering'],dtype='float32', name='metering')
                input_data.append(move_input)
                x3 = layers.Flatten(name = 'metering_flat')(move_input)
                fmodel = layers.concatenate(inputs = [fmodel,x3], name = 'concat_metering')

            fmodel = layers.Dense(20, activation='relu', name = 'last_dense')(fmodel)
            fmodel = layers.Dense(self.action_size, activation='linear', name = 'action_output')(fmodel)
            model  = models.Model(inputs=input_data, outputs=fmodel, name = '%s_%s_%s'%(self.name,T,self.kind))
            model.compile(loss='mse', optimizer = self.optimizer)

            return model
        
        else:
            if self.kind == 'm':
                # downstream state
                input_img1 = layers.Input(shape=self.state_collector['downstream'],dtype='float32', name = 'downstream')
                input_data.append(input_img1)
                x1 = layers.Conv2D(16, kernel_size=(5, 3), strides = (1, 1), activation = "relu", name ='downstream_cnn_1')(input_img1)
                x1 = layers.MaxPooling2D(pool_size=(2, 2), name = 'downstream_maxpool_1')(x1)
                x1 = layers.Conv2D(32, kernel_size=(1, 2), strides = (1, 1), activation = "relu", name ='downstream_cnn_2')(x1)
                x1 = layers.MaxPooling2D(pool_size=(1, 2), name = 'downstream_maxpool_2')(x1)
                x1 = layers.Flatten(name = 'downstream_flat')(x1)
                concat_layer.append(x1)

                # bus lane state
                input_img2 = layers.Input(shape=self.state_collector['buslane'],dtype='float32', name = 'buslane')
                input_data.append(input_img2)
                x2 = layers.Conv2D(16, kernel_size=(5, 3), strides = (1, 1), activation = "relu", name ='buslane_cnn_1')(input_img2)
                x2 = layers.MaxPooling2D(pool_size=(2, 2), name = 'buslane_maxpool_1')(x2)
                x2 = layers.Conv2D(32, kernel_size=(1, 2), strides = (1, 1), activation = "relu", name ='buslane_cnn_2')(x2)
                x2 = layers.MaxPooling2D(pool_size=(1, 2), name = 'buslane_maxpool_2')(x2)
                x2 = layers.Flatten(name = 'buslane_flat')(x2)
                concat_layer.append(x2)

                fmodel = layers.concatenate(inputs = concat_layer,name = 'concat')
                fmodel = layers.Dense(256, activation='relu',  name = 'dense_1')(fmodel)

            if self.kind == 'r':
                # upstream state
                input_img1 = layers.Input(shape=self.state_collector['upstream'],dtype='float32', name = 'upstream')
                input_data.append(input_img1)
                x1 = layers.Conv2D(16, kernel_size=(5, 3), strides = (1, 1), activation = "relu", name ='upstream_cnn_1')(input_img1)
                x1 = layers.MaxPooling2D(pool_size=(2, 2), name = 'upstream_maxpool_1')(x1)
                x1 = layers.Conv2D(32, kernel_size=(1, 2), strides = (1, 1), activation = "relu", name ='upstream_cnn_2')(x1)
                x1 = layers.MaxPooling2D(pool_size=(1, 2), name = 'upstream_maxpool_2')(x1)
                x1 = layers.Flatten(name = 'upstream_flat')(x1)
                concat_layer.append(x1)

                # bus lane state
                input_img2 = layers.Input(shape=self.state_collector['buslane'],dtype='float32', name = 'buslane')
                input_data.append(input_img2)
                x2 = layers.Conv2D(16, kernel_size=(5, 3), strides = (1, 1), activation = "relu", name ='buslane_cnn_1')(input_img2)
                x2 = layers.MaxPooling2D(pool_size=(2, 2), name = 'buslane_maxpool_1')(x2)
                x2 = layers.Conv2D(32, kernel_size=(1, 2), strides = (1, 1), activation = "relu", name ='buslane_cnn_2')(x2)
                x2 = layers.MaxPooling2D(pool_size=(1, 2), name = 'buslane_maxpool_2')(x2)
                x2 = layers.Flatten(name = 'buslane_flat')(x2)
                concat_layer.append(x2)

                fmodel = layers.concatenate(inputs = concat_layer,name = 'concat')
                fmodel = layers.Dense(144, activation='relu',  name = 'dense_1')(fmodel)
            
            if self.FP == True:
                # include fingerprints
                fp_input = layers.Input(shape=self.state_collector['fp'],dtype='float32', name='fp')
                input_data.append(fp_input)
                fmodel = layers.concatenate(inputs = [fmodel,fp_input], name = 'concat_FP')
            
            if self.move == True:
                # include other agent's action
                move_input = layers.Input(shape=self.state_collector['metering'],dtype='float32', name='metering')
                input_data.append(move_input)
                x3 = layers.Flatten(name = 'metering_flat')(move_input)
                fmodel = layers.concatenate(inputs = [fmodel,x3], name = 'concat_metering')

            fmodel = layers.Dense(64, activation='relu', name = 'dense_2')(fmodel)
            fmodel = layers.Dense(self.action_size, activation='linear', name = 'action_output')(fmodel)
            model  = models.Model(inputs=input_data, outputs=fmodel, name = '%s_%s_%s'%(self.name,T,self.kind))
            model.compile(loss='mse', optimizer = self.optimizer)

            return model

    def add_FP2obs(self,agent,state):# add fingerprints to state
        s = {}
        for k in state:
            s[k] = state[k]
        if self.FP:
            fingerprint = np.zeros(self.state_collector['fp'])
            fingerprint[0] = agent.epoch
            fingerprint[1] = agent.epsilon
            s['fp'] = fingerprint
        return s

    def load_state(self,agent,state):
        # state = {'downstream': ,'upstream': ,'FP_input'...}
        s = self.add_FP2obs(agent,state)
        temp = {}
        for k in self.state_collector:
            temp[k] = s[k]
        return temp
    
    def training(self):
        # Return a batch of experience
        # batch = [(trajectory,idx)] refer to buffer.py line 71
        batch, idx = self.exp_buffer.sample_batch(self.batch_size)
        
        # decomposition batch data into <s,a,r,s',FP>
        all_state = {} # s
        all_next_state = {} # s'
        for pos in self.state_collector:
            all_state[pos] = np.array([i[0][pos] for i in batch])
            all_next_state[pos] = np.array([i[3][pos] for i in batch])

        action = np.array([i[1] for i in batch])
        reward = np.array([i[2] for i in batch])

        y = self.Q.predict(all_state)
        next_Q = self.target_Q.predict(all_next_state)
        
        # double dqn strategy using main q network to find best action
        next_act = self.Q.predict(all_next_state)
        max_act = np.argmax(next_act, axis=1)

        discount_Q = [self.discount_factor * next_Q[index,max_act[index]] for index in range(self.batch_size)]
        target_y = reward + discount_Q
        if(self.exp_buffer.with_per):
            for i in range(len(action)):
                # calculate TD-error
                error = np.abs(target_y[i] - y[i][action[i]])
                self.exp_buffer.update(idx[i], error)
    
        for index in range(len(action)):
            y[index][int(action[index])] = target_y[index]

        # training Q network
        history = self.Q.fit(all_state,{'action_output': y}, epochs=1, verbose=0)
        # record loss (unnecessary)
        loss = history.history['loss']
        return loss
    
    def save(self,path):
        if self.kind == 'm':
            self.Q.save(path + (r"\mainline_model\epoch_%s_Q.h5"%(str(self.epoch))))
            self.target_Q.save(path + (r"\mainline_model\epoch_%s_targetQ.h5"%(str(self.epoch))))
        elif self.kind == 'r':
            self.Q.save(path + (r"\ramp_model\epoch_%s_Q.h5"%(str(self.epoch))))
            self.target_Q.save(path + (r"\ramp_model\epoch_%s_targetQ.h5"%(str(self.epoch))))
