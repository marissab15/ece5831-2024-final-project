#QB/RB Comparison Model

#Import packages
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

class QBvsRB():
    def __init__(self):
        self.dataset = None
        self.X = None
        self.y = None
        self.callbacks = None
        self.model = None

    def preprocess_train_data(self):
        #Standardize stats by # of games played
        self.dataset['Pass_Cmp_per_Game_1'] = self.dataset['Pass_Cmp_1']/self.dataset['QB_Games_1']
        self.dataset['Pass_Att_per_Game_1'] = self.dataset['Pass_Att_1']/self.dataset['QB_Games_1']
        self.dataset['Pass_TD_per_Game_1'] = self.dataset['Pass_TD_1']/self.dataset['QB_Games_1']
        self.dataset['Int_per_Game_1'] = self.dataset['Int_1']/self.dataset['QB_Games_1']

        self.dataset['Rush_Att_per_Game_1'] = self.dataset['Rush_Att_1']/self.dataset['RB_Games_1']
        self.dataset['Rush_Yds_per_Game_1'] = self.dataset['Rush_Yds_1']/self.dataset['RB_Games_1']
        self.dataset['Rush_TD_per_Game_1'] = self.dataset['Rush_TD_1']/self.dataset['RB_Games_1']
        self.dataset['Rec_per_Game_1'] = self.dataset['Rec_1']/self.dataset['RB_Games_1']
        self.dataset['Rcv_Yds_per_Game_1'] = self.dataset['Rcv_Yds_1']/self.dataset['RB_Games_1']
        self.dataset['Rcv_TD_per_Game_1'] = self.dataset['Rcv_TD_1']/self.dataset['RB_Games_1']
        self.dataset['ScrmgPlays_per_Game_1'] = self.dataset['ScrmgPlays_1']/self.dataset['RB_Games_1']
        self.dataset['Scrmg_Yds_per_Game_1'] = self.dataset['Scrmg_Yds_1']/self.dataset['RB_Games_1']
        self.dataset['Scrmg_TD_per_Game_1'] = self.dataset['Scrmg_TD_1']/self.dataset['RB_Games_1']

        #Create new column to indicate which player won the Heisman
        self.dataset['Winner_Coded'] = '999'
        for row in range(self.dataset.shape[0]):
            if self.dataset['Winner'][row] == self.dataset['QB_1'][row]:
                self.dataset.loc[row, 'Winner_Coded'] = 0.0
            if self.dataset['Winner'][row] == self.dataset['RB_1'][row]:
                self.dataset.loc[row, 'Winner_Coded'] = 1.0
        
        #Format conference fields 
        self.dataset['QB_Conf_1'] = self.dataset['QB_Conf_1'].str.strip()
        self.dataset['RB_Conf_1'] = self.dataset['RB_Conf_1'].str.strip()
        self.dataset['RB_Conf_1'] = self.dataset['RB_Conf_1'].replace({'Pac-10':'Pac-12'}) #was manually fixed for QB

        #Create dummy vars for  conferences
        pd.concat([self.dataset, pd.get_dummies(self.dataset['QB_Conf_1'],prefix='QB1_').astype(int)], axis=1)
        pd.concat([self.dataset, pd.get_dummies(self.dataset['RB_Conf_1'],prefix='RB1_').astype(int)], axis=1)

        #Drop non-standardized fields
        self.dataset.drop(columns = ['Pass_Cmp_1','Pass_Att_1','Pass_Yds_1', 'Pass_TD_1', 'Int_1', 'Rush_Att_1',
                                      'Rush_Yds_1', 'Rush_TD_1', 'Rec_1','Rcv_Yds_1','Rcv_TD_1','ScrmgPlays_1',
                                      'Scrmg_Yds_1','Scrmg_TD_1',
                                      'Winner', 'QB_Conf_1', 'RB_Conf_1'], inplace=True)
        
        #Drop variables we won't be using to predict
        self.dataset.drop(columns = ['QB_1', 'RB_1'], inplace=True)

    def load_train_data(self, url):
        self.dataset = pd.read_csv(url)
        self.preprocess_train_data()

    def make_x_y(self):
        self.X = self.dataset.drop(columns = ['Year','Winner_Coded'])
        self.y = self.dataset['Winner_Coded']  

        #Standardize X
        train_mean = self.X.mean(axis=0)
        self.X -= train_mean
        train_std = self.X.std(axis=0)
        self.X /= train_std
    
    def build_network(self):
        self.model = tf.keras.Sequential([
            layers.Dense(10, activation = 'relu'),
            layers.Dense(5, activation = 'relu'),
            layers.Dense(1, activation = 'sigmoid')
        ])
        self.model.compile(loss = 'binary_crossentropy',
                            optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1),
                            metrics = ['accuracy'])

    def leave_one_out_cross_val(self):
        self.callbacks =  [
        tf.keras.callbacks.EarlyStopping(
            monitor = 'loss', 
            patience=3
        ),
        tf.keras.callbacks.TensorBoard()
        ]

        losses = []
        accuracies = []
        for i in range(len(self.y)):
            #print(y.shape)
            y_train = np.asarray(self.y)
            y_test = np.asarray(self.y[i])
            y_train = np.delete(y_train, i)
            X_train = self.X.copy()
            X_test = pd.DataFrame(self.X.iloc[i,:]).transpose()
            X_train.drop([i], inplace=True)
            y_train = tf.convert_to_tensor(y_train, dtype='float')
            y_test = tf.convert_to_tensor(y_test, dtype='float')
            y_test = tf.reshape(y_test, [1,1])

            self.build_network()
            self.model.fit(X_train, y_train, epochs = 10, batch_size = 4, callbacks = self.callbacks)
            error = self.model.evaluate(X_test, y_test)
            losses.append(error[0])
            accuracies.append(error[1])
            print(f'Finished round {i+1}')

        print(f'Average Loss: {np.average(losses)}')
        print(f'Average Accuracy: {np.average(accuracies)}')

    def train_final_model(self):
        self.build_network()
        y_reformatted = tf.convert_to_tensor(self.y, dtype='float')
        self.model.fit(self.X, y_reformatted, epochs = 10, batch_size = 8, callbacks = self.callbacks)