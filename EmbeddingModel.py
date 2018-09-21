'''
    Embedding Model class

     - Main variables:
        self.model - keras model
        self.num_classes - number of outputs/classes
        self.class_weights - class weights got from get_class_weights()
                             used in weighted_cross_entropy
                             (if using imblearn this becomes simple cross_entropy)
        self.num_sites - number of sites from X w/ shape (num_examples, num_error, num_sites)
        self.num_error - number of errors from X w/ shape (num_examples, num_error, num_sites)
        self.num_embed - number of created embeddings for each site or error
        self.embed_epochs - number of epochs to train embeddings
        self.add_attention - if add [0,1] outputs for each site and error on which 
                             model attention was mostly focused 
        self.sites_attention_model - model that returns sites attention. returns shape (num_sites,)
        self.error_attention_model - model that returns error attention. returns shape (num_sites,)
        self.model_params - self.create_model() parameters, element names must be the same !
                             
     - Main functions:
        self.create_model - creates self.model
        self.train - trains self.model w/ selected parameters
        self.predict - predicts Y from X with self.model
        self.change_inputs - function that changes inputs before feeding to model
        self.load_model - loads model.h5 file from directory (dirpath)
        self.save_model - saves model.h5 and model.json to directory (dirpath)
        self.pretraining - creates embeddings
'''

import keras
from keras.layers import Input, Embedding, dot, Flatten, Dense, Dropout, Concatenate, Reshape, multiply
import numpy as np
from skopt.space import Real, Categorical, Integer

from utils.BaseModel import Model as BaseModel
from utils.losses import weighted_categorical_crossentropy
from utils.model_utils import get_class_weights


class EmbeddingModel(BaseModel):
    def __init__( self, X, num_classes, num_sites=142, num_error=54, num_embed=20, 
                  embedding_training_epochs=180, add_attention=False ):
        '''
        @param X w/ shape (num_examples, num_error, num_sites)
        @param num_classes - number of outputs/classes
        @param num_error - number of errors from X w/ shape (num_examples, num_error, num_sites)
        @param num_sites - number of sites from X w/ shape (num_examples, num_error, num_sites)
        @param num_embed - number of created embeddings for each site or error
        @param embedding_training_epochs - number of epochs to train self.pretraining()
        @param add_attention - if add [0,1] outputs for each site and error on which 
                               model attention was mostly focused 
        '''
        self.num_sites = num_sites
        self.num_error = num_error
        self.num_embed = num_embed
        self.num_classes = num_classes
        self.embed_epochs = embedding_training_epochs
        self.add_attention = add_attention
        self.pretraining(X)
        self.model_params = {
            'dense_layers':3,
            'dense_units':50,
            'dropout_value':0.2,
            'learning_rate':1e-3,
        }
        

    def create_model(self, dense_layers, dense_units, dropout_value, learning_rate):
        ''' creates feed forward neral network w/ inputs got from self.change_inputs():
            if self.add_attention == True:
                creates sites attention and error attention models
                (self.sites_attention_model, self.error_attention_model)
        '''
        m_input = Input( (self.num_sites + self.num_error, self.num_embed) )
        m = m_input

        if self.add_attention:
            m_a = Flatten()(m)
            m_a_sites = Dense(units=self.num_sites, activation='softmax', name='sites_attention')(m_a)
            m_a_error = Dense(units=self.num_error, activation='softmax', name='error_attention')(m_a)
            m_a = Concatenate(axis=1)([m_a_sites, m_a_error])
            m_a = Reshape([self.num_sites + self.num_error,1])(m_a)
            m = multiply([m, m_a])

        m = Flatten()(m)
        for _ in range(dense_layers):
            m = Dense(units=dense_units, activation='relu')(m)
            m = Dropout(dropout_value)(m)
        m_output = Dense(self.num_classes, activation='softmax')(m)
        
        self.model = keras.models.Model(inputs=m_input, outputs=m_output)
        self.model.compile( loss = weighted_categorical_crossentropy(self.class_weights),
                            optimizer = keras.optimizers.Adam(lr=learning_rate) )

        if self.add_attention:
            self.sites_attention_model = keras.models.Model(
                inputs = self.model.input,
                outputs = self.model.get_layer(name='sites_attention').output
            )
            self.error_attention_model = keras.models.Model(
                inputs = self.model.input,
                outputs = self.model.get_layer(name='error_attention').output
            )
        

    def pretraining(self, X):
        ''' creates error and sites embeddings from matrix* (num_error, num_sites):
                - self.error_embedding w/ shape (num_error, num_embed)
                - self.sites_embedding w/ shape (num_error, num_embed)
            * this matrix is from X w/ shape (num_examples, num_error, num_sites)
              where everything is summed across 1st axis to (num_error, num_sites)
              and all numbers that are greater then 0 are replaced w/ 1
        @param X w/ shape (num_examples, num_error, num_sites)
        '''
        sites_input = Input((self.num_sites,), name='sites_input') # (batch,142,1)
        sites_embed = Embedding(self.num_sites, self.num_embed, name='sites_embed')(sites_input) # (batch,142,5)
        error_input = Input((self.num_error,), name='error_input') # (batch,54,1)
        error_embed = Embedding(self.num_error, self.num_embed, name='error_embed')(error_input) # (batch,54,5)
        modl_output = dot(inputs=[error_embed, sites_embed], axes=2) # (batch,54,142)
        model = keras.models.Model([error_input, sites_input], modl_output)
        model.compile( loss='mse', optimizer = keras.optimizers.Adam(lr=1e-2) )

        a = np.arange(self.num_error).reshape((1,self.num_error))
        b = np.arange(self.num_sites).reshape((1,self.num_sites))

        targets = np.expand_dims( np.sum(X, axis=0), axis=0)
        targets[ targets > 0 ] = 1

        model.fit(x=[a,b], y=targets, epochs=self.embed_epochs, verbose=0)

        self.error_embedding = model.get_layer(name='error_embed').get_weights()[0] # (num_error,num_embed)
        self.sites_embedding = model.get_layer(name='sites_embed').get_weights()[0] # (num_sites,num_embed)


    def change_inputs(self, X, Y=None):
        ''' create input from X w/ shape (num_examples, num_error, num_sites)
            to X w/ (num_examples, num_error + num_sites, num_embed) where:
                1) each error index is converted to (num_embed,) vector from self.pretraining()
                   and is multiplied by sum of this error at all sites
                   (this happens at each example separately)
                2) each site index is converted to (num_embed,) vector from self.pretraining()
                   and is multiplied by sum of all errors at this site
                   (this happens at each example separately)
                3) matrix from 1) w/ shape (num_examples, num_error, num_embed)
                   and 2) w/ shape (num_examples, num_sites, num_embed)
                   are concatinated to shape (num_examples, num_error + num_sites, num_embed)
        @param X w/ shape (num_examples, num_error, num_sites)
        @param Y w/ shape (num_examples,)
        return (num_examples, num_error + num_sites, num_embed)
        '''
        X_error = []; X_sites = []
        for x in range(len(X)):
            e_arr = []
            for i in range(self.num_error): # 54
                e_sum = np.sum( X[x,i,:] )
                e_lat = self.error_embedding[i]
                e = e_lat * e_sum
                e_arr.append(e)
            s_arr = []
            for j in range(self.num_sites): # 142   
                s_sum = np.sum( X[x,:,j] )
                s_lat = self.sites_embedding[j]
                s = s_lat * s_sum
                s_arr.append(s)
            X_error.append(np.array(e_arr))
            X_sites.append(np.array(s_arr))
        X_sites = np.array(X_sites)
        X_error = np.array(X_error)
        X_new = np.concatenate((X_sites, X_error),axis=1)
        if Y is not None:
            Y = keras.utils.to_categorical(Y, num_classes=self.num_classes)
            return X_new, Y
        else:
            return X_new


    def predict(self, X, argmax=True):
        '''
        @param X w/ shape (num_examples, num_error, num_sites)
        return y_argmax np.array w/ shape (num_examples,), where each number represents class index
            or
            if self.add_attention == True:
        return y_argmax np.array w/ shape (num_examples,) and
               sites_att_output np.array w/ shape (num_examples, num_sites), where each 
                                number [0,1] represents how much attention was used on that site
               error_att_output np.array w/ shape (num_examples, num_error), where each 
                                number [0,1] represents how much attention was used on that error
        '''
        X = self.change_inputs(X)
        y_pred = self.model.predict(X) # (num_examples, num_outputs)
        if argmax:
            y_pred = np.argmax(y_pred, axis=-1) # (num_examples,)
        if self.add_attention:
            sites_att_output = self.sites_attention_model.predict(X) # (num_examples, 142)
            error_att_output = self.error_attention_model.predict(X) # (num_examples, 54)
            return y_pred, sites_att_output, error_att_output
        else:
            return y_pred


    def set_skopt_dimensions(self):
        ''' initializes self.dimensions list
            !!! order of elements must be the same as self.create_model() params !!!
            !!! name fields must be the same as keys in self.model_params dict   !!!
        '''
        self.dimensions = [
            Integer(     low=1,    high=15,                        name='dense_layers'      ),
            Integer(     low=5,    high=75,                        name='dense_units'       ),
            Real(        low=0.01, high=0.5,                       name='dropout_value'     ),
            Real(        low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate'     )
        ]




#