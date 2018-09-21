'''
    Simple NN model class

     - Main variables:
        self.model - keras model
        self.num_classes - number of outputs/classes
        self.class_weights - class weights got from get_class_weights()
                             used in weighted_cross_entropy
                             (if using imblearn this becomes simple cross_entropy)
        self.num_sites - number of sites from X w/ shape (num_examples, num_error, num_sites)
        self.num_error - number of errors from X w/ shape (num_examples, num_error, num_sites)
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
'''

import keras
from keras.layers import Input, Flatten, Dense, Dropout, Reshape, multiply
import numpy as np
from skopt.space import Real, Categorical, Integer

from utils.BaseModel import Model as BaseModel
from utils.losses import weighted_categorical_crossentropy
from utils.model_utils import get_class_weights


class SimpleModel(BaseModel):
    def __init__(self, num_classes, num_error=54, num_sites=142, add_attention=False):
        '''
        @param num_classes - number of outputs/classes
        @param num_error - number of errors from X w/ shape (num_examples, num_error, num_sites)
        @param num_sites - number of sites from X w/ shape (num_examples, num_error, num_sites)
        '''
        self.num_error = num_error
        self.num_sites = num_sites
        self.num_classes = num_classes
        self.add_attention = add_attention
        self.model_params = {
            'dense_layers':3,
            'dense_units':50,
            'regulizer_value':0.0015,
            'dropout_value':0.015,
            'learning_rate':1e-3,
        }


    def create_model( self, dense_layers, dense_units, regulizer_value, 
                      dropout_value, learning_rate ):
        ''' creates feed-forward neural network, where X matrix w/ shape 
            (num_examples, num_error, num_sites) is flatten to shape 
            (num_examples, num_error * num_sites) and used as an input
        '''
        m_input = Input((self.num_error,self.num_sites))
        m = m_input

        if self.add_attention:
            m_flatten = Flatten()(m)
            m_sites_a = Dense(self.num_sites, activation='softmax', name='sites_attention')(m_flatten)
            m_sites_a = Reshape([1, self.num_sites])(m_sites_a)
            m_error_a = Dense(self.num_error, activation='softmax', name='error_attention')(m_flatten)
            m_error_a = Reshape([self.num_error, 1])(m_error_a)
            m_attention = multiply([m_error_a, m_sites_a]) # (num_examples, num_error, num_sites)
            # multiple attention w/ input matrix
            m = multiply([m, m_attention])

        m = Flatten()(m)
        for _ in range(dense_layers):
            m = Dense( units=dense_units, activation='relu', 
                       kernel_initializer='lecun_normal',
                       kernel_regularizer=keras.regularizers.l2(regulizer_value) )(m)
            m = Dropout(dropout_value)(m)

        m_output = Dense( units=self.num_classes, activation='softmax', 
                          kernel_initializer='lecun_normal',
                          kernel_regularizer=keras.regularizers.l2(regulizer_value) )(m)
        
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


    def predict(self, X, argmax=True):
            '''
            @param X w/ shape (num_examples, num_errors, num_sites)
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
            Real(        low=1e-3, high=0.9,  prior="log-uniform", name='regulizer_value'   ),
            Real(        low=0.01, high=0.5,                       name='dropout_value'     ),
            Real(        low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate'     )
        ]




#