from keras.callbacks import Callback

class DataPlaceholder():
    def __init__(self):
        class DataItem():
            def __init__(self):
                self.x = None
                self.y = None
        self.train = DataItem()
        self.val = DataItem()
        self.test = DataItem()
    def print_shapes(self):
        print('Train shapes: x={} , y={}'.format(self.train.x.shape,self.train.y.shape))
        print('Val shapes  : x={} , y={}'.format(self.val.x.shape,self.val.y.shape))
        print('Test shapes : x={} , y={}'.format(self.test.x.shape,self.test.y.shape))


class PredictData(Callback):
    def __init__(self,model,x,y,log_word):
        self.x = x; self.y = y; 
        self.model = model
        self.log_word = log_word

    def on_epoch_end(self,epoch,logs={}):
        logs[self.log_word+'pred'] = self.model.predict(self.x)
        logs[self.log_word+'labels'] = self.y




