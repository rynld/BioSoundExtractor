
import numpy as np

class RBM:

    def __init__(self, visible, hidden, learning_rate = 0.0001, epochs=10, mini_batch_size=10, \
                 type_visible_layer = 'B',type_hidden_layer = 'B',gibb_sampling_step= 1, sigma = 1.0, \
                 learning_type = 'CD'):

        self.visible_count = visible
        self.hidden_count = hidden
        self.lr = learning_rate
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.weights = 0.1*np.random.randn(visible,hidden)
        self.bias_vis = np.zeros((visible,1))
        self.bias_hid = np.zeros((hidden,1))
        self.momentum = 0
        self.weight_cost = 0.0002
        self.gibb_sampling_step = gibb_sampling_step
        self.type_visible_layer = type_visible_layer
        self.type_hidden_layer = type_hidden_layer
        self.sigma = sigma
        self.functions = {'B':self.bernoulli,'G':self.normal}
        self.learning_type = learning_type


    def clone(self):
        return RBM(self.visible_count,self.hidden_count,self.lr,self.epochs,self.mini_batch_size,\
                   self.type_visible_layer,self.type_hidden_layer,self.gibb_sampling_step,self.sigma)

    def fit(self,X,normalize=False):

        self.link_parameters(np.copy(X))
        X_prime = np.copy(X)
        mean = np.mean(X_prime,axis=0)
        std = np.std(X_prime,axis=0)
        std = [x if x != 0 else 1.0 for x in std]
        X_prime = (X_prime-mean)/std

        self._fit(X_prime)

    def _fit(self,X):

        # if X is not np.matrix:
        #     raise Exception('')

        row,col = np.shape(X)

        for i in range(self.epochs):

            np.random.shuffle(X)

            batch = [X[j : j + self.mini_batch_size,:] for j in range(0,row,self.mini_batch_size)]

            error = 0.0

            dw = np.zeros(np.shape(self.weights))
            dbv = np.zeros(np.shape(self.bias_vis))
            dbh = np.zeros(np.shape(self.bias_hid))

            for b in batch:

                delta_w,delta_bias_v,delta_bias_h,e = self.gradient(b)

                self.weights += self.lr*(delta_w/(self.mini_batch_size + 0.0) - self.weight_cost*self.weights)

                self.bias_vis += self.lr*(delta_bias_v/(self.mini_batch_size + 0.0))
                self.bias_hid += self.lr*(delta_bias_h/(self.mini_batch_size + 0.0))

                dw,dbv,dbh = delta_w,delta_bias_v,delta_bias_h

                error += e

            print(i,error)

    def gradient(self,minibatch):

        delta_w = np.zeros(np.shape(self.weights))
        delta_bias_v = np.zeros(np.shape(self.bias_vis))
        delta_bias_h = np.zeros(np.shape(self.bias_hid))

        error = 0.0

        for i,x in enumerate(minibatch):

            pos_vis_act = np.reshape(x,(len(x),1))

            reconst_vis_act,pos_hidden_prob,neg_vis_prob, neg_hid_prob = \
                self._cd(pos_vis_act if self.learning_type == "CD" else self.persistent_markov_chain[i])

            positive = np.dot(pos_vis_act,np.transpose(pos_hidden_prob))

            negative = np.dot(neg_vis_prob,np.transpose(neg_hid_prob))

            delta_w += positive - negative
            delta_bias_v += pos_vis_act - neg_vis_prob
            delta_bias_h += pos_hidden_prob - neg_hid_prob

            error += sum(sum((pos_vis_act - neg_vis_prob)**2))

            if self.learning_type == "PCD":
                self.persistent_markov_chain[i] = reconst_vis_act


        return delta_w,delta_bias_v,delta_bias_h, error

    def _cd(self, visible):

        pos_hidden_prob,pos_hidden_act = self.function_hidden_layer(np.dot(np.transpose(self.weights),visible) + self.bias_hid)
        neg_hidden_prob,neg_hidden_act = pos_hidden_prob,pos_hidden_act

        for k in range(self.gibb_sampling_step):

            neg_vis_prob,neg_vis_act = self.function_visible_layer(np.dot(self.weights,neg_hidden_act) + self.bias_vis)
            neg_hidden_prob,neg_hidden_act = self.function_hidden_layer(np.dot(np.transpose(self.weights),neg_vis_act) + self.bias_hid)


        return neg_vis_act,pos_hidden_prob,neg_vis_prob,neg_hidden_prob

    def link_parameters(self,X):

        self.function_visible_layer = self.functions[self.type_visible_layer]
        self.function_hidden_layer = self.functions[self.type_hidden_layer]

        if self.learning_type == "PCD":
            np.random.shuffle(X)

            self.persistent_markov_chain = [np.reshape(x,(len(x),1)) for x in X[:self.mini_batch_size]]

    def sigmoide(self,x):
        return 1.0/(1.0 + np.exp(-x))

    def bernoulli(self,x):
        x = self.sigmoide(x)
        return x,x > np.random.uniform(0,1,np.shape(x))

    def normal(self,x):
        a,b = np.shape(x)
        res = x + self.sigma*np.random.randn(a,b)
        return res,res

    def transform(self,X):

        res = None
        try:
            res,_ = self.function_hidden_layer(np.dot(X,self.weights))
        except:
            for i in range(0,np.shape(X)[0],self.mini_batch_size):
                if res is None:
                    res = self.function_hidden_layer(np.dot(X[i:i+self.mini_batch_size,],self.weights))

                else:
                    res = np.hstack((res,self.function_hidden_layer(np.dot(X[i:i+self.mini_batch_size,],self.weights))))
                    # X = X[i:i+self.mini_batch_size,]
        return res

    def _gibbs_sampling(self,x):

        neg_hidden_prob,neg_hidden_act = self.function_hidden_layer(np.dot(np.transpose(self.weights),x) + self.bias_hid)
        neg_vis_prob,neg_vis_act = self.function_visible_layer(np.dot(self.weights,neg_hidden_act) + self.bias_vis)
        return neg_vis_act

    def generate_samples(self,x,samples = 99):

        burn_in = 100
        x = np.reshape(x,(len(x),1))

        res = x

        for i in range(burn_in):
            x = self._gibbs_sampling(x)

        for i in range(samples):
            x = self._gibbs_sampling(x)
            np.hstack((res,x))

        print(np.shape(res))
        return res









