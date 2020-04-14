from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
from scipy.interpolate import griddata
import time
import tensorflow as tf
np.random.seed(1234)
tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x, y, t, u, v, layers, save_dir):
        print(x.shape, y.shape)
        X = np.concatenate([x, y, t], 1)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.lb = X.min(0)
        self.ub = X.max(0)
        self.arecord=[]
        self.brecord=[]
        self.crecord=[]
        self.drecord=[]
        self.iter_record=[]
        self.X = X

        self.x = X[:, 0:1]
        self.y = X[:, 1:2]
        self.t = X[:, 2:3]

        self.u = u
        self.v = v

        self.layers = layers

        # Initialize NN
        #self.weights, self.biases = self.initialize_NN(layers)

        # Initialize parameters
        # self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        # self.lambda_2 = tf.Variable([0.0], dtype=tf.float32)
        #self.lambda_a = tf.Variable([1.0], dtype=tf.float32,trainable=False)
        self.lambda_b= tf.Variable([1.0], dtype=tf.float32)
        #self.lambda_c= tf.Variable([2.0], dtype=tf.float32)
        self.lambda_d= tf.Variable([1.0], dtype=tf.float32)
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])

        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]])

        self.pred=self.model(tf.concat([self.x_tf, self.y_tf,self.t_tf],1))
        self.u_pred, self.v_pred, self.p_pred, self.f_u_pred, self.f_v_pred = self.net_NS(self.x_tf, self.y_tf,self.t_tf,self.pred)

        self.loss = tf.reduce_sum(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_sum(tf.square(self.v_tf - self.v_pred)) + \
                    tf.reduce_sum(tf.square(self.f_u_pred)) + \
                    tf.reduce_sum(tf.square(self.f_v_pred))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        self.Saver = tf.train.Saver(max_to_keep=100)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def model(self, x_train, is_train=True, checkpoint_dir=None):
        x_train = 2.0 * (x_train - self.lb) / (self.ub - self.lb) - 1.0
        def normalize(train, test):
            mean, std = train.mean(), test.std()
            train = (train - mean) / std
            test = (test - mean) / std
            return train, test

        # def xavier_init(self, size):
        #     in_dim = size[0]
        #     out_dim = size[1]
        #     xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        #     return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

        # def bn_layer(x,is_train,moving_decay=0.9,eps=1e-5):
        #     shape=np.shape(x)
        #     pshape=shape[-1]
        #     with tf.name_scope('bn_layer'):
        #         gamma=tf.Variable(tf.ones(pshape,dtype=tf.float32),name='gamma')
        #         beta=tf.Variable(tf.zeros(pshape,dtype=tf.float32),name='beta')
        #         axes = list(range(len(shape) - 1))
        #         batch_mean,batch_var=tf.nn.moments(x,axes,name='moment')
        #         ema=tf.train.ExponentialMovingAverage(moving_decay)
        #         def mean_var_update():
        #             ema_op=ema.apply([batch_mean,batch_var])
        #             with tf.control_dependencies([ema_op]):
        #                 return tf.identity(batch_mean),tf.identity(batch_var)
        #         mean,var=tf.cond(tf.equal(is_train,True),lambda:mean_var_update(),lambda:(ema.average(batch_mean),ema.average(batch_var)))
        #         return tf.nn.batch_normalization(x,mean,var,beta,gamma,eps)

        def add_layer(inputs, in_size, out_size, is_train=True, bn_need=True, activation_function=None):
            w = tf.Variable(tf.truncated_normal(mean=0, stddev=1, shape=[in_size, out_size]))
            b = tf.Variable(tf.random_normal([1, out_size]))
            wx_plus_b = tf.matmul(tf.cast(inputs, tf.float32), w) + b
            # if bn_need==True:
            #    wx_plus_b=bn_layer(wx_plus_b,is_train)
            # tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(0.005)(w))
            if activation_function == None:
                return wx_plus_b
            else:
                return activation_function(wx_plus_b)

        with tf.name_scope('layer_1'):
            layer_1 = add_layer(x_train, 3, 20, is_train, bn_need=False, activation_function=tf.nn.tanh)
        with tf.name_scope('layer_2'):
            layer_2 = add_layer(layer_1, 20, 20, is_train, bn_need=False, activation_function=tf.nn.tanh)
        with tf.name_scope('layer_3'):
            layer_3 = add_layer(layer_2, 20, 20, is_train, bn_need=False, activation_function=tf.nn.tanh)
        with tf.name_scope('layer_4'):
            layer_4 = add_layer(layer_3, 20, 20, is_train, bn_need=False, activation_function=tf.nn.tanh)
        with tf.name_scope('layer_5'):
            layer_5 = add_layer(layer_4, 20, 20, is_train, bn_need=False, activation_function=tf.nn.tanh)
        with tf.name_scope('output'):
            outputs = add_layer(layer_5, 20, 2, is_train, activation_function=None)
        return outputs
    # def initialize_NN(self, layers):
    #     weights = []
    #     biases = []
    #     num_layers = len(layers)
    #     for l in range(0, num_layers - 1):
    #         W = self.xavier_init(size=[layers[l], layers[l + 1]])
    #         b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
    #         weights.append(W)
    #         biases.append(b)
    #     return weights, biases
    #
    # def xavier_init(self, size):
    #     in_dim = size[0]
    #     out_dim = size[1]
    #     xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
    #     return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    #
    # def neural_net(self, X, weights, biases):
    #     num_layers = len(weights) + 1
    #
    #     H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
    #     for l in range(0, num_layers - 2):
    #         W = weights[l]
    #         b = biases[l]
    #         H = tf.tanh(tf.add(tf.matmul(H, W), b))
    #     W = weights[-1]
    #     b = biases[-1]
    #     Y = tf.add(tf.matmul(H, W), b)
    #     return Y

    def net_NS(self, x, y,t, pred):
        #lambda_a = self.lambda_a
        lambda_b = self.lambda_b
        #lambda_c = self.lambda_c
        lambda_d = self.lambda_d
        psi_and_p=pred
        #psi_and_p = self.model(tf.concat([x, y, t], 1), self.weights, self.biases)
        print(psi_and_p)
        psi = psi_and_p[:, 0:1]
        p = psi_and_p[:, 1:2]

        u = tf.gradients(psi, y)[0]
        v = -tf.gradients(psi, x)[0]

        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]

        v_t = tf.gradients(v, t)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]

        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]
        f_u = u_t + lambda_b * (u * u_x + v * u_y) + p_x - lambda_d * (u_xx + u_yy)
        f_v = v_t + lambda_b * (u * v_x + v * v_y) + p_y - lambda_d * (v_xx + v_yy)
        #f_u = lambda_a*u_t + lambda_b * (u * u_x + v * u_y) + lambda_c*p_x - lambda_d * (u_xx + u_yy)
        #f_v = lambda_a*v_t + lambda_b * (u * v_x + v * v_y) + lambda_c*p_y - lambda_d * (v_xx + v_yy)

        return u, v, p, f_u, f_v

    # def callback(self, loss, lambda_1, lambda_2):
    #    print('Loss: %.3e, l1: %.3f, l2: %.5f' % (loss, lambda_1, lambda_2))

    def train(self, nIter):
        ckpt = tf.train.get_checkpoint_state(self.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # 加载模型和训练好的参
            self.Saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("加载模型成功：" + ckpt.model_checkpoint_path)

            # 通过文件名得到模型保存时迭代的轮数.格式：model.ckpt-6000.data-00000-of-00001
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

        else:
            tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t,
                   self.u_tf: self.u, self.v_tf: self.v}

            start_time = time.time()
            for it in range(nIter):
                self.sess.run(self.train_op_Adam, tf_dict)
                if it % 1000 == 0:
                    print(it, self.save_dir)
                    self.Saver.save(self.sess, os.path.join(self.save_dir, 'ns_model'), global_step=it)
                # Print
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    #lambda_a_value = self.sess.run(self.lambda_a)
                    lambda_b_value = self.sess.run(self.lambda_b)
                    #lambda_c_value = self.sess.run(self.lambda_c)
                    lambda_d_value = self.sess.run(self.lambda_d)
                    #self.arecord.append(lambda_a_value)
                    self.brecord.append(lambda_b_value)
                    #self.crecord.append(lambda_c_value)
                    self.drecord.append(lambda_d_value)
                    self.iter_record.append(it)
                    #print('It: %d, Loss: %.3e, la: %.3f, lb: %.5f,lc: %.3f, ld: %.5f, Time: %.2f' %
                    #      (it, loss_value, lambda_a_value, lambda_b_value,lambda_c_value , lambda_d_value, elapsed))
                    print('It: %d, Loss: %.3e,  lb: %.5f, ld: %.5f, Time: %.2f' %
                          (it, loss_value,  lambda_b_value, lambda_d_value, elapsed))
                    start_time = time.time()

        # self.optimizer.minimize(self.sess,
        #                         feed_dict = tf_dict,
        #                         fetches = [self.loss, self.lambda_1, self.lambda_2],
        #                         loss_callback = self.callback)
        #self.optimizer.minimize(self.sess,feed_dict=tf_dict,fetches=[self.loss, self.lambda_1, self.lambda_2])

    def predict(self, x_star, y_star, t_star):

        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star}

        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)

        return u_star, v_star, p_star




def save_fig(grid_x,grid_y,y_predict,i):
    plt.figure(figsize=(15, 4))
    plt.pcolormesh(grid_x,grid_y,y_predict,cmap=plt.cm.coolwarm,alpha=1,antialiased=True)
    #plt.clim(-0.12,0.02)
    plt.colorbar()
    plt.title('p_predict_iter_{}'.format(str(i)))
    plt.xlabel('x/h')
    plt.ylabel('y/h')
    plt.legend()
    plt.savefig("./prediction_{}.jpg".format(str(i)))
    plt.show()
if __name__=='__main__':
    N_train = 5000

    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]

    # Load Data
    data = scipy.io.loadmat('../cylinder_nektar_wake.mat')

    U_star = data['U_star']  # N x 2 x T
    P_star = data['p_star']  # N x T
    t_star = data['t']  # T x 1
    print('t',len(t_star))
    X_star = data['X_star']  # N x 2

    N = X_star.shape[0]
    T = t_star.shape[0]
    print("N,T",N,T)
    #sys.exit(0)
    # Rearrange Data
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T

    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    PP = P_star  # N x T

    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    t = TT.flatten()[:, None]  # NT x 1

    u = UU.flatten()[:, None]  # NT x 1
    v = VV.flatten()[:, None]  # NT x 1
    p = PP.flatten()[:, None]  # NT x 1

    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    # Training Data
    idx = np.random.choice(N * T, N_train, replace=False)
    x_train = x[idx, :]
    y_train = y[idx, :]
    t_train = t[idx, :]
    u_train = u[idx, :]
    v_train = v[idx, :]
    #save_dir = ('../models_abandon_1_halft2/')
    save_dir = ('../check/')
    #save_noisy_dir = ('./models_abandon_noise_1/')
    # Training

    model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers, save_dir)
    model.train(200000)
    #model.train(200000)
    # Test Data
    '''
    plt.figure(1)
    plt.xlabel('iterarion')
    #plt.plot(model.iter_record,model.arecord,'r',label='a')
    #plt.plot(model.iter_record,model.brecord,'g',label='b')
    #plt.plot(model.iter_record,model.crecord,'b',label='c')
    plt.plot(model.iter_record,model.drecord,'y',label='d')
    plt.legend()
    plt.show()
    #sys.exit(0)
    '''
    for i in range(0,200):
        snap = np.array([i])
        x_star = X_star[:, 0:1]
        y_star = X_star[:, 1:2]
        t_star = TT[:, snap]

        u_star = U_star[:, 0, snap]
        v_star = U_star[:, 1, snap]
        p_star = P_star[:, snap]

        # Prediction

        u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)
        #p_pred=u_pred
        #p_star=u_star
        #lambda_1_value = model.sess.run(model.lambda_1)
        #lambda_2_value = model.sess.run(model.lambda_2)
        # Error
        '''
        error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
        error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
        error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)
        
        #error_lambda_1 = np.abs(lambda_1_value - 1.0) * 100
        #error_lambda_2 = np.abs(lambda_2_value - 0.01) / 0.01 * 100
        
        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))
        print('Error p: %e' % (error_p))
        #print('Error l1: %.5f%%' % (error_lambda_1))
        #print('Error l2: %.5f%%' % (error_lambda_2))
        '''
        # Plot Results
        #    plot_solution(X_star, u_pred, 1)
        #    plot_solution(X_star, v_pred, 2)
        #    plot_solution(X_star, p_pred, 3)
        #    plot_solution(X_star, p_star, 4)
        #    plot_solution(X_star, p_star - p_pred, 5)

        # Predict for plotting
        lb = X_star.min(0)
        ub = X_star.max(0)
        nn = 200
        x = np.linspace(lb[0], ub[0], nn)
        y = np.linspace(lb[1], ub[1], nn)
        X, Y = np.meshgrid(x, y)
        #p_pred=p_star
        #UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
        #VV_star = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')
        PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method='cubic')
        P_exact = griddata(X_star, p_star.flatten(), (X, Y), method='cubic')
        fig=plt.figure()
        ######## Row 2: Pressure #######################
        ########      Predicted p(t,x,y)     ###########

        ax = plt.subplot(1,2,1)
        h = ax.imshow(PP_star, interpolation='nearest', cmap='rainbow',
                      extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
                      origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        fig.colorbar(h, cax=cax)
        plt.xlabel('x')
        plt.ylabel('y')
        ax.set_aspect('equal', 'box')
        #ax.set_title('Predicted pressure', fontsize=10)
        ax.set_title('predicted P time{}'.format(i), fontsize=10)

        ########     Exact p(t,x,y)     ###########
        ax2 = plt.subplot(1,2,2)
        h = ax2.imshow(P_exact, interpolation='nearest', cmap='rainbow',
                      extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
                      origin='lower', aspect='auto')
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        fig.colorbar(h, cax=cax)
        plt.xlabel('x')
        plt.ylabel('y')
        ax2.set_aspect('equal', 'box')
        ax2.set_title('Exact P time{}'.format(i), fontsize=10)
        plt.savefig("./check_P/{}.jpg".format(str(i)))
        #plt.savefig("./pic_half2/{}.jpg".format(str(i)))
        #ax.axis('off')
        #plt.show()