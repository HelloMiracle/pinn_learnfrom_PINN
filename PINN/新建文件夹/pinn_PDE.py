import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class periodic_pinn:
    def __init__(self,x_train,iter,save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.lossrecord=[]
        self.l1record=[]
        self.l2record=[]
        self.iterrecord=[]
        self.x=x_train[:,0].reshape(-1,1)
        self.y=x_train[:,1].reshape(-1,1)
        self.u=x_train[:,2].reshape(-1,1)
        self.v=x_train[:,3].reshape(-1,1)
        self.rsxx=x_train[:,4].reshape(-1,1)
        print(self.rsxx)
        self.rsxy = x_train[:, 5].reshape(-1, 1)
        self.rsyy = x_train[:, 6].reshape(-1, 1)
        X=x_train[:,0:2]
        self.lb = X.min(0)
        self.ub = X.max(0)
        self.iter=iter
        self.lambda_1 = tf.Variable([1.0], dtype=tf.float32,trainable=False)
        self.lambda_2 = tf.Variable([0.1], dtype=tf.float32,trainable=True)
        print(self.x.shape)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
        self.is_train=tf.placeholder(tf.bool)
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]])
        self.rsxx_tf = tf.placeholder(tf.float32, shape=[None, self.rsxx.shape[1]])
        self.rsxy_tf = tf.placeholder(tf.float32, shape=[None, self.rsxy.shape[1]])
        self.rsyy_tf = tf.placeholder(tf.float32, shape=[None, self.rsyy.shape[1]])
        self.pred = self.model(tf.concat([self.x_tf, self.y_tf], 1))
        self.u_pred, self.v_pred, self.p_pred, self.f_u_pred, self.f_v_pred = self.add_pde(self.x_tf, self.y_tf,self.rsxx_tf,self.rsxy_tf,self.rsyy_tf,self.pred)

       #with tf.variable_scope('loss'):
        self.loss = tf.reduce_sum(tf.square(self.u_tf - self.u_pred)) +tf.reduce_sum(tf.square(self.v_tf - self.v_pred)) + \
                    tf.reduce_sum(tf.square(self.f_u_pred)) + tf.reduce_sum(tf.square(self.f_v_pred))
            #tf.add_to_collection('losses', self.loss)
            #self.final_loss = tf.add_n(tf.get_collection('losses'))
            #tf.summary.scalar('final_loss',self.loss)
            #tf.summary.scalar('loss',self.loss)
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,method='L-BFGS-B', options={'maxiter': 50000, 'maxfun': 50000,
                                                                         'maxcor': 50, 'maxls': 50, 'ftol': 1.0 * np.finfo(float).eps})
        self.train_op=tf.train.AdamOptimizer().minimize(self.loss)
        #self.train_op=tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(self.loss)
        self.merge=tf.summary.merge_all()
        self.Saver = tf.train.Saver(max_to_keep=100)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def add_pde(self,x,y,rsxx,rsxy,rsyy,pred):
        psi_and_p = pred
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        psi = psi_and_p[:, 0:1]
        p = psi_and_p[:, 1:2]
        u = tf.gradients(psi, y)[0]
        v = -tf.gradients(psi, x)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]
        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]
        rsxx_x=tf.gradients(rsxx, x)[0]
        rsxy_x = tf.gradients(rsxy, x)[0]
        rsxy_y = tf.gradients(rsxy, y)[0]
        rsyy_y=tf.gradients(rsyy, y)[0]
        f_u = (lambda_1) * (u * u_x + v * u_y) + p_x - (lambda_2/10) * (u_xx + u_yy)
        f_v = (lambda_1) * (u * v_x + v * v_y) + p_y - (lambda_2/10) * (v_xx + v_yy)
        return u, v, p, f_u, f_v

    def model(self,x,is_train=True, checkpoint_dir=None):
        x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
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

        def add_layer(inputs,in_size,out_size,is_train=True,bn_need=True,activation_function=None):
            w=tf.Variable(tf.truncated_normal(mean=0,stddev=1,shape=[in_size,out_size]))
            b=tf.Variable(tf.random_normal([1,out_size]))
            wx_plus_b=tf.matmul(tf.cast(inputs,tf.float32),w)+b
            #if bn_need==True:
            #    wx_plus_b=bn_layer(wx_plus_b,is_train)
            #tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(0.005)(w))
            if activation_function==None:
                return wx_plus_b
            else:
                return activation_function(wx_plus_b)

        with tf.name_scope('layer_1'):
            layer_1=add_layer(x,2,30,is_train,bn_need=False,activation_function=tf.nn.tanh)
        with tf.name_scope('layer_2'):
            layer_2=add_layer(layer_1,30,30,is_train,bn_need=False,activation_function=tf.nn.tanh)
        with tf.name_scope('layer_3'):
            layer_3=add_layer(layer_2,30,30,is_train,bn_need=False,activation_function=tf.nn.tanh)
        with tf.name_scope('layer_4'):
            layer_4 = add_layer(layer_3, 30, 30, is_train, bn_need=False, activation_function=tf.nn.tanh)
        with tf.name_scope('layer_5'):
            layer_5 = add_layer(layer_4, 30, 30, is_train, bn_need=False, activation_function=tf.nn.tanh)
        with tf.name_scope('output'):
            outputs=add_layer(layer_5,30,2,is_train,activation_function=None)

        # with tf.name_scope('layer_lin_1'):
        #     layer_1_lin=add_layer(input,3,50,bn_need=False,activation_function=None)
        # #layer_2_lin=add_layer(layer_1_lin,100,100,bn_need=False,activation_function=None)
        # with tf.name_scope('output_lin'):
        #     outputs_lin=add_layer(layer_1_lin,50,1,bn_need=False,activation_function=None)
        # with tf.name_scope('combine'):
        #     #predict=outputs
        #     predict=(1-a)*outputs+a*outputs_lin
        return outputs

    # def train(self,x_train,x_test,i):
    #     #     #input_quenu=tf.train.slice_input_producer([x_train],shuffle=True,seed=1)
    #     #     #x_batch_train,y_batch_train=tf.train.shuffle_batch(input_quenu,batch_size=256,capacity=512,num_threads=1,min_after_dequeue=32)
    #     #     #coord = tf.train.Coordinator()
    #     #     #threads = tf.train.start_queue_runners(self.sess, coord)
    #     #     train_writer=tf.summary.FileWriter('./pinns/pinn_train',self.sess.graph)
    #     #     test_writer = tf.summary.FileWriter('/pinns/pinn_test')
    #     #     tf_dict={self.x_tf:self.x,self.y_tf:self.y,self.u_tf:self.u,self.v_tf:self.v,self.is_train:tf.bool(True)}
    #     #     if self.is_train:
    #     #         for i in range(i):
    #     #             #batch_x,batch_y=self.sess.run([x_batch_train,y_batch_train])
    #     #             _,summary=self.sess.run([self.train_op,self.merge],feed_dict=tf_dict)
    #     #             #_,summary=sess.run([train_op,merge],feed_dict={input:x_train,real:y_train,is_train:tf.cast(True,tf.bool).eval()})
    #     #             train_writer.add_summary(summary,i)
    #     #             if i%50==0:
    #     #                 #pre_loss=sess.run(self.loss,feed_dict={input:x_test,is_train:tf.cast(False,tf.bool).eval()})
    #     #                 train_loss = self.sess.run(self.loss, feed_dict=tf_dict)
    #     #                 print("iter:{}--train loss:{}".format(i,train_loss))
    #     #             if (i+1)%100==0:
    #     #                 self.Saver.save(self.sess,r'C:\Users\mike\PycharmProjects\pinn\pinns\model\model.cpkt',global_step=i+1)
    #     #     else:
    #     #         ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
    #     #         if ckpt and ckpt.model_checkpoint_path:
    #     #             print(ckpt)
    #     #             self.Saver.restore(self.sess,ckpt.model_checkpoint_path)
    #     #     # coord.request_stop()
    #     #     # # 等待线程终止
    #     #     # coord.join(threads)
    def predict(self,XP,YP):
        tf_dict = {self.x_tf:XP, self.y_tf:YP}
        y_predict=self.sess.run(self.p_pred,feed_dict=tf_dict)
        #y_predict=pd.DataFrame({'predict':pd.Series(y_predict.T[0])},index=list(range(0,len(y_predict))))
        print(y_predict)
        return y_predict

    def train(self, nIter):

        tf_dict = {self.x_tf: self.x, self.y_tf: self.y,self.u_tf: self.u, self.v_tf: self.v,self.rsxx_tf:self.rsxx,self.rsxy_tf:self.rsxy,self.rsyy_tf:self.rsyy}
        print(tf_dict)
        print(tf_dict)
        for it in range(nIter):
            self.sess.run(self.train_op, tf_dict)
            if it % 100 == 0:
                #print(it, self.save_dir)
                self.Saver.save(self.sess, os.path.join(self.save_dir, 'qiu_ns_model'), global_step=it)
            # Print
            #if it/20==1:
            #    print(self.sess.run(self.f_u_pred,feed_dict=tf_dict))
            if it % 20 == 0:
                loss_value = self.sess.run(self.loss, feed_dict=tf_dict)
                lambda_1_value = self.sess.run(self.lambda_1)
                lambda_2_value = self.sess.run(self.lambda_2)
                self.lossrecord.append(loss_value)
                #self.l1record.append(lambda_1_value)
                self.l2record.append(lambda_2_value)
                self.iterrecord.append(it)
                print('It: %d, Loss: %.3e, l1: %.3f, l2: %.5f' %(it, loss_value, lambda_1_value, lambda_2_value))
            #print('d0123')
        #self.optimizer.minimize(self.sess,
                                #feed_dict=tf_dict,
                                #fetches=[self.loss, self.lambda_1, self.lambda_2])
        return
def transform_shape(x):
    print(x)
    #y=np.array(x['predict'])
    #y=y.reshape(127,159)
    y=x.reshape(117,149)
    return y

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

if __name__=="__main__":
    data = pd.read_csv(r'C:\Users\mike\PycharmProjects\pinn\qiuling\data_u_p_dns.csv', index_col=[0])
    grid_x = np.loadtxt(r'C:\Users\mike\PycharmProjects\pinn\qiuling\Grid_X')
    grid_y = np.loadtxt(r'C:\Users\mike\PycharmProjects\pinn\qiuling\Grid_Y')
    grid_u = np.loadtxt(r'C:\Users\mike\PycharmProjects\pinn\qiuling\U_DNS_X')
    grid_v = np.loadtxt(r'C:\Users\mike\PycharmProjects\pinn\qiuling\U_DNS_Y')
    grid_p = np.loadtxt(r'C:\Users\mike\PycharmProjects\pinn\qiuling\p_DNS')
    rs_xx = np.loadtxt(r'C:\Users\mike\PycharmProjects\pinn\qiuling\RS_DNS_XX')
    rs_xy = np.loadtxt(r'C:\Users\mike\PycharmProjects\pinn\qiuling\RS_DNS_XY')
    rs_yy = np.loadtxt(r'C:\Users\mike\PycharmProjects\pinn\qiuling\RS_DNS_YY')
    #remove boundary 5points
    x=grid_x[5:122,5:154]
    y = grid_y[5:122, 5:154]
    u = grid_u[5:122, 5:154]
    v = grid_v[5:122, 5:154]
    p = grid_p[5:122, 5:154]
    rs_xx=rs_xx[5:122, 5:154]
    rs_xy = rs_xy[5:122, 5:154]
    rs_yy = rs_yy[5:122, 5:154]

    x_ = x.reshape(-1, 1)
    y_ = y.reshape(-1, 1)
    u_ = u.reshape(-1, 1)
    v_ = v.reshape(-1, 1)
    p_ = p.reshape(-1, 1)
    xx_=rs_xx.reshape(-1,1)
    xy_ = rs_xy.reshape(-1, 1)
    yy_ = rs_yy.reshape(-1, 1)
    #x = grid_x[5:122, 5:154]
    print(x.shape)
    XYUVP=np.concatenate([x_,y_,u_,v_,xx_,xy_,yy_],1)
    print(XYUVP)
    t=np.array(data['p_dns']).reshape(127,159)
    np.savetxt(r"C:\Users\mike\PycharmProjects\pinn\qiuling\t.txt",t)
    test_x=np.array(data['grid_x']).reshape(-1,1)
    test_y=np.array(data['grid_y']).reshape(-1,1)
    #print(test_x.reshape(127,159),grid_x)
    x_train, x_test, y_train, y_test = train_test_split(XYUVP,p_,train_size=0.3, random_state=0)
    #min_max_scaler = preprocessing.MinMaxScaler()
    #x_scaled = min_max_scaler.fit_transform(x)
    y_test = tf.reshape(y_test, [-1, 1])
    #x_train=min_max_scaler.fit_transform(x_train[['grid_y','grid_x', 'u_x','u_y']])
    #x_test=min_max_scaler.fit_transform(x_test[['grid_y','grid_x', 'u_x','u_y']])

    i=101
    checkpoint_dir=r'C:\Users\mike\PycharmProjects\pinn\pinns\model_mut10'
    model=periodic_pinn(x_train,i,checkpoint_dir)
    model.train(i)
    ####plot l1,l2
    plt.figure(1)
    plt.xlabel('iterarion')
    #plt.plot(model.iter_record,model.arecord,'r',label='a')
    #plt.plot(model.iterrecord,model.l1record,'g',label='l1')
    #plt.plot(model.iter_record,model.crecord,'b',label='c')
    plt.plot(model.iterrecord,model.l2record,'y',label='l2')
    plt.title('l2')
    plt.legend()
    plt.figure(2)
    plt.xlabel('iterarion')
    plt.plot(model.iterrecord, model.lossrecord, 'g', label='loss')
    plt.show()


    print('do')
    y_predict=model.predict(x_,y_)
    #y_predict.to_csv(r"C:\Users\mike\PycharmProjects\pinn\qiuling\y_")
    y_predict=transform_shape(y_predict)
    #np.savetxt(r'C:\Users\mike\PycharmProjects\pinn\qiuling\y_tran_iter{}_regularize{}.txt'.format(str(i),'0.001'),y_predict)
    save_fig(x,y,y_predict,i)
