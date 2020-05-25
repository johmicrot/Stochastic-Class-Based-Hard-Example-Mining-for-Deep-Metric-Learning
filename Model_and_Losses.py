import seaborn as sns;
sns.set()
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, GlobalAveragePooling2D, Conv2D, Layer
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from Helper_Functions import *


class SCHEM():

    def __init__(self,
                 run_name,
                 out_features,
                 optim,
                 load_im_size,
                 cs_weight,
                 fe_weight,
                 img_location,
                 crop_size,
                 init_W,
                 end_layer,
                 scale_CS_W = False,
                 l2_norm = False,
                 schsm_sampling ='hard',
                 batch_size=60,
                 LR=0.0001,
                 K_constant=5,
                 beta=5,
                 eta=10
                 ):
        """

        :param run_name:            Allows you to give unique names to each run
        :param out_features:        Number of output features
        :param optim:               Learning rate model optemizer
        :param load_im_size:        Size of image to load
        :param cs_weight:           Coefficent in front of L_c
        :param fe_weight:           Coefficent in front of L_t
        :param img_location:        Location of folder containing .npy files
        :param crop_size:           Paper uses 224
        :param init_W:              Weight initialization
        :param K_constant:          Same value from paper
        :param eta:                 Same value from paper
        :param beta:                Same value from paper
        :param end_layer:           Number of layers to cut off from MobileNetV2
        :param scale_CS_W:          Boolean to scale CS weights to have a magnitude of 1.
        :param l2_norm:             Boolean that controls if the feature map G has l2-normalization applied
        :param schsm_sampling:  
        """
        self.schsm_sampling_method = schsm_sampling
        self.l2_norm = l2_norm
        self.batch_size = batch_size
        self.scale_sig = scale_CS_W
        self.init_W = init_W
        self.end_layer = end_layer
        self.K_constant = K_constant
        self.beta = beta
        self.cs_weight = cs_weight
        self.fe_weight = fe_weight
        self.optim = optim

        self.out_features = out_features
        self.LR = LR
        self.init_W = init_W
        self.model_location = None
        self.img_location = img_location
        self.tf_m = tf.constant(self.batch_size * 3.0)
        self.load_im_size = load_im_size
        self.crop_size = crop_size
        self.in_shape = (crop_size, crop_size, 3)
        self.prepare_data()
        self.eta = eta
        self.n_batch_size = batch_size - eta

        self.run_name = run_name
        self.name_mod = '(%.1f,%.1f)_el(%i)%s' % (fe_weight,
                                                  cs_weight,
                                                  end_layer,
                                                  self.run_name)

        self.save_mod = '%s/' % self.name_mod
        self.results_dir = 'results(%i)' % crop_size
        self.tsne_out = '%s/plots/tsne/' % self.results_dir + self.save_mod
        self.loss_out = '%s/plots/loss/' % self.results_dir + self.save_mod
        self.csv_out = '%s/csv/' % self.results_dir + self.save_mod
        self.acc_out = '%s/plots/acc/' % self.results_dir + self.save_mod
        self.acc_out_bin = '%s/plots_bin' % self.results_dir
        self.model_out = '%s/model_weights/' % self.results_dir + self.save_mod
        make_dirs([self.tsne_out,
                   self.loss_out,
                   self.acc_out,
                   self.model_out,
                   self.acc_out_bin])

    def base_network(self):
        mob_net = tf.keras.applications.MobileNetV2(weights='imagenet',
                                                    include_top=False,
                                                    input_shape=self.in_shape)
        fmg = Layer(name="Feature_map_G_1")(mob_net.layers[-self.end_layer].output)
        fmg = Conv2D(self.out_features, (1, 1), name='Feature_map_G_2',activation='relu')(fmg)
        fmg = tf.keras.layers.BatchNormalization(axis=1, trainable=False, name='Feature_map_G_3')(fmg)
        x = GlobalAveragePooling2D()(fmg)
        if self.l2_norm:
            x = Lambda(lambda a: tf.math.l2_normalize(a, axis=1), name='l2_norm')(x)

        outmodel = Model(inputs=mob_net.input, outputs=x, name='base_FE_network')
        self.map_G = Model(inputs=mob_net.input, outputs=fmg, name='base_FE_network')
        return outmodel


    def create_model_and_compile(self):
        input_shape = self.in_shape
        self.base_model = self.base_network()
        self.cs_layer = Dense(self.num_classes, use_bias=False, name='CS_layer')
        qpn_in = Input(shape=input_shape, name='in_qpn')
        qpny = Input(shape=(self.num_classes,), name='in_qpny')
        qpny_label = Input(shape=(1,), name='in_qpny_label')

        qpn_out = self.base_model(qpn_in)
        qpn_cls_sig = self.cs_layer(qpn_out)


        fe_loss = Lambda(self.triplet_loss_all_combinations, name='triplet_loss_FE')(qpn_out) * self.fe_weight
        fe_accuracy = Lambda(self.FE_accuracy, name='FE_accuracy_metric')(qpn_out)

        cs_loss = Lambda(self.manual_CS_loss, name='CS_loss_calc')([qpn_cls_sig, qpny]) * self.cs_weight
        cs_accuracy = Lambda(self.CS_accuracy, name='CS_Acc')([qpn_cls_sig, qpny])

        total_loss = fe_loss + cs_loss

        model = Model(inputs=[qpn_in, qpny, qpny_label], outputs= [qpn_cls_sig], name='FEModel')

        if 'adm' in self.optim:
          optm = Adam(lr=self.LR)
        elif 'ranger' in self.optim:  # option to use a newer optimizer
          radam = tfa.optimizers.RectifiedAdam(lr=self.LR, min_lr=1e-7)
          optm = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)


        model.add_loss(total_loss)
        model.compile(optimizer=optm)

        # Metrics to track the accuracy and loss progression
        model.add_metric(fe_accuracy, name='fe_a', aggregation='mean')
        model.add_metric(cs_accuracy, name='cs_a', aggregation='mean')
        model.add_metric(fe_loss, name='fe_loss', aggregation='mean')
        model.add_metric(cs_loss, name='cs_loss_out', aggregation='mean')

        return model, optm

    def row_wise_subtraction_and_comparison(self, i, qp, n):
        """

        :param i: index to grab from qp
        :param qp: all QP images
        :param n:  all N images
        :return:  all elements from a specific row of QP subtracted from all elements in n
        """
        # Takes a row and subtracts n from all elemnts

        i = tf.cast(i, tf.int32)
        q_elm = tf.gather(qp, i)
        qp_elm_sub = tf.expand_dims(q_elm, axis=0) - qp
        qp_elm_mag = tf.math.sqrt(tf.reduce_sum(tf.math.square(qp_elm_sub), axis=1))
        n_sub = tf.expand_dims(q_elm, axis=0) - n
        n_mag = tf.math.sqrt(tf.reduce_sum(tf.math.square(n_sub), axis=1))

        return tf.expand_dims(qp_elm_mag, axis=1) - n_mag

    def triplet_loss_all_combinations(self, args):

        qp = args[:self.eta, :]
        n = args[self.eta:, :]

        all_combinations = tf.map_fn(lambda i: self.row_wise_subtraction_and_comparison(i, qp, n), elems=tf.range(self.eta), dtype=tf.float32)

        loss = all_combinations + tf.constant(0.2) # Hinge loss margin in 0.2

        loss = K.maximum(loss, tf.constant(0.0))

        # Using weighting scheme to give more weight to the hard triplets.
        num_non_zero = tf.math.count_nonzero(loss, dtype=tf.dtypes.float32)
        l_out = tf.cond(tf.equal(num_non_zero,tf.constant(0.0)), lambda: tf.constant(0.0), lambda :tf.divide(tf.reduce_sum(loss), num_non_zero))
        return tf.reduce_mean(loss)

    def manual_CS_loss(self, args):
        #Paper implementation of L_c loss
        cs, qpn_y = args
        qpn_yf = tf.cast(qpn_y, tf.float32)
        a = tf.nn.softmax_cross_entropy_with_logits(qpn_yf, cs, axis=1, name='cross_entropy')
        return tf.reduce_mean(a)

    def get_cs_row(self, i, qpn_y, cse_out):
        i = tf.cast(i, tf.int32)
        curr_class = tf.where(tf.gather(qpn_y, i))
        row = tf.gather(cse_out, i)
        elm = tf.gather(row, curr_class)
        row_denom = tf.reduce_sum(row)
        elm_norm = elm / tf.maximum(row_denom, tf.keras.backend.epsilon())

        return tf.squeeze(tf.math.log(elm_norm))

    def get_class_pool(self, qx, W, qp_idx, alpha):
        # Calculating P_c from paper
        W = tf.math.l2_normalize(W, axis=0)
        qx = tf.math.l2_normalize(qx, axis=1)
        sim_space = tf.tensordot(qx, W, axes=1)
        sim_space_max = tf.reduce_max(sim_space, axis=0)

        closes_class_idx = tf.argsort(sim_space_max, direction='DESCENDING')
        out = tf.boolean_mask(closes_class_idx, ~tf.equal(closes_class_idx, qp_idx))

        return out[:alpha*(self.K_constant-1)]

    def CS_accuracy(self, args):
        qpncs, qpny = args
        qpt = tf.cast(tf.equal(tf.argmax(qpny, axis=1), tf.argmax(qpncs, axis=1)), tf.float32)
        return K.mean(qpt)

    def rand_flip_and_crop(self,batch):
            batch = tf.map_fn(lambda x: tf.image.random_crop(x, size=[self.crop_size, self.crop_size, 3]), batch)
            batch = tf.map_fn(tf.image.random_flip_left_right, batch)
            return batch

    def center_crop(self,batch):
        # Used during testing
        start_point = self.load_im_size - self.crop_size
        batch = tf.image.crop_to_bounding_box(batch,
                                              start_point,
                                              start_point,
                                              self.crop_size,
                                              self.crop_size)

        return batch

    def prepare_data(self):

        x_train = np.load('%s/trnx_CUB_%s.npy' % (self.img_location, self.load_im_size))
        y_train = np.load('%s/trny_CUB_%s.npy' % (self.img_location, self.load_im_size))
        y_train_OH = np.load('%s/trny_OH_CUB_%s.npy' % (self.img_location, self.load_im_size))
        x_test = np.load('%s/tstx_CUB_%s.npy' % (self.img_location, self.load_im_size))
        y_test = np.load('%s/tsty_CUB_%s.npy' % (self.img_location, self.load_im_size))

        y_train_num_label, y_test_num_label = [], []
        for elm in y_train_OH:
            y_train_num_label.append(list(elm).index(1))
        for elm in y_test:
            y_test_num_label.append(int(elm.split('.')[0]))
        self.all_c = list(set(y_train_num_label))
        self.num_classes = len(self.all_c)
        y_train_num_label = np.array(y_train_num_label)
        y_test_num_label = np.array(y_test_num_label)

        y_train_OH = y_train_OH.astype(bool)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        self.x_train = x_train
        self.y_train_OH = y_train_OH
        self.y_train = y_train_num_label
        self.x_test = x_test
        self.y_test = y_test_num_label

        return x_train, y_train_OH, y_train_num_label, x_test, y_test_num_label

    def generate_G(self, images, mode):
        # Generate the feature map G.  Done in batches due to memeory restrictions
        if mode == 'test':
            x_shaped = self.center_crop(images)
        else:
            x_shaped = self.rand_flip_and_crop(images)
        # This split is done due to help wiht memory overflow
        test_batch_split = np.array_split(np.arange(len(x_shaped)), 15)
        feats = np.array([])
        for t_bat in test_batch_split:
            tmp = tf.keras.backend.batch_flatten(self.map_G(tf.gather(x_shaped, t_bat)))
            if tf.size(feats) == 0:
                feats = tmp
            else:
                feats = tf.concat([feats, tmp], axis=0)
        return feats

    def generate_out(self, images, mode):
        # Generate the network output layer that is trained on
        # Used just for debugging
        if mode == 'test':
            x_shaped = self.center_crop(images)
        else:
            x_shaped = self.rand_flip_and_crop(images)
        tmp = 15
        if self.load_im_size == 256:
            tmp = 50
        test_batch_split = np.array_split(np.arange(len(x_shaped)), tmp)
        feats = np.array([])
        for t_bat in test_batch_split:
            tmp = tf.keras.backend.batch_flatten(self.base_model(tf.gather(x_shaped, t_bat)))
            if tf.size(feats) == 0:
                feats = tmp
            else:
                feats = tf.concat([feats, tmp], axis=0)
        return feats

    def get_batch(self):
        # x_train, y_train, y_train_OH
        qp_idx = self.q_idx
        xqp_inds = np.where(self.y_train_OH[:, qp_idx])
        all_Xqp = self.x_train[xqp_inds]
        all_qpy_label = self.y_train[xqp_inds]
        all_qpy = self.y_train_OH[xqp_inds]

        batch_qp_idx = np.random.choice(np.arange(len(all_Xqp)), self.eta, replace=False)

        batch_qp = all_Xqp[batch_qp_idx]
        batch_qp = self.rand_flip_and_crop(batch_qp)

        batch_qpy = all_qpy[batch_qp_idx]
        batch_qpy_label = all_qpy_label[batch_qp_idx]
        return batch_qp, batch_qpy, batch_qpy_label


    def feature_map_G_statistics(self):
        xd = self.x_test
        y_labels = self.y_test
        g_feats = self.generate_G(xd, mode='test')
        feats1 = tf.math.l2_normalize(g_feats, axis=1)
        sim_matrix = tf.tensordot(feats1, tf.transpose(feats1), axes=1)
        closest_neighbor = tf.argsort(sim_matrix, direction='DESCENDING', axis=1)[:, 1]  # k = 1 here
        grab_class = tf.gather(y_labels, closest_neighbor)
        acc = tf.cast(tf.math.equal(y_labels, grab_class), tf.float32)
        acc1 = tf.reduce_mean(acc)


        ###################### compare with output R@1
        out_feats = self.generate_out(xd, mode='test')
        feats1 = tf.math.l2_normalize(out_feats, axis=1)
        sim_matrix = tf.tensordot(feats1, tf.transpose(feats1), axes=1)
        closest_neighbor = tf.argsort(sim_matrix, direction='DESCENDING', axis=1)[:, 1]  # k = 1 here
        grab_class = tf.gather(y_labels, closest_neighbor)
        acc = tf.cast(tf.math.equal(y_labels, grab_class), tf.float32)
        acc5 = tf.reduce_mean(acc)



        return acc1.numpy(), acc5.numpy()


    def generate_all_statistics(self):
        xd = self.x_test
        y_labels = self.y_test
        feats = self.generate_G(xd, mode='test')
        feats1 = tf.math.l2_normalize(feats, axis=1)
        sim_matrix = tf.tensordot(feats1, tf.transpose(feats1), axes=1)
        # doing [:, 1]  to remove the first column which is the correlation with itself
        closest_neighbor = tf.argsort(sim_matrix, direction='DESCENDING', axis=1)[:, 1] 
        grab_class = tf.gather(y_labels, closest_neighbor)
        acc = tf.reduce_mean(tf.cast(tf.math.equal(y_labels, grab_class), tf.float32))

        return acc.numpy()


    def FE_accuracy(self, args):
        qp = args[:self.eta, :]
        n = args[self.eta:, :]

        indicator = tf.map_fn(lambda i: self.row_wise_subtraction_and_comparison(i, qp, n), elems=tf.range(self.eta),
                              dtype=tf.float32)
        comparison = tf.cast(tf.math.greater(tf.constant(0.0), indicator), tf.float32)

        return tf.reduce_mean(comparison)

    def schem_sample(self,
                     batch_qp
                     ):
        x_train = self.x_train
        y_train = self.y_train
        y_train_OH = self.y_train_OH

        if 'hard' in self.schsm_sampling_method:  # Hard sampling
            qp_features = tf.keras.backend.batch_flatten(self.base_model(batch_qp))
            qp_features = tf.math.l2_normalize(qp_features, axis=1)
            Pc = self.get_class_pool(qp_features, self.cs_layer.get_weights()[0], self.q_idx, self.alpha)
            n_indexes = []
            # Could automate this in a tensor, but probably don't need to
            for elm in Pc:
                indexes_for_class = np.where(y_train == elm)[0]
                n_indexes += list(indexes_for_class)

            # Generate feature space for all N elements.
            Ps_all = x_train[n_indexes]
            n_all_y = y_train_OH[n_indexes]
            n_all_y_label = y_train[n_indexes]
            Ps_features = self.generate_out(Ps_all, mode='train')
            Ps_features = tf.math.l2_normalize(Ps_features, axis=1)
            similarity = tf.tensordot(qp_features, tf.transpose(Ps_features), axes=1)

            # Take top similarities from all q-inputs.
            max_sim = tf.reduce_max(similarity, axis=0)
            max_sum = tf.reduce_sum(similarity, axis=0)
            similar_indicies_sorted = tf.argsort(max_sum, axis=0, direction='DESCENDING')

            top_similar_indicies = tf.argsort(max_sim, axis=0, direction='DESCENDING')[:self.beta * self.n_batch_size]

            n_idx = np.random.choice(top_similar_indicies, self.n_batch_size)


            n_bat = tf.gather(Ps_all, n_idx)
            n_bat = self.rand_flip_and_crop(n_bat)  # maybe i should crop before and not randomly re-crop

            n_y = tf.gather(n_all_y, n_idx)
            n_y_label = tf.gather(n_all_y_label, n_idx)
        else:  # Naive sampling
            n_idxs = np.random.choice(np.setdiff1d(np.arange(self.num_classes), self.q_idx), 10)
            n_bat = np.array([])
            n_y = np.array([])
            n_y_label = np.array([])
            for n_idx in n_idxs:
                tot_elms = sum(y_train_OH[:, n_idx])
                xqp_inds = np.where(y_train_OH[:, n_idx])
                all_Xn = x_train[xqp_inds]
                all_yn = y_train_OH[xqp_inds]
                all_yn_label = y_train[xqp_inds]
                b_n_idxs = np.random.choice(np.arange(tot_elms), 2, replace=False)
                if n_bat.size == 0:
                    n_bat = all_Xn[b_n_idxs]
                    n_y = all_yn[b_n_idxs]
                    n_y_label = all_yn_label[b_n_idxs]
                else:
                    n_bat = np.append(n_bat, all_Xn[b_n_idxs], axis=0)
                    n_y = np.append(n_y, all_yn[b_n_idxs], axis=0)
                    n_y_label = np.append(n_y_label, all_yn_label[b_n_idxs], axis=0)
            n_bat = self.rand_flip_and_crop(n_bat)
        return n_bat, n_y, n_y_label
