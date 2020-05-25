from Model_and_Losses import SCHEM
import pandas as pd
from Helper_Functions import *
import time as t
import sys

args = process_input(sys.argv)

hem = SCHEM(run_name=args['runname'],
            optim='adm',
            end_layer=args['end_lyr'],
            out_features=args['out_features'],
            img_location=args['img_location'],
            load_im_size=args['img_load_size'],
            crop_size=args['img_crop_size'],
            init_W=args['inital_weights'],
            fe_weight=1.0,
            cs_weight=1.0,
            scale_CS_W=args['sclCS'],
            )

rand_classes = np.arange(hem.num_classes)
qpn_model, optm = hem.create_model_and_compile()

train_list, test_list = [], []
cs_acc_list, fe_acc_list = [], []

for epoch in range(0, args['epocs']):
    np.random.shuffle(rand_classes)
    if epoch > 200:
        optm.learning_rate.assign(optm.learning_rate * 0.9675)
        print('optim LR = ', optm.learning_rate)
    for idx, cls_idx in enumerate(rand_classes):
        hem.q_idx = cls_idx
        # hem.alpha = np.random.randint(3, 6)
        hem.alpha = np.random.randint(1, 2)
        batch_qp, batch_qpy, batch_qpy_label = hem.get_batch()
        n_bat, n_y_OH, n_y_label = SCHEM.schem_sample(hem, batch_qp)
        qpn_batch = tf.concat([batch_qp, n_bat], axis=0)
        qpn_y_boolOH = tf.concat([batch_qpy, n_y_OH], axis=0)
        qpn_y_label = tf.concat([batch_qpy_label, n_y_label], axis=0)

        model_out = qpn_model.train_on_batch([qpn_batch, qpn_y_boolOH, qpn_y_label])
        print(model_out)
        i_t = epoch + (idx + 1) / hem.num_classes
        train_list.append([i_t] + model_out)

    end_time = t.time()
    accuracy = hem.generate_all_statistics()

    print([epoch, accuracy])
    test_list.append([epoch, accuracy])
    break
if args['saveW']:
    qpn_model.save_weights(hem.model_out + 'weights_epoc(%i).tf' % epoch)

# Create directories after testing has finished to prevent empty folders
make_dirs([hem.csv_out, hem.csv_out2])
trn_pd = pd.DataFrame(train_list, columns=['time', 'L_t', 'A_fe', 'A_cs', 'L_fe', 'L_cs',])
trn_pd.to_csv(hem.csv_out + 'train_results.csv', index=None)
tst_pd = pd.DataFrame(test_list, columns=['time', 'R1'])
tst_pd.to_csv(hem.csv_out + 'acc_results.csv', index=None)