import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import loggers as pl_loggers
from dataBuilders.data_builder import SearchDataModule
from model.transformerLightning import TransformerModel
from model.transformerLightning_mixed import TransformerModel_Mixed
import numpy as np
import os
import sys
sys.path.append('./src/')
from evaluation.evaluation_model import Evaluation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data path and output files
    parser.add_argument('-data_path', default='./dataset/processdata/dataset_amazon', type=str)
    parser.add_argument('-index_folder', default='./dataset/processdata/', type=str)
    parser.add_argument('-index_file', default='splitlist_all_amazon.txt', type=str) # all_time_better

    parser.add_argument('-testing_dataset_choice', default='amazon', type=str)  # wine, yogurt, amazon,all, irregular
    parser.add_argument('-training_dataset_choice', default='amazon', type=str)  # wine, yogurt, amazon,all
    parser.add_argument('-leave_one_comb_out', default=0, type=int)
    parser.add_argument('-leave_one_comb_out_tgt_id', default=0, type=int)
    parser.add_argument('-leave_one_comb_out_layout_id', default=0, type=int)
    parser.add_argument('-spp', default=0, type=int) # 0: no spp, 2, 3, 4 represent level

    parser.add_argument('-checkpoint', default='None', type=str)
    #parser.add_argument('-posOption', default=2, type=int) # choices: 1, 2, 3, 4
    parser.add_argument('-alpha', type=float, default=0.9)
    parser.add_argument('-functionChoice', default='learned', type=str) # choices: linear, exp1, exp2, original, original_update, learned
    parser.add_argument('-changeX', default='None', type=str) # None, False, True
    parser.add_argument('-CA_version', default=3, type=int)  # valid values atm: 0, 3
    # 0: no cross attention, 1: add padding to input, 2: extra FC stream, 3: add pad prob in logits
    parser.add_argument('-CA_head', default=2, type=int) # the number of cross attention heads
    parser.add_argument('-CA_dk', default=512, type=int) # 512, 64, scaling factor in attention matrix

    parser.add_argument('-PE_matrix', default='./src/model/amazon_learned_random_PE.npy', type=str)
    parser.add_argument('-log_name', default='amazon_pamformer', type=str)
    parser.add_argument('-output_postfix', type=str, default='') # better to start with '_'
    parser.add_argument('-stochastic_iteration', type=int, default=100)
    parser.add_argument('-write_output', type=str, default='True')

    # model settings and hyperparameters
    parser.add_argument('-model', default='Transformer', type=str) #BaseModel, Gazeformer
    parser.add_argument('-learning_rate', default=1e-4, type=float)
    parser.add_argument('-scheduler_lambda1', default=1, type=int)
    parser.add_argument('-scheduler_lambda2', default=1.0, type=float)
    parser.add_argument('-grad_accumulate', type=int, default=1)
    parser.add_argument('-clip_val', default=1.0, type=float)
    parser.add_argument('-limit_val_batches', default=1.0, type=float)
    parser.add_argument('-val_check_interval', default=1.0, type=float)

    # training settings
    parser.add_argument('-gpus', default='-1', type=str)
    parser.add_argument('-batch_size', type=int, default=20)
    parser.add_argument('-num_epochs', type=int, default=500)
    parser.add_argument('-random_seed', type=int, default=1234)
    parser.add_argument('-early_stop_patience', type=int, default=30)

    parser.add_argument('-monitor', type=str, default='validation_delta_each_epoch') #'validation_loss_each_epoch'
    parser.add_argument('-do_train', type=str, default='True')
    parser.add_argument('-do_test', type=str, default='True')

    args = parser.parse_args()

    if args.training_dataset_choice == args.testing_dataset_choice and args.testing_dataset_choice != 'all':
        if args.testing_dataset_choice == 'wine':
            args.package_size = 22
            args.shelf_row = 2
            args.shelf_col = 11
        elif args.testing_dataset_choice == 'yogurt':
            args.package_size = 27
            args.shelf_row = 3
            args.shelf_col = 9
        elif args.testing_dataset_choice == 'amazon':
            args.package_size = 84
            args.shelf_row = 6
            args.shelf_col = 14
    else: # zero=yogurt, one=wine
        args.package_size = np.array([27, 22, 84])
        args.shelf_row = np.array([3, 2, 6])
        args.shelf_col = np.array([9, 11, 14])
        args.batch_size *= 2

    args.output_path = './dataset/checkEvaluation/' + args.log_name + '/'

    # random seed
    seed_everything(args.random_seed)

    # create new directory of saving results
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    # set logger
    logger = pl_loggers.TensorBoardLogger(f'./lightning_logs/{args.log_name}')

    mode = 'min'
    if args.monitor == 'validation_sim_each_epoch':
        mode = 'max'
    # # save checkpoint & early stopping & learning rate decay & learning rate monitor
    checkpoint_callback = ModelCheckpoint(monitor=args.monitor,
                                          save_last=False,
                                          save_top_k=1,
                                          mode=mode,)

    early_stop_callback = EarlyStopping(
                            monitor=args.monitor,
                            min_delta=0.00,
                            patience=args.early_stop_patience,
                            verbose=False,
                            mode=mode
                            )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # make dataloader & model
    hasExpectedFile = True
    if args.model == 'Transformer':
        search_data = SearchDataModule(args)
        if args.testing_dataset_choice == 'irregular':
            irregular_data = IrregularShelfModule(args)

        if args.training_dataset_choice == args.testing_dataset_choice and args.training_dataset_choice != "all":
            model = TransformerModel(args, search_data.max_len)
        elif args.testing_dataset_choice == 'irregular':
            model = TransformerModel_Mixed_Irregular(args, search_data.max_len, irregular_data.max_len)
        else:
            model = TransformerModel_Mixed(args, search_data.max_len)
    elif args.model == 'BaseModel':
        model = BaseModel(args)
        search_data = BaseSearchDataModule(args)
    elif args.model == 'Gazeformer':
        search_data = GazeformerDataModule(args)
        model = TransformerModel_Gazeformer(args, 20)
        hasExpectedFile = False
    else:
        print('Invalid model')

    if args.checkpoint == 'None':
        args.checkpoint = None
    trainer = Trainer(deterministic=True,
                      num_sanity_val_steps=10,
                      resume_from_checkpoint=args.checkpoint,
                      logger=logger,
                      gpus=args.gpus,
                      #distributed_backend='ddp',
                      #plugins=DDPPlugin(find_unused_parameters=True),
                      gradient_clip_val=args.clip_val,
                      max_epochs=args.num_epochs,
                      limit_val_batches=args.limit_val_batches,
                      val_check_interval=args.val_check_interval,
                      accumulate_grad_batches=args.grad_accumulate,
                      fast_dev_run=False,
                      callbacks=[lr_monitor, checkpoint_callback, early_stop_callback])

    # Fit the instantiated model to the data
    if args.do_train == 'True':
        trainer.fit(model, search_data.train_loader, search_data.val_loader)
        trainer.test(model=model, dataloaders=search_data.test_loader)
    elif args.do_test == 'True':
        if args.testing_dataset_choice == 'irregular':
            model = model.load_from_checkpoint(args.checkpoint, args=args, max_len=search_data.max_len, irregular_max_len=irregular_data.max_len)
            trainer.test(model=model, dataloaders=irregular_data.test_loader)
        else:
            model = model.load_from_checkpoint(args.checkpoint, args=args, max_len=search_data.max_len)
            trainer.test(model=model, dataloaders=search_data.test_loader)

    e = Evaluation(args.training_dataset_choice, args.testing_dataset_choice, args.output_path,
                   args.data_path, args.index_folder+args.index_file,
                   ITERATION=args.stochastic_iteration, hasExpectedFile=hasExpectedFile)
    e.evaluation()

