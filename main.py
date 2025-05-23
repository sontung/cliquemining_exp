from pathlib import Path

import faiss
import numpy as np
import pytorch_lightning as pl

from aachen import SampleDataset, test
from vpr_model import VPRModel
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule

if __name__ == '__main__':        
    datamodule = GSVCitiesDataModule(
        batch_size=30,
        img_per_place=4,
        min_img_per_place=4,
        shuffle_all=False, # shuffle all images or keep shuffling in-city only
        random_sample_from_each_place=True,
        image_size=(224, 224),
        num_workers=10,
        show_data_stats=True,
        val_set_names=[], # pitts30k_val, pitts30k_test, msls_val
        clique_mapillary_args={
            'same_place_threshold': 25.0,
            # We create more batches than required so
            # that we can shuffle the dataset after each epoch
            'num_batches': 4000,
            'num_processes': 10,
        }
    )
    
    model = VPRModel(
        #---- Encoder
        backbone_arch='dinov2_vitb14',
        backbone_config={
            'num_trainable_blocks': 4,
            'return_token': True,
            'norm_layer': True,
        },
        agg_arch='SALAD',
        agg_config={
            'num_channels': 768,
            'num_clusters': 64,
            'cluster_dim': 128,
            'token_dim': 256,
        },
        lr = 6e-5,
        optimizer='adamw',
        weight_decay=9.5e-9, # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': 4000,
        },

        #----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name='MultiSimilarityLoss',
        miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1,
        faiss_gpu=False
    )

    # model params saving using Pytorch Lightning
    # we save the best 3 models accoring to Recall@1 on pittsburg val
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor='pitts30k_val/R1',
        filename=f'{model.encoder_arch}' + '_({epoch:02d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        save_last=True,
        mode='max'
    )

    #------------------
    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        default_root_dir=f'./logs/', # Tensorflow can be used to viz 
        num_nodes=1,
        num_sanity_val_steps=0, # runs a validation step before stating training
        precision='16-mixed', # we use half precision to reduce  memory usage
        max_epochs=4,
        check_val_every_n_epoch=1, # run validation every epoch
        callbacks=[checkpoint_cb],# we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
        log_every_n_steps=20,
    )

    # we call the trainer, we give it the model and the datamodule
    trainer.fit(model=model, datamodule=datamodule)

    model.eval()
    model.cuda()
    ds_path = Path("../glace_experiment/datasets/aachen")
    train_data = SampleDataset(
        ds_path / "train", "../covis_graph/checkpoints/pose_overlap.npz", batch_size=32, nb_iterations=10
    )
    mat1 = test(train_data, model, 8448)

    train_data = SampleDataset(
        ds_path / "test", "../covis_graph/checkpoints/pose_overlap.npz", batch_size=32, nb_iterations=10
    )
    mat2 = test(train_data, model, 8448)
    whiten = False
    if whiten:
        from cuml import PCA

        pca_torch = PCA(n_components=256, copy=False)
        pca_torch.fit(mat1)
        print(f"Explained variance: {pca_torch.explained_variance_ratio_.sum()}")
        mat1 = pca_torch.transform(mat1)
        mat2 = pca_torch.transform(mat2)

    db_desc0 = np.load("../covis_graph/checkpoints/desc_salad_db_cl.npy")
    test_desc0 = np.load("../covis_graph/checkpoints/desc_salad_test_cl.npy")
    db_desc0 = np.ascontiguousarray(db_desc0)
    test_desc0 = np.ascontiguousarray(test_desc0)
    index = faiss.IndexFlatL2(db_desc0.shape[1])
    index.add(db_desc0.astype(np.float32))
    _, indices = index.search(test_desc0.astype(np.float32), 1)
    indices0 = indices.flatten()

    mat1 = np.ascontiguousarray(mat1)
    mat2 = np.ascontiguousarray(mat2)
    index = faiss.IndexFlatL2(mat1.shape[1])
    index.add(mat1.astype(np.float32))
    _, indices = index.search(mat2.astype(np.float32), 1)
    indices = indices.flatten()

    acc = np.sum(indices == indices0) / len(indices)
    print(f"Accuracy when compared to salad: {acc:.3f}")

