# Code for the ICLR 2024 submission: </br></br>"Hiding in Plain Sight: Disguising Data Stealing Attacks in Federated Learning"
## Prerequisites
- Install Anaconda. 
- Create the conda environment:<br>
> conda env create -f environment.yml
- Enable the created environment:<br>
> conda activate seer


## SEER experiments 

### Parameters
- *BATCH\_SIZE* - the batch size to use e.g **256**.
- *NUM\_CLIENTS* - the number of clients to aggregate e.g **8**.
- *PROPERTY* - the disaggregation property. Must be one of **bright**, **dark**, **red**,**blue**,**green**, **hedge**, **vedge**,**rand_conv**.
- *CHECKPOINT* - the trained model file e.g **../models/seer1.pt**.
- *DATASET* - the dataset to use. Must be one of **Cifar10**, **Cifar100**, **imagenet_full**, **TinyImageNet**, **TinyImageNet_rsz**, **Isic2019**, **Cifar10_2**, **Cifar10_1**, **CIFAR_C_***\<corruption\>***_***\<severity\>*.
  - where *\<corruption\>* is one of **brightness**, **contrast**, **defocus_blur**, **elastic_transform**, **fog**, **frost**, **gaussian_blur**, **gaussian_noise**, **glass_blur**, **impulse_noise**, **jpeg_compression**, **motion_blur**, **pixelate**, **saturate**, **shot_noise**, **snow**, **spatter**, **speckle_noise**, **zoom_blur**
  - and *\<severity\>* is one of **1**, **2**, **3**, **4**, **5**.


### Commands
- To run the single-client reconstruction experiment:<br>

        > bash train_rel.sh BATCH_SIZE --dataset DATASET --prop PROPERTY
        > bash test_rel.sh BATCH_SIZE --dataset DATASET --prop PROPERTY --checkpoint CHECKPOINT
- To run the multi-client reconstruction experiment:<br>
        
        > bash train.sh BATCH_SIZE --num_clients NUM_CLIENTS --dataset DATASET --prop PROPERTY
        > bash test.sh BATCH_SIZE --num_clients NUM_CLIENTS --dataset DATASET --prop PROPERTY --checkpoint CHECKPOINT

- To compare against the attack described in *Fishing for user data in large-batch federated learning via gradient magnification. In ICML, 2022*:<br>
        
        > bash compare_SEER_Fishing.sh

Example: Imagenet

        > bash train_rel.sh 64 --dataset imagenet_full --prop bright --acc_grad 10 --attack_cfg modern_cos_proj --case_cfg imagenet_sanity_check --par_sel_frac 0.02 --par_sel_size 9800
