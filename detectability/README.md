# Code for the ICLR 2024 submission: </br></br>"Hiding in Plain Sight: Disguising Data Stealing Attacks in Federated Learning"
## Prerequisites
- Install Anaconda. 
- Create the conda environment:<br>
> conda env create -f environment.yml
- Enable the created environment:<br>
> conda activate seer

## Detectability (Figure 1)
- `detectability/checkpoints/` contains some benign checkpoints (trained with `train_resnet.py`) and 4 SEER checkpoints
- `detectability/detectability.py` loads all checkpoints and saves D-SNR and transmit results to `json/`
- The example disaggregation (fishing) attack was run by adding `breaching_dsnr.py` to the root of the `breaching` library (commit `3e4d336ed3c631b8bf70974a9a5e1f923ddb039a`), which produced `images/fishing_*` photos; as that experiment did not set the random seed, running `breaching_dsnr.py` will not produce exactly the same results, but similar ones (D-SNR of 79, 33, 20, 8.7, 0.77 for multiplier values of 16, 15, 14, 13, 12, attempting to reconstruct the same car image)
- `plot_dsnr.py` loads the json file and the `images/fishing_*` and plots Figure 1 from our paper; it also requires the 3 icons in `images/` as well as the example SEER reconstruction `img4_s2n=0.65.png` obtained by running `detectability.py` with `PLOT_CAR=True` (requires full checkpoints (~6GB))
- `plot_transmit.py` does the same for the transmit-SNR shown in our appendix based on the corresponding json
