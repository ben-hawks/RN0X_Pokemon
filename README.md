# RN0X_Pokemon
Tiny, Quantized Neural Network, orignially based on ResNet8, trained to recognize pokemon. Originally developed/optimized for the TinyMLPerf v0.7/1.0 Image Classification Benchmark, using the CIFAR-10 Dataset. 

## Run Training
To Train a given model:
`python train_pokemon.py -c config/<config>.yml`
where `<config>` is one of the yml files in the config directory. RN06_Poke10.yml is what was used for the DEFCON30 Webcam Demo. If you are intending to retrain this model for deployment on a Pynq-Z2 or similar FPGA platform (Xilinx Zynq 7020), it is reccomended to use RN06 if you need to have other large/major IP blocks in your firmware (such as HDMI, etc.) and RN07 if the neural network is the only thing you intend to run on the PL/Fabric. 

## Pretrained Models
Multiple pretrained models are in this repo. The one used in the DEFCON30 Webcam Demo is `trained_model/deploy/resnet_v1_eembc_RN06_bilinear/small_model_best.h5`, though most variations in `trained_model/deploy` are simple variations on the preprocessing of the training set. 


## Datasets
The pokemon datasets used in this training are mentioned in `data/sources.txt`, with the training dataset(s) availible on Kaggle, and the test dataset needing to be scraped. The (very janky, cobbled together, annoying to use) tool can be found in [this repo.](https://www.github.com/ben-hawks/pokedex_scraper)

## Conversion to FPGA Firmware via hls4ml
This repo does not contain the scripts required to convert and deploy this model onto an FPGA. They are located in [this repo.](https://github.com/fastmachinelearning/hls4ml-live-demo)