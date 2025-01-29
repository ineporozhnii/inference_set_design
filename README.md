# Efficient Biological Data Acquisition through Inference Set Design

This is the code for the paper ["Efficient Biological Data Acquisition through Inference Set Design"](https://arxiv.org/abs/2410.19631) accepted at ICLR 2025. 

### Installation

```
pip install -r requirements.txt
pip install -e .
```

### Training a model

To train a model, simply run the `main.py`, and edit the configuration defined in the script to run a particular experiment. The configuration is self-documented in the `config.py` file. You can find tutorials how to train models and visualize results in `notebooks/tutorials`. 

### Citation

If you use this code in your research, please cite the following paper:

```
@article{neporozhnii2024efficientbiologicaldataacquisition,
         title={Efficient Biological Data Acquisition through Inference Set Design}, 
         author={Ihor Neporozhnii and Julien Roy and Emmanuel Bengio and Jason Hartford},
         year={2024},
         eprint={2410.19631},
         archivePrefix={arXiv},
         primaryClass={cs.LG},
         url={https://arxiv.org/abs/2410.19631}, 
}
```