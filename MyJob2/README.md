# CRoC: Context Refactoring Contrast for Graph Anomaly Detection with Limited Supervision



This is the code for reproducing the results reported in the ECAI2025 [paper](https://arxiv.org/pdf/2508.12278).

**More details of the code will be updated soon.**

Requirements:

* PyTorch
* DGL
* Numpy
* Scipy
* scikit-learn



## Quick Start



For quick reproduction of the CRoC(GIN) results in Yelp, you can type the following command in the terminal:

`python main.py`



## Reproduce Results



To reproduce other reported results, you can specify the dataset, model and hyper-parameters, e.g., reproduce the results on **Amazon**:

`python main.py --model CRoCSAGE --dataset amazon --alpha 0.5 --gamma 0.2 --eta 0.5 --n_epoch 200`




Note that **Yelp** and **Amazon** can be downloaded through DGL, while **T-Soc**, **T-Fin** and **DGraph-Fin** should be manually downloaded and placed under the ./dataset folder.



You can download these three datasets via:

* T-Soc and T-Fin: https://github.com/squareRoot3/Rethinking-Anomaly-Detection
* DGraph: https://dgraph.xinye.com/dataset


Hyper-parameters of experiments in each dataset is provided in Table 10 of the [arxiv paper](https://arxiv.org/pdf/2508.12278).


## Citation

```
@article{xie2025croc,
  title={CRoC: Context Refactoring Contrast for Graph Anomaly Detection with Limited Supervision},
  author={Xie, Siyue and Tam, Da Sun Handason and Lau, Wing Cheong},
  journal={arXiv preprint arXiv:2508.12278},
  year={2025}
}
```

