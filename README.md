# Big Learning Expectation Maximization  

The official code for the AAAI 2024 paper "[Big Learning Expectation Maximization](https://arxiv.org/abs/2312.11926)" by Yulai Cong and Sijia Li.

![converge_wrt_seed](https://github.com/YulaiCong/Big-Learning-Expectation-Maximization/assets/34753772/bae13ef1-a6be-4ac3-93b0-194951ecfb61)

## Abstract
Mixture models serve as one fundamental tool with versatile applications. However, their training techniques, like the popular Expectation Maximization (EM) algorithm, are notoriously sensitive to parameter initialization and often suffer from bad local optima that could be arbitrarily worse than the optimal. To address the long-lasting bad-local-optima challenge, we draw inspiration from the recent ground-breaking foundation models and propose to leverage their underlying big learning principle to upgrade the EM. Specifically, we present the Big Learning EM (BigLearn-EM), an EM upgrade that simultaneously performs joint, marginal, and orthogonally transformed marginal matchings between data and model distributions. Through simulated experiments, we empirically show that the BigLearn-EM is capable of delivering the optimal with high probability; comparisons on benchmark clustering datasets further demonstrate its effectiveness and advantages over existing techniques.

## Directory Explanation
```
filetree 
├── BL_vs_deepClustering
│  ├── dataset
│  ├── function.py
│  ├── method.py
|  ├── main_BigLearnEM.py
├── dataset
├── function.py
├── method.py
├── main_biglearnEM_vs_EM_v1.ipynb
├── main_realworld_clustering.ipynb
```

## Usage

### 1. Joint-EM vs BigLearn-EM on the 25-GMM simulation
- Run `main_biglearnEM_vs_EM_v1.ipynb`

### 2. Real-World Clustering Experiments
- Prepare the dataset: The Glass, Letter, Pendigits, and Vehicle datasets are given in the Directory 'dataset'. [Click here to download other datasets](https://www.csie.ntu.edu.tw/~cjlin/libsvm/index.html)
- Run `main_realworld_clustering.ipynb`.
- Note different hyperparameter settings should be used for different datasets. If 'out of memory' occurs, modify 'data_size' or 'chunk_size'. For datasets without an official testing set, set 'split_data' to 'True' to randomly select data samples to form one.
  
### 3. BigLearn-EM vs Deep Clustering Methods on the FashionMNIST dataset (BL_vs_deepClustering)
- Download the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset and place it into the `BL_vs_deepClustering/dataset` directory
- Run `main_BigLearnEM.py`
- Run `python main_BigLearnEM.py --device cuda:1 --Niter 70 --NITnei 5 --eps 0.05 --P1 0.4 --P2 0.5 --out_dir [path for training weights] --txt_dir [path for training records]`
- The experiment is conducted based on the [code](https://github.com/JinyuCai95/EDESC-pytorch) of the CVPR22 paper ["Efficient Deep Embedded Subspace Clustering"](https://openaccess.thecvf.com/content/CVPR2022/papers/Cai_Efficient_Deep_Embedded_Subspace_Clustering_CVPR_2022_paper.pdf).

## Reference
Please consider citing our paper if you refer to this code in your research.
```
@misc{cong2023big,
      title={Big Learning Expectation Maximization}, 
      author={Yulai Cong and Sijia Li},
      year={2023},
      eprint={2312.11926},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

​     
