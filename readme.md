## Dataset
To investigate the effectiveness of AMPLE, we adopt three vulnerability datasets from these paper: 
* Fan et al. [1]: <https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing>
* Reveal [2]: https://drive.google.com/drive/folders/1KuIYgFcvWUXheDhT--cBALsfy1I4utOyF
* FFMPeg+Qemu [3]: https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF

## Requirement
Our code is based on Python3 (>= 3.7). There are a few dependencies to run the code. The major libraries are listed as follows:
* torch  (==1.9.0)
* dgl  (==0.7.2)
* numpy  (==1.22.3)
* sklearn  (==0.0)
* pandas  (==1.4.1)
* tqdm

**Default settings in AMPLE**:
* Training configs: 
    * batch_size = 64, lr = 0.0001, epoch = 100, patience = 20
    * opt ='RAdam', weight_decay=1e-6




## Attention weight
We provide all the attention weights learned by our proposed model AMPLE for the test samples. Each dataset corresponds to a json file under ```attention weight\``` folder.




## References
[1] Jiahao Fan, Yi Li, Shaohua Wang, and Tien Nguyen. 2020. A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries. In The 2020 International Conference on Mining Software Repositories (MSR). IEEE.

[2] Saikat Chakraborty, Rahul Krishna, Yangruibo Ding, and Baishakhi Ray. 2020. Deep Learning based Vulnerability Detection: Are We There Yet? arXiv preprint arXiv:2009.07235 (2020).

[3] Yaqin Zhou, Shangqing Liu, Jingkai Siow, Xiaoning Du, and Yang Liu. 2019. Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks. In Advances in Neural Information Processing Systems. 10197–10207.

[4] M. Fu and C. Tantithamthavorn. 2022. Linevul: A transformer-based line-level vulnerability prediction. In The 2022 International Conference on Mining Software Repositories (MSR). IEEE.

