

<div align='center'>
<h1>Preference Diffusion for Recommdation (ICLR 2025)</h1>
 <a href='https://scholar.google.com/citations?user=sRoqbLwAAAAJ&hl=en'>Shuo Liu<sup>1,2</sup></a>,
 <a href='https://github.com/anzhang314'>An Zhang<sup>2*</sup></a>,
 <a href='https://hugo-chinn.github.io/'>Guoqing Hu<sup>3</sup></a>,
 <a href='https://faculty.ecnu.edu.cn/_s16/qh_en/main.psp'>Hong Qian<sup>1</sup></a>,
   <a href='https://www.chuatatseng.com/'>Tat-Seng Chua<sup>2</sup></a>,
    <br>
    <sup>1</sup>East China Normal University, <sup>2</sup>National University of Singapore, 
    <br>
    <sup>3</sup>University of Science and Technology of China, (*Correspondence )
</div>







<img src='imgs/moti.svg' />

------

:smile_cat: Welcome to PreferDiff, this is a implementation of ***Preference Diffusion for Recommendation***







## :one:  ​ Guide for Running PreferDiff



### :walking_man: Single GPU 

```sh
python main.py --model=PreferDiff --sd=O --td=O --loss_type=cosine  --lamda=0.4 --w=2 --hidden_size=3072  --ab=iids
```



### :runner: Multi-GPU

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port=12330 main.py --model=PreferDiff --sd=O --td=O --loss_type=cosine  --lamda=0.4 --w=2 --hidden_size=3072 --ab=iids
```



## :two: Best Hyperparameters

| Dataset    | learning rate | Weight Decay | lambda | w    | Embedding Size |
| ---------- | ------------- | ------------ | ------ | ---- | -------------- |
| **Sports** | 1e-4          | 0            | 0.4    | 2    | 3072           |
| **Beauty** | 1e-4          | 0            | 0.4    | 8    | 3072           |
| **Toys**   | 1e-4          | 0            | 0.6    | 6    | 3072           |



## :three: Guide for Running Baselines



```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port=12330 main.py --model=SASRec --sd=O --td=O 
```

# Bib

@article{Liu2024PreferDiff,
  title={Preference Diffusion for Recommendation},
  author={Liu, Shuo and Zhang, An and Hu, Guoqing and Qian, Hong and Chua, Tat-seng},
  journal={arXiv preprint arXiv:2410.13117},
  year={2024}
}