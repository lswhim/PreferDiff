

<div align='center'>
<h1>Preference Diffusion for Recommdation</h1>
 <a href='https://scholar.google.com/citations?user=sRoqbLwAAAAJ&hl=en'>Shuo Liu<sup>1</sup></a>,
 <a href='https://github.com/anzhang314'>An Zhang*</a>,
 <a href='https://hugo-chinn.github.io/'>Guoqing Hu</a>,
 <a href='https://faculty.ecnu.edu.cn/_s16/qh_en/main.psp'>Hong Qian</a>,
   <a href='https://www.chuatatseng.com/'>Tat-Seng Chua</a>,
(*Correspondence )
    <sup>1</sup>East China Normal University, <sup>2</sup>National University of Singapore, <sup>3</sup>University of Science and Technology of China
</div>





<img src='imgs/moti.svg' />

------

:smile_cat: Welcome to PreferDiff, this is a implementation of ***Preference Diffusion for Recommendation***







## :one:  ​ Guide for Running PreferDiff



### :walking_man: Single GPU 

```sh
python main.py --model=PDSRec --sd=O --td=O --loss_type=cosine  --lamda=0.4 --w=2 --hidden_size=3072 
```



### :runner: Multi-GPU

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port=12330 main.py --model=PDSRec --sd=O --td=O --loss_type=cosine  --lamda=0.4 --w=2 --hidden_size=3072 
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

