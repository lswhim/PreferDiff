

<div align='center'>
<h1>Preference Diffusion for Recommdation Submitted to ICLR 2025</h1>
</div>



<img src='imgs/moti.svg' />

------

:smile_cat: Welcome to PreferDiff, this is a implementation of ***Preference Diffusion for Recommendation***







## Running Guide for PreferDiff

## Single-GPU

```sh
python main.py --model=PDSRec --sd=O --td=O --loss_type=cosine  --lamda=0.5 --hidden_size=3072 
```

### Multi-GPU

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port=12330 main.py --model=PDSRec --sd=O --td=O --loss_type=cosine  --lamda=0.5 --hidden_size=3072 
```









