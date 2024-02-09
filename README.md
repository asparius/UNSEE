## Install dependencies and then build the library


```
pip install -r requirements.txt
pip install -e .

```` 

## Training

````
python train.py
````
## Model Weights
You can access model weights from [Hugging Face](https://huggingface.co/asparius).

## Running MTEB
Running STS benchmark in MTEB for VICReg.

````
python run_mteb_english.py
````
## BibTex Citation
Please cite the following works if you use this repository.

````
@misc{çağatan2024unsee,
      title={UNSEE: Unsupervised Non-contrastive Sentence Embeddings}, 
      author={Ömer Veysel Çağatan},
      year={2024},
      eprint={2401.15316},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}


````



## Acknowledgements

Repository is adapted from the repo of the ACL 21 paper [Bootstrapped Unsupervised Sentence Representation Learning](https://github.com/yanzhangnlp/BSL/tree/main) 
