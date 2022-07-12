# BadHash

The implementation of our ACM MM 2022 paper "**BadHash: Invisible Backdoor Attacks against Deep Hashing with Clean Label**" 

## Abstract

Due to its powerful feature learning capability and high efficiency, deep hashing has achieved great success in large-scale image retrieval. Meanwhile, extensive works have demonstrated that deep neural networks (DNNs) are susceptible to adversarial examples, and exploring  adversarial attack against deep hashing has attracted many research efforts. Nevertheless, backdoor attack, another famous threat to DNNs, has not been studied for deep hashing yet. Although various backdoor attacks have been proposed in the field of image classification, existing approaches failed to realize a truly imperceptive backdoor attack that enjoys invisible triggers and clean label setting simultaneously, and they cannot meet the intrinsic demand of image retrieval backdoor. 

In this paper, we propose BadHash, the first  imperceptible backdoor attack against deep hashing, which can effectively generate invisible and input-specific poisoned images with clean label. We first propose a new conditional generative adversarial network (cGAN) pipeline to effectively generate poisoned samples. For any given benign image, it seeks to generate a natural-looking poisoned counterpart with a unique invisible trigger. In order to improve the attack effectiveness, we introduce a label-based contrastive learning network LabCLN to exploit the semantic characteristics of different labels, which are subsequently used  for confusing and misleading the target model to learn the embedded trigger. We finally explore the mechanism of backdoor attacks on image retrieval in the hash space. Extensive experiments on multiple benchmark datasets verify that BadHash can generate imperceptible poisoned samples with strong attack ability and transferability over state-of-the-art deep hashing schemes.

## Requirements   

- python 
- torch==1.8.0
- torchvision==0.9.0
- lpips==0.1.4

## Method

### Train Trigger Genrator

Before running the code, please download the trained deep hash mode from [Baidudisk](https://pan.baidu.com/s/1hMZa54IKA0HAhz7s-SILcQ) (code:d78r), and put them into `checkpoint` folder.

```shell 
# the detailed attack pipeline implementation.
python train_tri_gen.py
```

### Generate Poisoned Samples

```shell 
# use the trigger generator to generator poisoned samples.
python implement_tri_gen.py
```

Write the paths of the generated poisoning samples into the training set to create the poisoning dataset. 

### Backdoor Training

```shell 
# train the backdoored model.
python CSQ.py
```
