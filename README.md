# CoGen
<div align=center><img src="https://github.com/zaixizhang/CoGen/blob/main/flow generation.png" width="700"/></div>

## Abstract   
Designing molecules with desirable physicochemical and pharmacological properties is a long-standing challenge in chemistry
and drug discovery. Recently, machine learning-based generative models have emerged as a powerful approach to generate
novel molecules with desired properties. However most existing approaches either treat molecules as 2D graphs (atoms as
nodes and covalent bonds as edges) or 3D point clouds (only contains atom types and 3D coordinates), lacking a unified
modeling. In this work, we present CoGen, a SE(3)-equivariant model that Concurrently Generates 2D molecular graphs and
3D structures. Specifically, CoGen sequentially generates atom types, chemical bonds, and 3D coordinates. An equivariant
3D graph neural network and a novel multi-head self-attention module with bond encodings are used to extract conditional
information from intermediate 3D graphs. In the generation process, chemical rules are leveraged to check whether the
generated 2D graphs are valid. Extensive experimental results show that CoGen outperforms previous methods on random
molecular geometry generation, targeted molecule discovery, and conditional molecular conformation generation tasks. Our
investigation demonstrates that the information of the 2D graphs and 3D structures are intrinsically complementary, and the
best molecular generation performance can only be obtained when both are considered.

## Requirements

```
matplotlib==3.1.1
numpy==1.17.1
torch==1.2.0
scipy==1.3.1
networkx==2.4
tqdm==4.47.0
pickle==0.7.5
```

## Datasets
Download QM9 data from https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz, unzip the file and put the file gdb9.sdf under the folder qm9/

## Run the code  
```
git clone https://github.com/zaixizhang/CoGen.git
cd CoGen
sh train.sh 
```
