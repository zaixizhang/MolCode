# MolCode
<div align=center><img src="https://github.com/zaixizhang/MolCode/blob/main/flow generation.png" width="700"/></div>

## Abstract   
Designing molecules with desirable physiochemical properties and functionalities is a long-standing challenge in chemistry,
material science, and drug discovery. Recently, machine learning-based generative models have emerged as promising
approaches for de novo molecule design. However, further refinement of methodology is highly desired as most existing
methods lack unified modeling of 2D topology and 3D geometry information and fail to efficiently learn the structure-property
relationship for molecule generation. Here we present MolCode, a roto-translation equivariant generative framework for
Molecular graph-structure Co-design. In MolCode, 3D geometric information is leveraged for molecular 2D graph generation,
which in turn helps guide the prediction of molecular 3D structure. Extensive experimental results show that MolCode
outperforms previous methods on a series of tasks including de novo molecule design, targeted molecule discovery, and
structure-based drug design. Particularly, MolCode not only consistently generates valid (99.95% Validity) and diverse (98.75%
Uniqueness) molecular graphs/structures with desirable properties, but also generate drug-like molecules with high affinity
to target proteins (61.8% high affinity ratio), which demonstrate MolCode’s potential applications in material design and drug
discovery. Our investigation reveals that the 2D topology and 3D geometry contain intrinsically complementary information in
molecule generation, and provides new insights into machine learning-based molecule representatio

## Datasets
Download QM9 data from https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz, unzip the file and put the file gdb9.sdf under the folder qm9/

The CrossDocked2020 datasets is public available at https://drive.google.com/drive/folders/1CzwxmTpjbrt83z_wBzcQncq84OVDPurM

## Run the code  
```
git clone https://github.com/zaixizhang/MolCode.git
cd MolCode
sh start.sh 
```
