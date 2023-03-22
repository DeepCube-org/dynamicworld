# DynamicWorld Wrapper

## Description

Python Package to wrap the DynamicWorld model.

Classes:

- Water
- Trees
- Grass
- Crops
- Shrub & Scrub
- Flooded Vegetation
- Built-Up Area
- Bare Ground
- Snow & Ice

## Example

./dynamicworld/examples/usage.ipynb

## Installation

### PIP
```
pip install git+https://github.com/DeepCube-org/dynamicworld.git
```

### Docker
```
docker build -f Dockerfile -t dynamicworld .
```
### Clouds

The parameter ```cloud``` in the inference script allows the "blending" of the DynamicWorld output and the output of the s2cloudless model. 
DynamicWorld was trained considering only sentinel-2 images without clouds... it means that cloudy images could be condiered OOD for it. 
So we could assume that the DynamicWorld model has been trained to approximate the lulc distribution of cloudless images (C=0):

$$DW(x, y) \approx p(Y=y|X=x, C=0)$$

But we would like to model the unconditional distribution:

$$p(Y=y|X=x)$$

Assuming that it is ok to have an uniform distribution over the lulc classes if C=1:

$$p(Y=y|X=x,C=1)=\frac{1}{K}$$

And assuming that the model:

$$p(C=c|X=x)$$

is approximated by another model (s2cloudless in our case):

$$s2cloudless(x) \approx p(C=1|X=x)$$

We can approximatge the desidered distribution by:
```math
\begin{align}
p(Y=y|X=x) & = \sum\limits_{c=0}^{1}p(Y=y,C=c|X=x) = \\
& = \sum\limits_{c=0}^{1}\frac{p(Y=y,C=c,X=x)}{p(X=x)} = \\
& = \sum\limits_{c=0}^{1}p(Y=y|C=c,X=x)p(C=c|X=x) = \\
& = p(Y=y|C=0,X=x)p(C=0|X=x) + p(Y=y|C=1,X=x)p(C=1|X=x) \approx \\
& \approx DW(x, y)(1-s2cloudless(x))+\frac{1}{K}s2cloudless(x)
\end{align}
```
