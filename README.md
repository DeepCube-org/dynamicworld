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

It is possible to address clouds in two different ways by the ```cloud``` parameter.
If ```cloud = None```, the cloud prob is not computed and it is not used in the inference process. So the result that is returned is the same of the DynamicWorld model. If ```cloud = 'mix'``` the Cloud Mixing strategy is used, if ```cloud = 'add'``` the Cloud Adding strategy is used.

#### Cloud Mixing
The parameter ```cloud``` in the inference script allows the "blending" of the DynamicWorld output and the output of the s2cloudless model. 
DynamicWorld was trained considering only sentinel-2 images without clouds... it means that cloudy pixels could be condiered OOD for it. 
So we could assume that the DynamicWorld model has been trained to approximate the lulc distribution of cloudless pixels (C=0):

$$DW(x, y) \approx p(Y=y|X=x, C=0)$$

But we would like to model the unconditional distribution:

$$p(Y=y|X=x)$$

Assuming that it is ok to have an uniform distribution over the lulc classes (1,...K) if the pixel is cloudy (C=1):

$$p(Y=y|X=x,C=1)=\frac{1}{K}$$

And assuming that the model:

$$p(C=c|X=x)$$

is approximated by another model (s2cloudless in our case):

$$s2cloudless(x) \approx p(C=1|X=x)$$

We can approximatge the desidered distribution by:
```math
\begin{align}
p(Y=y|X=x) & = p(Y=y,C=0|X=x) + p(Y=y,C=1|X=x) = \\
& = \frac{p(Y=y,C=0,X=x)}{p(X=x)} + \frac{p(Y=y,C=1,X=x)}{p(X=x)} = \\
& = p(Y=y|C=0,X=x)p(C=0|X=x) + p(Y=y|C=1,X=x)p(C=0|X=x) \approx \\
& \approx DW(x, y)(1-s2cloudless(x))+\frac{1}{K}s2cloudless(x)
\end{align}
```
#### Cloud Adding
A cloud class is added to the output of the DynamicWorld model in a very simple way:
```math
\begin{align}
p(Y=c|X=x) & \approx s2cloudless(x)\\
p(Y=y|X=x) & \approx (1-s2cloudless(x))DW(x, y) \forall y != c
\end{align}
```
The idea is very simple, the probability of the cloud class is defined by ```s2cloudless(x)```, the rest of the probability mass ```1-s2cloudless(x)``` is distributed in the other classes according to the output of the DynamicWorld model. 