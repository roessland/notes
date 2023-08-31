# notes
Jupyter Notebooks, mostly. Physics and numerical analysis.

![](./quad-spring.mov)

## Setup

```shell
mamba env create -f environment.yml
```

## Cleanup or refresh dependencies

```shell
mamba env remove -n notes
mamba env create -f environment.yml
```

## Keep working

```shell
mamba activate notes
```

## Save changes to environment

```shell
mamba env export > environment.yml
```