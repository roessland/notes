# notes
Jupyter Notebooks, mostly. Physics and numerical analysis.


## Demos
Quad spring animation using BDF solver for stiff ODEs.

<video src="https://github.com/roessland/notes/assets/210712/7a414b21-2cdc-44f9-8996-d3c8af16b377" controls="controls" style="max-width: 730px; " autoplay muted>
</video>



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
