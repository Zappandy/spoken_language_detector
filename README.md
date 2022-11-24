# Project

## Dataset

[spoken language dataset](https://www.kaggle.com/datasets/toponowicz/spoken-language-identification?resource=download). Be mindful that it's a 16 GB dataset

## Extra optional tools

You can check this thread of [jupyter lab vs. jupyter notebooks](https://stackoverflow.com/questions/50982686/what-is-the-difference-between-jupyter-notebook-and-jupyterlab)
```shell
pip3 install jupyterlab
```

```shell
pip3 install notebook
```

These tools were only used to run .ipynb files and facilitate visualizations!

[Jupyterlab guide](https://jupyter.org/install)

Python script to jupyter-notebook [converter](https://laptrinhx.com/convert-python-script-to-jupyter-notebook-and-vice-versa-1653154340/)

## Exploratory steps

How did we clean up the files?

```bash
ls <dir> | grep -o '.....$' | uniq
<dir> | grep -o '^es.*'  # finds the spanish ones
```
For our work, we used the test set found in local dirs such as
```
/media/andres/2D2DA2454B8413B5/test/test/
```

The final version is the file\_cleaner script found in this dir. That one copies the Spanish files to a new given dir as its second argument

## Theory

Tutorial on [mel spectograms](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53)
