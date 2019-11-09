# hackholyoke 2019

## Installation

First, make sure that Python 3 is installed and working on your computer.
(Try running `python --version` and seeing the output.)
Next, clone this repo and `cd` into the new directory. Then, run 

```sh
> python -m venv . && source ./bin/activate && pip install -r requirements.txt
```

Now, you have to download the [dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) and unarchive it into the `./dataset` directory.
Your organization should look like

```
- dataset
| - fer2013.csv
- src
- .gitignore
- LICENSE
- README.md
- requirements.txt