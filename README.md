# InsideOut

InsideOut was built by Cole Graber-Mitchell, Ahlaam Abbasi, and Khe Le at HackHolyoke 2019.

## What is this?

InsideOut is a machine learning mood ring.
Just like the 50 cent rings that you can find in gift shops, InsideOut can tell you your mood---even if you don't know it!
Simply plug in an arduino with leds wired to pins 2 through 8, train the machine learning model, and run the program.
Using your webcam, InsideOut will scan your face and reveal your deepest secrets (well, maybe just if you're feeling one of the 7 emotions it supports).

Using Haar cascades to identify faces, then cropping those faces and feeding them into a custom convolutional neural network, InsideOut identifies the emotions of every face in view.
The CNN was trained on a dataset of about 30,000 faces linked in below (which also contains instructions for installation and running).
Each sample came with a 48x48 greyscale image of an emotive face and a number that identified one of 7 emotions.

## What inspired InsideOut?

Everyone in our group was fascinated by computer vision, especially when it's used on humans.
Emotions are extraordinarily complex, yet we seem to understand them with ease.
This makes them a great candidate for machine learning.
Once we started thinking about recognizing emotions, we realized that we had the chance to build a real mood ring, with actual data behind it.
Unlike toy mood rings, this one actually works!

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
- model
- etc...
```

## Usage

### Jupyter

Most of the "workshopping code" is in Jupyter notebooks.
To run them, you must make sure that Jupyter can run python in the virtualenv.
Assuming you've installed the requirements as above, run

```sh
> ipython kernel install --user --name=venv
> jupyter lab
```

This will open up a new tab in your browser with the Jupyter instance.
Navigate to `./notebooks` and open up which ever you want to run.
Make sure that the kernel on the upper right of the editor says `venv` (switch it if it says `python3`) and you should be good to go.

### InsideOut proper

The application itself is in the `./src` directory.
First, run `python src/train.py`, which should train the neural network and store it in `./model/cnn.h5`.
Now, run `python src/main.py`, which will start the program.
If you have an arduino wired correctly and plugged into your computer, find its serial port and edit the end of `./src/main.py` with the required information.