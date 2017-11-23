# Having Fun with Recurrent Neural Networks (RNNs)
(Tutorial for PyData 2017 by Sergio G. Burdisso - sergio.burdisso@gmail.com)

**Description:** https://pydata.org/sanluis2017/schedule/presentation/3/

**Static Slides:** http://tworld-ai.com/extra/pydata/recurrently-happy-rnn.slides.html


---
### First things first
Make sure you have installed the following before opening the notebook.
* **[SciPy Stack](https://www.scipy.org/install.html)**
* **[TensorFlow](https://www.tensorflow.org/install/) >= 1.0**: (Tested on version 1.3.0. - _**GPU version recommended**_)
* **[RISE Slideshow Extension](https://github.com/damianavila/RISE)**: This notebook is meant to be shown as an Slideshow, install RISE using these commands:
```
~$ pip install RISE
~$ jupyter-nbextension install rise --py --sys-prefix
~$ jupyter-nbextension enable rise --py --sys-prefix
```
* **ABC player**: to be able to play ABC songs, we'll use [abcmidi](http://abc.sourceforge.net/abcMIDI/original/) and [timidity](https://sfxpt.wordpress.com/2015/02/02/how-to-play-midi-files-under-ubuntu-linux/) commands. Try to install them on ubuntu by:
```
~$ sudo apt install abcmidi
~$ sudo apt-get install timidity timidity-interfaces-extra
```

* **Finally**, you should download the needed files:

1. Download the **dataset** from here: [dataset.zip](http://tworld-ai.com/extra/pydata/dataset.zip)
2. Download the **pretrained models** from here: [trained_models.zip](http://tworld-ai.com/extra/pydata/trained_models.zip)
3. Extract those .zip files inside the notebook directory, so that you end up with something like this:
```
recurrently-happy-rnn
    ├──dataset
    ├──trained_models
    ├──images
    ├──recurrently-happy-rnn.ipynb
    ...
```
4. Run the Jupyter notebook inside this directory (otherwise, images aren't gonna display properly).

And that's it.
Enjoy, have fun and happy hacking! :D
