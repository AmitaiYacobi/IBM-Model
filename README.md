# Assignment 1 IBM Models

For running the model you should run the following command **inside** IBM-Model directory:<br>
`python3 main.py`<br>
with the following flags:<br>
`-e /path/to/english_file`<br>
`-f /path/to/french_file`<br>
`-m model1 or model2`<br>
`-i number of iterations`<br>
`-n number of sentence pairs to train with (amount of data)`<br>

Running example - `python3 main.py -m model1 -f data/hansards.f -e data/hansards.e -i 15 -n 100000`<br>
There are default values for each flag in case you won't use it.
