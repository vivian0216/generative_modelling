# generative_modelling
The project for course DSAIT4030: Generative Modelling.

## Setup
1) To setup the project install uv, from this site https://docs.astral.sh/uv/getting-started/installation/.
2) Run the following command in the terminal of the project to set up a venv:
```
cd generative_modelling
uv venv
```
3) To install all the necessary requirements, run the following:
```
uv pip install .
```
4) To run the scripts:
```
uv run src/script.py
```
5) To update the venv:
```
uv pip install . --upgrade
```


## Evaluation between JEM and baseline model (JEM0)
1) Make sure to have the two models trained.
```
uv run src/train2.py --models CNN       #for the baseline JEM0
uv run src/train2.py --models CCF       #for the JEM model
```
2) Move jem_final_model_pth to map evaluation_models\jem or jem0 respectively.
3) To evaluate the JEM model witn a baseline model (JEM0) without an energy block, you will have to run the following commands:
```
uv run src/evaluation.py
```

