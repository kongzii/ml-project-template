# Machine Learning project template

This repository could serve as a starting point for your machine learning projects. 
It contains ready to train (Keras) and serve (FastAPI) model implementation for the IMDB Movie Reviews dataset.

Using a few simple commands, you can fire up `MLFlow`, `TensorBoard`, `Jupyter`, `training`, or `serving` the model. 
Metrics are automatically logged in `MLFlow` and `TensorBoard`.

## Requirements

- make **
- docker
- docker-compose

** Make is used only to shortcut docker-compose commands in `Makefile`.

## TLDR

Run the following commands.

```
make dataset
make train
make test
make serve
```

And then, check results at http://0.0.0.0:8080/ (MLFlow) or play with API at http://localhost:8000/docs (FastAPI).

## Usage details

### Tensorboard

Start with `make tensorboard`, or it will be started automatically with other components.

Available at http://0.0.0.0:8070/.

### MLFlow

Start with `make mlflow`, or it will be started automatically with other components.

Available at http://0.0.0.0:8080/.

### Jupyter

Start with `make jupyter`.

Available at http://0.0.0.0:8888/.

### Serve via FastAPI

After training is complete, run `make serve`. This will start the server at http://localhost:8000. 

You can use it to easily inference with models:

API documentation can be found at http://localhost:8000/docs (you can even make requests there).

### Clean experiment

Run `make clean`, this will stop running containers and delete all created logs, files, etc.
