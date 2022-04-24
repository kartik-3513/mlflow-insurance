# MLFLOW PIPELINE

MLFlow is an open source end to end machine learning lifecycle management tool.
It has 3 core features:-

### MLFlow Tracking

It allows us to run experiments on our dataset and model while it keeps track of each run. It stores the parameters used, metrics achieved, artifacts generated and even the model for each run as specified in the code. It also records the source file and the user that executed the code. These informations can be presented in a convient way through the UI provided by the tracking server.

To run the tracking server on the localhost execute `mlflow ui` on the commandline

### MLFlow Projects

MlFlow projects allow data scientists to share the project configuration and dependencies with each other such that the results obtained on any environment or system is always the same.

We define a file named MLProject that specifies the environment in which the project should execute, what are the possible entry points into the projects (that can be run to start experiments), the parameters these entry points require and the commands to execute these.

The conda environmnet specifies the dependencies needed to execute the experiment. When an experiment is run, firstly, conda creates this environment with the specified dependencies and then executes the code within this environment.

To run an expriment of the project from a particular entry point execute `mlflow run .`. This will run the main entry point with default parameters. If we want to add parameters we use the -P flag, like `mlfow run . -P alpha=0.2`.

### MLFlow Models

MlFlow models provided a serialized binary version of ML models from various libraries. These models are highly reliable (consistent in their results) accross deployment platforms because they are run in the specified conda environment.

To deploy a model to localhost run `mlflow models serve --model-uri runs:/e6814e4514f24189942e80148291899b/model -p 3000`. f97b6a22eef24fb291f37c72b80057e8 is the runid of the model we chose to deploy.

After deploying, send a curl request using: 
curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["age","bmi","children","sex_female","sex_male","smoker_no","smoker_yes","region_northeast","region_northwest","region_southeast","region_southwest"],"data":[[20, 23.3, 2, 0, 1, 0, 1, 1, 0, 0, 0]]}' http://127.0.0.1:3000/invocations
