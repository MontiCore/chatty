# SE-CHATTY
The main branch of SE-Chatty is deployed at:

https://studentprojects.se-rwth.de/se-chatty-dev/

The dev branch of SE-Chatty is deployed at:

https://studentprojects.se-rwth.de/se-chatty/

## Running Locally
### Requirements:
* Docker needs to be installed to run the database services.
* Create a ``.env`` file in the src directory (or directly set environmental variables)
```
ANONYMIZED_TELEMETRY=False

LUKNET_OPENAI_KEY=
LUKNET_OPENAI_ENDPOINT=https://luknet.openai.azure.com/

SWEDEN_OPENAI_KEY=
SWEDEN_OPENAI_ENDPOINT=https://se-sweden.openai.azure.com/

SCIEBO_UNAME=
SCIEBO_PSWD=
```
* When starting SE-Chatty without docker use a python version in the range of 3.8-3.11 (3.10 is confirmend working). 
Then run the following command to setup the databases and install the requirements (for Windows):

``call setup_local.bat`` 

## Run app:
Start with docker:

``docker compose up --build``

SE-Chatty can run without docker, which is generally faster. 
Make sure the database containers are running (`docker-compose up -d mongodb weaviate`) and all scripts in the `setup_local.bat` have run succesfully.
Then in the `src` folder run:

``streamlit run app.py -- --local``

## Common problems:
* Some IDEs change the linebreaks of the `start.sh` file, this will result in the scripts failing when executed in docker.
Change this in the IDE settings if necessary.
