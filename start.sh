echo "Wating for DB services to start"
python3 ../scripts/wait_for_db.py
echo "Initializing MongoDB"
python3 ../scripts/setup_mongo.py
echo "Initializing VectorDB"
python3 ../scripts/prep_vectordb.py
echo "Downloading Syntax Checking JAR"
curl "https://monticore.de/download/MCCD.jar" -o ../MCCD.jar -L -k &
echo "Starting App"
streamlit run app.py