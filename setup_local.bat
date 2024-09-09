docker-compose up -d mongodb weaviate
pip install -r requirements.txt
cd src
python ../scripts/wait_for_db.py --local
python ../scripts/setup_mongo.py --local
python ../scripts/prep_vectordb.py --force
curl "https://monticore.de/download/MCCD.jar" -o ../MCCD.jar -L -k &
streamlit run app.py -- --local