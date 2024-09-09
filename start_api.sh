echo "Wating for DB services to start"
python3 ../scripts/wait_for_db.py
echo "Starting API"
python3 api.py
