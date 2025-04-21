psql -U postgres -d recsys -f src/scripts/setup_db.sql
psql -U postgres -d recsys -f src/scripts/preprocess_data.sql
