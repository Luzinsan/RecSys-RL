psql -U postgres -d recsys -f src/scripts/setup_db.sql

for datafile in "datasets/prepared4db"/events_*.csv; do
  if [ -f "$datafile" ]; then
    echo "Загрузка файла: '$datafile'"

    psql -U postgres -d recsys -c "\\COPY e_commerce.events FROM '${datafile}' WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',');"

    if [ $? -ne 0 ]; then
      echo "Ошибка при загрузке файла '$datafile'. Скрипт прерван."
    else
      echo "Файл '$datafile' успешно загружен."
    fi
  else
     echo "Пропуск элемента: '$datafile' (не является файлом или не найден)"
  fi
done

echo "Загрузка данных завершена."

psql -U postgres -d recsys -c "ALTER TABLE e_commerce.events ADD PRIMARY KEY (event_time, product_id, user_session);"

psql -U postgres -d recsys -f src/scripts/preprocess_data.sql
