SET search_path TO e_commerce, public;
DROP TABLE IF EXISTS e_commerce.events_processed;

-- Создаем новую таблицу для хранения обработанных событий
CREATE TABLE e_commerce.events_processed (
    -- Оригинальные колонки
    event_time TIMESTAMP NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    product_id INTEGER NOT NULL,
    category_id BIGINT,
    category_code VARCHAR(40),
    brand VARCHAR(20),
    price REAL,
    user_id BIGINT NOT NULL,
    user_session VARCHAR(36) NOT NULL,
    -- Исторические фичи
    session_event_num INTEGER NOT NULL, -- Номер события в сессии
    user_global_event_num BIGINT NOT NULL, -- Глобальный номер события пользователя
    user_views_before BIGINT, -- Просмотры пользователя до тек. события
    user_carts_before BIGINT, -- Корзины пользователя до тек. события
    user_purchases_before BIGINT, -- Покупки пользователя до тек. события
    product_views_before BIGINT, -- Просмотры продукта до тек. события
    product_purchases_before BIGINT, -- Покупки продукта до тек. события
    product_avg_price REAL NOT NULL, -- Накопл. средняя цена продукта (вкл. тек. событие)
    category_views_before BIGINT, -- Просмотры категории до тек. события
    category_avg_price REAL NOT NULL, -- Накопл. средняя цена категории (вкл. тек. событие)
    -- Простые временные фичи
    hour_sin REAL NOT NULL,
    hour_cos REAL NOT NULL,
    day_of_week_sin REAL NOT NULL,
    day_of_week_cos REAL NOT NULL,
    day_sin REAL NOT NULL,
    day_cos REAL NOT NULL,
    month_sin REAL NOT NULL,
    month_cos REAL NOT NULL,
    year INTEGER NOT NULL,
    is_weekend INTEGER NOT NULL,
    -- Фичи праздников
    is_holiday INTEGER NOT NULL,
    days_to_next_holiday INTEGER NOT NULL,
    days_from_last_holiday INTEGER NOT NULL,
    holiday_name VARCHAR(50)
);





INSERT INTO e_commerce.events_processed (
    event_time, event_type, product_id, category_id, category_code, brand, price, user_id, user_session,
    session_event_num, user_global_event_num,
    user_views_before, user_carts_before, user_purchases_before,
    product_views_before, product_purchases_before, product_avg_price,
    category_views_before, category_avg_price,
    hour_sin, hour_cos, day_of_week_sin, day_of_week_cos, day_sin, day_cos, month_sin, month_cos, year, is_weekend,
    is_holiday, days_to_next_holiday, days_from_last_holiday, holiday_name
)
WITH FilteredEvents AS (
    -- Сначала фильтруем сессии, оставляя те, где > 2 событий
    SELECT *
    FROM e_commerce.events
    WHERE user_session IN (
        SELECT user_session
        FROM e_commerce.events
        GROUP BY user_session
        HAVING COUNT(*) > 2
    )
)
SELECT
    event_time,
    event_type,
    product_id,
    category_id,
    category_code,
    brand,
    price,
    user_id,
    user_session,
    -- Статистики пользователя
    -- Номер события внутри сессии
    ROW_NUMBER() OVER (PARTITION BY user_session ORDER BY event_time) AS session_event_num,
    -- Номер события пользователя за всю историю
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_time) AS user_global_event_num,
    -- Рассчитываем количество просмотров пользователя ДО текущего события
    SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) OVER user_window AS user_views_before,
    -- Рассчитываем количество добавлений в корзину пользователя ДО текущего события
    SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) OVER user_window AS user_carts_before,
    -- Рассчитываем количество покупок пользователя ДО текущего события
    SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) OVER user_window AS user_purchases_before,
    -- Статистики продукта ДО текущего события
    -- Количество просмотров продукта ДО текущего события
    SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) OVER product_window AS product_views_before,
    -- Количество покупок продукта ДО текущего события
    SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) OVER product_window AS product_purchases_before,
    -- Накопленная средняя цена продукта, ВКЛЮЧАЯ текущее событие
    AVG(price) OVER (
        PARTITION BY product_id
        ORDER BY event_time
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS product_avg_price,
    -- Статистики категории
    -- Количество просмотров категории ДО текущего события
    SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) OVER category_window AS category_views_before,
    -- Накопленная средняя цена в категории (ВКЛЮЧАЯ текущее событие)
    AVG(price) OVER (
        PARTITION BY category_code
        ORDER BY event_time
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS category_avg_price,
    -- Простые временные фичи
    SIN(2 * PI() * EXTRACT(HOUR FROM fe.event_time) / 24.0) AS hour_sin,
    COS(2 * PI() * EXTRACT(HOUR FROM fe.event_time) / 24.0) AS hour_cos,
    SIN(2 * PI() * EXTRACT(ISODOW FROM fe.event_time) / 7.0) AS day_of_week_sin, -- ISODOW: 1=Пн..7=Вс
    COS(2 * PI() * EXTRACT(ISODOW FROM fe.event_time) / 7.0) AS day_of_week_cos,
    SIN(2 * PI() * EXTRACT(DAY FROM fe.event_time) / 31.0) AS day_sin,
    COS(2 * PI() * EXTRACT(DAY FROM fe.event_time) / 31.0) AS day_cos,
    SIN(2 * PI() * EXTRACT(MONTH FROM fe.event_time) / 12.0) AS month_sin,
    COS(2 * PI() * EXTRACT(MONTH FROM fe.event_time) / 12.0) AS month_cos,
    EXTRACT(YEAR FROM fe.event_time)::INTEGER AS year,
    CASE WHEN EXTRACT(ISODOW FROM fe.event_time) IN (6, 7) THEN 1 ELSE 0 END AS is_weekend,
    -- Фичи праздников
    CASE WHEN date IS NOT NULL THEN 1 ELSE 0 END AS is_holiday,
    -- Используем COALESCE для замены NULL (нет след./пред. праздника) на большое число
    COALESCE(next_holiday_date - CAST(event_time AS DATE), 9999) AS days_to_next_holiday,
    COALESCE(CAST(event_time AS DATE) - prev_holiday_date, 9999) AS days_from_last_holiday,
    holiday_name
FROM
    FilteredEvents fe
-- Присоединяем название праздника (если день праздничный)
LEFT JOIN
    e_commerce.holidays rh ON CAST(event_time AS DATE) = rh.date
-- Находим следующую дату праздника
LEFT JOIN LATERAL (
    SELECT MIN(h.date) as next_holiday_date
    FROM e_commerce.holidays h
    WHERE h.date > CAST(event_time AS DATE)
) next_h ON true
-- Находим предыдущую дату праздника
LEFT JOIN LATERAL (
    SELECT MAX(h.date) as prev_holiday_date
    FROM e_commerce.holidays h
    WHERE h.date < CAST(event_time AS DATE)
) prev_h ON true
WINDOW
	-- Окно для пользовательских фичей (ДО текущего события)
    user_window AS (PARTITION BY user_id ORDER BY event_time ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
    -- Окно для продуктовых фичей просмотров/покупок (ДО текущего события)
    product_window AS (PARTITION BY product_id ORDER BY event_time ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
    -- Окно для фичей категорий (ДО текущего события)
    category_window AS (PARTITION BY category_code ORDER BY event_time ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
ORDER BY
    user_id, event_time;



-- Добавляем Primary Key ПОСЛЕ вставки данных (быстрее)
ALTER TABLE e_commerce.events_processed
ADD PRIMARY KEY (event_time, product_id, user_session);

-- Индексы на events_processed для ускорения следующего шага
CREATE INDEX IF NOT EXISTS idx_processed_brand ON e_commerce.events_processed (brand); -- Для DENSE_RANK
CREATE INDEX IF NOT EXISTS idx_processed_user_session ON e_commerce.events_processed (user_session); -- Для DENSE_RANK
CREATE INDEX IF NOT EXISTS idx_processed_holiday_name ON e_commerce.events_processed (holiday_name); -- Для DENSE_RANK
CREATE INDEX IF NOT EXISTS idx_processed_event_time ON e_commerce.events_processed (event_time); -- по приколу


DROP TABLE IF EXISTS e_commerce.events_encoded;
-- Создаем финальную таблицу для закодированных и масштабированных данных
CREATE TABLE e_commerce.events_encoded (
    event_time TIMESTAMP NOT NULL, -- оставляю для первичного ключа
    price REAL NOT NULL,
    -- Масштабированные исторические фичи (из events_processed)
    session_event_num REAL NOT NULL,
    user_global_event_num REAL NOT NULL,
    user_views_before REAL,
    user_carts_before REAL,
    user_purchases_before REAL,
    product_views_before REAL,
    product_purchases_before REAL,
    product_avg_price REAL NOT NULL,
    category_views_before REAL,
    category_avg_price REAL NOT NULL,
    -- Масштабированные простые временные фичи(из events_processed)
    hour_sin REAL NOT NULL,
    hour_cos REAL NOT NULL,
    day_of_week_sin REAL NOT NULL,
    day_of_week_cos REAL NOT NULL,
    day_sin REAL NOT NULL,
    day_cos REAL NOT NULL,
    month_sin REAL NOT NULL,
    month_cos REAL NOT NULL,
    year REAL NOT NULL,
    is_weekend REAL NOT NULL,
    -- Масштабированные фичи праздников
    is_holiday REAL NOT NULL,
    days_to_next_holiday REAL NOT NULL,
    days_from_last_holiday REAL NOT NULL,
    -- holiday_name INTEGER NOT NULL,

    -- Закодированные категориальные признаки 
    -- (или не закодированные, но сам факт, что категориальные и не скейлятся))
    -- Оригинальные колонки
    -- единственный незакодированный признак, 
    -- чтобы во время обучения можно было подобрать маппинг для reward
    event_type VARCHAR(20) NOT NULL, -- reward
    product_id INTEGER NOT NULL,
    category_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    brand INTEGER NOT NULL, -- Label Encoding
    user_session BIGINT NOT NULL, -- Label Encoding
    holiday_name INTEGER NOT NULL, -- Label Encoding
    product_id_idx INTEGER NOT NULL -- Embedding слой
    
);


-- Вставляем данные с кодированием и масштабированием
INSERT INTO e_commerce.events_encoded (
    -- Перечисляем ВСЕ колонки целевой таблицы events_encoded в нужном порядке
    event_time, price,
    session_event_num, user_global_event_num, user_views_before, user_carts_before, user_purchases_before,
    product_views_before, product_purchases_before, product_avg_price, category_views_before, category_avg_price,
    hour_sin, hour_cos, day_of_week_sin, day_of_week_cos, day_sin, day_cos, month_sin, month_cos, year, is_weekend,
    is_holiday, days_to_next_holiday, days_from_last_holiday,
    event_type, product_id, category_id, user_id, brand, user_session, holiday_name, product_id_idx
)
WITH SourceData AS (
    -- Читаем все из обновленной таблицы events_processed
    SELECT * FROM e_commerce.events_processed
),
EncodedFeatures AS (
    -- Рассчитываем целочисленные индексы для категориальных признаков
    SELECT
        sd.*, -- Выбираем все колонки из SourceData для использования ниже
        -- Кодируем brand
        DENSE_RANK() OVER (ORDER BY COALESCE(sd.brand, '__MISSING__')) AS brand_idx,
        -- Кодируем user_session (даже если это UUID, для целочисленного представления)
        DENSE_RANK() OVER (ORDER BY sd.user_session) AS user_session_idx,
        -- Кодируем holiday_name (добавленный из Python)
        DENSE_RANK() OVER (ORDER BY COALESCE(sd.holiday_name, '__MISSING__')) AS holiday_name_idx,
        -- category_code не кодируем, т.к. его нет в целевой таблице events_encoded (и он уже закодирован в category_id)
        -- product_id, category_id, user_id уже являются числовыми идентификаторами
        -- Кодируем product_id (для Embedding слоя)
        DENSE_RANK() OVER (ORDER BY sd.product_id) AS product_id_idx
    FROM SourceData sd
)
-- Финальный SELECT для вставки в events_encoded
SELECT
    -- Выбираем колонки в ТОЧНОМ СООТВЕТСТВИИ с порядком в INSERT INTO events_encoded
    ef.event_time,
    ef.price,
    -- Масштабированные числовые признаки (исходные+исторические)
    (ef.session_event_num::REAL - AVG(ef.session_event_num::REAL) OVER ()) / NULLIF(STDDEV_SAMP(ef.session_event_num::REAL) OVER (), 0), -- session_event_num REAL NOT NULL
    (ef.user_global_event_num::REAL - AVG(ef.user_global_event_num::REAL) OVER ()) / NULLIF(STDDEV_SAMP(ef.user_global_event_num::REAL) OVER (), 0), -- user_global_event_num REAL NOT NULL
    (ef.user_views_before::REAL - AVG(ef.user_views_before::REAL) OVER ()) / NULLIF(STDDEV_SAMP(ef.user_views_before::REAL) OVER (), 0), -- user_views_before REAL
    (ef.user_carts_before::REAL - AVG(ef.user_carts_before::REAL) OVER ()) / NULLIF(STDDEV_SAMP(ef.user_carts_before::REAL) OVER (), 0), -- user_carts_before REAL
    (ef.user_purchases_before::REAL - AVG(ef.user_purchases_before::REAL) OVER ()) / NULLIF(STDDEV_SAMP(ef.user_purchases_before::REAL) OVER (), 0), -- user_purchases_before REAL
    (ef.product_views_before::REAL - AVG(ef.product_views_before::REAL) OVER ()) / NULLIF(STDDEV_SAMP(ef.product_views_before::REAL) OVER (), 0), -- product_views_before REAL
    (ef.product_purchases_before::REAL - AVG(ef.product_purchases_before::REAL) OVER ()) / NULLIF(STDDEV_SAMP(ef.product_purchases_before::REAL) OVER (), 0), -- product_purchases_before REAL
    (ef.product_avg_price - AVG(ef.product_avg_price) OVER ()) / NULLIF(STDDEV_SAMP(ef.product_avg_price) OVER (), 0), -- product_avg_price REAL NOT NULL
    (ef.category_views_before::REAL - AVG(ef.category_views_before::REAL) OVER ()) / NULLIF(STDDEV_SAMP(ef.category_views_before::REAL) OVER (), 0), -- category_views_before REAL
    (ef.category_avg_price - AVG(ef.category_avg_price) OVER ()) / NULLIF(STDDEV_SAMP(ef.category_avg_price) OVER (), 0), -- category_avg_price REAL NOT NULL

    -- Масштабированные простые временные фичи
    (ef.hour_sin - AVG(ef.hour_sin) OVER ()) / NULLIF(STDDEV_SAMP(ef.hour_sin) OVER (), 0), -- hour_sin REAL NOT NULL
    (ef.hour_cos - AVG(ef.hour_cos) OVER ()) / NULLIF(STDDEV_SAMP(ef.hour_cos) OVER (), 0), -- hour_cos REAL NOT NULL
    (ef.day_of_week_sin - AVG(ef.day_of_week_sin) OVER ()) / NULLIF(STDDEV_SAMP(ef.day_of_week_sin) OVER (), 0), -- day_of_week_sin REAL NOT NULL
    (ef.day_of_week_cos - AVG(ef.day_of_week_cos) OVER ()) / NULLIF(STDDEV_SAMP(ef.day_of_week_cos) OVER (), 0), -- day_of_week_cos REAL NOT NULL
    (ef.day_sin - AVG(ef.day_sin) OVER ()) / NULLIF(STDDEV_SAMP(ef.day_sin) OVER (), 0), -- day_sin REAL NOT NULL
    (ef.day_cos - AVG(ef.day_cos) OVER ()) / NULLIF(STDDEV_SAMP(ef.day_cos) OVER (), 0), -- day_cos REAL NOT NULL
    (ef.month_sin - AVG(ef.month_sin) OVER ()) / NULLIF(STDDEV_SAMP(ef.month_sin) OVER (), 0), -- month_sin REAL NOT NULL
    (ef.month_cos - AVG(ef.month_cos) OVER ()) / NULLIF(STDDEV_SAMP(ef.month_cos) OVER (), 0), -- month_cos REAL NOT NULL
    (ef.year::REAL - AVG(ef.year::REAL) OVER ()) / NULLIF(STDDEV_SAMP(ef.year::REAL) OVER (), 0), -- year REAL NOT NULL
    (ef.is_weekend::REAL - AVG(ef.is_weekend::REAL) OVER ()) / NULLIF(STDDEV_SAMP(ef.is_weekend::REAL) OVER (), 0), -- is_weekend REAL NOT NULL

    -- Масштабированные фичи праздников
    (ef.is_holiday::REAL - AVG(ef.is_holiday::REAL) OVER ()) / NULLIF(STDDEV_SAMP(ef.is_holiday::REAL) OVER (), 0), -- is_holiday REAL NOT NULL
    
    (ef.days_to_next_holiday::REAL - AVG(ef.days_to_next_holiday:: REAL) OVER ()) / STDDEV_SAMP(ef.days_to_next_holiday::REAL) OVER (), -- days_to_next_holiday REAL NOT NULL
    (ef.days_from_last_holiday::REAL - AVG(ef.days_from_last_holiday:: REAL) OVER ()) / STDDEV_SAMP(ef.days_from_last_holiday::REAL) OVER (), -- days_from_last_holiday REAL NOT NULL

    -- Категориальные/ID признаки (некоторые закодированы, некоторые оригинальные)
    ef.event_type::VARCHAR,           -- reward
    ef.product_id::INTEGER,           -- product_id INTEGER NOT NULL (оригинальный)
    ef.category_id::BIGINT,           -- category_id BIGINT NOT NULL (оригинальный)
    ef.user_id::BIGINT,               -- user_id BIGINT NOT NULL (оригинальный)
    ef.brand_idx::INTEGER,            -- brand INTEGER NOT NULL (закодированный)
    ef.user_session_idx::BIGINT,      -- user_session BIGINT NOT NULL (закодированный)
    ef.holiday_name_idx::INTEGER,     -- holiday_name INTEGER NOT NULL (закодированный)
    ef.product_id_idx::INTEGER        -- product_id_idx INTEGER NOT NULL (для Embedding слоя)
FROM EncodedFeatures ef;

ALTER TABLE e_commerce.events_encoded
ADD PRIMARY KEY (event_time, product_id, user_session);

-- Основной индекс для загрузки последовательностей пользователя
CREATE INDEX IF NOT EXISTS idx_encoded_user_time ON e_commerce.events_encoded (user_id, event_time);
-- Индекс для возможной фильтрации по времени
CREATE INDEX IF NOT EXISTS idx_encoded_event_time ON e_commerce.events_encoded (event_time);
-- Индекс по ID сессии (используем закодированный ID)
CREATE INDEX IF NOT EXISTS idx_encoded_session_idx ON e_commerce.events_encoded (user_session);
-- Индекс по ID продукта
CREATE INDEX IF NOT EXISTS idx_encoded_product_id ON e_commerce.events_encoded (product_id);
-- Индекс по ID пользователя
CREATE INDEX IF NOT EXISTS idx_encoded_user_id ON e_commerce.events_encoded (user_id);
-- Индекс по ID продукта для Embedding слоя
CREATE INDEX IF NOT EXISTS idx_encoded_product_id_idx ON e_commerce.events_encoded (product_id_idx);