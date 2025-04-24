SET search_path TO e_commerce, public;
DROP TABLE IF EXISTS e_commerce.events_processed;

-- Create a new table for storing processed events
CREATE TABLE e_commerce.events_processed (
    -- Original columns
    event_time TIMESTAMP NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    product_id INTEGER NOT NULL,
    category_id BIGINT,
    category_code VARCHAR(40),
    brand VARCHAR(20),
    price REAL,
    user_id BIGINT NOT NULL,
    user_session VARCHAR(36) NOT NULL,
    -- Historical features
    session_event_num INTEGER NOT NULL, -- Number of event in session
    user_global_event_num BIGINT NOT NULL, -- Global number of event for user
    user_views_before BIGINT, -- User views before current event
    user_carts_before BIGINT, -- User carts before current event
    user_purchases_before BIGINT, -- User purchases before current event
    product_views_before BIGINT, -- Product views before current event
    product_purchases_before BIGINT, -- Product purchases before current event
    product_avg_price REAL NOT NULL, -- Accumulated product average price (includes current event)
    category_views_before BIGINT, -- Category views before current event
    category_avg_price REAL NOT NULL, -- Accumulated category average price (includes current event)
    -- Simple time features
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
    -- Holiday features
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
    -- User statistics
    -- Number of event in session
    ROW_NUMBER() OVER (PARTITION BY user_session ORDER BY event_time) AS session_event_num,
    -- Number of event for user in all history
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_time) AS user_global_event_num,
    -- Calculate number of user views before current event
    SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) OVER user_window AS user_views_before,
    -- Calculate number of user carts before current event
    SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) OVER user_window AS user_carts_before,
    -- Calculate number of user purchases before current event
    SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) OVER user_window AS user_purchases_before,
    -- Product statistics before current event
    -- Number of product views before current event
    SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) OVER product_window AS product_views_before,
    -- Number of product purchases before current event
    SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) OVER product_window AS product_purchases_before,
    -- Accumulated product average price, including current event
    AVG(price) OVER (
        PARTITION BY product_id
        ORDER BY event_time
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS product_avg_price,
    -- Category statistics
    -- Number of category views before current event
    SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) OVER category_window AS category_views_before,
    -- Accumulated category average price, including current event
    AVG(price) OVER (
        PARTITION BY category_code
        ORDER BY event_time
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS category_avg_price,
    -- Simple time features
    SIN(2 * PI() * EXTRACT(HOUR FROM fe.event_time) / 24.0) AS hour_sin,
    COS(2 * PI() * EXTRACT(HOUR FROM fe.event_time) / 24.0) AS hour_cos,
    SIN(2 * PI() * EXTRACT(ISODOW FROM fe.event_time) / 7.0) AS day_of_week_sin, -- ISODOW: 1=Mon..7=Sun
    COS(2 * PI() * EXTRACT(ISODOW FROM fe.event_time) / 7.0) AS day_of_week_cos,
    SIN(2 * PI() * EXTRACT(DAY FROM fe.event_time) / 31.0) AS day_sin,
    COS(2 * PI() * EXTRACT(DAY FROM fe.event_time) / 31.0) AS day_cos,
    SIN(2 * PI() * EXTRACT(MONTH FROM fe.event_time) / 12.0) AS month_sin,
    COS(2 * PI() * EXTRACT(MONTH FROM fe.event_time) / 12.0) AS month_cos,
    EXTRACT(YEAR FROM fe.event_time)::INTEGER AS year,
    CASE WHEN EXTRACT(ISODOW FROM fe.event_time) IN (6, 7) THEN 1 ELSE 0 END AS is_weekend,
    -- Holiday features
    CASE WHEN date IS NOT NULL THEN 1 ELSE 0 END AS is_holiday,
    -- Use COALESCE to replace NULL (no next/prev holiday) with a large number
    COALESCE(next_holiday_date - CAST(event_time AS DATE), 9999) AS days_to_next_holiday,
    COALESCE(CAST(event_time AS DATE) - prev_holiday_date, 9999) AS days_from_last_holiday,
    holiday_name
FROM
    FilteredEvents fe
-- Join holiday name (if day is holiday)
LEFT JOIN
    e_commerce.holidays rh ON CAST(event_time AS DATE) = rh.date
-- Find next holiday date
LEFT JOIN LATERAL (
    SELECT MIN(h.date) as next_holiday_date
    FROM e_commerce.holidays h
    WHERE h.date > CAST(event_time AS DATE)
) next_h ON true
-- Find previous holiday date
LEFT JOIN LATERAL (
    SELECT MAX(h.date) as prev_holiday_date
    FROM e_commerce.holidays h
    WHERE h.date < CAST(event_time AS DATE)
) prev_h ON true
WINDOW
    -- Window for user features (before current event)
    user_window AS (PARTITION BY user_id ORDER BY event_time ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
    -- Window for product features (before current event)
    product_window AS (PARTITION BY product_id ORDER BY event_time ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
    -- Window for category features (before current event)
    category_window AS (PARTITION BY category_code ORDER BY event_time ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
ORDER BY
    user_id, event_time;


ALTER TABLE e_commerce.events_processed
ADD PRIMARY KEY (event_time, product_id, user_session);

-- Indexes on events_processed for faster next step
CREATE INDEX IF NOT EXISTS idx_processed_brand ON e_commerce.events_processed (brand); -- For DENSE_RANK
CREATE INDEX IF NOT EXISTS idx_processed_user_session ON e_commerce.events_processed (user_session); -- For DENSE_RANK
CREATE INDEX IF NOT EXISTS idx_processed_holiday_name ON e_commerce.events_processed (holiday_name); -- For DENSE_RANK
CREATE INDEX IF NOT EXISTS idx_processed_event_time ON e_commerce.events_processed (event_time); -- for fun


DROP TABLE IF EXISTS e_commerce.events_encoded;
-- Create final table for encoded and scaled data
CREATE TABLE e_commerce.events_encoded (
    event_time TIMESTAMP NOT NULL, -- leave for primary key
    price REAL NOT NULL,
    -- Scaled historical features (from events_processed)
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
    -- Scaled simple time features (from events_processed)
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
    -- Scaled holiday features
    is_holiday REAL NOT NULL,
    days_to_next_holiday REAL NOT NULL,
    days_from_last_holiday REAL NOT NULL,
    -- holiday_name INTEGER NOT NULL,

    -- Encoded categorical features 
    -- (or not encoded, but the fact that categorical and not scaled)
    -- Original columns
    -- only one not encoded feature, 
    -- to be able to choose mapping for reward during training
    event_type VARCHAR(20) NOT NULL, -- reward
    product_id INTEGER NOT NULL,
    category_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    brand INTEGER NOT NULL, -- Label Encoding
    user_session BIGINT NOT NULL, -- Label Encoding
    holiday_name INTEGER NOT NULL, -- Label Encoding
    product_id_idx INTEGER NOT NULL -- Embedding слой
    
);


-- Insert data with encoding and scaling
INSERT INTO e_commerce.events_encoded (
    -- List all columns of the target table events_encoded in the correct order
    event_time, price,
    session_event_num, user_global_event_num, user_views_before, user_carts_before, user_purchases_before,
    product_views_before, product_purchases_before, product_avg_price, category_views_before, category_avg_price,
    hour_sin, hour_cos, day_of_week_sin, day_of_week_cos, day_sin, day_cos, month_sin, month_cos, year, is_weekend,
    is_holiday, days_to_next_holiday, days_from_last_holiday,
    event_type, product_id, category_id, user_id, brand, user_session, holiday_name, product_id_idx
)
WITH SourceData AS (
    -- Read all from the updated table events_processed
    SELECT * FROM e_commerce.events_processed
),
EncodedFeatures AS (
    -- Calculate integer indices for categorical features
    SELECT
        sd.*, -- Select all columns from SourceData for use below
        -- Encode brand
        DENSE_RANK() OVER (ORDER BY COALESCE(sd.brand, '__MISSING__')) AS brand_idx,
        -- Encode user_session (even if it's UUID, for integer representation)
        DENSE_RANK() OVER (ORDER BY sd.user_session) AS user_session_idx,
        -- Encode holiday_name (added from Python)
        DENSE_RANK() OVER (ORDER BY COALESCE(sd.holiday_name, '__MISSING__')) AS holiday_name_idx,
        -- category_code not encoded, because it's not in the target table events_encoded (and it's already encoded in category_id)
        -- product_id, category_id, user_id are already numerical identifiers
        -- Encode product_id (for Embedding layer)
        DENSE_RANK() OVER (ORDER BY sd.product_id) AS product_id_idx
    FROM SourceData sd
)
SELECT
    -- Select columns in EXACTLY the same order as in INSERT INTO events_encoded
    ef.event_time,
    ef.price,
    -- Scaled numerical features (original + historical)
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

    -- Scaled simple time features
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

    -- Scaled holiday features
    (ef.is_holiday::REAL - AVG(ef.is_holiday::REAL) OVER ()) / NULLIF(STDDEV_SAMP(ef.is_holiday::REAL) OVER (), 0), -- is_holiday REAL NOT NULL
    
    (ef.days_to_next_holiday::REAL - AVG(ef.days_to_next_holiday:: REAL) OVER ()) / STDDEV_SAMP(ef.days_to_next_holiday::REAL) OVER (), -- days_to_next_holiday REAL NOT NULL
    (ef.days_from_last_holiday::REAL - AVG(ef.days_from_last_holiday:: REAL) OVER ()) / STDDEV_SAMP(ef.days_from_last_holiday::REAL) OVER (), -- days_from_last_holiday REAL NOT NULL

    -- Categorical/ID features (some encoded, some original)
    ef.event_type::VARCHAR,           -- reward
    ef.product_id::INTEGER,           -- product_id INTEGER NOT NULL (original)
    ef.category_id::BIGINT,           -- category_id BIGINT NOT NULL (original)
    ef.user_id::BIGINT,               -- user_id BIGINT NOT NULL (original)
    ef.brand_idx::INTEGER,            -- brand INTEGER NOT NULL (encoded)
    ef.user_session_idx::BIGINT,      -- user_session BIGINT NOT NULL (encoded)
    ef.holiday_name_idx::INTEGER,     -- holiday_name INTEGER NOT NULL (encoded)
    ef.product_id_idx::INTEGER        -- product_id_idx INTEGER NOT NULL (for Embedding layer)
FROM EncodedFeatures ef;

ALTER TABLE e_commerce.events_encoded
ADD PRIMARY KEY (event_time, product_id, user_session);

-- Main index for loading user sequences
CREATE INDEX IF NOT EXISTS idx_encoded_user_time ON e_commerce.events_encoded (user_id, event_time);
-- Index for possible time filtering
CREATE INDEX IF NOT EXISTS idx_encoded_event_time ON e_commerce.events_encoded (event_time);
-- Index by session ID (use encoded ID)
CREATE INDEX IF NOT EXISTS idx_encoded_session_idx ON e_commerce.events_encoded (user_session);
-- Index by product ID
CREATE INDEX IF NOT EXISTS idx_encoded_product_id ON e_commerce.events_encoded (product_id);
-- Index by user ID
CREATE INDEX IF NOT EXISTS idx_encoded_user_id ON e_commerce.events_encoded (user_id);
-- Index by product ID for Embedding layer
CREATE INDEX IF NOT EXISTS idx_encoded_product_id_idx ON e_commerce.events_encoded (product_id_idx);