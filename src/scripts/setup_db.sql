CREATE SCHEMA IF NOT EXISTS e_commerce;
SET search_path TO e_commerce, public;

DROP TABLE IF EXISTS e_commerce.events;
CREATE TABLE e_commerce.events (
    event_time TIMESTAMP NOT NULL,      -- Timestamp of the event
    event_type VARCHAR(50) NOT NULL,    -- Type of event (e.g., view, cart, purchase)
    product_id INTEGER NOT NULL,        -- Product identifier
    category_id BIGINT,                 -- Category identifier
    category_code VARCHAR(40),          -- Full category path (can be long)
    brand VARCHAR(20),                  -- Brand name (allow slightly more space)
    price REAL NOT NULL,                -- Price (float32 equivalent)
    user_id BIGINT NOT NULL,            -- User identifier
    user_session VARCHAR(36) NOT NULL   -- User session identifier (can be long/complex)
);

\COPY e_commerce.events FROM 'datasets/prepared4db/events_clean.csv' WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',');

-- Добавляем Primary Key ПОСЛЕ вставки данных (быстрее)
ALTER TABLE e_commerce.events
ADD PRIMARY KEY (event_time, product_id, user_session);

-- Одиночные на events 
CREATE INDEX IF NOT EXISTS idx_events_user_id ON e_commerce.events (user_id);
CREATE INDEX IF NOT EXISTS idx_events_product_id ON e_commerce.events (product_id);
CREATE INDEX IF NOT EXISTS idx_events_event_time ON e_commerce.events (event_time);
CREATE INDEX IF NOT EXISTS idx_events_user_session ON e_commerce.events (user_session);
-- Составные на events
CREATE INDEX IF NOT EXISTS idx_events_user_time ON e_commerce.events (user_id, event_time);
CREATE INDEX IF NOT EXISTS idx_events_product_time ON e_commerce.events (product_id, event_time);
CREATE INDEX IF NOT EXISTS idx_events_category_time ON e_commerce.events (category_code, event_time);
CREATE INDEX IF NOT EXISTS idx_events_session_time ON e_commerce.events (user_session, event_time);

---------------------- HOLIDAYS ----------------------

DROP TABLE IF EXISTS e_commerce.holidays;
CREATE TABLE e_commerce.holidays (
    date DATE NOT NULL,                   -- Date of the holiday
    holiday_name VARCHAR(50) NOT NULL    -- Name of the holiday
);


\COPY e_commerce.holidays FROM 'datasets/prepared4db/holidays.csv' WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',');

-- Добавляем Primary Key ПОСЛЕ вставки данных (быстрее)
ALTER TABLE e_commerce.holidays
ADD PRIMARY KEY (date);

-- На holidays 
CREATE INDEX IF NOT EXISTS idx_holidays_date ON e_commerce.holidays (date);

