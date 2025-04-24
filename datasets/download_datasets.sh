#!/bin/bash

curl -L -o ./datasets/ecommerce-behavior-data-from-multi-category-store.zip\
  https://www.kaggle.com/api/v1/datasets/download/mkechinov/ecommerce-behavior-data-from-multi-category-store
unzip ./datasets/ecommerce-behavior-data-from-multi-category-store.zip \
    -d ./datasets/original/ecommerce-behavior-data-from-multi-category-store/
rm ./datasets/ecommerce-behavior-data-from-multi-category-store.zip


curl -L -o ./datasets/ecommerce-events-history-in-cosmetics-shop.zip\
  https://www.kaggle.com/api/v1/datasets/download/mkechinov/ecommerce-events-history-in-cosmetics-shop
unzip ./datasets/ecommerce-events-history-in-cosmetics-shop.zip \
    -d ./datasets/original/ecommerce-events-history-in-cosmetics-shop/
rm ./datasets/ecommerce-events-history-in-cosmetics-shop.zip


curl -L -o ./datasets/ecommerce-events-history-in-electronics-store.zip\
  https://www.kaggle.com/api/v1/datasets/download/mkechinov/ecommerce-events-history-in-electronics-store
unzip ./datasets/ecommerce-events-history-in-electronics-store.zip \
    -d ./datasets/original/ecommerce-events-history-in-electronics-store/
rm ./datasets/ecommerce-events-history-in-electronics-store.zip