## 0.3.2 (2025-05-04)

- Fixed crash with Fiddle 1.1.7+
- Fixed memory leaks

## 0.3.1 (2024-12-29)

- Removed dependency on `base64` gem for serialization

## 0.3.0 (2024-10-23)

- Changed dataset directory to match XDG Base Directory Specification
- Removed dependency on `csv` gem for `load_movielens`
- Dropped support for marshal serialization
- Dropped support for Ruby < 3.1

## 0.2.1 (2022-07-11)

- Added support for JSON serialization

## 0.2.0 (2022-06-14)

- Updated cmfrec to 3.4.2
- Fixed missing item ids with `load_movielens`
- Dropped support for Ruby < 2.7

## 0.1.7 (2022-03-22)

- Improved ARM detection
- Fixed error with `load_movielens`
- Fixed duplicates in `item_info` with `load_movielens`

## 0.1.6 (2021-08-12)

- Added `user_ids` and `item_ids` methods
- Added `user_id` argument to `user_factors`
- Added `item_id` argument to `item_factors`
- Added `user_id` argument to `user_bias`
- Added `item_id` argument to `item_bias`
- Added `item_ids` argument to `new_user_recs`
- Fixed order for `user_recs`

## 0.1.5 (2021-08-10)

- Fixed issue with `user_recs` and `new_user_recs` returning rated items
- Fixed error with `new_user_recs`

## 0.1.4 (2021-02-04)

- Added support for saving and loading recommenders
- Added `similar_users` and `similar_items`
- Improved ARM detection

## 0.1.3 (2020-12-28)

- Added ARM shared library for Mac

## 0.1.2 (2020-12-09)

- Added `load_movielens` method
- Updated cmfrec to 2.4.1

## 0.1.1 (2020-11-28)

- Added `predict` method

## 0.1.0 (2020-11-27)

- First release
