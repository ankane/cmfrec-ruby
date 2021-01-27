# cmfrec

:fire: Recommendations for Ruby, powered by [cmfrec](https://github.com/david-cortes/cmfrec)

- Supports side information :tada:
- Works with explicit and implicit feedback
- Uses high-performance matrix factorization

[![Build Status](https://github.com/ankane/cmfrec/workflows/build/badge.svg?branch=master)](https://github.com/ankane/cmfrec/actions)

## Installation

Add this line to your application’s Gemfile:

```ruby
gem 'cmfrec'
```

For Windows, also follow [these instructions](#windows-installation).

## Getting Started

Create a recommender

```ruby
recommender = Cmfrec::Recommender.new
```

If users rate items directly, this is known as explicit feedback. Fit the recommender with:

```ruby
recommender.fit([
  {user_id: 1, item_id: 1, rating: 5},
  {user_id: 2, item_id: 1, rating: 3}
])
```

> IDs can be integers, strings, or any other data type

If users don’t rate items directly (for instance, they’re purchasing items or reading posts), this is known as implicit feedback. Leave out the rating, or use a value like number of purchases, number of page views, or time spent on page:

```ruby
recommender.fit([
  {user_id: 1, item_id: 1, value: 1},
  {user_id: 2, item_id: 1, value: 1}
])
```

> Use `value` instead of rating for implicit feedback

Get recommendations for a user in the training data

```ruby
recommender.user_recs(user_id)
```

Get recommendations for a new user

```ruby
recommender.new_user_recs([
  {item_id: 1, value: 5},
  {item_id: 2, value: 3}
])
```

Use the `count` option to specify the number of recommendations (default is 5)

```ruby
recommender.user_recs(user_id, count: 3)
```

Get predicted ratings for specific users and items

```ruby
recommender.predict([{user_id: 1, item_id: 2}, {user_id: 2, item_id: 4}])
```

## Side Information

Add side information about users, items, or both

```ruby
user_info = [
  {user_id: 1, cats: 1, dogs: 0},
  {user_id: 2, cats: 2, dogs: 1},
]
item_info = [
  {item_id: 1, genre_comedy: 1, genre_drama: 0},
  {item_id: 2, genre_comedy: 0, genre_drama: 1},
]
recommender.fit(ratings, user_info: user_info, item_info: item_info)
```

Get recommendations for a new user with ratings and side information

```ruby
ratings = [
  {item_id: 1, rating: 5},
  {item_id: 2, rating: 3}
]
recommender.new_user_recs(ratings, user_info: {cats: 0, dogs: 2})
```

Get recommendations with only side information

```ruby
recommender.new_user_recs([], user_info: {cats: 0, dogs: 2})
```

## Similarity [master]

Add this line to your application’s Gemfile:

```ruby
gem 'ngt'
```

Get similar users

```ruby
recommender.similar_users(user_id)
```

Get similar items - “users who liked this item also liked”

```ruby
recommender.similar_items(item_id)
```

## Examples

### MovieLens

Load the data

```ruby
ratings, user_info, item_info = Cmfrec.load_movielens
```

Create a recommender and get predictions

```ruby
recommender = Cmfrec::Recommender.new(factors: 20)
recommender.fit(ratings.first(80000), user_info: user_info, item_info: item_info)
recommender.predict(ratings.last(20000))
```

## Options

Specify the number of factors and epochs

```ruby
Cmfrec::Recommender.new(factors: 8, epochs: 20)
```

If recommendations look off, trying changing `factors`. The default is 8, but 3 could be good for some applications and 300 good for others.

### Explicit Feedback

Add implicit features

```ruby
Cmfrec::Recommender.new(add_implicit_features: true)
```

Disable bias

```ruby
Cmfrec::Recommender.new(user_bias: false, item_bias: false)
```

## Data

Data can be an array of hashes

```ruby
[{user_id: 1, item_id: 1, rating: 5}, {user_id: 2, item_id: 1, rating: 3}]
```

Or a Rover data frame

```ruby
Rover.read_csv("ratings.csv")
```

## Storing Recommenders [master]

Store the recommender

```ruby
bin = Marshal.dump(recommender)
File.binwrite("recommender.bin", bin)
```

> You can save it to a file, database, or any other storage system

Load a recommender

```ruby
bin = File.binread("recommender.bin")
recommender = Marshal.load(bin)
```

## Reference

Get the global mean

```ruby
recommender.global_mean
```

Get the factors

```ruby
recommender.user_factors
recommender.item_factors
```

Get the bias

```ruby
recommender.user_bias
recommender.item_bias
```

## Windows Installation

On Windows, build the [cmfrec C shared library](https://github.com/david-cortes/cmfrec#instalation) and set:

```ruby
Cmfrec.ffi_lib = "path/to/cmfrec.dll"
```

## History

View the [changelog](https://github.com/ankane/cmfrec/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/cmfrec/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/cmfrec/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

To get started with development:

```sh
git clone https://github.com/ankane/cmfrec.git
cd cmfrec
bundle install
bundle exec rake vendor:all
bundle exec rake test
```
