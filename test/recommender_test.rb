require_relative "test_helper"

class RecommenderTest < Minitest::Test
  def test_example
    data = [
      {user_id: 0, item_id: 1, rating: 3},
      {user_id: 0, item_id: 4, rating: 3},
      {user_id: 0, item_id: 5, rating: 4},
      {user_id: 0, item_id: 6, rating: 5},
      {user_id: 1, item_id: 2, rating: 2},
      {user_id: 2, item_id: 0, rating: 1},
      {user_id: 2, item_id: 3, rating: 5},
      {user_id: 2, item_id: 5, rating: 1},
      {user_id: 3, item_id: 4, rating: 4},
      {user_id: 3, item_id: 5, rating: 2},
      {user_id: 3, item_id: 7, rating: 3},
      {user_id: 4, item_id: 1, rating: 3},
      {user_id: 5, item_id: 0, rating: 1},
      {user_id: 5, item_id: 2, rating: 3},
      {user_id: 5, item_id: 3, rating: 2},
      {user_id: 5, item_id: 6, rating: 5}
    ]

    recommender = Cmfrec::Recommender.new(factors: 3, user_bias: false, item_bias: false, verbose: false)
    recommender.fit(data)
    assert_in_delta 2.9375, recommender.global_mean
  end

  def test_no_bias
    data = read_csv("ratings")
    recommender = Cmfrec::Recommender.new(user_bias: false, item_bias: false, verbose: false)
    recommender.fit(data)
    assert_nil recommender.user_bias
    assert_nil recommender.item_bias
  end

  def test_user_recs_item_ids
    recommender = Cmfrec::Recommender.new(verbose: false)
    recommender.fit([
      {user_id: 1, item_id: 1, rating: 5},
      {user_id: 1, item_id: 2, rating: 3}
    ])
    assert_equal [2], recommender.user_recs(1, item_ids: [2]).map { |r| r[:item_id] }
  end

  # Python library gets a_vec from -1 index (bug?)
  def test_user_recs_new_user
    data = read_csv("ratings")
    recommender = Cmfrec::Recommender.new(verbose: false)
    recommender.fit(data)
    assert_empty recommender.user_recs(1000)
  end

  # only return items that exist
  def test_user_recs_new_item
    data = read_csv("ratings")
    recommender = Cmfrec::Recommender.new(verbose: false)
    recommender.fit(data)
    assert_empty recommender.user_recs(3, item_ids: [1000])
  end

  def test_predict
    data = read_csv("ratings")
    recommender = Cmfrec::Recommender.new(verbose: false)
    recommender.fit(data)

    predict_data = [{user_id: 3, item_id: 2}, {user_id: 3, item_id: 4}]
    assert_elements_in_delta [2.59874401, 2.82454054], recommender.predict(predict_data)
  end

  def test_predict_new_user
    data = read_csv("ratings")
    recommender = Cmfrec::Recommender.new(verbose: false)
    recommender.fit(data)

    bias_index = recommender.instance_variable_get(:@item_map)[2]
    expected = recommender.global_mean + recommender.item_bias[bias_index]
    assert_elements_in_delta [expected], recommender.predict([{user_id: 1000, item_id: 2}])
  end

  def test_predict_new_item
    data = read_csv("ratings")
    recommender = Cmfrec::Recommender.new(verbose: false)
    recommender.fit(data)

    bias_index = recommender.instance_variable_get(:@user_map)[3]
    expected = recommender.global_mean + recommender.user_bias[bias_index]
    assert_elements_in_delta [expected], recommender.predict([{user_id: 3, item_id: 1000}])
  end

  def test_predict_new_user_and_item
    data = read_csv("ratings")
    recommender = Cmfrec::Recommender.new(verbose: false)
    recommender.fit(data)

    expected = recommender.global_mean
    assert_elements_in_delta [expected], recommender.predict([{user_id: 1000, item_id: 1000}])
  end

  def test_no_training_data
    recommender = Cmfrec::Recommender.new
    error = assert_raises(ArgumentError) do
      recommender.fit([])
    end
    assert_equal "No training data", error.message
  end

  def test_missing_user_id
    recommender = Cmfrec::Recommender.new
    error = assert_raises(ArgumentError) do
      recommender.fit([{item_id: 1, rating: 5}])
    end
    assert_equal "Missing user_id", error.message
  end

  def test_missing_item_id
    recommender = Cmfrec::Recommender.new
    error = assert_raises(ArgumentError) do
      recommender.fit([{user_id: 1, rating: 5}])
    end
    assert_equal "Missing item_id", error.message
  end

  def test_user_info_missing_user_id
    data = read_csv("ratings")
    recommender = Cmfrec::Recommender.new
    error = assert_raises(ArgumentError) do
      recommender.fit(data, user_info: [{a: 1}])
    end
    assert_equal "Missing user_id", error.message
  end

  def test_item_info_missing_item_id
    data = read_csv("ratings")
    recommender = Cmfrec::Recommender.new
    error = assert_raises(ArgumentError) do
      recommender.fit(data, item_info: [{a: 1}])
    end
    assert_equal "Missing item_id", error.message
  end

  def test_not_fit
    recommender = Cmfrec::Recommender.new
    error = assert_raises do
      recommender.user_recs(1)
    end
    assert_equal "Not fit", error.message
  end

  def test_rover
    data = Rover.read_csv("test/support/ratings.csv")
    user_info = Rover.read_csv("test/support/user_info.csv")
    item_info = Rover.read_csv("test/support/item_info.csv")
    recommender = Cmfrec::Recommender.new(verbose: false)
    recommender.fit(data, user_info: user_info, item_info: item_info)
    recommender.new_user_recs(data)
    recommender.predict(data)
  end
end
