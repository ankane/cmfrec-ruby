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

  def test_explicit
    data = read_csv("ratings")
    user_info = read_csv("user_info")
    item_info = read_csv("item_info")

    recommender = Cmfrec::Recommender.new(factors: 3, verbose: false)
    recommender.fit(data, user_info: user_info, item_info: item_info)

    assert_explicit(recommender, data, user_info, item_info)
  end

  def test_explicit_marshal
    data = read_csv("ratings")
    user_info = read_csv("user_info")
    item_info = read_csv("item_info")

    recommender = Cmfrec::Recommender.new(factors: 3, verbose: false)
    recommender.fit(data, user_info: user_info, item_info: item_info)

    recommender = Marshal.load(Marshal.dump(recommender))
    assert_explicit(recommender, data, user_info, item_info)
  end

  def assert_explicit(recommender, data, user_info, item_info)
    assert_in_delta 2.6053429099247047, recommender.global_mean
    assert_kind_of Array, recommender.user_factors
    assert_kind_of Array, recommender.item_factors

    expected = [0.06021799829871035, -0.08009341941042669, -0.020419767632974168, 0.0]
    assert_elements_in_delta expected, recommender.user_bias

    expected = [-0.09391428617760242, -0.14812063585826798, 0.013820866767180017, -0.05169852652784643, 0.23961739305039456]
    assert_elements_in_delta expected, recommender.item_bias

    recs = recommender.user_recs(3, item_ids: [2, 4])
    assert_equal [2, 4], recs.map { |r| r[:item_id] }
    assert_elements_in_delta [2.59874401, 2.82454054], recs.map { |r| r[:score] }

    recs = recommender.user_recs(3)
    assert_equal [4, 2, 3, 0, 1], recs.map { |r| r[:item_id] }
    assert_elements_in_delta [2.82454054, 2.59874401, 2.53322462, 2.49100886, 2.43680251], recs.map { |r| r[:score] }

    new_data = data.select { |d| d[:user_id] == 3 }.map(&:dup)
    new_data.each { |d| d.delete(:user_id) }
    new_user_info = user_info.find { |d| d[:user_id] == 3 }

    # data + user info
    recs = recommender.new_user_recs(new_data, user_info: new_user_info)
    assert_equal [4, 2, 3, 0, 1], recs.map { |r| r[:item_id] }

    # data
    recs = recommender.new_user_recs(new_data)
    assert_equal [4, 2, 3, 0, 1], recs.map { |r| r[:item_id] }

    # user info
    recs = recommender.new_user_recs([], user_info: new_user_info)
    assert_equal [4, 2, 3, 0, 1], recs.map { |r| r[:item_id] }
  end

  def test_implicit
    data = read_csv("ratings")
    data.each { |v| v.delete(:rating) }
    user_info = read_csv("user_info")
    item_info = read_csv("item_info")

    recommender = Cmfrec::Recommender.new(factors: 3, verbose: false)
    recommender.fit(data, user_info: user_info, item_info: item_info)
    assert_implicit(recommender, data, user_info, item_info)
  end

  def test_implicit_marshal
    data = read_csv("ratings")
    data.each { |v| v.delete(:rating) }
    user_info = read_csv("user_info")
    item_info = read_csv("item_info")

    recommender = Cmfrec::Recommender.new(factors: 3, verbose: false)
    recommender.fit(data, user_info: user_info, item_info: item_info)

    recommender = Marshal.load(Marshal.dump(recommender))
    assert_implicit(recommender, data, user_info, item_info)
  end

  def assert_implicit(recommender, data, user_info, item_info)
    assert_equal 0, recommender.global_mean
    assert_kind_of Array, recommender.user_factors
    assert_kind_of Array, recommender.item_factors
    assert_nil recommender.user_bias
    assert_nil recommender.item_bias

    recs = recommender.user_recs(3, item_ids: [2, 4])
    assert_equal [2, 4], recs.map { |r| r[:item_id] }
    # assert_elements_in_delta [0.66010979, 0.27917186], recs.map { |r| r[:score] }

    recs = recommender.user_recs(3)
    assert_equal [4, 2, 3, 1, 0], recs.map { |r| r[:item_id] }

    new_data = data.select { |d| d[:user_id] == 3 }.map(&:dup)
    new_data.each { |d| d.delete(:user_id) }
    new_user_info = user_info.find { |d| d[:user_id] == 3 }

    # data + user info
    recs = recommender.new_user_recs(new_data, user_info: new_user_info)
    assert_equal [4, 2, 3, 1, 0], recs.map { |r| r[:item_id] }

    # data
    recs = recommender.new_user_recs(new_data)
    assert_equal [4, 1, 0, 3, 2], recs.map { |r| r[:item_id] }

    # user info
    recs = recommender.new_user_recs([], user_info: new_user_info)
    assert_equal [4, 2, 3, 1, 0], recs.map { |r| r[:item_id] }
  end

  def test_no_bias
    data = read_csv("ratings")
    recommender = Cmfrec::Recommender.new(user_bias: false, item_bias: false, verbose: false)
    recommender.fit(data)
    assert_nil recommender.user_bias
    assert_nil recommender.item_bias
  end

  # TODO better test
  def test_add_implicit_features
    data = read_csv("ratings")
    recommender = Cmfrec::Recommender.new(add_implicit_features: true, verbose: false)
    recommender.fit(data)

    recs = recommender.user_recs(3, item_ids: [2, 4])
    assert_equal [2, 4], recs.map { |r| r[:item_id] }
    assert_elements_in_delta [2.59874401, 2.82454054], recs.map { |r| r[:score] }
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

  def read_csv(name)
    require "csv"
    CSV.read("test/support/#{name}.csv", headers: true, converters: :numeric, header_converters: :symbol).map(&:to_h)
  end
end
