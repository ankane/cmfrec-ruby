require_relative "test_helper"

class ExplicitTest < Minitest::Test
  def test_explicit
    data = read_csv("ratings")
    user_info = read_csv("user_info")
    item_info = read_csv("item_info")

    recommender = Cmfrec::Recommender.new(factors: 3, verbose: false)
    recommender.fit(data, user_info: user_info, item_info: item_info)

    assert_explicit(recommender, data, user_info, item_info)
  end

  def test_explicit_json
    data = read_csv("ratings")
    user_info = read_csv("user_info")
    item_info = read_csv("item_info")

    recommender = Cmfrec::Recommender.new(factors: 3, verbose: false)
    recommender.fit(data, user_info: user_info, item_info: item_info)

    recommender = Cmfrec::Recommender.load_json(recommender.to_json)
    assert_explicit(recommender, data, user_info, item_info)
  end

  # TODO better test
  def test_add_implicit_features
    data = read_csv("ratings")
    recommender = Cmfrec::Recommender.new(add_implicit_features: true, verbose: false)
    recommender.fit(data)

    recs = recommender.user_recs(3, item_ids: [2, 4])
    assert_equal [4, 2], recs.map { |r| r[:item_id] }
    assert_elements_in_delta [2.82454054, 2.59874401], recs.map { |r| r[:score] }
  end

  def assert_explicit(recommender, data, user_info, item_info)
    assert_in_delta 2.6053429099247047, recommender.global_mean
    assert_kind_of Array, recommender.user_factors
    assert_kind_of Array, recommender.item_factors
    assert_kind_of Array, recommender.user_bias
    assert_kind_of Array, recommender.item_bias

    expected = [-0.08009341941042647, -0.020419767633096483, 0.06021799829862086, 0.0]
    assert_elements_in_delta expected, recommender.user_bias

    expected = [-0.09391428617760313, -0.14812063585792457, 0.23961739305090726, 0.01382086676717879, -0.05169852652770836]
    assert_elements_in_delta expected, recommender.item_bias

    recs = recommender.user_recs(3, item_ids: [2, 4])
    assert_equal [4, 2], recs.map { |r| r[:item_id] }
    assert_elements_in_delta [2.82454054, 2.59874401], recs.map { |r| r[:score] }

    recs = recommender.user_recs(3, item_ids: [1, 2, 4], count: 2)
    assert_equal [4, 2], recs.map { |r| r[:item_id] }

    recs = recommender.user_recs(3)
    assert_equal [2, 3, 0], recs.map { |r| r[:item_id] }
    assert_elements_in_delta [2.59874401, 2.53322462, 2.49100886], recs.map { |r| r[:score] }

    new_data = data.select { |d| d[:user_id] == 3 }.map(&:dup)
    new_data.each { |d| d.delete(:user_id) }
    new_user_info = user_info.find { |d| d[:user_id] == 3 }

    # data + user info
    recs = recommender.new_user_recs(new_data, user_info: new_user_info)
    assert_equal [2, 3, 0], recs.map { |r| r[:item_id] }

    # data
    recs = recommender.new_user_recs(new_data)
    assert_equal [2, 3, 0], recs.map { |r| r[:item_id] }

    # user info
    recs = recommender.new_user_recs([], user_info: new_user_info)
    assert_equal [4, 2, 3, 0, 1], recs.map { |r| r[:item_id] }

    # user bias
    recommender.user_ids.zip(recommender.user_bias) do |user_id, bias|
      assert_equal bias, recommender.user_bias(user_id)
    end
    assert_nil recommender.user_bias("unknown")

    # item bias
    recommender.item_ids.zip(recommender.item_bias) do |item_id, bias|
      assert_equal bias, recommender.item_bias(item_id)
    end
    assert_nil recommender.item_bias("unknown")
  end
end
