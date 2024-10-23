require_relative "test_helper"

class ImplicitTest < Minitest::Test
  def test_implicit
    data = read_csv("ratings")
    data.each { |v| v.delete(:rating) }
    user_info = read_csv("user_info")
    item_info = read_csv("item_info")

    recommender = Cmfrec::Recommender.new(factors: 3, verbose: false)
    recommender.fit(data, user_info: user_info, item_info: item_info)
    assert_implicit(recommender, data, user_info, item_info)
  end

  def test_implicit_json
    data = read_csv("ratings")
    data.each { |v| v.delete(:rating) }
    user_info = read_csv("user_info")
    item_info = read_csv("item_info")

    recommender = Cmfrec::Recommender.new(factors: 3, verbose: false)
    recommender.fit(data, user_info: user_info, item_info: item_info)

    recommender = Cmfrec::Recommender.load_json(recommender.to_json)
    assert_implicit(recommender, data, user_info, item_info)
  end

  def assert_implicit(recommender, data, user_info, item_info)
    assert_equal 0, recommender.global_mean
    assert_kind_of Array, recommender.user_factors
    assert_kind_of Array, recommender.item_factors
    assert_nil recommender.user_bias
    assert_nil recommender.item_bias

    recs = recommender.user_recs(3, item_ids: [2, 4])
    assert_equal [4, 2], recs.map { |r| r[:item_id] }
    # assert_elements_in_delta [0.66010979, 0.27917186], recs.map { |r| r[:score] }

    recs = recommender.user_recs(3)
    assert_equal [0, 3, 2], recs.map { |r| r[:item_id] }

    new_data = data.select { |d| d[:user_id] == 3 }.map(&:dup)
    new_data.each { |d| d.delete(:user_id) }
    new_user_info = user_info.find { |d| d[:user_id] == 3 }

    # data + user info
    recs = recommender.new_user_recs(new_data, user_info: new_user_info)
    assert_equal [0, 3, 2], recs.map { |r| r[:item_id] }

    # data
    recs = recommender.new_user_recs(new_data)
    assert_equal [0, 3, 2], recs.map { |r| r[:item_id] }

    # user info
    recs = recommender.new_user_recs([], user_info: new_user_info)
    assert_equal [4, 0, 1, 3, 2], recs.map { |r| r[:item_id] }

    # user bias
    recommender.user_ids.each do |user_id|
      assert_nil recommender.user_bias(user_id)
    end
    assert_nil recommender.user_bias("unknown")

    # item bias
    recommender.item_ids.each do |item_id|
      assert_nil recommender.item_bias(item_id)
    end
    assert_nil recommender.item_bias("unknown")
  end
end
