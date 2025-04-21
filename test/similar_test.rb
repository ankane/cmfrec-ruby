require_relative "test_helper"

class SimilarTest < Minitest::Test
  def setup
    # ngt not supported
    skip if RUBY_ENGINE == "truffleruby"
  end

  def test_similar_users
    data = read_csv("ratings")
    user_info = read_csv("user_info")
    item_info = read_csv("item_info")

    recommender = Cmfrec::Recommender.new(factors: 20, verbose: false)
    recommender.fit(data, user_info: user_info, item_info: item_info)

    refute_empty recommender.similar_users(data.first[:user_id])
    assert_empty recommender.similar_users("missing")
  end

  def test_similar_items
    data = read_csv("ratings")
    user_info = read_csv("user_info")
    item_info = read_csv("item_info")

    recommender = Cmfrec::Recommender.new(factors: 20, verbose: false)
    recommender.fit(data, user_info: user_info, item_info: item_info)

    refute_empty recommender.similar_items(data.first[:item_id])
    assert_empty recommender.similar_items("missing")
  end
end
