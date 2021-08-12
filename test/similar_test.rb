require_relative "test_helper"

class SimilarTest < Minitest::Test
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

  def test_user_similarity
    data = read_csv("ratings")
    recommender = Cmfrec::Recommender.new(factors: 20, verbose: false)
    recommender.fit(data)
    assert_in_delta(-0.12189805507659912, recommender.user_similarity(1, 3))

    # ensure consistent with similar_users
    similar_user = recommender.similar_users(1).first
    assert_equal 3, similar_user[:item_id]
    assert_in_delta(-0.12189805507659912, similar_user[:score])

    recommender.user_ids.each do |user_id|
      assert_in_delta 1, recommender.user_similarity(user_id, user_id)
    end
  end

  def test_item_similarity
    data = read_csv("ratings")
    recommender = Cmfrec::Recommender.new(factors: 20, verbose: false)
    recommender.fit(data)
    assert_in_delta 0.12198162078857422, recommender.item_similarity(0, 4)

    # ensure consistent with similar_items
    similar_item = recommender.similar_items(0).first
    assert_equal 4, similar_item[:item_id]
    assert_in_delta 0.12198162078857422, similar_item[:score]

    recommender.item_ids.each do |item_id|
      assert_in_delta 1, recommender.item_similarity(item_id, item_id)
    end
  end
end
