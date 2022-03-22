require_relative "test_helper"

class DataTest < Minitest::Test
  def test_load_movielens
    ratings, user_info, item_info = Cmfrec.load_movielens
    assert_equal 100000, ratings.size
    assert_equal 943, user_info.size
    assert_equal 1682, item_info.size
  end
end
