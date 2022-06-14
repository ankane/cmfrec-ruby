require_relative "test_helper"

class DataTest < Minitest::Test
  def test_load_movielens
    ratings, user_info, item_info = Cmfrec.load_movielens
    assert_equal 100000, ratings.size
    assert_equal 943, user_info.size
    assert_equal 1664, item_info.size
    assert ratings.all? { |v| v[:user_id] }
    assert ratings.all? { |v| v[:item_id] }
  end
end
