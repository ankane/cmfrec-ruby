require "bundler/setup"
Bundler.require(:default)
require "minitest/autorun"
require "minitest/pride"

class Minitest::Test
  def setup
    # autoload before GC.stress
    Cmfrec::FFI if stress?

    GC.stress = true if stress?
  end

  def teardown
    GC.stress = false if stress?
  end

  def stress?
    ENV["STRESS"]
  end

  def assert_elements_in_delta(expected, actual)
    assert_equal expected.size, actual.size
    expected.zip(actual) do |exp, act|
      assert_in_delta exp, act
    end
  end

  def read_csv(name)
    require "csv"
    CSV.read("test/support/#{name}.csv", headers: true, converters: :numeric, header_converters: :symbol).map(&:to_h)
  end
end
