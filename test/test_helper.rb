require "bundler/setup"
Bundler.require(:default)
require "minitest/autorun"
require "minitest/pride"

class Minitest::Test
  def setup
    if stress?
      # autoload before GC.stress
      Cmfrec::FFI.name
      skip if is_a?(DataTest)
      GC.stress = true
    end
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

  FILES = {}

  def read_csv(name)
    require "csv"

    FILES[name] ||= CSV.read("test/support/#{name}.csv", headers: true, converters: :numeric, header_converters: :symbol).map { |v| v.to_h.freeze }.freeze
  end
end
