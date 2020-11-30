# stdlib
require "etc"
require "fiddle/import"

# modules
require "cmfrec/data"
require "cmfrec/recommender"
require "cmfrec/version"

module Cmfrec
  class Error < StandardError; end

  extend Data

  class << self
    attr_accessor :ffi_lib
  end
  lib_name =
    if Gem.win_platform?
      "cmfrec.dll"
    elsif RbConfig::CONFIG["host_os"] =~ /darwin/i
      "libcmfrec.dylib"
    else
      "libcmfrec.so"
    end
  vendor_lib = File.expand_path("../vendor/#{lib_name}", __dir__)
  self.ffi_lib = [vendor_lib]

  # friendlier error message
  autoload :FFI, "cmfrec/ffi"
end
