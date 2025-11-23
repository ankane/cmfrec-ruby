# stdlib
require "etc"
require "fiddle/import"

# modules
require_relative "cmfrec/data"
require_relative "cmfrec/recommender"
require_relative "cmfrec/version"

module Cmfrec
  class Error < StandardError; end

  extend Data

  class << self
    attr_accessor :ffi_lib
  end
  lib_path =
    if Gem.win_platform?
      "x64-mingw/cmfrec.dll"
    elsif RbConfig::CONFIG["host_os"] =~ /darwin/i
      if RbConfig::CONFIG["host_cpu"] =~ /arm|aarch64/i
        "arm64-darwin/libcmfrec.dylib"
      else
        "x86_64-darwin/libcmfrec.dylib"
      end
    else
      if RbConfig::CONFIG["host_cpu"] =~ /arm|aarch64/i
        "aarch64-linux/libcmfrec.so"
      else
        "x86_64-linux/libcmfrec.so"
      end
    end
  vendor_lib = File.expand_path("../vendor/#{lib_path}", __dir__)
  self.ffi_lib = [vendor_lib]

  # friendlier error message
  autoload :FFI, "cmfrec/ffi"
end
