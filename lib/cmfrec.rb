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
      "x86_64-linux/libcmfrec.so"
    end
  vendor_lib = File.expand_path("../vendor/#{lib_path}", __dir__)
  self.ffi_lib = [vendor_lib]

  # friendlier error message
  autoload :FFI, "cmfrec/ffi"
end
