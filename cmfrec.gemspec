require_relative "lib/cmfrec/version"

Gem::Specification.new do |spec|
  spec.name          = "cmfrec"
  spec.version       = Cmfrec::VERSION
  spec.summary       = "Recommendations for Ruby using collective matrix factorization"
  spec.homepage      = "https://github.com/ankane/cmfrec"
  spec.license       = "MIT"

  spec.author        = "Andrew Kane"
  spec.email         = "andrew@chartkick.com"

  spec.files         = Dir["*.{md,txt}", "{lib,vendor}/**/*"]
  spec.require_path  = "lib"

  spec.required_ruby_version = ">= 2.5"
end
