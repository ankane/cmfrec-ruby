require "bundler/gem_tasks"
require "rake/testtask"

task default: :test
Rake::TestTask.new do |t|
  t.libs << "test"
  t.pattern = "test/**/*_test.rb"
end

def download_file(file, sha256)
  require "open-uri"

  url = "https://github.com/ankane/ml-builds/releases/download/cmfrec-2.4.1/#{file}"
  puts "Downloading #{file}..."
  contents = URI.open(url).read

  computed_sha256 = Digest::SHA256.hexdigest(contents)
  raise "Bad hash: #{computed_sha256}" if computed_sha256 != sha256

  dest = "vendor/#{file}"
  File.binwrite(dest, contents)
  puts "Saved #{dest}"
end

namespace :vendor do
  task :linux do
    download_file("libcmfrec.so", "63ff30f54b4b051d7a7296b92810e2eb19667d42121ddb16307c9758c2a9242a")
  end

  task :mac do
    download_file("libcmfrec.dylib", "e35240f6c7b6460fed9d8319157e8bf1f773f8f012a1f718935eda134f984dab")
    download_file("libcmfrec.arm64.dylib", "e78869445422b9e6a375fee20ef0658c839c2dd58a5a20a668aa9ddc080aa717")
  end

  task :windows do
    # download_file("cmfrec.dll")
  end

  task all: [:linux, :mac, :windows]

  task :platform do
    if Gem.win_platform?
      Rake::Task["vendor:windows"].invoke
    elsif RbConfig::CONFIG["host_os"] =~ /darwin/i
      Rake::Task["vendor:mac"].invoke
    else
      Rake::Task["vendor:linux"].invoke
    end
  end
end
