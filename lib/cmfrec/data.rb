module Cmfrec
  module Data
    def load_movielens
      require "csv"

      data_path = download_file("ml-100k/u.data", "https://files.grouplens.org/datasets/movielens/ml-100k/u.data",
        file_hash: "06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490")
      user_path = download_file("ml-100k/u.user", "https://files.grouplens.org/datasets/movielens/ml-100k/u.user",
        file_hash: "f120e114da2e8cf314fd28f99417c94ae9ddf1cb6db8ce0e4b5995d40e90e62c")
      item_path = download_file("ml-100k/u.item", "https://files.grouplens.org/datasets/movielens/ml-100k/u.item",
        file_hash: "553841ebc7de3a0fd0d6b62a204ea30c1e651aacfb2814c7a6584ac52f2c5701")

      # convert u.item to utf-8
      movies_str = File.read(item_path).encode("UTF-8", "binary", invalid: :replace, undef: :replace, replace: "")

      user_info = []
      CSV.foreach(user_path, col_sep: "|") do |row|
        user = {user_id: row[0].to_i}
        10.times do |i|
          user[:"region#{i}"] = row[4][0] == i.to_s ? 1 : 0
        end
        user_info << user
      end

      item_info = []
      movies = {}
      genres = %w(unknown action adventure animation childrens comedy crime documentary drama fantasy filmnoir horror musical mystery romance scifi thriller war western)
      CSV.parse(movies_str, col_sep: "|", converters: [:numeric]) do |row|
        movies[row[0]] = row[1]
        item = {item_id: row[1], year: row[2] ? Date.parse(row[2]).year : 1970}
        genres.each_with_index do |genre, i|
          item[:"genre_#{genre}"] = row[i + 5]
        end
        item_info << item
      end

      data = []
      CSV.foreach(data_path, col_sep: "\t", converters: [:numeric]) do |row|
        data << {
          user_id: row[0],
          item_id: movies[row[1]],
          rating: row[2]
        }
      end

      [data, user_info, item_info]
    end

    private

    def download_file(fname, origin, file_hash:)
      require "digest"
      require "fileutils"
      require "net/http"
      require "tmpdir"

      # TODO handle this better
      raise "No HOME" unless ENV["HOME"]
      dest = "#{ENV["HOME"]}/.cmfrec/#{fname}"
      FileUtils.mkdir_p(File.dirname(dest))

      return dest if File.exist?(dest)

      temp_path = "#{Dir.tmpdir}/cmfrec-#{Time.now.to_f}" # TODO better name

      digest = Digest::SHA2.new

      uri = URI(origin)

      # Net::HTTP automatically adds Accept-Encoding for compression
      # of response bodies and automatically decompresses gzip
      # and deflateresponses unless a Range header was sent.
      # https://ruby-doc.org/stdlib-2.6.4/libdoc/net/http/rdoc/Net/HTTP.html
      Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == "https") do |http|
        request = Net::HTTP::Get.new(uri)

        puts "Downloading data from #{origin}"
        File.open(temp_path, "wb") do |f|
          http.request(request) do |response|
            response.read_body do |chunk|
              f.write(chunk)
              digest.update(chunk)
            end
          end
        end
      end

      if digest.hexdigest != file_hash
        raise Error, "Bad hash: #{digest.hexdigest}"
      end

      puts "Hash verified: #{file_hash}"

      FileUtils.mv(temp_path, dest)

      dest
    end
  end
end
