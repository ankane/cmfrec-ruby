module Cmfrec
  module Data
    def load_movielens
      data_path = download_file("ml-100k/u.data", "https://files.grouplens.org/datasets/movielens/ml-100k/u.data",
        file_hash: "06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490")
      user_path = download_file("ml-100k/u.user", "https://files.grouplens.org/datasets/movielens/ml-100k/u.user",
        file_hash: "f120e114da2e8cf314fd28f99417c94ae9ddf1cb6db8ce0e4b5995d40e90e62c")
      item_path = download_file("ml-100k/u.item", "https://files.grouplens.org/datasets/movielens/ml-100k/u.item",
        file_hash: "553841ebc7de3a0fd0d6b62a204ea30c1e651aacfb2814c7a6584ac52f2c5701")

      user_info = []
      File.foreach(user_path) do |line|
        row = line.split("|")
        user = {user_id: row[0].to_i}
        10.times do |i|
          user[:"region#{i}"] = row[4][0] == i.to_s ? 1 : 0
        end
        user_info << user
      end

      item_info = []
      movies = {}
      movie_names = {}
      genres = %w(unknown action adventure animation childrens comedy crime documentary drama fantasy filmnoir horror musical mystery romance scifi thriller war western)
      File.foreach(item_path) do |line|
        row = line.encode("UTF-8", "ISO-8859-1").split("|")
        movies[row[0]] = row[1]

        # filter duplicates
        next if movie_names[row[1]]
        movie_names[row[1]] = true

        item = {item_id: row[1], year: !row[2].empty? ? Date.strptime(row[2], "%d-%b-%Y").year : 1970}
        genres.each_with_index do |genre, i|
          item[:"genre_#{genre}"] = row[i + 5].to_i
        end
        item_info << item
      end

      data = []
      File.foreach(data_path) do |line|
        row = line.split("\t")
        data << {
          user_id: row[0].to_i,
          item_id: movies[row[1]],
          rating: row[2].to_i
        }
      end

      [data, user_info, item_info]
    end

    private

    def download_file(fname, origin, file_hash:)
      require "digest"
      require "fileutils"
      require "open-uri"
      require "tmpdir"

      cache_home = ENV["XDG_CACHE_HOME"] || "#{ENV.fetch("HOME")}/.cache"
      dest = "#{cache_home}/cmfrec/#{fname}"
      FileUtils.mkdir_p(File.dirname(dest))

      return dest if File.exist?(dest)

      Dir.mktmpdir do |dir|
        temp_path = "#{dir}/cmfrec"

        puts "Downloading data from #{origin}"
        IO.copy_stream(URI.parse(origin).open(redirect: false), temp_path)

        digest = Digest::SHA2.file(temp_path)
        if digest.hexdigest != file_hash
          raise Error, "Bad hash: #{digest.hexdigest}"
        end
        puts "Hash verified: #{file_hash}"

        FileUtils.mv(temp_path, dest)
      end

      dest
    end
  end
end
