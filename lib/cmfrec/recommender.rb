module Cmfrec
  class Recommender
    attr_reader :global_mean

    def initialize(factors: 8, epochs: 10, verbose: true, user_bias: true, item_bias: true, add_implicit_features: false)
      set_params(
        k: factors,
        niter: epochs,
        verbose: verbose,
        user_bias: user_bias,
        item_bias: item_bias,
        add_implicit_features: add_implicit_features
      )

      @fit = false
      @user_map = {}
      @item_map = {}
      @user_info_map = {}
      @item_info_map = {}
    end

    def fit(train_set, user_info: nil, item_info: nil)
      reset
      partial_fit(train_set, user_info: user_info, item_info: item_info)
    end

    def predict(data)
      check_fit

      data = to_dataset(data)

      u = data.map { |v| @user_map[v[:user_id]] || @user_map.size }
      i = data.map { |v| @item_map[v[:item_id]] || @item_map.size }

      row = int_ptr(u)
      col = int_ptr(i)
      n_predict = data.size
      predicted = Fiddle::Pointer.malloc(n_predict * Fiddle::SIZEOF_DOUBLE, Fiddle::RUBY_FREE)

      if @implicit
        check_status FFI.predict_X_old_collective_implicit(
          row, col, predicted, n_predict,
          @a, @b,
          @k, @k_user, @k_item, @k_main,
          @m, @n,
          @nthreads
        )
      else
        check_status FFI.predict_X_old_collective_explicit(
          row, col, predicted, n_predict,
          @a, @bias_a,
          @b, @bias_b,
          @global_mean,
          @k, @k_user, @k_item, @k_main,
          @m, @n,
          @nthreads
        )
      end

      predictions = real_array(predicted)
      predictions.map! { |v| v.nan? ? @global_mean : v } if @implicit
      predictions
    end

    def user_recs(user_id, count: 5, item_ids: nil)
      check_fit
      user = @user_map[user_id]

      if user
        a_vec = @a[user * @k * Fiddle::SIZEOF_DOUBLE, @k * Fiddle::SIZEOF_DOUBLE]
        a_bias = @bias_a ? @bias_a[user * Fiddle::SIZEOF_DOUBLE, Fiddle::SIZEOF_DOUBLE].unpack1("d") : 0
        # @rated[user] will be nil for recommenders saved before 0.1.5
        top_n(a_vec: a_vec, a_bias: a_bias, count: count, rated: (@rated[user] || {}).keys, item_ids: item_ids, row_index: user)
      else
        # no items if user is unknown
        # TODO maybe most popular items
        []
      end
    end

    def new_user_recs(data, count: 5, user_info: nil, item_ids: nil)
      check_fit

      data = to_dataset(data)
      user_info = to_dataset(user_info) if user_info

      # remove unknown items
      data, unknown_data = data.partition { |d| @item_map[d[:item_id]] }

      if unknown_data.any?
        # TODO warn for unknown items?
        # warn "[cmfrec] Unknown items: #{unknown_data.map { |d| d[:item_id] }.join(", ")}"
      end

      rated_ids = data.map { |d| @item_map[d[:item_id]] }

      nnz = data.size

      u_vec_sp = []
      u_vec_x_col = []
      if user_info
        user_info.each do |k, v|
          next if k == :user_id

          uc = @user_info_map[k]
          raise "Bad key: #{k}" unless uc

          u_vec_x_col << uc
          u_vec_sp << v
        end
      end
      p_ = @user_info_map.size
      nnz_u_vec = u_vec_sp.size
      u_vec_x_col = int_ptr(u_vec_x_col)
      u_vec_sp = real_ptr(u_vec_sp)

      u_vec = nil
      u_bin_vec = nil
      pbin = 0

      weight = nil
      lam_unique = nil
      l1_lam_unique = nil
      n_max = @n

      if data.any?
        if @implicit
          ratings = data.map { |d| d[:value] || 1 }
        else
          ratings = data.map { |d| d[:rating] }
          check_ratings(ratings)
        end
        xa = real_ptr(ratings)
        x_col = int_ptr(rated_ids)
      else
        xa = nil
        x_col = nil
      end
      xa_dense = nil

      rated = rated_ids.uniq

      prep = prepare_top_n(count: count, rated: rated, item_ids: item_ids)
      return [] if prep.empty?
      include_ix, n_include, exclude_ix, n_exclude, outp_ix, outp_score, count = prep

      if @implicit
        args = [
          @n,
          u_vec, p_,
          u_vec_sp, u_vec_x_col, nnz_u_vec,
          @na_as_zero_user,
          @nonneg,
          @u_colmeans,
          @b, @c,
          xa, x_col, nnz,
          @k, @k_user, @k_item, @k_main,
          @lambda_, @l1_lambda, @alpha, @w_main, @w_user,
          @w_main_multiplier,
          @apply_log_transf,
          nil, #BeTBe,
          nil, #BtB,
          nil, #BeTBeChol,
          nil, #CtUbias,
          include_ix, n_include,
          exclude_ix, n_exclude,
          outp_ix, outp_score,
          count, @nthreads
        ]
        check_status FFI.topN_new_collective_implicit(*fiddle_args(args))
      else
        cb = nil
        scaling_bias_a = 0

        args = [
          @user_bias,
          u_vec, p_,
          u_vec_sp, u_vec_x_col, nnz_u_vec,
          u_bin_vec, pbin,
          @na_as_zero_user, @na_as_zero,
          @nonneg,
          @c, cb,
          @global_mean, @bias_b,
          @u_colmeans,
          xa, x_col, nnz,
          xa_dense, @n,
          weight,
          @b,
          @bi, @add_implicit_features,
          @k, @k_user, @k_item, @k_main,
          @lambda_, lam_unique,
          @l1_lambda, l1_lam_unique,
          @scale_lam, @scale_lam_sideinfo,
          @scale_bias_const, scaling_bias_a,
          @w_main, @w_user, @w_implicit,
          n_max, @include_all_x,
          nil, #BtB,
          nil, #TransBtBinvBt,
          nil, #BtXbias,
          nil, #BeTBeChol,
          nil, #BiTBi,
          nil, #CtCw,
          nil, #TransCtCinvCt,
          nil, #CtUbias,
          nil, #B_plus_bias,
          include_ix, n_include,
          exclude_ix, n_exclude,
          outp_ix, outp_score,
          count, @nthreads
        ]
        check_status FFI.topN_new_collective_explicit(*fiddle_args(args))
      end

      top_n_output(outp_ix, outp_score)
    end

    def user_ids
      @user_map.keys
    end

    def item_ids
      @item_map.keys
    end

    def user_factors(user_id = nil)
      read_factors(@a, [@m, @m_u].max, @k_user + @k + @k_main, user_id, @user_map)
    end

    def item_factors(item_id = nil)
      read_factors(@b, [@n, @n_i].max, @k_item + @k + @k_main, item_id, @item_map)
    end

    def user_bias(user_id = nil)
      read_bias(@bias_a, user_id, @user_map) if @bias_a
    end

    def item_bias(item_id = nil)
      read_bias(@bias_b, item_id, @item_map) if @bias_b
    end

    def similar_items(item_id, count: 5)
      check_fit
      similar(item_id, @item_map, item_factors, count, item_index)
    end
    alias_method :item_recs, :similar_items

    def similar_users(user_id, count: 5)
      check_fit
      similar(user_id, @user_map, user_factors, count, user_index)
    end

    def to_json
      require "json"

      obj = {
        implicit: @implicit
      }

      # options
      obj[:factors] = @k
      obj[:epochs] = @niter
      obj[:verbose] = @verbose

      # factors
      obj[:user_ids] = @user_map.keys
      obj[:item_ids] = @item_map.keys
      obj[:rated] = @user_map.map { |_, u| (@rated[u] || {}).keys }
      obj[:user_factors] = json_dump_ptr(@a)
      obj[:item_factors] = json_dump_ptr(@b)

      # bias
      obj[:user_bias] = json_dump_ptr(@bias_a)
      obj[:item_bias] = json_dump_ptr(@bias_b)

      # mean
      obj[:global_mean] = @global_mean

      unless (@user_info_map.keys + @item_info_map.keys).all? { |v| v.is_a?(Symbol) }
        raise "Side info keys must be symbols to save"
      end

      # side info
      obj[:user_info_ids] = @user_info_map.keys
      obj[:item_info_ids] = @item_info_map.keys
      obj[:user_info_factors] = json_dump_ptr(@c)
      obj[:item_info_factors] = json_dump_ptr(@d)

      # implicit features
      obj[:add_implicit_features] = @add_implicit_features
      obj[:user_factors_implicit] = json_dump_ptr(@ai)
      obj[:item_factors_implicit] = json_dump_ptr(@bi)

      unless @implicit
        obj[:min_rating] = @min_rating
        obj[:max_rating] = @max_rating
      end

      obj[:user_means] = json_dump_ptr(@u_colmeans)

      JSON.generate(obj)
    end

    def self.load_json(json)
      require "json"

      obj = JSON.parse(json)

      recommender = new
      recommender.send(:json_load, obj)
      recommender
    end

    private

    def user_index
      @user_index ||= create_index(user_factors)
    end

    def item_index
      @item_index ||= create_index(item_factors)
    end

    def create_index(factors)
      require "ngt"

      index = Ngt::Index.new(@k, distance_type: "Cosine")
      index.batch_insert(factors)
      index
    end

    # TODO include bias
    def similar(id, map, factors, count, index)
      i = map[id]
      if i
        keys = map.keys
        result = index.search(factors[i], size: count + 1)[1..-1]
        result.map do |v|
          {
            # ids from batch_insert start at 1 instead of 0
            item_id: keys[v[:id] - 1],
            # convert cosine distance to cosine similarity
            score: 1 - v[:distance]
          }
        end
      else
        []
      end
    end

    def reset
      @fit = false
      @user_map.clear
      @item_map.clear
      @user_info_map.clear
      @item_info_map.clear
      @user_index = nil
      @item_index = nil
    end

    # TODO resize pointers as needed and reset values for new memory
    def partial_fit(train_set, user_info: nil, item_info: nil)
      train_set = to_dataset(train_set)

      unless @fit
        @implicit = !train_set.any? { |v| v[:rating] }
      end

      unless @implicit
        ratings = train_set.map { |o| o[:rating] }
        check_ratings(ratings)
      end

      check_training_set(train_set)
      update_maps(train_set)

      x_row = []
      x_col = []
      x_val = []
      value_key = @implicit ? :value : :rating
      @rated = Hash.new { |hash, key| hash[key] = {} }
      train_set.each do |v|
        u = @user_map[v[:user_id]]
        i = @item_map[v[:item_id]]
        @rated[u][i] = true

        x_row << u
        x_col << i
        x_val << (v[value_key] || 1)
      end
      @rated.default = nil

      @m = @user_map.size
      @n = @item_map.size
      nnz = train_set.size

      x_row = int_ptr(x_row)
      x_col = int_ptr(x_col)
      x = real_ptr(x_val)

      x_full = nil
      weight = nil
      lam_unique = nil
      l1_lam_unique = nil

      uu = nil
      ii = nil

      # side info
      u_row, u_col, u_sp, nnz_u, @m_u, p_ = process_info(user_info, @user_map, @user_info_map, :user_id)
      i_row, i_col, i_sp, nnz_i, @n_i, q = process_info(item_info, @item_map, @item_info_map, :item_id)

      @precompute_for_predictions = false

      # initialize w/ normal distribution
      reset_values = !@fit

      @a = Fiddle::Pointer.malloc([@m, @m_u].max * (@k_user + @k + @k_main) * Fiddle::SIZEOF_DOUBLE, Fiddle::RUBY_FREE)
      @b = Fiddle::Pointer.malloc([@n, @n_i].max * (@k_item + @k + @k_main) * Fiddle::SIZEOF_DOUBLE, Fiddle::RUBY_FREE)
      @c = p_ > 0 ? Fiddle::Pointer.malloc(p_ * (@k_user + @k) * Fiddle::SIZEOF_DOUBLE, Fiddle::RUBY_FREE) : nil
      @d = q > 0 ? Fiddle::Pointer.malloc(q * (@k_item + @k) * Fiddle::SIZEOF_DOUBLE, Fiddle::RUBY_FREE) : nil

      @bias_a = nil
      @bias_b = nil

      u_colmeans = Fiddle::Pointer.malloc(p_ * Fiddle::SIZEOF_DOUBLE, Fiddle::RUBY_FREE)
      i_colmeans = Fiddle::Pointer.malloc(q * Fiddle::SIZEOF_DOUBLE, Fiddle::RUBY_FREE)

      if @implicit
        set_implicit_vars

        args = [
          @a, @b,
          @c, @d,
          reset_values, @random_state,
          u_colmeans, i_colmeans,
          @m, @n, @k,
          x_row, x_col, x, nnz,
          @lambda_, lam_unique,
          @l1_lambda, l1_lam_unique,
          uu, @m_u, p_,
          ii, @n_i, q,
          u_row, u_col, u_sp, nnz_u,
          i_row, i_col, i_sp, nnz_i,
          @na_as_zero_user, @na_as_zero_item,
          @k_main, @k_user, @k_item,
          @w_main, @w_user, @w_item, real_ptr([@w_main_multiplier]),
          @alpha, @adjust_weight, @apply_log_transf,
          @niter, @nthreads, @verbose, @handle_interrupt,
          @use_cg, @max_cg_steps, @precondition_cg, @finalize_chol,
          @nonneg, @max_cd_steps, @nonneg_c, @nonneg_d,
          @precompute_for_predictions,
          nil, #precomputedBtB,
          nil, #precomputedBeTBe,
          nil, #precomputedBeTBeChol
          nil  #precomputedCtUbias
        ]
        check_status FFI.fit_collective_implicit_als(*fiddle_args(args))

        @global_mean = 0
      else
        @bias_a = Fiddle::Pointer.malloc([@m, @m_u].max * Fiddle::SIZEOF_DOUBLE, Fiddle::RUBY_FREE) if @user_bias
        @bias_b = Fiddle::Pointer.malloc([@n, @n_i].max * Fiddle::SIZEOF_DOUBLE, Fiddle::RUBY_FREE) if @item_bias

        if @add_implicit_features
          @ai = Fiddle::Pointer.malloc([@m, @m_u].max * (@k + @k_main) * Fiddle::SIZEOF_DOUBLE, Fiddle::RUBY_FREE)
          @bi = Fiddle::Pointer.malloc([@n, @n_i].max * (@k + @k_main) * Fiddle::SIZEOF_DOUBLE, Fiddle::RUBY_FREE)
        else
          @ai = nil
          @bi = nil
        end

        glob_mean = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE, Fiddle::RUBY_FREE)

        # TODO add
        scaling_bias_a = nil
        scaling_bias_b = nil

        args = [
          @bias_a, @bias_b,
          @a, @b,
          @c, @d,
          @ai, @bi,
          @add_implicit_features,
          reset_values, @random_state,
          glob_mean,
          u_colmeans, i_colmeans,
          @m, @n, @k,
          x_row, x_col, x, nnz,
          x_full,
          weight,
          @user_bias, @item_bias, @center,
          @lambda_, lam_unique,
          @l1_lambda, l1_lam_unique,
          @scale_lam, @scale_lam_sideinfo,
          @scale_bias_const, scaling_bias_a, scaling_bias_b,
          uu, @m_u, p_,
          ii, @n_i, q,
          u_row, u_col, u_sp, nnz_u,
          i_row, i_col, i_sp, nnz_i,
          @na_as_zero, @na_as_zero_user, @na_as_zero_item,
          @k_main, @k_user, @k_item,
          @w_main, @w_user, @w_item, @w_implicit,
          @niter, @nthreads, @verbose, @handle_interrupt,
          @use_cg, @max_cg_steps, @precondition_cg, @finalize_chol,
          @nonneg, @max_cd_steps, @nonneg_c, @nonneg_d,
          @precompute_for_predictions,
          @include_all_x,
          nil, #B_plus_bias,
          nil, #precomputedBtB,
          nil, #precomputedTransBtBinvBt,
          nil, #precomputedBtXbias
          nil, #precomputedBeTBeChol,
          nil, #precomputedBiTBi,
          nil, #precomputedTransCtCinvCt,
          nil, #precomputedCtCw
          nil  #precomputedCtUbias
        ]
        check_status FFI.fit_collective_explicit_als(*fiddle_args(args))

        @global_mean = real_array(glob_mean).first
      end

      @u_colmeans = u_colmeans

      @fit = true

      self
    end

    def set_params(
      k: 40, lambda_: 10.0, method: "als", use_cg: true,
      user_bias: true, item_bias: true, center: true, add_implicit_features: false,
      scale_lam: false, scale_lam_sideinfo: false, scale_bias_const: false,
      k_user: 0, k_item: 0, k_main: 0,
      w_main: 1.0, w_user: 1.0, w_item: 1.0, w_implicit: 0.5,
      l1_lambda: 0.0, center_u: true, center_i: true,
      maxiter: 800, niter: 10, parallelize: "separate", corr_pairs: 4,
      max_cg_steps: 3, precondition_cg: false, finalize_chol: true,
      na_as_zero: false, na_as_zero_user: false, na_as_zero_item: false,
      nonneg: false, nonneg_c: false, nonneg_d: false, max_cd_steps: 100,
      precompute_for_predictions: true, include_all_x: true,
      use_float: true, random_state: 1, verbose: true, print_every: 10,
      handle_interrupt: true, produce_dicts: false, nthreads: -1
    )
      @k = k
      @k_user = k_user
      @k_item = k_item
      @k_main = k_main
      @lambda_ = lambda_
      @w_main = w_main
      @w_user = w_user
      @w_item = w_item
      @w_implicit = w_implicit
      @user_bias = !!user_bias
      @item_bias = !!item_bias
      @method = method
      @add_implicit_features = !!add_implicit_features
      @use_cg = !!use_cg
      @max_cg_steps = max_cg_steps.to_i
      @max_cd_steps = max_cd_steps.to_i
      @finalize_chol = !!finalize_chol
      @maxiter = maxiter
      @niter = niter
      @parallelize = parallelize
      @na_as_zero = !!na_as_zero
      @na_as_zero_user = !!na_as_zero_user
      @na_as_zero_item = !!na_as_zero_item
      @nonneg = !!nonneg
      @nonneg_c = !!nonneg_c
      @nonneg_d = !!nonneg_d
      @precompute_for_predictions = !!precompute_for_predictions
      @include_all_x = true
      @use_float = !!use_float
      @verbose = !!verbose
      @print_every = print_every
      @corr_pairs = corr_pairs
      @random_state = random_state.to_i
      @produce_dicts = !!produce_dicts
      @handle_interrupt = !!handle_interrupt
      nthreads = Etc.nprocessors if nthreads < 0
      @nthreads = nthreads

      @center = center
      @scale_lam = scale_lam
      @scale_lam_sideinfo = scale_lam_sideinfo
      @scale_bias_const = scale_bias_const
      @l1_lambda = l1_lambda
      @precondition_cg = precondition_cg

      # TODO center_u, center_i
    end

    def update_maps(train_set)
      raise ArgumentError, "Missing user_id" if train_set.any? { |v| v[:user_id].nil? }
      raise ArgumentError, "Missing item_id" if train_set.any? { |v| v[:item_id].nil? }

      train_set.each do |v|
        @user_map[v[:user_id]] ||= @user_map.size
        @item_map[v[:item_id]] ||= @item_map.size
      end
    end

    def check_ratings(ratings)
      unless ratings.all? { |r| !r.nil? }
        raise ArgumentError, "Missing ratings"
      end
      unless ratings.all? { |r| r.is_a?(Numeric) }
        raise ArgumentError, "Ratings must be numeric"
      end
    end

    def check_training_set(train_set)
      raise ArgumentError, "No training data" if train_set.empty?
    end

    def check_fit
      raise "Not fit" unless @fit
    end

    def to_dataset(dataset)
      if defined?(Rover::DataFrame) && dataset.is_a?(Rover::DataFrame)
        # convert keys to symbols
        dataset = dataset.dup
        dataset.keys.each do |k, v|
          dataset[k.to_sym] ||= dataset.delete(k)
        end
        dataset.to_a
      elsif defined?(Daru::DataFrame) && dataset.is_a?(Daru::DataFrame)
        # convert keys to symbols
        dataset = dataset.dup
        new_names = dataset.vectors.to_a.map { |k| [k, k.to_sym] }.to_h
        dataset.rename_vectors!(new_names)
        dataset.to_a[0]
      else
        dataset
      end
    end

    def read_factors(ptr, d1, d2, id, map)
      width = d2 * Fiddle::SIZEOF_DOUBLE
      if id
        i = map[id]
        ptr[i * width, width].unpack("d*") if i
      else
        arr = []
        offset = 0
        d1.times do |i|
          arr << ptr[offset, width].unpack("d*")
          offset += width
        end
        arr
      end
    end

    def read_bias(ptr, id, map)
      if id
        i = map[id]
        ptr[i * Fiddle::SIZEOF_DOUBLE, Fiddle::SIZEOF_DOUBLE].unpack1("d") if i
      else
        real_array(ptr)
      end
    end

    def prepare_top_n(count: nil, rated: nil, item_ids: nil)
      if item_ids
        # remove missing ids
        item_ids = item_ids.map { |v| @item_map[v] }.compact
        return [] if item_ids.empty?

        include_ix = int_ptr(item_ids)
        n_include = item_ids.size

        count = n_include if n_include < count
      else
        include_ix = nil
        n_include = 0
      end

      if rated && !item_ids
        # assumes rated is unique and all items are known
        # calling code is responsible for this
        exclude_ix = int_ptr(rated)
        n_exclude = rated.size
        remaining = @item_map.size - n_exclude
        return [] if remaining == 0
        count = remaining if remaining < count
      else
        exclude_ix = nil
        n_exclude = 0
      end

      outp_ix = Fiddle::Pointer.malloc(count * Fiddle::SIZEOF_INT, Fiddle::RUBY_FREE)
      outp_score = Fiddle::Pointer.malloc(count * Fiddle::SIZEOF_DOUBLE, Fiddle::RUBY_FREE)

      [include_ix, n_include, exclude_ix, n_exclude, outp_ix, outp_score, count]
    end

    def top_n(a_vec:, a_bias:, count:, rated: nil, item_ids: nil, row_index:)
      prep = prepare_top_n(count: count, rated: rated, item_ids: item_ids)
      return [] if prep.empty?
      include_ix, n_include, exclude_ix, n_exclude, outp_ix, outp_score, count = prep

      if @implicit
        check_status FFI.topN_old_collective_implicit(
          a_vec,
          @a, row_index,
          @b,
          @k, @k_user, @k_item, @k_main,
          include_ix, n_include,
          exclude_ix, n_exclude,
          outp_ix, outp_score,
          count, @n, @nthreads
        )
      else
        # TODO add param
        n_max = @n

        check_status FFI.topN_old_collective_explicit(
          a_vec, a_bias,
          @a, @bias_a, row_index,
          @b,
          @bias_b,
          @global_mean,
          @k, @k_user, @k_item, @k_main,
          include_ix, n_include,
          exclude_ix, n_exclude,
          outp_ix, outp_score,
          count, @n, n_max, @include_all_x ? 1 : 0, @nthreads
        )
      end

      top_n_output(outp_ix, outp_score)
    end

    def top_n_output(outp_ix, outp_score)
      imap = @item_map.map(&:reverse).to_h
      item_ids = int_array(outp_ix).map { |v| imap[v] }
      scores = real_array(outp_score)

      item_ids.zip(scores).map do |item_id, score|
        {item_id: item_id, score: score}
      end
    end

    # convert boolean to int
    def fiddle_args(args)
      args.map { |v| v == true || v == false ? (v ? 1 : 0) : v }
    end

    def check_status(ret_val)
      case ret_val
      when 0
        # success
      when 1
        raise "Could not allocate sufficient memory"
      else
        raise "Bad status: #{ret_val}"
      end
    end

    def process_info(info, map, info_map, key)
      return [nil, nil, nil, 0, 0, 0] unless info

      info = to_dataset(info)

      row = []
      col = []
      val = []
      info.each do |ri|
        rk = ri[key]
        raise ArgumentError, "Missing #{key}" unless rk

        r = (map[rk] ||= map.size)
        ri.each do |k, v|
          next if k == key
          row << r
          col << (info_map[k] ||= info_map.size)
          val << v
        end
      end
      [int_ptr(row), int_ptr(col), real_ptr(val), val.size, map.size, info_map.size]
    end

    def int_ptr(v)
      v.pack("i*")
    end

    def real_ptr(v)
      v.pack("d*")
    end

    def int_array(ptr)
      ptr.to_s(ptr.size).unpack("i*")
    end

    def real_array(ptr)
      ptr.to_s(ptr.size).unpack("d*")
    end

    def set_implicit_vars
      @w_main_multiplier = 1.0
      @alpha = 1.0
      @adjust_weight = false # downweight?
      @apply_log_transf = false

      # different defaults
      @lambda_ = 1e0
      @w_user = 10
      @w_item = 10
      @finalize_chol = false
    end

    def json_dump_ptr(ptr)
      [ptr.to_s(ptr.size)].pack("m0") if ptr
    end

    def json_load_ptr(str)
      Fiddle::Pointer[str.unpack1("m0")] if str
    end

    def json_load(obj)
      @implicit = obj["implicit"]

      # options
      set_params(
        k: obj["factors"],
        niter: obj["epochs"],
        verbose: obj["verbose"],
        user_bias: !obj["user_bias"].nil?,
        item_bias: !obj["item_bias"].nil?,
        add_implicit_features: obj["add_implicit_features"]
      )

      # factors
      @user_map = obj["user_ids"].map.with_index.to_h
      @item_map = obj["item_ids"].map.with_index.to_h
      @rated = obj["rated"].map.with_index.to_h { |r, i| [i, r.to_h { |v| [v, true] }] }
      @a = json_load_ptr(obj["user_factors"])
      @b = json_load_ptr(obj["item_factors"])

      # bias
      @bias_a = json_load_ptr(obj["user_bias"])
      @bias_b = json_load_ptr(obj["item_bias"])

      # mean
      @global_mean = obj["global_mean"]

      # side info
      @user_info_map = obj["user_info_ids"].map(&:to_sym).map.with_index.to_h
      @item_info_map = obj["item_info_ids"].map(&:to_sym).map.with_index.to_h
      @c = json_load_ptr(obj["user_info_factors"])
      @d = json_load_ptr(obj["item_info_factors"])

      # implicit features
      @add_implicit_features = obj["add_implicit_features"]
      @ai = json_load_ptr(obj["user_factors_implicit"])
      @bi = json_load_ptr(obj["item_factors_implicit"])

      unless @implicit
        @min_rating = obj["min_rating"]
        @max_rating = obj["max_rating"]
      end

      @u_colmeans = json_load_ptr(obj["user_means"])

      @m = @user_map.size
      @n = @item_map.size
      @m_u = @user_info_map.size
      @n_i = @item_info_map.size

      set_implicit_vars if @implicit

      @fit = @m > 0
    end
  end
end
