require "ffi"

module Cmfrec
  module FFI
    extend ::FFI::Library

    puts "try loading"
    # p File.exist?(Cmfrec.ffi_lib.first)
    blas_path = File.expand_path("../../blas/libblas.dll", __dir__)
    p File.exist?(blas_path)
    ffi_lib [blas_path]
    puts "blas loaded"

    lapack_path = File.expand_path("../../blas/liblapack.dll", __dir__)
    p File.exist?(lapack_path)
    ffi_lib [lapack_path]
    puts "lapack loaded"

    ffi_lib Cmfrec.ffi_lib
    puts "cmfrec loaded"

    # extend Fiddle::Importer

    # libs = Cmfrec.ffi_lib.dup
    # begin
    #   puts "load"
    #   p File.exist?(libs.first)
    #   dlload Fiddle.dlopen(libs.shift)
    # rescue Fiddle::DLError => e
    #   retry if libs.any?
    #   raise e
    # end

    # typealias "bool", "char"
    # # determined by CMakeLists.txt
    # typealias "int_t", "int"
    # typealias "real_t", "double"

    # extern "int_t fit_collective_explicit_als(real_t *biasA, real_t *biasB, real_t *A, real_t *B, real_t *C, real_t *D, real_t *Ai, real_t *Bi, bool add_implicit_features, bool reset_values, int_t seed, real_t *glob_mean, real_t *U_colmeans, real_t *I_colmeans, int_t m, int_t n, int_t k, int_t X_row[], int_t X_col[], real_t *X, size_t nnz, real_t *Xfull, real_t *weight, bool user_bias, bool item_bias, real_t lam, real_t *lam_unique, real_t *U, int_t m_u, int_t p, real_t *II, int_t n_i, int_t q, int_t U_row[], int_t U_col[], real_t *U_sp, size_t nnz_U, int_t I_row[], int_t I_col[], real_t *I_sp, size_t nnz_I, bool NA_as_zero_X, bool NA_as_zero_U, bool NA_as_zero_I, int_t k_main, int_t k_user, int_t k_item, real_t w_main, real_t w_user, real_t w_item, real_t w_implicit, int_t niter, int_t nthreads, bool verbose, bool handle_interrupt, bool use_cg, int_t max_cg_steps, bool finalize_chol, bool nonneg, int_t max_cd_steps, bool nonneg_C, bool nonneg_D, bool precompute_for_predictions, bool include_all_X, real_t *B_plus_bias, real_t *precomputedBtB, real_t *precomputedTransBtBinvBt, real_t *precomputedBeTBeChol, real_t *precomputedBiTBi, real_t *precomputedTransCtCinvCt, real_t *precomputedCtCw)"
    # extern "int_t fit_collective_implicit_als(real_t *A, real_t *B, real_t *C, real_t *D, bool reset_values, int_t seed, real_t *U_colmeans, real_t *I_colmeans, int_t m, int_t n, int_t k, int_t X_row[], int_t X_col[], real_t *X, size_t nnz, real_t lam, real_t *lam_unique, real_t *U, int_t m_u, int_t p, real_t *II, int_t n_i, int_t q, int_t U_row[], int_t U_col[], real_t *U_sp, size_t nnz_U, int_t I_row[], int_t I_col[], real_t *I_sp, size_t nnz_I, bool NA_as_zero_U, bool NA_as_zero_I, int_t k_main, int_t k_user, int_t k_item, real_t w_main, real_t w_user, real_t w_item, real_t *w_main_multiplier, real_t alpha, bool adjust_weight, bool apply_log_transf, int_t niter, int_t nthreads, bool verbose, bool handle_interrupt, bool use_cg, int_t max_cg_steps, bool finalize_chol, bool nonneg, int_t max_cd_steps, bool nonneg_C, bool nonneg_D, bool precompute_for_predictions, real_t *precomputedBtB, real_t *precomputedBeTBe, real_t *precomputedBeTBeChol)"
    # extern "int_t factors_collective_explicit_single(real_t *a_vec, real_t *a_bias, real_t *u_vec, int_t p, real_t *u_vec_sp, int_t u_vec_X_col[], size_t nnz_u_vec, real_t *u_bin_vec, int_t pbin, bool NA_as_zero_U, bool NA_as_zero_X, bool nonneg, real_t *C, real_t *Cb, real_t glob_mean, real_t *biasB, real_t *U_colmeans, real_t *Xa, int_t X_col[], size_t nnz, real_t *Xa_dense, int_t n, real_t *weight, real_t *B, real_t *Bi, bool add_implicit_features, int_t k, int_t k_user, int_t k_item, int_t k_main, real_t lam, real_t *lam_unique, real_t w_main, real_t w_user, real_t w_implicit, int_t n_max, bool include_all_X, real_t *TransBtBinvBt, real_t *BtB, real_t *BeTBeChol, real_t *BiTBi, real_t *CtCw, real_t *TransCtCinvCt, real_t *B_plus_bias)"
    # extern "int_t factors_collective_implicit_single(real_t *a_vec, real_t *u_vec, int_t p, real_t *u_vec_sp, int_t u_vec_X_col[], size_t nnz_u_vec, bool NA_as_zero_U, bool nonneg, real_t *U_colmeans, real_t *B, int_t n, real_t *C, real_t *Xa, int_t X_col[], size_t nnz, int_t k, int_t k_user, int_t k_item, int_t k_main, real_t lam, real_t alpha, real_t w_main, real_t w_user, real_t w_main_multiplier, bool apply_log_transf, real_t *BeTBe, real_t *BtB, real_t *BeTBeChol)"
    # extern "void predict_multiple(real_t *restrict A, int_t k_user, real_t *restrict B, int_t k_item, real_t *restrict biasA, real_t *restrict biasB, real_t glob_mean, int_t k, int_t k_main, int_t m, int_t n, int_t predA[], int_t predB[], size_t nnz, real_t *restrict outp, int_t nthreads)"
    # extern "int_t predict_X_old_collective_explicit(int_t row[], int_t col[], real_t *restrict predicted, size_t n_predict, real_t *restrict A, real_t *restrict biasA, real_t *restrict B, real_t *restrict biasB, real_t glob_mean, int_t k, int_t k_user, int_t k_item, int_t k_main, int_t m, int_t n_max, int_t nthreads)"
    # extern "int_t topN(real_t *restrict a_vec, int_t k_user, real_t *restrict B, int_t k_item, real_t *restrict biasB, real_t glob_mean, real_t biasA, int_t k, int_t k_main, int_t *restrict include_ix, int_t n_include, int_t *restrict exclude_ix, int_t n_exclude, int_t *restrict outp_ix, real_t *restrict outp_score, int_t n_top, int_t n, int_t nthreads)"
  end
end
