module SpiderETVariationFit

using Dates
using DataFrames
using JLD2
using LinearAlgebra
using Measurements
using NPZ
using Random
using Statistics

const SPIDER_REPO = normpath(joinpath(@__DIR__, ".."))
const MADMAX_ROOT = normpath(joinpath(SPIDER_REPO, ".."))
include(joinpath(MADMAX_ROOT, "MADBead", "MADBead.jl"))

const DEFAULT_DATA_FILE = joinpath(SPIDER_REPO, "data", "ET_results_variations.npz")
const DEFAULT_CACHE_DIR = joinpath(SPIDER_REPO, "data", "spider_et_variation_fit_cache")
const DEFAULT_RESULTS_FILE = joinpath(SPIDER_REPO, "data", "5_result_bf_determination_spider_ET_variations.jld2")
const CONFIGURATION_LOOKUP_COLUMNS = ["configuration_index", "line_count", "z_slice_count", "gap_count"]
const REQUIRED_VARIATION_KEYS = [
    "ETs_reduced",
    "z_reduced",
    "z_count",
    "ETs_full",
    "z_full",
    "frequencies",
    "configuration_indices",
    "configuration_lookup",
    "z_ixs_used",
    "z_ixs_used_count",
]

const DEFAULT_N_MC = 100
const DEFAULT_REL_ERR = 0.05
const DEFAULT_OUTLIER_WINDOW = 51
const DEFAULT_OUTLIER_THRESHOLD = 12.0
const DEFAULT_SEED = 904230
const DEFAULT_SECONDARY_BAND = (19e9, 20e9)

value_of(x) = hasproperty(x, :val) ? getfield(x, :val) : x
error_of(x) = hasproperty(x, :err) ? getfield(x, :err) : zero(value_of(x))

function timestamp()
    return Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS")
end

function full_cache_path(cache_dir=DEFAULT_CACHE_DIR)
    return joinpath(cache_dir, "full_reference.jld2")
end

function config_cache_path(cache_dir, configuration_index)
    return joinpath(cache_dir, "config_$(lpad(string(configuration_index), 3, "0")).jld2")
end

function metrics_csv_path(cache_dir)
    return joinpath(cache_dir, "configuration_metrics.csv")
end

function assert_finite_vector(name, values)
    @assert ndims(values) == 1 "$(name) must be a vector."
    @assert !isempty(values) "$(name) must not be empty."
    @assert all(isfinite.(values)) "$(name) contains non-finite values."
    return values
end

function orient_ET(name, ET_zf, ν, z_rel)
    @assert ndims(ET_zf) == 2 "$(name) must be a 2D array with shape (z, frequency)."
    @assert size(ET_zf, 1) == length(z_rel) "$(name) z dimension does not match its z array."
    @assert size(ET_zf, 2) == length(ν) "$(name) frequency dimension does not match frequencies."
    finite_mask = isfinite.(real.(ET_zf)) .& isfinite.(imag.(ET_zf))
    bad_count = count(.!finite_mask)
    if bad_count > 0
        @warn "$(name) contains $(bad_count) non-finite ET value(s); cleanup will skip affected reduced frequencies."
    end
    return permutedims(ET_zf, (2, 1))
end

function make_ET_measurements(ET_fz; rel_err=DEFAULT_REL_ERR)
    @assert 0 <= rel_err < 1 "rel_err should be a fractional uncertainty."
    real_err = rel_err .* abs.(real.(ET_fz))
    imag_err = rel_err .* abs.(imag.(ET_fz))
    return (real.(ET_fz) .± real_err) .+ 1im .* (imag.(ET_fz) .± imag_err)
end

function build_dataset(label, ET_zf, z_rel, ν; z_mirror, rel_err)
    z_rel = vec(Float64.(z_rel))
    ET_zf = Array{ComplexF64}(ET_zf)
    ET_fz = orient_ET(label, ET_zf, ν, z_rel)
    z_abs = z_mirror .- z_rel
    @assert all(z_abs .< z_mirror) "$(label) z values must lie left of the mirror."

    order = sortperm(z_abs)
    ET_fz = ET_fz[:, order]
    z_rel = z_rel[order]
    z_abs = z_abs[order]

    return (
        label=label,
        z_rel=z_rel,
        z_abs=z_abs,
        ET=ET_fz,
        ET_meas=make_ET_measurements(ET_fz; rel_err=rel_err),
    )
end

function configuration_label(configuration_index, line_count, z_slice_count, gap_count)
    return "cfg$(configuration_index)_L$(line_count)_Z$(z_slice_count)_G$(gap_count)"
end

function config_z_ixs_used(z_ixs_pad, z_ixs_count, row)
    if ndims(z_ixs_pad) == 2
        counts = vec(z_ixs_count)
        @assert size(z_ixs_pad, 1) == length(counts) "z_ixs_used row count mismatch."
        n = counts[row]
        @assert 0 <= n <= size(z_ixs_pad, 2) "z_ixs_used_count contains an invalid value at row $(row)."
        return n == 0 ? Int[] : collect(vec(z_ixs_pad[row, 1:n]))
    elseif ndims(z_ixs_pad) == 3
        @assert ndims(z_ixs_count) == 2 "3D z_ixs_used requires 2D z_ixs_used_count."
        @assert size(z_ixs_pad, 1) == size(z_ixs_count, 1) "z_ixs_used configuration count mismatch."
        @assert size(z_ixs_pad, 2) == size(z_ixs_count, 2) "z_ixs_used gap count mismatch."
        z_ixs = Int[]
        for gap in axes(z_ixs_count, 2)
            n = z_ixs_count[row, gap]
            @assert 0 <= n <= size(z_ixs_pad, 3) "z_ixs_used_count contains an invalid value at row $(row), gap $(gap)."
            if n > 0
                append!(z_ixs, vec(z_ixs_pad[row, gap, 1:n]))
            end
        end
        return collect(z_ixs[z_ixs .>= 0])
    else
        error("z_ixs_used must be either 2D or 3D.")
    end
end

function load_ET_variation_results(data_file=DEFAULT_DATA_FILE; z_mirror=2.298, rel_err=DEFAULT_REL_ERR)
    @assert isfile(data_file) "Missing ET variation data file: $(data_file)"
    data = npzread(data_file)
    for key in REQUIRED_VARIATION_KEYS
        @assert haskey(data, key) "Missing key $(key) in $(data_file)."
    end

    ν = assert_finite_vector("frequencies", vec(Float64.(data["frequencies"])))
    @assert issorted(ν) "frequencies must be sorted in ascending order."

    z_full = assert_finite_vector("z_full", vec(Float64.(data["z_full"])))
    ET_full = Array{ComplexF64}(data["ETs_full"])
    @assert size(ET_full) == (length(z_full), length(ν)) "ETs_full must have shape (z_full, frequency)."
    full = build_dataset("full", ET_full, z_full, ν; z_mirror=z_mirror, rel_err=rel_err)

    ET_pad = ComplexF64.(data["ETs_reduced"])
    z_pad = Float64.(data["z_reduced"])
    z_count = Int.(vec(data["z_count"]))
    z_ixs_pad = Int.(data["z_ixs_used"])
    z_ixs_count = Int.(data["z_ixs_used_count"])
    lookup = Int.(data["configuration_lookup"])
    indices = Int.(vec(data["configuration_indices"]))

    @assert ndims(ET_pad) == 3 "ETs_reduced must have shape (configuration, max_z, frequency)."
    @assert ndims(z_pad) == 2 "z_reduced must have shape (configuration, max_z)."
    @assert size(lookup, 2) == length(CONFIGURATION_LOOKUP_COLUMNS) "Unexpected configuration lookup shape."
    @assert size(lookup, 1) == size(ET_pad, 1) == size(z_pad, 1) == length(z_count) == length(indices) "Configuration counts do not agree."
    @assert size(ET_pad, 2) == size(z_pad, 2) "Reduced ET and z padding dimensions do not agree."
    @assert size(ET_pad, 3) == length(ν) "Padded reduced ET frequency dimension mismatch."
    @assert all(0 .< z_count .<= size(ET_pad, 2)) "z_count contains invalid values."
    @assert all(z_ixs_count .>= 0) "z_ixs_used_count contains invalid values."
    @assert size(z_ixs_pad, 1) == size(ET_pad, 1) "z_ixs_used configuration count mismatch."

    reduced_configurations = NamedTuple[]
    for row in axes(lookup, 1)
        configuration_index, line_count, z_slice_count, gap_count = lookup[row, :]
        @assert configuration_index == indices[row] "configuration_indices and configuration_lookup disagree at row $(row)."
        n_z = z_count[row]
        ET_zf = ET_pad[row, 1:n_z, :]
        z_rel = vec(z_pad[row, 1:n_z])
        @assert all(isfinite.(z_rel)) "Configuration $(configuration_index) contains non-finite z values."
        z_ixs_used = config_z_ixs_used(z_ixs_pad, z_ixs_count, row)
        label = configuration_label(configuration_index, line_count, z_slice_count, gap_count)
        dataset = build_dataset(label, ET_zf, z_rel, ν; z_mirror=z_mirror, rel_err=rel_err)
        push!(
            reduced_configurations,
            merge(
                dataset,
                (
                    configuration_index=configuration_index,
                    line_count=line_count,
                    z_slice_count=z_slice_count,
                    gap_count=gap_count,
                    lookup_row=row,
                    z_ixs_used=collect(z_ixs_used),
                ),
            ),
        )
    end

    return ν, reduced_configurations, full, lookup
end

function robust_median_mad(values)
    finite_values = values[isfinite.(values)]
    if isempty(finite_values)
        return 0.0, 0.0
    end
    med = median(finite_values)
    scale = 1.4826 * median(abs.(finite_values .- med))
    return med, scale
end

function hampel_component_flags(values; window=DEFAULT_OUTLIER_WINDOW, threshold=DEFAULT_OUTLIER_THRESHOLD)
    @assert isodd(window) "Hampel window must be odd."
    @assert window >= 3 "Hampel window must contain at least 3 points."
    x = Float64.(values)
    flags = falses(length(x))
    scores = zeros(Float64, length(x))

    global_med, global_scale = robust_median_mad(x)
    half_window = window ÷ 2

    for i in eachindex(x)
        if !isfinite(x[i])
            flags[i] = true
            scores[i] = Inf
            continue
        end

        lo = max(firstindex(x), i - half_window)
        hi = min(lastindex(x), i + half_window)
        neighbor_idx = vcat(collect(lo:i-1), collect(i+1:hi))
        neighbors = x[neighbor_idx]
        finite_neighbors = neighbors[isfinite.(neighbors)]
        local_med, local_scale = isempty(finite_neighbors) ? (global_med, global_scale) : robust_median_mad(finite_neighbors)
        scale = max(local_scale, global_scale, eps(Float64))
        scores[i] = abs(x[i] - local_med) / scale
        flags[i] = scores[i] > threshold
    end

    return flags, scores
end

function detect_reduced_ET_outliers(dataset, ν; window=DEFAULT_OUTLIER_WINDOW, threshold=DEFAULT_OUTLIER_THRESHOLD)
    @assert size(dataset.ET, 1) == length(ν) "Reduced ET frequency dimension must match ν."

    point_mask = falses(size(dataset.ET))
    real_score = zeros(Float64, size(dataset.ET))
    imag_score = zeros(Float64, size(dataset.ET))

    for z_idx in axes(dataset.ET, 2)
        real_flags, real_scores = hampel_component_flags(real.(dataset.ET[:, z_idx]); window=window, threshold=threshold)
        imag_flags, imag_scores = hampel_component_flags(imag.(dataset.ET[:, z_idx]); window=window, threshold=threshold)
        point_mask[:, z_idx] .= real_flags .| imag_flags
        real_score[:, z_idx] .= real_scores
        imag_score[:, z_idx] .= imag_scores
    end

    frequency_skip_mask = vec(any(point_mask, dims=2))
    keep_frequency_mask = .!frequency_skip_mask

    return (
        point_mask=point_mask,
        real_score=real_score,
        imag_score=imag_score,
        frequency_skip_mask=frequency_skip_mask,
        keep_frequency_mask=keep_frequency_mask,
        skipped_frequencies=ν[frequency_skip_mask],
        window=window,
        threshold=threshold,
    )
end

function apply_frequency_mask(values::AbstractVector, keep_frequency_mask)
    @assert length(values) == length(keep_frequency_mask) "Frequency mask length mismatch."
    return values[keep_frequency_mask]
end

function apply_frequency_mask(dataset::NamedTuple, keep_frequency_mask; label_suffix="clean")
    @assert size(dataset.ET, 1) == length(keep_frequency_mask) "Dataset frequency dimension does not match mask."
    return merge(
        dataset,
        (
            label="$(dataset.label)_$(label_suffix)",
            ET=dataset.ET[keep_frequency_mask, :],
            ET_meas=dataset.ET_meas[keep_frequency_mask, :],
        ),
    )
end

function outlier_summary_table(configurations, outliers, ν)
    df = DataFrame(
        configuration_index=Int[],
        line_count=Int[],
        z_slice_count=Int[],
        gap_count=Int[],
        n_reduced_z=Int[],
        n_skipped_frequency=Int[],
        n_kept_frequency=Int[],
        kept_frequency_fraction=Float64[],
        first_skipped_GHz=Union{Missing, Float64}[],
        last_skipped_GHz=Union{Missing, Float64}[],
    )

    for (dataset, outlier) in zip(configurations, outliers)
        skipped = outlier.skipped_frequencies
        push!(
            df,
            (
                dataset.configuration_index,
                dataset.line_count,
                dataset.z_slice_count,
                dataset.gap_count,
                length(dataset.z_abs),
                count(outlier.frequency_skip_mask),
                count(outlier.keep_frequency_mask),
                count(outlier.keep_frequency_mask) / length(ν),
                isempty(skipped) ? missing : first(skipped) * 1e-9,
                isempty(skipped) ? missing : last(skipped) * 1e-9,
            ),
        )
    end
    return df
end

function prepare_inputs(; data_file=DEFAULT_DATA_FILE, z_mirror=2.298, rel_err=DEFAULT_REL_ERR, outlier_window=DEFAULT_OUTLIER_WINDOW, outlier_threshold=DEFAULT_OUTLIER_THRESHOLD)
    ν, reduced_configurations, full_data, configuration_lookup = load_ET_variation_results(data_file; z_mirror=z_mirror, rel_err=rel_err)
    outliers_by_config = [detect_reduced_ET_outliers(dataset, ν; window=outlier_window, threshold=outlier_threshold) for dataset in reduced_configurations]
    ν_clean_by_config = [apply_frequency_mask(ν, outlier.keep_frequency_mask) for outlier in outliers_by_config]
    reduced_configurations_clean = [apply_frequency_mask(dataset, outlier.keep_frequency_mask) for (dataset, outlier) in zip(reduced_configurations, outliers_by_config)]
    df_outlier_summary = outlier_summary_table(reduced_configurations, outliers_by_config, ν)

    return (
        ν=ν,
        reduced_configurations=reduced_configurations,
        full_data=full_data,
        configuration_lookup=configuration_lookup,
        outliers_by_config=outliers_by_config,
        ν_clean_by_config=ν_clean_by_config,
        reduced_configurations_clean=reduced_configurations_clean,
        df_outlier_summary=df_outlier_summary,
    )
end

function model_priors(; z_mirror_measurement=2.298 ± 0.0, N_mc=DEFAULT_N_MC)
    ϵ_b = 9.23 ± 0.2
    r_b = (2.93e-3 ± 15e-6) / 2
    P_in = 1
    r_d = 0.15
    A = π * r_d^2
    σ_Al = (5.0 ± 1) * 1e7

    ϵ_d = 9.3 ± 0.1
    tanD_d = 1e-5 ± 1e-6
    d_d = (1 ± 0.05) * 1e-3

    n_disk = 3 ± 0
    E0_0 = 1 ± 1
    d_v_i = [8.265 ± 0.1, 9.813 ± 0.1, 9.813 ± 0.1] * 1e-3

    p0_all = Dict("E_0"=>E0_0, "z_m"=>z_mirror_measurement, "σ_m"=>σ_Al, "n_disk"=>n_disk)
    for i in 1:Int(n_disk.val)
        p0_all["d_v_$i"] = d_v_i[i]
        p0_all["d_d_$i"] = d_d
        p0_all["ϵ_d_$i"] = ϵ_d
        p0_all["tanD_d_$i"] = tanD_d
    end
    p0_all["r_b"] = r_b
    p0_all["ϵ_b"] = ϵ_b

    keys_optim = ["E_0", "d_v_1", "d_v_2", "d_v_3"]
    keys_helper = ["n_disk"]
    keys_fixed = [setdiff(keys(p0_all), keys_optim, keys_helper)...]

    return (
        z_m_FM504_M=z_mirror_measurement,
        P_in=P_in,
        A=A,
        p0_all=p0_all,
        keys_optim=keys_optim,
        keys_helper=keys_helper,
        keys_fixed=keys_fixed,
        N_mc=N_mc,
    )
end

function final_parameter_table(ν, p_all_ν_mc, p0_all)
    p_final_ν = DataFrame("f"=>ν)
    for key in keys(p0_all)
        p_final_ν[!, key] = mean.(p_all_ν_mc[!, key]) .± std.(p_all_ν_mc[!, key])
    end
    return p_final_ν
end

function integrate_mc_fields(ν, p_all_ν_mc, p0_all, N_mc)
    int_dz_E_mc = zeros(ComplexF64, length(ν), N_mc)
    Threads.@threads for f in eachindex(ν)
        for i in 1:N_mc
            p_dict = Dict(key=>p_all_ν_mc[f, key][i] for key in keys(p0_all))
            int_dz_E_mc[f, i] = MADBead.int_dz_E_param(p_dict; f=ν[f])
        end
    end
    return int_dz_E_mc
end

function boostfactor_from_int_dz(ν, int_dz_E_mc; P_in, A, J_0=1)
    ∫dV_E = mean(abs.(int_dz_E_mc), dims=2)[:] .± std(abs.(int_dz_E_mc), dims=2)[:]
    P_sig = J_0^2 / (16 * P_in) .* abs2.(∫dV_E)
    P_0 = MADBead.c_const * A * J_0^2 ./ (2 * MADBead.ϵ0 * (2π .* ν).^2)
    boostfactor = P_sig ./ P_0
    return ∫dV_E, boostfactor
end

function ET_dataframe(ν, dataset)
    df_ET = DataFrame("f"=>ν)
    df_ET[!, "ET"] = [dataset.ET_meas[f, :] for f in 1:length(ν)]
    df_ET[!, "z_abs"] = [dataset.z_abs for _ in 1:length(ν)]
    df_ET[!, "z_rel"] = [dataset.z_rel for _ in 1:length(ν)]
    return df_ET
end

function run_boostfactor_analysis(dataset, ν, priors)
    println("Running $(dataset.label) fit with $(length(dataset.z_abs)) z points, $(length(ν)) frequencies, N_mc=$(priors.N_mc), threads=$(Threads.nthreads()).")
    p_all_ν_mc = MADBead.fit_E_z_MC(ν, dataset.z_abs, dataset.ET_meas, priors.p0_all, priors.keys_optim, priors.keys_fixed, priors.keys_helper, priors.N_mc)
    p_final_ν = final_parameter_table(ν, p_all_ν_mc, priors.p0_all)
    int_dz_E_mc = integrate_mc_fields(ν, p_all_ν_mc, priors.p0_all, priors.N_mc)
    ∫dV_E, boostfactor = boostfactor_from_int_dz(ν, int_dz_E_mc; P_in=priors.P_in, A=priors.A)

    df_bf_analysis = DataFrame("f"=>ν, "bf"=>boostfactor, "int_dV_E"=>∫dV_E)
    df_bf_analysis = innerjoin(df_bf_analysis, p_final_ν, on="f")

    df_bf_mc_analysis = DataFrame("f"=>ν)
    df_bf_mc_analysis[!, "int_dV_E_mc"] = [int_dz_E_mc[f, :] for f in 1:length(ν)]
    df_bf_mc_analysis = innerjoin(df_bf_mc_analysis, p_all_ν_mc, on="f")

    return (
        label=dataset.label,
        p_all_ν_mc=p_all_ν_mc,
        p_final_ν=p_final_ν,
        int_dz_E_mc=int_dz_E_mc,
        ∫dV_E=∫dV_E,
        boostfactor=boostfactor,
        df_ET=ET_dataframe(ν, dataset),
        df_bf_analysis=df_bf_analysis,
        df_bf_mc_analysis=df_bf_mc_analysis,
    )
end

function p_nominal_at_frequency(result, row, p0_all)
    return Dict(key=>Float64(value_of(result.p_final_ν[row, key])) for key in keys(p0_all))
end

function fit_quality_metrics(dataset, ν, result, p0_all; rel_floor=DEFAULT_REL_ERR)
    residual_sq_sum = 0.0
    reference_sq_sum = 0.0
    chi_sq_sum = 0.0
    rel_abs_errors = Float64[]
    n_points = 0

    for f_idx in eachindex(ν)
        p = p_nominal_at_frequency(result, f_idx, p0_all)
        δ = MADBead.calc_δ_e_mie(p["ϵ_b"], p["r_b"]; f=ν[f_idx])
        δc = MADBead.calc_δ_c_mie(p["ϵ_b"], p["r_b"]; f=ν[f_idx])
        model = sqrt.(abs.(MADBead.E_field2_conv_1D_z_param(dataset.z_abs, p; f=ν[f_idx], δ=δ, δc=δc)))
        reference = abs.(dataset.ET[f_idx, :])
        finite_mask = isfinite.(model) .& isfinite.(reference)
        if !any(finite_mask)
            continue
        end

        model = model[finite_mask]
        reference = reference[finite_mask]
        denom = max.(abs.(reference), eps(Float64))
        residual = model .- reference
        sigma = max.(rel_floor .* denom, eps(Float64))

        residual_sq_sum += sum(abs2, residual)
        reference_sq_sum += sum(abs2, reference)
        chi_sq_sum += sum(abs2, residual ./ sigma)
        append!(rel_abs_errors, abs.(residual) ./ denom)
        n_points += length(reference)
    end

    return (
        fit_abs_nrmse=n_points == 0 ? NaN : sqrt(residual_sq_sum / max(reference_sq_sum, eps(Float64))),
        fit_rel_abs_median=isempty(rel_abs_errors) ? NaN : median(rel_abs_errors),
        fit_rel_abs_p90=isempty(rel_abs_errors) ? NaN : quantile(rel_abs_errors, 0.9),
        fit_chi_rms=n_points == 0 ? NaN : sqrt(chi_sq_sum / n_points),
        fit_n_points=n_points,
    )
end

function reference_indices(ν_query, ν_reference)
    indices = Vector{Int}(undef, length(ν_query))
    for (i, f) in enumerate(ν_query)
        j = searchsortedfirst(ν_reference, f)
        if j > length(ν_reference) || ν_reference[j] != f
            j = argmin(abs.(ν_reference .- f))
            @assert abs(ν_reference[j] - f) <= max(abs(f), 1.0) * 1e-12 "Could not match frequency $(f) to reference grid."
        end
        indices[i] = j
    end
    return indices
end

function boostfactor_agreement_metrics(ν_candidate, bf_candidate, ν_reference, bf_reference; band=(minimum(ν_reference), maximum(ν_reference)))
    band_mask = (ν_candidate .>= band[1]) .& (ν_candidate .<= band[2])
    if !any(band_mask)
        return (bf_nrmse=NaN, bf_rel_abs_median=NaN, bf_rel_abs_p90=NaN, bf_corr=NaN, bf_peak_delta_GHz=NaN, bf_n_points=0)
    end

    ν_band = ν_candidate[band_mask]
    candidate = Float64.(value_of.(bf_candidate[band_mask]))
    reference = Float64.(value_of.(bf_reference[reference_indices(ν_band, ν_reference)]))
    finite_mask = isfinite.(candidate) .& isfinite.(reference)
    if !any(finite_mask)
        return (bf_nrmse=NaN, bf_rel_abs_median=NaN, bf_rel_abs_p90=NaN, bf_corr=NaN, bf_peak_delta_GHz=NaN, bf_n_points=0)
    end

    ν_band = ν_band[finite_mask]
    candidate = candidate[finite_mask]
    reference = reference[finite_mask]
    residual = candidate .- reference
    rel_abs = abs.(residual) ./ max.(abs.(reference), eps(Float64))
    peak_delta_GHz = (ν_band[argmax(candidate)] - ν_band[argmax(reference)]) * 1e-9

    return (
        bf_nrmse=sqrt(sum(abs2, residual) / max(sum(abs2, reference), eps(Float64))),
        bf_rel_abs_median=median(rel_abs),
        bf_rel_abs_p90=quantile(rel_abs, 0.9),
        bf_corr=length(candidate) > 1 ? cor(candidate, reference) : NaN,
        bf_peak_delta_GHz=peak_delta_GHz,
        bf_n_points=length(candidate),
    )
end

function configuration_metrics_dataframe(configurations, clean_configurations, ν_clean_by_config, results, outliers, full_result, ν_full, priors; secondary_band=DEFAULT_SECONDARY_BAND)
    rows = NamedTuple[]
    full_bf = full_result.boostfactor
    full_band = (minimum(ν_full), maximum(ν_full))

    for (dataset, clean_dataset, ν_clean, result, outlier) in zip(configurations, clean_configurations, ν_clean_by_config, results, outliers)
        fit_metrics = fit_quality_metrics(clean_dataset, ν_clean, result, priors.p0_all; rel_floor=DEFAULT_REL_ERR)
        bf_full = boostfactor_agreement_metrics(ν_clean, result.boostfactor, ν_full, full_bf; band=full_band)
        bf_secondary = boostfactor_agreement_metrics(ν_clean, result.boostfactor, ν_full, full_bf; band=secondary_band)

        push!(
            rows,
            merge(
                (
                    configuration_index=dataset.configuration_index,
                    line_count=dataset.line_count,
                    z_slice_count=dataset.z_slice_count,
                    gap_count=dataset.gap_count,
                    n_reduced_z=length(dataset.z_abs),
                    n_kept_frequency=length(ν_clean),
                    n_skipped_frequency=count(outlier.frequency_skip_mask),
                    kept_frequency_fraction=length(ν_clean) / length(ν_full),
                    fit_has_enough_z=length(dataset.z_abs) >= length(priors.keys_optim) + 1,
                ),
                fit_metrics,
                (
                    bf_nrmse_fullband=bf_full.bf_nrmse,
                    bf_rel_abs_median_fullband=bf_full.bf_rel_abs_median,
                    bf_rel_abs_p90_fullband=bf_full.bf_rel_abs_p90,
                    bf_corr_fullband=bf_full.bf_corr,
                    bf_peak_delta_GHz_fullband=bf_full.bf_peak_delta_GHz,
                    bf_n_points_fullband=bf_full.bf_n_points,
                    bf_nrmse_19_20GHz=bf_secondary.bf_nrmse,
                    bf_rel_abs_median_19_20GHz=bf_secondary.bf_rel_abs_median,
                    bf_rel_abs_p90_19_20GHz=bf_secondary.bf_rel_abs_p90,
                    bf_corr_19_20GHz=bf_secondary.bf_corr,
                    bf_peak_delta_GHz_19_20GHz=bf_secondary.bf_peak_delta_GHz,
                    bf_n_points_19_20GHz=bf_secondary.bf_n_points,
                ),
            ),
        )
    end

    df = DataFrame(rows)
    df[!, :rank_bf_fullband] = invperm(sortperm(df.bf_nrmse_fullband))
    df[!, :rank_fit_abs_nrmse] = invperm(sortperm(df.fit_abs_nrmse))
    return df
end

function select_frequency_subset(ν, dataset, limit)
    if limit <= 0 || limit >= length(ν)
        return ν, dataset
    end
    idx = collect(1:limit)
    return ν[idx], merge(dataset, (ET=dataset.ET[idx, :], ET_meas=dataset.ET_meas[idx, :]))
end

function config_position_by_index(configurations, configuration_index)
    pos = findfirst(dataset -> dataset.configuration_index == configuration_index, configurations)
    pos === nothing && error("Unknown configuration index $(configuration_index).")
    return pos
end

function run_full_fit(; cache_dir=DEFAULT_CACHE_DIR, data_file=DEFAULT_DATA_FILE, N_mc=DEFAULT_N_MC, seed=DEFAULT_SEED, force=false, frequency_limit=0)
    mkpath(cache_dir)
    path = full_cache_path(cache_dir)
    if isfile(path) && !force
        println("Full cache already exists: $(path)")
        return path
    end

    Random.seed!(seed)
    inputs = prepare_inputs(data_file=data_file)
    priors = model_priors(N_mc=N_mc)
    ν_fit, full_fit_data = select_frequency_subset(inputs.ν, inputs.full_data, frequency_limit)
    full_result = run_boostfactor_analysis(full_fit_data, ν_fit, priors)

    JLD2.jldsave(
        path;
        kind="full",
        created_at=timestamp(),
        seed,
        N_mc,
        data_file,
        ν_full=ν_fit,
        full_data=full_fit_data,
        full_result,
        priors,
    )
    println("Saved full cache to $(path)")
    return path
end

function run_config_fit(; configuration_index, cache_dir=DEFAULT_CACHE_DIR, data_file=DEFAULT_DATA_FILE, N_mc=DEFAULT_N_MC, seed=DEFAULT_SEED + 1000 + configuration_index, force=false, frequency_limit=0)
    mkpath(cache_dir)
    path = config_cache_path(cache_dir, configuration_index)
    if isfile(path) && !force
        println("Configuration cache already exists: $(path)")
        return path
    end

    Random.seed!(seed)
    inputs = prepare_inputs(data_file=data_file)
    priors = model_priors(N_mc=N_mc)
    pos = config_position_by_index(inputs.reduced_configurations, configuration_index)
    ν_clean = inputs.ν_clean_by_config[pos]
    clean_dataset = inputs.reduced_configurations_clean[pos]
    ν_fit, fit_dataset = select_frequency_subset(ν_clean, clean_dataset, frequency_limit)
    result = run_boostfactor_analysis(fit_dataset, ν_fit, priors)
    @assert isempty(intersect(result.df_bf_analysis.f, inputs.outliers_by_config[pos].skipped_frequencies)) "Skipped frequencies leaked into configuration $(configuration_index) fit output."

    JLD2.jldsave(
        path;
        kind="config",
        created_at=timestamp(),
        seed,
        N_mc,
        data_file,
        configuration_index,
        configuration=inputs.reduced_configurations[pos],
        clean_configuration=fit_dataset,
        outliers=inputs.outliers_by_config[pos],
        ν_clean=ν_fit,
        config_result=result,
        priors,
    )
    println("Saved configuration $(configuration_index) cache to $(path)")
    return path
end

function save_metrics_csv(df, path)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(names(df)), ","))
        for row in eachrow(df)
            values = String[]
            for name in names(df)
                value = row[name]
                push!(values, ismissing(value) ? "" : string(value))
            end
            println(io, join(values, ","))
        end
    end
    return path
end

function save_final_results(results_file, inputs, priors, full_result, configuration_results, df_config_metrics; cache_dir=DEFAULT_CACHE_DIR, data_file=DEFAULT_DATA_FILE)
    mkpath(dirname(results_file))
    df_config_metrics_by_bf = sort(df_config_metrics, :bf_nrmse_fullband)
    df_config_metrics_by_fit = sort(df_config_metrics, :fit_abs_nrmse)
    config_position_by_index = Dict(dataset.configuration_index => i for (i, dataset) in enumerate(inputs.reduced_configurations))
    best_bf_pos = config_position_by_index[df_config_metrics_by_bf.configuration_index[1]]
    best_fit_pos = config_position_by_index[df_config_metrics_by_fit.configuration_index[1]]

    ν = inputs.ν
    ν_full = inputs.ν
    ν_by_config = inputs.ν_clean_by_config
    z_m_FM504_M = priors.z_m_FM504_M
    rel_err = DEFAULT_REL_ERR
    N_mc = priors.N_mc
    p0_all = priors.p0_all
    keys_optim = priors.keys_optim
    keys_helper = priors.keys_helper
    keys_fixed = priors.keys_fixed
    primary_bf_band = :full
    secondary_bf_band = DEFAULT_SECONDARY_BAND
    configuration_lookup = inputs.configuration_lookup
    configuration_metadata = select(df_config_metrics, :configuration_index, :line_count, :z_slice_count, :gap_count, :n_reduced_z)
    df_outlier_summary = inputs.df_outlier_summary

    outlier_point_mask_by_config = [outlier.point_mask for outlier in inputs.outliers_by_config]
    outlier_real_score_by_config = [outlier.real_score for outlier in inputs.outliers_by_config]
    outlier_imag_score_by_config = [outlier.imag_score for outlier in inputs.outliers_by_config]
    outlier_frequency_skip_mask_by_config = [outlier.frequency_skip_mask for outlier in inputs.outliers_by_config]
    outlier_keep_frequency_mask_by_config = [outlier.keep_frequency_mask for outlier in inputs.outliers_by_config]
    skipped_frequencies_by_config = [outlier.skipped_frequencies for outlier in inputs.outliers_by_config]

    z_reduced_abs_by_config = [dataset.z_abs for dataset in inputs.reduced_configurations]
    z_reduced_rel_by_config = [dataset.z_rel for dataset in inputs.reduced_configurations]
    z_ixs_used_by_config = [dataset.z_ixs_used for dataset in inputs.reduced_configurations]
    ET_reduced_raw_by_config = [dataset.ET for dataset in inputs.reduced_configurations]
    ET_reduced_meas_raw_by_config = [dataset.ET_meas for dataset in inputs.reduced_configurations]
    ET_reduced_clean_by_config = [dataset.ET for dataset in inputs.reduced_configurations_clean]
    ET_reduced_meas_clean_by_config = [dataset.ET_meas for dataset in inputs.reduced_configurations_clean]

    p_all_ν_mc_by_config = [result.p_all_ν_mc for result in configuration_results]
    p_final_ν_by_config = [result.p_final_ν for result in configuration_results]
    int_dz_E_mc_by_config = [result.int_dz_E_mc for result in configuration_results]
    ∫dV_E_by_config = [result.∫dV_E for result in configuration_results]
    boostfactor_by_config = [result.boostfactor for result in configuration_results]
    df_ET_by_config = [result.df_ET for result in configuration_results]
    df_ET_raw_by_config = [ET_dataframe(ν, dataset) for dataset in inputs.reduced_configurations]
    df_bf_analysis_by_config = [result.df_bf_analysis for result in configuration_results]
    df_bf_mc_analysis_by_config = [result.df_bf_mc_analysis for result in configuration_results]

    z_full_abs = inputs.full_data.z_abs
    z_full_rel = inputs.full_data.z_rel
    ET_full = inputs.full_data.ET
    ET_full_meas = inputs.full_data.ET_meas
    p_all_ν_mc_full = full_result.p_all_ν_mc
    p_final_ν_full = full_result.p_final_ν
    int_dz_E_mc_full = full_result.int_dz_E_mc
    ∫dV_E_full = full_result.∫dV_E
    boostfactor_full = full_result.boostfactor
    df_ET_full = full_result.df_ET
    df_bf_analysis_full = full_result.df_bf_analysis
    df_bf_mc_analysis_full = full_result.df_bf_mc_analysis

    full_data = inputs.full_data
    reduced_configurations = inputs.reduced_configurations
    reduced_configurations_clean = inputs.reduced_configurations_clean
    outliers_by_config = inputs.outliers_by_config
    ν_clean_by_config = inputs.ν_clean_by_config

    JLD2.jldsave(
        results_file;
        DATA_FILE=data_file,
        cache_dir,
        created_at=timestamp(),
        ν,
        ν_full,
        ν_by_config,
        ν_clean_by_config,
        z_m_FM504_M,
        rel_err,
        N_mc,
        p0_all,
        keys_optim,
        keys_helper,
        keys_fixed,
        primary_bf_band,
        secondary_bf_band,
        configuration_lookup,
        configuration_metadata,
        config_position_by_index,
        best_bf_pos,
        best_fit_pos,
        df_outlier_summary,
        df_config_metrics,
        df_config_metrics_by_bf,
        df_config_metrics_by_fit,
        full_data,
        reduced_configurations,
        reduced_configurations_clean,
        outliers_by_config,
        full_result,
        configuration_results,
        outlier_point_mask_by_config,
        outlier_real_score_by_config,
        outlier_imag_score_by_config,
        outlier_frequency_skip_mask_by_config,
        outlier_keep_frequency_mask_by_config,
        skipped_frequencies_by_config,
        outlier_window=DEFAULT_OUTLIER_WINDOW,
        outlier_threshold=DEFAULT_OUTLIER_THRESHOLD,
        z_reduced_abs_by_config,
        z_reduced_rel_by_config,
        z_ixs_used_by_config,
        ET_reduced_raw_by_config,
        ET_reduced_meas_raw_by_config,
        ET_reduced_clean_by_config,
        ET_reduced_meas_clean_by_config,
        p_all_ν_mc_by_config,
        p_final_ν_by_config,
        int_dz_E_mc_by_config,
        ∫dV_E_by_config,
        boostfactor_by_config,
        df_ET_by_config,
        df_ET_raw_by_config,
        df_bf_analysis_by_config,
        df_bf_mc_analysis_by_config,
        z_full_abs,
        z_full_rel,
        ET_full,
        ET_full_meas,
        p_all_ν_mc_full,
        p_final_ν_full,
        int_dz_E_mc_full,
        ∫dV_E_full,
        boostfactor_full,
        df_ET_full,
        df_bf_analysis_full,
        df_bf_mc_analysis_full,
    )

    save_metrics_csv(df_config_metrics_by_bf, metrics_csv_path(cache_dir))
    println("Saved assembled results to $(results_file)")
    println("Saved metrics CSV to $(metrics_csv_path(cache_dir))")
    return results_file
end

function assemble_results(; cache_dir=DEFAULT_CACHE_DIR, data_file=DEFAULT_DATA_FILE, results_file=DEFAULT_RESULTS_FILE, N_mc=DEFAULT_N_MC)
    inputs = prepare_inputs(data_file=data_file)
    priors = model_priors(N_mc=N_mc)

    full_path = full_cache_path(cache_dir)
    isfile(full_path) || error("Missing full reference cache: $(full_path)")
    full_result = JLD2.load(full_path, "full_result")
    @assert length(full_result.df_bf_analysis.f) == length(inputs.ν) "Full cache frequency grid does not match full production grid."

    configuration_results = Vector{Any}(undef, length(inputs.reduced_configurations))
    for dataset in inputs.reduced_configurations
        path = config_cache_path(cache_dir, dataset.configuration_index)
        isfile(path) || error("Missing configuration cache: $(path)")
        loaded = JLD2.load(path)
        result = loaded["config_result"]
        @assert isempty(intersect(result.df_bf_analysis.f, inputs.outliers_by_config[dataset.lookup_row].skipped_frequencies)) "Skipped frequencies leaked into $(dataset.label) fit output."
        @assert length(result.df_bf_analysis.f) == length(inputs.ν_clean_by_config[dataset.lookup_row]) "Configuration $(dataset.configuration_index) cache frequency length mismatch."
        configuration_results[dataset.lookup_row] = result
    end

    df_config_metrics = configuration_metrics_dataframe(
        inputs.reduced_configurations,
        inputs.reduced_configurations_clean,
        inputs.ν_clean_by_config,
        configuration_results,
        inputs.outliers_by_config,
        full_result,
        inputs.ν,
        priors;
        secondary_band=DEFAULT_SECONDARY_BAND,
    )

    return save_final_results(results_file, inputs, priors, full_result, configuration_results, df_config_metrics; cache_dir=cache_dir, data_file=data_file)
end

function subset_by_original_indices(dataset, indices)
    return merge(dataset, (ET=dataset.ET[indices, :], ET_meas=dataset.ET_meas[indices, :]))
end

function run_smoke(; cache_dir=mktempdir(), data_file=DEFAULT_DATA_FILE, configuration_index=47, N_mc=2, seed=DEFAULT_SEED, force=true)
    mkpath(cache_dir)
    Random.seed!(seed)
    inputs = prepare_inputs(data_file=data_file)
    priors = model_priors(N_mc=N_mc)
    pos = config_position_by_index(inputs.reduced_configurations, configuration_index)
    kept_original_indices = findall(inputs.outliers_by_config[pos].keep_frequency_mask)
    @assert length(kept_original_indices) >= 2 "Need at least two kept frequencies for smoke test."
    idx = kept_original_indices[1:2]
    ν_smoke = inputs.ν[idx]
    full_smoke = merge(subset_by_original_indices(inputs.full_data, idx), (label="smoke_full",))
    config_smoke = merge(inputs.reduced_configurations[pos], (
        label="smoke_$(inputs.reduced_configurations[pos].label)",
        ET=inputs.reduced_configurations[pos].ET[idx, :],
        ET_meas=inputs.reduced_configurations[pos].ET_meas[idx, :],
    ))

    full_result = run_boostfactor_analysis(full_smoke, ν_smoke, priors)
    config_result = run_boostfactor_analysis(config_smoke, ν_smoke, priors)
    df_config_metrics = configuration_metrics_dataframe(
        [inputs.reduced_configurations[pos]],
        [config_smoke],
        [ν_smoke],
        [config_result],
        [inputs.outliers_by_config[pos]],
        full_result,
        ν_smoke,
        priors;
        secondary_band=(minimum(ν_smoke), maximum(ν_smoke)),
    )
    smoke_results_file = joinpath(cache_dir, "smoke_result.jld2")
    JLD2.jldsave(
        full_cache_path(cache_dir);
        kind="smoke_full",
        created_at=timestamp(),
        seed,
        N_mc,
        data_file,
        ν_full=ν_smoke,
        full_data=full_smoke,
        full_result,
        priors,
    )
    JLD2.jldsave(
        config_cache_path(cache_dir, configuration_index);
        kind="smoke_config",
        created_at=timestamp(),
        seed=seed + 1000 + configuration_index,
        N_mc,
        data_file,
        configuration_index,
        configuration=inputs.reduced_configurations[pos],
        clean_configuration=config_smoke,
        outliers=inputs.outliers_by_config[pos],
        ν_clean=ν_smoke,
        config_result,
        priors,
    )
    JLD2.jldsave(
        smoke_results_file;
        kind="smoke",
        created_at=timestamp(),
        seed,
        N_mc,
        configuration_index,
        ν_smoke,
        full_result,
        config_result,
        df_config_metrics,
    )
    println("SMOKE_OK result=$(smoke_results_file) bf_nrmse=$(df_config_metrics.bf_nrmse_fullband[1]) fit_n_points=$(df_config_metrics.fit_n_points[1])")
    return smoke_results_file
end

end # module
