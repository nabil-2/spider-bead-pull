#!/usr/bin/env julia

include(joinpath(@__DIR__, "SpiderETVariationFit.jl"))
using .SpiderETVariationFit

function parse_cli(args)
    isempty(args) && error("Usage: run_spider_et_variation_fit.jl MODE [options], where MODE is full, config, assemble, or smoke.")
    mode = args[1]
    opts = Dict{String,Any}(
        "mode" => mode,
        "cache_dir" => SpiderETVariationFit.DEFAULT_CACHE_DIR,
        "data_file" => SpiderETVariationFit.DEFAULT_DATA_FILE,
        "results_file" => SpiderETVariationFit.DEFAULT_RESULTS_FILE,
        "n_mc" => SpiderETVariationFit.DEFAULT_N_MC,
        "seed" => SpiderETVariationFit.DEFAULT_SEED,
        "config_index" => nothing,
        "frequency_limit" => 0,
        "force" => false,
    )

    i = 2
    while i <= length(args)
        arg = args[i]
        if arg == "--force"
            opts["force"] = true
            i += 1
        elseif startswith(arg, "--")
            key = replace(arg[3:end], "-" => "_")
            i += 1
            i <= length(args) || error("Missing value for $(arg).")
            opts[key] = args[i]
            i += 1
        else
            error("Unexpected argument: $(arg)")
        end
    end

    for key in ("n_mc", "seed", "frequency_limit")
        opts[key] = parse(Int, string(opts[key]))
    end
    if opts["config_index"] !== nothing
        opts["config_index"] = parse(Int, string(opts["config_index"]))
    end
    return opts
end

function main(args=ARGS)
    opts = parse_cli(args)
    println("mode=$(opts["mode"]) threads=$(Threads.nthreads()) cache_dir=$(opts["cache_dir"])")

    if opts["mode"] == "full"
        SpiderETVariationFit.run_full_fit(
            cache_dir=opts["cache_dir"],
            data_file=opts["data_file"],
            N_mc=opts["n_mc"],
            seed=opts["seed"],
            force=opts["force"],
            frequency_limit=opts["frequency_limit"],
        )
    elseif opts["mode"] == "config"
        opts["config_index"] === nothing && error("--config-index is required for config mode.")
        seed = haskey(opts, "config_seed") ? parse(Int, string(opts["config_seed"])) : opts["seed"] + 1000 + opts["config_index"]
        SpiderETVariationFit.run_config_fit(
            configuration_index=opts["config_index"],
            cache_dir=opts["cache_dir"],
            data_file=opts["data_file"],
            N_mc=opts["n_mc"],
            seed=seed,
            force=opts["force"],
            frequency_limit=opts["frequency_limit"],
        )
    elseif opts["mode"] == "assemble"
        SpiderETVariationFit.assemble_results(
            cache_dir=opts["cache_dir"],
            data_file=opts["data_file"],
            results_file=opts["results_file"],
            N_mc=opts["n_mc"],
        )
    elseif opts["mode"] == "smoke"
        config_index = opts["config_index"] === nothing ? 47 : opts["config_index"]
        SpiderETVariationFit.run_smoke(
            cache_dir=opts["cache_dir"],
            data_file=opts["data_file"],
            configuration_index=config_index,
            N_mc=opts["n_mc"],
            seed=opts["seed"],
            force=true,
        )
    else
        error("Unknown mode $(opts["mode"]). Expected full, config, assemble, or smoke.")
    end
end

main()
