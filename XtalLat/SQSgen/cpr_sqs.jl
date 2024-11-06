using Random
using LinearAlgebra
using NPZ
using Combinatorics
using Statistics
using JLD2
using Distributed
using Base.Threads
@everywhere include("basis_func_quin.jl")

function array_product(array, repeat)
    return collect(Iterators.product(ntuple(_ -> array, repeat)...))
end 

function pointcpr_sqs(compo::Vector{Float64},
                    spin_denote::Vector{Float64}=Vector{Float64}([2,1,0,-1,-2]))
    
    cpr_mat = hcat([phi_vec(spin) for spin in spin_denote]...)'
    return compo'*cpr_mat
end


function cpr(spin_list::Vector{Float64}, cluster_type::String, config_info::config)
    """
    return cluster correlation function / ( one cluster ⋅ one embedding )
    """
    outer_prod = kron([phi_vec(spin) for spin in spin_list]...)
    #! deprecate the `cluster_degeneracy` term
    # cluster_degeneracy = cluster_symoperator(config_info, cluster_type, spin_list)
    return outer_prod
end

function cpr_sqs(compo::Vector{Float64},
                cluster_type::String,
                config_info::config,
                len_spin_per_cluster::Int64,
                spin_denote::Vector{Float64}=Vector{Float64}([2,1,0,-1,-2]))
    
    """
    return correlation function for single cluster given composition
    """
    n_element = length(spin_denote)

    cluster_buffer = array_product(spin_denote, len_spin_per_cluster)
    cluster_buffer = reshape(cluster_buffer, (n_element^len_spin_per_cluster))

    prob_cluster = collect(kron([compo for _ in 1:len_spin_per_cluster]...))
    
    len_cluster = length(prob_cluster)
    cpr_vec = 0
    for i in 1:len_cluster
        spin_ = [spin for spin in cluster_buffer[i]]
        spin_corfunc = cpr(spin_, cluster_type, config_info)
        cpr_vec = cpr_vec .+ prob_cluster[i] * spin_corfunc
    end

    basis_mergeind = basis_symoperator(config_info, cluster_type)
    len_basis = length(basis_mergeind)
    cpr_vec = Vector{Float64}([mean(cpr_vec[basis_mergeind[i]]) for i ∈ 1:len_basis])

    return cpr_vec
end

function cluster_extract_sqs(compo::Vector{Float64},
                        cluster_type_list::Vector{Any},
                        config_info::config)
    
    """
    return cluster correlation function for all clusters
    """
    cpr_vec = [Vector{Float64}() for _ in 1:length(cluster_type_list)]
    len_symbol = length(cluster_type_list)

    for count in 1:len_symbol
        cluster_type = cluster_type_list[count]
        len_spin_per_cluster = length(config_info.ind_cluster[cluster_type][1])
        cpr_ = cpr_sqs(compo, cluster_type, config_info, len_spin_per_cluster)
        cpr_vec[count] = cpr_
    end

    #* flatten cpr_vec
    cpr_vec = reduce(vcat, cpr_vec)
    append!(cpr_vec, pointcpr_sqs(compo))

    return cpr_vec
end

function main_sqs(config_info::config,
                compo_list::Matrix{Float64},)
                

    """
    Derive the correlation function vector for virtual SQS given the composition
    """

    key_list_ = config_info.symbol_list
    num_compo = size(compo_list, 1)
    cpr_mat = [Vector{Float64}() for _ in 1:num_compo]
    for i in 1:num_compo
        compo_i = compo_list[i,:]
        cpr_vec = cluster_extract_sqs(compo_i, key_list_, config_info)
        cpr_mat[i] = cpr_vec
    end 

    return cpr_mat 
end
