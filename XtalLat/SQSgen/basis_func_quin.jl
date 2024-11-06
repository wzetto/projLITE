function ϕ1(σ)
    return sqrt(2)/2 .* σ
end

function ϕ2(σ)
    return -sqrt(10/7) .+ sqrt(5/14) .* σ.^2
end

function ϕ3(σ)
    return -17/(6*sqrt(2)).*σ .+ 5/(6*sqrt(2)).*σ.^3
end

function ϕ4(σ)
    return 3*sqrt(14)/7 .- 155*sqrt(14)/168*σ.^2 .+ 5*sqrt(14)/24*σ.^4
end

function phi_vec(σ::Float64)
    return Vector{Float64}([ϕ1(σ), ϕ2(σ), ϕ3(σ), ϕ4(σ)])
end

function phi_vec(σ::Vector{Float64})
    return Vector{Float64}([mean(ϕ1(σ)), mean(ϕ2(σ)), mean(ϕ3(σ)), mean(ϕ4(σ))])
end

function ele_list_gen(compo_list::Vector{Float64},
                    num_cell=150,
                    cell_sequence_num::Int64=1,
                    mode::String="randchoice",
                    max_iter::Int64=100,
                    )

    """
    Quite old version
    New one can be found in `cevander.jl`
    """

    @assert abs(sum(compo_list)-1) < 0.001 "Make sure the sum of atomic concentration = 1"

    a_c, b_c, c_c, d_c, e_c = compo_list
    seq_buffer = zeros(Float64, cell_sequence_num, num_cell)
    for i in 1:cell_sequence_num
        iter = 0
        while true
            if cmp("randchoice", mode) == 0
                len_a = rand(range(convert(Int, round(a_c*num_cell))-1, convert(Int, round(a_c*num_cell))+1, step=1))
                len_a = zero_clip(len_a, a_c)
                len_b = rand(range(convert(Int, round(b_c*num_cell))-1, convert(Int, round(b_c*num_cell))+1, step=1))
                len_b = zero_clip(len_b, b_c)
                len_c = rand(range(convert(Int, round(c_c*num_cell))-1, convert(Int, round(c_c*num_cell))+1, step=1))
                len_c = zero_clip(len_c, c_c)
                len_d = rand(range(convert(Int, round(d_c*num_cell))-1, convert(Int, round(d_c*num_cell))+1, step=1))
                len_d = zero_clip(len_d, d_c)
            elseif cmp("int", mode) == 0
                len_a = convert(Int, round(a_c*num_cell))
                len_b = convert(Int, round(b_c*num_cell))
                len_c = convert(Int, round(c_c*num_cell))
                len_d = convert(Int, round(d_c*num_cell))
            end

            len_e = num_cell-len_a-len_b-len_c-len_d
            len_e_ideal = copy(len_e)
            len_e = zero_clip(len_e, e_c)

            if abs(len_e-num_cell*e_c) <= 1 && len_e == len_e_ideal
                a_list = zeros(Float64, len_a) .+ 2
                b_list = zeros(Float64, len_b) .+ 1
                c_list = zeros(Float64, len_c)
                d_list = zeros(Float64, len_d) .- 1
                e_list = zeros(Float64, len_e) .- 2
                ele_list_raw = cat(a_list, b_list, c_list, d_list, e_list, dims=1)
                seq_buffer[i,:] = shuffle(ele_list_raw)
                break
            end
            
            iter += 1
            if iter > max_iter
                println("Max iteration reached")
                println(len_e, len_e_ideal, num_cell*e_c)
                break
            end

        end
    end
    return seq_buffer
end