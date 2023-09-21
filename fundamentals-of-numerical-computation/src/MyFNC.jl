module MyFNC

using LinearAlgebra

"""
backsub(U,b)

Solve the upper triangular linear system with matrix `U` and
right-hand side vector `b`. Ux=b
"""
function backsub(U,b)
    n = length(b)
    x = zeros(Float64, n)
    x[n] = b[n]/U[n,n]
    for i = n-1:-1:1
        s = U[i,i+1:n] ⋅ x[i+1:n]
        if U[i,i] == 0
            throw(LinearAlgebra.SingularException(i))
        end
        x[i] = (b[i] - s)/U[i,i]
    end
    return x
end



"""
forwardsub(L, b)

Solve the lower triangular linear system with matrix `L` and right-hand side `b`.
"""
function forwardsub(L, b)
    n = length(b)
    x = zeros(Float64, n)
    x[1] = b[1]/L[1,1]
    for i = 2:n
        s = L[i,1:i-1] ⋅ x[1:i-1]
        if L[i,i] == 0
            throw(LinearAlgebra.SingularException(i))
        end
        x[i] = (b[i] - s)/L[i,i]
    end
    return x
end

end # module