module DFTKWannierExt

using DFTK
using LinearAlgebra
using Unitful
using UnitfulAtomic
import Wannier

"""
Build a Wannier.jl model that can be used for Wannierization.

The frozen window `dis_froz_min` to `dis_froz_max` for disentanglement is given in
Hartree, or as a `Unitful` energy. It is only used if `n_bands > n_wannier`.
`kmesh_tol` is the tolerance used by Wannier.jl to determine the b-vector shells.
"""
@DFTK.timing function Wannier.Model(scfres;
                                    n_bands=scfres.n_bands_converge,
                                    n_wannier=n_bands,
                                    projections=DFTK.default_wannier_centers(n_wannier),
                                    dis_froz_max=nothing,
                                    dis_froz_min=-Inf,
                                    kmesh_tol=1e-6)
    # TODO None of the routines consider spin at the moment
    @assert scfres.basis.model.spin_polarization in (:none, :spinless)
    @assert length(projections) == n_wannier
    scfres.basis.kgrid isa MonkhorstPack || error("The basis must be constructed from a MP grid.")

    # TODO Use band_data_to_dict to get this easily MPI compatible.

    # Undo symmetry operations to get full k-point list
    scfres_unfold = DFTK.unfold_bz(scfres)
    basis = scfres_unfold.basis
    ψ = scfres_unfold.ψ
    model = basis.model

    # Wannier.jl uses Ångström; DFTK lattice is in bohr -> convert
    lattice = Wannier.Mat3(ustrip.(auconvert.(u"Å", model.lattice)))

    # 旧 Wannier.get_recip_lattice の置き換え:
    # 逆格子 B は A' * B = 2π I を満たすので B = 2π * inv(A)'。
    # A の列が格子ベクトル、という規約は Model/KspaceStencil と同じ。
    # CrystalBase.reciprocal_lattice は KspaceStencil 等に対するメソッドしか
    # src/ に無いため、ここでは依存を増やさず自前で計算する。
    recip_lattice = Wannier.Mat3(2π * inv(Matrix(lattice))')

    # 旧コードは atom_positions / kpoints を行列 (3 x N) で作っていたが、
    # 新 Model / KspaceStencil は Vector{Vec3} を要求するので変換する。
    atom_positions = [Wannier.Vec3{Float64}(Tuple(p)...) for p in model.positions]
    atom_labels    = [string(DFTK.element_symbol(element)) for element in model.atoms]
    kgrid          = Wannier.Vec3{Int}(basis.kgrid.kgrid_size)
    kpoints        = [Wannier.Vec3{Float64}(Tuple(kpt.coordinate)...) for kpt in basis.kpoints]

    # Finite-difference stencil (b-vectors) connecting each k to its neighbours k+b
    # 旧 Wannier.get_bvectors(kpoints, recip_lattice; kmesh_tol) の置き換え:
    #   kstencil = generate_kspace_stencil(recip, kgrid, kpoints; atol=kmesh_tol)
    # shells 側のキーワードは kmesh_tol ではなく atol であることに注意。
    # (generate_kspace_stencil のシグネチャは kstencil.jl:526-590 参照。)
    kstencil = Wannier.generate_kspace_stencil(recip_lattice, kgrid, kpoints; atol=kmesh_tol)
    n_kpts   = length(basis.kpoints)
    n_bvecs  = length(kstencil.bvectors)

    # Overlaps M[ik][ib] = ⟨u_{m,k} | u_{n,k+b}⟩
    # 旧: M[:, :, ib, ik] (4次元配列)。新: overlaps は Vector{Vector{Matrix}}。
    # kpb_k / kpb_G は [ik][ib] 順。旧コードの [ib, ik] / [:, ib, ik] と逆なので注意。
    # G_shift の定義は旧 kpb_b と同じ:
    #   k + b = kpoints[kpb_k[ik][ib]] + kpb_G[ik][ib]
    overlaps = map(1:n_kpts) do ik
        map(1:n_bvecs) do ib
            ik_plus_b = kstencil.kpb_k[ik][ib]
            G_shift   = collect(kstencil.kpb_G[ik][ib])  # Vec3{Int} -> Vector{Int}
            DFTK.overlap_Mmn_k_kpb(basis, ψ, ik, ik_plus_b, G_shift, n_bands)
        end
    end

    # Initial gauge from projections, Löwdin-orthonormalised
    # 旧: U = Wannier.orthonorm_lowdin(A) (A は 3次元配列)。
    # 新: orthonorm_lowdin は AbstractMatrix 専用 + AbstractVector 版あり。
    #     Vector{Matrix} で渡すと各 k点ごとに Lowdin 直交化される。
    A = map(1:n_kpts) do ik
        DFTK.compute_amn_kpoint(basis, basis.kpoints[ik], ψ[ik], projections, n_bands)
    end
    gauges = Wannier.orthonorm_lowdin(A)

    E    = reduce(hcat, [εk[1:n_bands] for εk in scfres_unfold.eigenvalues])
    E_eV = ustrip.(auconvert.(u"eV", E))  # Wannier.jl uses eV
    # 新: eigenvalues は Vector{Vector}
    eigenvalues = [E_eV[:, ik] for ik in 1:n_kpts]

    if n_bands != n_wannier && !isnothing(dis_froz_max) && !isnothing(dis_froz_min)
        dis_froz_min_eV::Float64 = ustrip(auconvert(u"eV", austrip(dis_froz_min)))
        dis_froz_max_eV::Float64 = ustrip(auconvert(u"eV", austrip(dis_froz_max)))
        # get_frozen_bands は E::Vector を要求し、Vector{BitVector} を返す
        frozen_bands = Wannier.get_frozen_bands(eigenvalues, dis_froz_max_eV, dis_froz_min_eV)
    else
        # 新: frozen_bands は Vector{BitVector}。旧 falses(n_bands, n_kpts) 行列は不可。
        frozen_bands = [falses(n_bands) for _ in 1:n_kpts]
    end

    # 新 Model コンストラクタ:
    #   Model(lattice, atom_positions, atom_labels, kstencil,
    #         overlaps, gauges, eigenvalues, frozen_bands)
    # entangled_bands はデフォルト (default_entangled_bands(gauges)) に任せる。
    # kgrid / kpoints / bvectors は kstencil に内包されているため引数にしない。
    return Wannier.Model(lattice, atom_positions, atom_labels, kstencil,
                         overlaps, gauges, eigenvalues, frozen_bands)
end

end
