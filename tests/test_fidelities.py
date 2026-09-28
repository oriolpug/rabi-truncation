import numpy as np
import pytest
from numpy.testing import assert_allclose
from src.fidelities import compare_states, atom_density_matrix
from src.states import FockBasis


def embed_dense(v,basis,k,union,N):
    out=np.zeros(2*(N+1)**len(union),complex)
    for i,n in enumerate(basis.states):
        occupation=[0]*len(union)
        for momentum,number in zip(k,n):
            occupation[union.index(momentum)]=number
        index=2*np.ravel_multi_index(tuple(occupation),(N+1,)*len(union))
        out[index:index+2]=v[2*i:2*i+2]
    return out


def test_multimode_overlap_is_retained_between_full_and_totalcap():
    a,b=FockBasis(2,2,'full'),FockBasis(2,2,'full+totalcap')
    va,vb=np.zeros(a.dim),np.zeros(b.dim)
    va[2*a.index[(1,1)]]=vb[2*b.index[(1,1)]]=1
    result=compare_states(va,a,[-1,1],vb,b,[-1,1])
    assert_allclose(list(result.values()),[1,1])


@pytest.mark.parametrize('sa,sb',[('truncated','full+totalcap'),('full','full+totalcap'),('full','full')])
def test_implicit_embedding_matches_explicit_full_space_and_qubit_formula(sa,sb):
    a,b=FockBasis(2,2,sa),FockBasis(3,3,sb)
    ka,kb=[1.,-1.],[-1.,0.,1.]
    rng=np.random.default_rng(8)
    va=rng.normal(size=a.dim)+1j*rng.normal(size=a.dim)
    vb=rng.normal(size=b.dim)+1j*rng.normal(size=b.dim)
    va/=np.linalg.norm(va); vb/=np.linalg.norm(vb)
    ea,eb=embed_dense(va,a,ka,kb,3),embed_dense(vb,b,kb,kb,3)
    expected=abs(np.vdot(ea,eb))**2
    ra,rb=atom_density_matrix(va),atom_density_matrix(vb)
    atom_expected=np.trace(ra@rb).real+2*np.sqrt(max(0,np.linalg.det(ra).real*np.linalg.det(rb).real))
    result=compare_states(va,a,ka,vb,b,kb)
    assert_allclose(result['F_state'],expected,atol=1e-14)
    assert_allclose(result['F_atom'],atom_expected,atol=1e-12)
    assert result['F_state'] <= result['F_atom']+1e-12


def test_identical_TLS_does_not_imply_identical_global_states():
    basis=FockBasis(3,1,'full+totalcap')
    va,vb=np.zeros(basis.dim),np.zeros(basis.dim)
    va[2*basis.index[(1,0,0)]]=vb[2*basis.index[(0,0,1)]]=1
    result=compare_states(va,basis,[-1,0,1],vb,basis,[-1,0,1])
    assert_allclose([result['F_state'],result['F_atom']],[0,1])


def test_roundoff_alignment_and_vacuum_in_missing_modes():
    a,b=FockBasis(1,1,'full'),FockBasis(2,1,'full+totalcap')
    va,vb=np.zeros(a.dim),np.zeros(b.dim)
    va[2*a.index[(1,)]]=vb[2*b.index[(0,1)]]=1
    assert_allclose(compare_states(va,a,[1.],vb,b,[0.,1.+1e-14])['F_state'],1.)
