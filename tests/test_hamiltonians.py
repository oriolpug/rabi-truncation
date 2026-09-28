import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.states import FockBasis
from src.hamiltonians import Hamiltonian


def tensor(items):
    result=np.array([[1.]])
    for item in items:
        result=np.kron(result,item)
    return result


def dense_oracle(k,N,atom,rwa):
    M=len(k); d=N+1; field_dim=d**M
    hfield=np.zeros((field_dim,field_dim),complex)
    interaction=np.zeros((2*field_dim,2*field_dim),complex)
    annihilation=np.diag(np.sqrt(np.arange(1,d)),1)
    raising=np.array([[0,0],[1,0]],complex)
    for m,km in enumerate(k):
        factors=[np.eye(d) for _ in range(M)]; factors[m]=annihilation
        a=tensor(factors)
        hfield+=abs(km)*a.conj().T@a
        f=np.sqrt(abs(km)) if atom['coupling']=='sqrt' else 1.
        u=1j*f*np.exp(-1j*km*atom['x_tls'])/np.sqrt(atom['L'])
        if rwa:
            interaction+=np.kron(u*a,raising)+np.kron(u.conjugate()*a.conj().T,raising.conj().T)
        else:
            interaction+=np.kron(u*a+u.conjugate()*a.conj().T,raising+raising.conj().T)
    return np.kron(hfield,np.eye(2))+np.kron(np.eye(field_dim),np.diag([0,atom['omega_0']]))+atom['D']*interaction


@pytest.mark.parametrize('scheme',['full','full+totalcap','truncated'])
@pytest.mark.parametrize('rwa',[False,True])
@pytest.mark.parametrize('profile',['sqrt','flat'])
@pytest.mark.parametrize('x',[0.,.37])
def test_projected_hamiltonian_against_independent_tensor_operators(scheme,rwa,profile,x):
    k=np.array([-1.,0.,1.]); N=2
    basis=FockBasis(3,N,scheme)
    atom={'L':6.,'omega_0':1.2,'D':.3,'x_tls':x,'coupling':profile}
    full=dense_oracle(k,N,atom,rwa)
    indices=[2*np.ravel_multi_index(n,(N+1,)*3)+s for n in basis.states for s in (0,1)]
    expected=full[np.ix_(indices,indices)]
    actual=Hamiltonian(basis,k,atom,rwa).build_hamiltonian().full()
    assert_allclose(actual,expected,atol=1e-14)
    assert_allclose(actual,actual.conj().T,atol=1e-14)
    total=np.repeat(basis.photon_numbers,2)+np.tile([0,1],len(basis.states))
    parity=(-1.)**total
    assert_allclose(actual*parity[None,:],parity[:,None]*actual,atol=1e-14)
    if rwa:
        assert_allclose(actual*total[None,:],total[:,None]*actual,atol=1e-14)


def test_spatial_phase_is_a_field_gauge_transform():
    basis=FockBasis(3,2,'full+totalcap'); k=np.array([-1.,0.,1.])
    atom={'L':6.,'omega_0':1.,'D':.3,'x_tls':0.,'coupling':'sqrt'}
    h0=Hamiltonian(basis,k,atom).build_hamiltonian().full()
    x=.4
    hx=Hamiltonian(basis,k,{**atom,'x_tls':x}).build_hamiltonian().full()
    u=np.repeat(np.exp(1j*x*(np.array(basis.states)@k)),2)
    assert_allclose(hx,u[:,None]*h0*u.conj()[None,:])
