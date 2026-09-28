from math import comb, factorial
import numpy as np
import pytest
from numpy.testing import assert_allclose
from src.states import FockBasis, initial_state


@pytest.mark.parametrize('M,N', [(1,0),(1,3),(3,2),(5,1)])
def test_basis_dimensions_and_nested_sets(M, N):
    a, b, c = [FockBasis(M,N,s) for s in ('truncated','full+totalcap','full')]
    assert a.dim == 2*(1+M*N)
    assert b.dim == 2*comb(M+N,N)
    assert c.dim == 2*(N+1)**M
    assert set(a.states) <= set(b.states) <= set(c.states)
    for basis in (a,b,c):
        assert len(set(basis.states)) == len(basis.states)
        for i, occupation in enumerate(basis.states):
            assert basis.index[occupation] == i


@pytest.mark.parametrize('scheme', ['truncated','full+totalcap','full'])
@pytest.mark.parametrize('k0', [-.4,0,.4])
def test_number_packet_keeps_all_signed_and_zero_modes(scheme,k0):
    basis = FockBasis(3,2,scheme)
    k = np.array([-1.,0.,1.])
    p = {'k_0':k0,'sigma_k':.7,'x_0':.3,'n':1}
    v = initial_state(basis,k,p,{'initial_state':'+'}).full()[:,0]
    packet = np.exp(-(k-k0)**2/(4*.7**2))*np.exp(-1j*k*.3)
    packet /= np.linalg.norm(packet)
    for m in range(3):
        n = [0]*3
        n[m] = 1
        assert_allclose(v[2*basis.index[tuple(n)]:2*basis.index[tuple(n)]+2],
                        packet[m]*np.ones(2)/np.sqrt(2))
        assert abs(v[2*basis.index[tuple(n)]]) > 0
    assert_allclose(np.linalg.norm(v),1)


@pytest.mark.parametrize('scheme', ['truncated','full+totalcap','full'])
def test_coherent_preparation_is_normalized_product_projection(scheme):
    basis = FockBasis(3,2,scheme)
    k=np.array([-1.,0.,1.]); alpha=.4+.2j
    p={'k_0':0.,'sigma_k':.7,'x_0':.3,'state':'coherent','alpha':alpha}
    v=initial_state(basis,k,p,{'initial_state':'e'}).full()[:,0]
    packet=np.exp(-k**2/(4*.7**2))*np.exp(-1j*k*.3)
    packet/=np.linalg.norm(packet)
    expected=np.zeros(basis.dim,dtype=complex)
    for i,n in enumerate(basis.states):
        expected[2*i+1]=np.exp(-abs(alpha)**2/2)*np.prod([(alpha*c)**v/np.sqrt(factorial(v)) for c,v in zip(packet,n)])
    expected/=np.linalg.norm(expected)
    assert_allclose(v,expected,atol=1e-14)


def test_vacuum_and_underflow_safe_packet():
    basis=FockBasis(3,1,'full+totalcap')
    p={'k_0':100.,'sigma_k':1e-5,'x_0':0.,'n':0}
    v=initial_state(basis,np.array([-1.,0.,1.]),p,{'initial_state':'g'}).full()[:,0]
    assert v[2*basis.index[(0,0,0)]] == 1
    assert_allclose(np.linalg.norm(v),1)
