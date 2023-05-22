# projected time-dependent Variational Monte Carlo (p-tVMC)

This is a package for the projected time-dependent Variational Monte Carlo (p-tVMC) method based on infidelity optimization for variational simulation of quantum dynamics. 
See the paper "Unbiasing time-dependent Variational Monte Carlo with projected quantum evolution" for reference. 

The p-tVMC can be used to simulate the evolution generated by an arbitrary transformation U, by iteratively minimizing the infidelity among the variational ansatz with free parameters |ψ⟩ and the state U|ϕ⟩ where U is an arbitrary transformation and |ϕ⟩ is a known state (such as an ansatz with known parameters). 
There are no restrictions on U from the moment a function that computes its connected elements is implemented.
The package supports the possibility to sample from the states |ψ⟩ and U|ϕ⟩, which can be used for any transformation U (unitary and non-unitary), and to sample from the states |ψ⟩ and |ϕ⟩, which is possible only for a unitary U (exploiting the norm conservation).
To sample from U|ϕ⟩ a `jax` compatible operator for U must be used, and the package exports few examples of them (the Ising Transverse Field Ising Hamiltonian, Rx and Ry single qubit rotations and the Hadamard gate). 
In addition, the code includes the possibility to use the Control Variates (CV) correction on the infidelity stochastic estimator to improve its signal to noise ratio and reduce the sampling overhead by orders of magnitudes.

## Content of the repository

- **netket_fidelity** : folder containing the following several subfolders: 
    - **infidelity**: contains the infidelity operator. 
    - **operator**: contains the `jax`-compatible operators for U.
    - **driver**: contains the driver for infidelity optimization. 
- **test**: folder containing tests for the infidelity stochastic estimation and for the `jax`-compatible rotation operators.
- **Examples**: folder containing some examples of application. 

## Example of usage

```
import netket as nk
import netket_fidelity as nkf

# Create the Hilbert space and the variational states |ψ⟩ and |ϕ⟩
hi = nk.hilbert.Spin(0.5, 4)
sampler = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=16)
model = nk.models.RBM(alpha=1, param_dtype=complex)
phi = nk.vqs.MCState(sampler=sampler, model=model, n_samples=100)
psi = nk.vqs.MCState(sampler=sampler, model=model, n_samples=100)

# Transformation U
U = nkf.operator.Hadamard(hi, 0)

# Instantiate the operator to optimize the infidelity with U|ϕ⟩ by sampling from |ψ⟩ and |ϕ⟩
I_op = nkf.infidelity.InfidelityOperator(phi, U=U, U_dagger=U, is_unitary=True, cv_coeff=-1/2)

# Create the driver
optimizer = nk.optimizer.Sgd(learning_rate=0.01)
te =  nkf.driver.infidelity_optimizer.InfidelityOptimizer(phi, U, psi, optimizer, U_dagger=U, is_unitary=True, cv_coeff=-0.5)

# Run the driver
te.run(n_iter=100)
```

