import numpy as np
import matplotlib.pyplot as plt
import logging
from qiskit_optimization.problems import QuadraticProgram
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import Sampler  # Updated for compatibility
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Set up debugging
logging.basicConfig(level=logging.DEBUG)

# Define problem parameters
returns = np.array([3, 4, -1])
covariances = np.array([
    [0.9, 0.5, -0.7],
    [0.5, 0.9, -0.2],
    [-0.7, -0.2, 0.9]
])
total_budget = 6
num_assets = len(returns)

# Create Qiskit QuadraticProgram
qp = QuadraticProgram("portfolio_optimization")

# Add integer variables
for i in range(num_assets):
    qp.integer_var(name=f"w_{i}", lowerbound=0, upperbound=7)

# Define quadratic objective: minimize w^T * Σ * w - μ^T * w
quadratic = {}
for i in range(num_assets):
    for j in range(num_assets):
        quadratic[(f"w_{i}", f"w_{j}")] = covariances[i, j]
linear = {f"w_{i}": -returns[i] for i in range(num_assets)}
qp.minimize(linear=linear, quadratic=quadratic)

# Add budget constraint: sum(w_i) <= total_budget
qp.linear_constraint(linear={f"w_{i}": 1 for i in range(num_assets)}, sense="<=", rhs=total_budget, name="budget")

# Run classical solution
print("=== Running Classical Solution ===")
classical_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
classical_result = classical_solver.solve(qp)
print(f"Classical solution: x = {classical_result.x}, cost = {classical_result.fval}")

# QAOA setup with updated sampler
qaoa_success = False
try:
    print("\n=== Attempting QAOA with Aer Sampler ===")
    sampler = Sampler(run_options={"shots": 1024, "seed": 42})  # Add shots for stability
    optimizer = COBYLA(maxiter=200)
    
    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=2  # Increased for better approximation
    )
    
    # Convert to QUBO for QAOA (handles integer to binary encoding)
    converter = QuadraticProgramToQubo()  # Can set penalty=100 if auto-penalty insufficient
    qubo = converter.convert(qp)
    
    qaoa_optimizer = MinimumEigenOptimizer(qaoa)
    qaoa_result = qaoa_optimizer.solve(qubo)
    
    # Interpret back to original variables
    qaoa_x = converter.interpret(qaoa_result.x)
    qaoa_fval = qp.objective.evaluate(qaoa_x)
    
    # Check constraint satisfaction
    budget_used = sum(qaoa_x)
    if budget_used > total_budget:
        print(f"Warning: QAOA solution violates budget constraint (used {budget_used}/{total_budget})")
    else:
        print(f"QAOA solution: x = {qaoa_x}, cost = {qaoa_fval}")
        qaoa_success = True
    
except Exception as e:
    print(f"QAOA failed: {e}")

# Create visualization and analysis
plt.figure(figsize=(12, 8))

if qaoa_success:
    # Cost Comparison
    plt.subplot(2, 2, 1)
    solutions = ['QAOA', 'Classical']
    costs = [qaoa_fval, classical_result.fval]
    colors = ['blue', 'red']
    
    bars = plt.bar(solutions, costs, color=colors, alpha=0.7)
    plt.ylabel('Cost', fontsize=12)
    plt.title('Cost Comparison', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, cost in zip(bars, costs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + abs(cost) * 0.01, 
                 f'{cost:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Portfolio allocation comparison
    plt.subplot(2, 2, 2)
    x_pos = np.arange(num_assets)
    width = 0.35
    
    plt.bar(x_pos - width/2, qaoa_x, width, label='QAOA', alpha=0.7, color='blue')
    plt.bar(x_pos + width/2, classical_result.x, width, label='Classical', alpha=0.7, color='red')
    
    plt.xlabel('Assets')
    plt.ylabel('Allocation')
    plt.title('Portfolio Allocations')
    plt.xticks(x_pos, [f'Asset {i+1}' for i in range(num_assets)])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Risk-Return analysis
    plt.subplot(2, 2, 3)
    solutions_data = [
        ('QAOA', qaoa_x, 'blue'),
        ('Classical', classical_result.x, 'red')
    ]
    
    for name, weights, color in solutions_data:
        portfolio_return = np.dot(returns, weights)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariances, weights)))
        plt.scatter(portfolio_risk, portfolio_return, s=100, alpha=0.7, 
                    color=color, label=name, edgecolors='black')
        plt.text(portfolio_risk, portfolio_return + 0.1, name, 
                 ha='center', fontsize=10, fontweight='bold')
    
    plt.xlabel('Risk (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.title('Risk-Return Profile')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Budget utilization
    plt.subplot(2, 2, 4)
    qaoa_budget = sum(qaoa_x)
    classical_budget = sum(classical_result.x)
    
    plt.bar(['QAOA', 'Classical'], [qaoa_budget, classical_budget], 
            color=['blue', 'red'], alpha=0.7)
    plt.axhline(y=total_budget, color='green', linestyle='--', 
                label=f'Total Budget ({total_budget})')
    plt.ylabel('Budget Used')
    plt.title('Budget Utilization')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print(f"\n=== Detailed Results ===")
    print(f"QAOA - Asset weights: {qaoa_x}")
    print(f"Classical - Asset weights: {classical_result.x}")
    print(f"QAOA - Total budget used: {qaoa_budget}/{total_budget}")
    print(f"Classical - Total budget used: {classical_budget}/{total_budget}")
    
    # Calculate portfolio metrics
    for name, weights in [('QAOA', qaoa_x), ('Classical', classical_result.x)]:
        portfolio_return = np.dot(returns, weights)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariances, weights)))
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        print(f"{name} - Return: {portfolio_return:.3f}, Risk: {portfolio_risk:.3f}, Sharpe: {sharpe_ratio:.3f}")

else:
    print("\n=== QAOA could not run successfully or violated constraints ===")
    print("Falling back to classical results only.")
    
    # Show only classical results
    plt.subplot(1, 2, 1)
    plt.bar(range(num_assets), classical_result.x, alpha=0.7, color='red')
    plt.xlabel('Assets')
    plt.ylabel('Allocation')
    plt.title('Classical Portfolio Allocation')
    plt.xticks(range(num_assets), [f'Asset {i+1}' for i in range(num_assets)])
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    portfolio_return = np.dot(returns, classical_result.x)
    portfolio_risk = np.sqrt(np.dot(classical_result.x, np.dot(covariances, classical_result.x)))
    
    plt.bar(['Return', 'Risk'], [portfolio_return, portfolio_risk], 
            alpha=0.7, color=['green', 'orange'])
    plt.ylabel('Value')
    plt.title('Portfolio Metrics')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Classical solution: x = {classical_result.x}, cost = {classical_result.fval}")
    print(f"Portfolio return: {portfolio_return:.3f}")
    print(f"Portfolio risk: {portfolio_risk:.3f}")
    print(f"Budget used: {sum(classical_result.x)}/{total_budget}")

print("\n=== Updated Troubleshooting Tips ===")
print("1. Ensure Qiskit >=1.0 and qiskit-aer >=0.13; pip install --upgrade qiskit qiskit-algorithms qiskit-aer qiskit-optimization")
print("2. Use qiskit_aer.primitives.Sampler for local simulation.")
print("3. If QAOA doesn't converge to optimal, try reps=3, more shots (e.g., 4096), or a different optimizer like SLSQP.")
print("4. If constraint violated, set explicit penalty in QuadraticProgramToQubo(penalty=1000).")
print("5. For larger problems, consider IBM Quantum hardware via Qiskit Runtime.")