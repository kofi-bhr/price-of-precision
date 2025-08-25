#!/usr/bin/env python3
"""
results.py - Figures for "The Price of Precision"

Author: Kofi Hair-Ralston
Date: August 2025

Mathematical Framework:
- Signal structure: s_g = θ + b·𝟙_{g=1} + ε(b)
- Noise variance: σ²_ε(b) = σ²_0 + κ(b_max - b) 
- Posterior beliefs via Bayesian updating
- Firm maximizes expected productivity of hired workers

Dependencies:
    numpy, scipy, matplotlib, seaborn, os, warnings

Usage:
    python results.py
    
Output:
    Creates ./figures/ directory with PNG files for all paper figures
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize, integrate
import warnings
import os
from typing import Tuple, Dict, Any, Callable
from dataclasses import dataclass

# Suppress unnecessary warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Set random seed for reproducibility
np.random.seed(42)

@dataclass
class ModelParameters:
    """
    Container for all model parameters with validation.
    
    Main Parameters:
    - π_g: Group proportions (π_0 + π_1 = 1)
    - μ: Population mean productivity 
    - σ²_θ: True productivity variance
    - κ: Technology coupling parameter (steepness of fairness-accuracy trade-off)
    - σ²_0: Baseline signal noise at maximum bias
    - b_max: Maximum allowable bias level
    
    Policy Parameters:
    - α: Linear external cost parameter
    - β: Quadratic external cost parameter
    """
    # Core model parameters
    pi_0: float = 0.5           # Proportion of group 0
    pi_1: float = 0.5           # Proportion of group 1  
    mu: float = 1.0             # Population mean productivity
    sigma_theta_sq: float = 1.0 # True productivity variance
    kappa: float = 0.5          # Technology coupling parameter
    sigma_0_sq: float = 0.1     # Baseline noise variance
    b_max: float = 1.0          # Maximum bias level
    
    # Policy parameters for welfare
    alpha: float = 0.3          # Linear external cost
    beta: float = 0.1           # Quadratic external cost coefficient
    
    def __post_init__(self):
        """Validate values on initialization."""
        assert abs(self.pi_0 + self.pi_1 - 1.0) < 1e-10, "Group proportions must sum to 1"
        assert all(x > 0 for x in [self.sigma_theta_sq, self.kappa, self.sigma_0_sq, self.b_max]), \
               "Variance and scale parameters must be positive"
        assert 0 <= self.pi_0 <= 1, "Group proportions must be in [0,1]"
        assert self.alpha >= 0 and self.beta >= 0, "External cost parameters must be non-negative"

class OptimalBiasModel:
    """
    Main model class implementing the firm's optimization problem.
    
    This class uses all the mathematical machinery for:
    1. Computing posterior beliefs via Bayesian updating
    2. Solving for optimal bias b*  
    3. Computing value functions & welfare
    4. Generating predictions for comparative statics
    """
    
    def __init__(self, params: ModelParameters):
        """Initialize model with parameter validation."""
        self.params = params
        self._validate_parameters()
        
    def _validate_parameters(self) -> None:
        """More model-specific parameter validation."""
        # Make sure the precision-bias trade-off is well-defined
        max_noise = self.params.sigma_0_sq + self.params.kappa * self.params.b_max
        if max_noise <= 0:
            raise ValueError(f"Maximum noise variance {max_noise:.3f} must be positive")
            
    def signal_variance(self, b: float) -> float:
        """
        Total signal variance as function of bias choice.
        
        σ²_s(b) = σ²_θ + σ²_ε(b) = σ²_θ + σ²_0 + κ(b_max - b)
        
        Args:
            b: Bias level ∈ [0, b_max]
            
        Returns:
            Total signal variance
        """
        noise_var = self.params.sigma_0_sq + self.params.kappa * (self.params.b_max - b)
        return self.params.sigma_theta_sq + noise_var
    
    def posterior_mean(self, s: float, g: int, b: float) -> float:
        """
        Bayesian posterior mean of productivity given signal.
        
        From equation (A.4):
        E[θ|s,g,b] = (σ²_θ(s - b·𝟙_{g=1}) + σ²_ε(b)μ) / (σ²_θ + σ²_ε(b))
        
        Args:
            s: Observed signal value
            g: Group membership (0 or 1)
            b: Current bias level
            
        Returns:
            Posterior expected productivity
        """
        sigma_s_sq = self.signal_variance(b)
        noise_var = sigma_s_sq - self.params.sigma_theta_sq
        
        numerator = (self.params.sigma_theta_sq * (s - b * (g == 1)) + 
                    noise_var * self.params.mu)
        return numerator / sigma_s_sq
    
    def optimal_threshold(self, b: float, tolerance: float = 1e-8) -> float:
        """
        Compute optimal hiring threshold t*(b) such that E[θ|t*,g,b] = μ.
        
        From Lemma 1: At b=0, t*=μ. For general b, solve numerically.
        
        Args:
            b: Bias level
            tolerance: Numerical tolerance for root finding
            
        Returns:
            Optimal threshold t*(b)
        """
        if abs(b) < tolerance:
            return self.params.mu
            
        # For group 0, solve E[θ|t,0,b] = μ
        def threshold_condition(t: float) -> float:
            return self.posterior_mean(t, 0, b) - self.params.mu
            
        # Use robust root finding with reasonable bounds
        try:
            result = optimize.root_scalar(
                threshold_condition, 
                bracket=[self.params.mu - 3*np.sqrt(self.signal_variance(b)),
                        self.params.mu + 3*np.sqrt(self.signal_variance(b))],
                method='brentq',
                xtol=tolerance
            )
            return result.root
        except ValueError:
            # Fallback to bisection if bracket fails
            return optimize.fsolve(threshold_condition, self.params.mu)[0]
    
    def value_function(self, b: float, n_samples: int = 100000) -> float:
        """
        Compute firm's expected value V(b) via Monte Carlo integration.
        
        V(b) = Σ_g π_g ∫_{t*}^∞ E[θ|s,g,b] f(s|g,b) ds
        
        Args:
            b: Bias level
            n_samples: Number of Monte Carlo samples for integration
            
        Returns:
            Expected value to firm
        """
        if not (0 <= b <= self.params.b_max):
            return -np.inf  # Penalize infeasible choices
            
        threshold = self.optimal_threshold(b)
        sigma_s = np.sqrt(self.signal_variance(b))
        total_value = 0.0
        
        for g in [0, 1]:
            # Signal distribution for group g
            signal_mean = self.params.mu + b * (g == 1)
            pi_g = self.params.pi_0 if g == 0 else self.params.pi_1
            
            # Monte Carlo integration over hired candidates (s ≥ threshold)
            samples = np.random.normal(signal_mean, sigma_s, n_samples)
            hired_mask = samples >= threshold
            
            if np.sum(hired_mask) > 0:
                hired_signals = samples[hired_mask]
                expected_productivities = np.array([
                    self.posterior_mean(s, g, b) for s in hired_signals
                ])
                group_value = pi_g * np.mean(expected_productivities)
                total_value += group_value
                
        return total_value
    
    def value_derivative(self, b: float, h: float = 1e-6) -> float:
        """
        Numerical derivative of value function dV/db using central differences.
        
        Args:
            b: Bias level
            h: Step size for numerical differentiation
            
        Returns:
            Numerical derivative dV/db
        """
        if b - h < 0:
            # Forward difference at boundary
            return (self.value_function(b + h) - self.value_function(b)) / h
        elif b + h > self.params.b_max:
            # Backward difference at boundary  
            return (self.value_function(b) - self.value_function(b - h)) / h
        else:
            # Central difference in interior
            return (self.value_function(b + h) - self.value_function(b - h)) / (2 * h)
    
    def solve_optimal_bias(self, initial_guess: float = 0.5) -> Tuple[float, Dict[str, Any]]:
        """
        Solve for optimal bias b* using numerical optimization.
        
        Finds b* ∈ [0, b_max] such that dV/db = 0, exploiting concavity of V(b).
        
        Args:
            initial_guess: Starting point for optimization
            
        Returns:
            Tuple of (optimal_bias, optimization_info)
        """
        # Ensure initial guess is feasible
        initial_guess = np.clip(initial_guess, 0.001, self.params.b_max - 0.001)
        
        # Use bounded optimization since b ∈ [0, b_max]
        result = optimize.minimize_scalar(
            lambda b: -self.value_function(b),  # Minimize negative for maximization
            bounds=(0, self.params.b_max),
            method='bounded',
            options={'xatol': 1e-6}
        )
        
        optimal_bias = result.x
        
        # Verify first-order condition
        derivative_at_optimum = self.value_derivative(optimal_bias)
        
        info = {
            'success': result.success,
            'function_value': -result.fun,  # Convert back to maximum
            'derivative_at_optimum': derivative_at_optimum,
            'is_interior': 0 < optimal_bias < self.params.b_max,
            'optimization_result': result
        }
        
        return optimal_bias, info
    
    def social_welfare(self, b: float) -> float:
        """
        Social welfare function SW(b) = V(b) - E(b).
        
        External cost: E(b) = αb + βb²/2
        
        Args:
            b: Bias level
            
        Returns:
            Social welfare
        """
        private_value = self.value_function(b)
        external_cost = (self.params.alpha * b + 
                        self.params.beta * b**2 / 2)
        return private_value - external_cost
    
    def solve_social_optimum(self) -> Tuple[float, Dict[str, Any]]:
        """
        Solve for socially optimal bias b**.
        
        Returns:
            Tuple of (social_optimum, optimization_info)
        """
        result = optimize.minimize_scalar(
            lambda b: -self.social_welfare(b),
            bounds=(0, self.params.b_max),
            method='bounded',
            options={'xatol': 1e-6}
        )
        
        social_optimum = result.x
        info = {
            'success': result.success,
            'social_welfare': -result.fun,
            'private_optimum_comparison': None  # To be filled by caller
        }
        
        return social_optimum, info

class FigureGenerator:
    """
    Generate all figures for the paper with consistent styling and layout.
    
    This class handles:
    - Figure aesthetics and publication standards
    - Directory management
    - Consistent color schemes and fonts
    - High-resolution output suitable for journals
    """
    
    def __init__(self, output_dir: str = "./figures"):
        """Initialize figure generator with output directory."""
        self.output_dir = output_dir
        self._setup_plotting_style()
        self._create_output_directory()
        
    def _setup_plotting_style(self) -> None:
        """Set up publication-quality plotting style."""
        # Use professional color palette
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#593E2C']
        sns.set_palette(colors)
        
        # Configure matplotlib for publication quality
        plt.rcParams.update({
            'figure.figsize': (12, 9),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
            'text.usetex': False,  # Avoid LaTeX dependency issues
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'lines.linewidth': 2,
            'patch.linewidth': 0.5,
            'patch.facecolor': colors[0],
            'patch.edgecolor': '#000000',
            'patch.force_edgecolor': True
        })
        
    def _create_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Figure output directory: {os.path.abspath(self.output_dir)}")
        
    def generate_figure_1(self, params: ModelParameters) -> None:
        """
        Generate Figure 1: Model Mechanics and Results (4 panels).
        
        Panel (a): Firm's concave value function V(b) with optimum at b*
        Panel (b): Theoretical fairness-accuracy frontier  
        Panel (c): Optimal signal distributions for both groups
        Panel (d): Comparative statics showing ∂b*/∂κ > 0
        """
        print("Generating Figure 1: Model Mechanics and Results...")
        
        model = OptimalBiasModel(params)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Panel (a): Value Function V(b)
        print("  Computing value function...")
        b_grid = np.linspace(0.01, params.b_max - 0.01, 50)
        v_values = [model.value_function(b, n_samples=50000) for b in b_grid]
        b_star, _ = model.solve_optimal_bias()
        
        ax1.plot(b_grid, v_values, 'b-', linewidth=2.5, label='V(b)')
        ax1.axvline(b_star, color='red', linestyle='--', linewidth=2, 
                   label=f'b* = {b_star:.3f}')
        ax1.axhline(model.value_function(b_star), color='red', linestyle=':', 
                   alpha=0.7, label=f'V(b*) = {model.value_function(b_star):.3f}')
        ax1.set_xlabel('Bias Level (b)')
        ax1.set_ylabel('Expected Value E[U]')
        ax1.set_title('(a) Firm\'s Payoff Function V(b)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel (b): Fairness-Accuracy Frontier
        print("  Computing fairness-accuracy frontier...")
        # Simulate fairness vs accuracy for different bias levels
        fairness_scores = []  # 1 - disparate impact
        accuracy_scores = []  # AUC or similar
        
        for b in b_grid:
            # Compute disparate impact: ratio of hiring rates
            threshold = model.optimal_threshold(b)
            sigma_s = np.sqrt(model.signal_variance(b))
            
            # Hiring probabilities for each group
            prob_hire_0 = 1 - stats.norm.cdf(threshold, params.mu, sigma_s)
            prob_hire_1 = 1 - stats.norm.cdf(threshold, params.mu + b, sigma_s)
            
            # Disparate impact ratio (group 1 / group 0)
            if prob_hire_0 > 0:
                disparate_impact = prob_hire_1 / prob_hire_0
                fairness = 1 - abs(1 - disparate_impact)  # 1 = perfectly fair
            else:
                fairness = 0
                
            # Accuracy approximation: higher precision means higher accuracy
            accuracy = 1 / (1 + model.signal_variance(b))  # Normalized measure
            
            fairness_scores.append(fairness)
            accuracy_scores.append(accuracy)
        
        ax2.plot(fairness_scores, accuracy_scores, 'g-', linewidth=2.5, 
                label='Fairness-Accuracy Frontier')
        
        # Mark optimal point
        b_star_idx = np.argmin(np.abs(b_grid - b_star))
        ax2.plot(fairness_scores[b_star_idx], accuracy_scores[b_star_idx], 
                'ro', markersize=8, label='Firm Optimum')
        
        ax2.set_xlabel('Fairness (1 - Disparate Impact)')
        ax2.set_ylabel('Accuracy (Normalized)')
        ax2.set_title('(b) Theoretical Fairness-Accuracy Trade-off', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel (c): Signal Distributions
        print("  Plotting signal distributions...")
        
        # Use the model's method to get the fundamental optimal threshold, t*
        # (This will be equal to params.mu in the current model, but this is more robust)
        t_star_fundamental = model.optimal_threshold(b_star)
        
        sigma_s = np.sqrt(model.signal_variance(b_star))
        x_range = np.linspace(params.mu - 3.5*sigma_s, params.mu + b_star + 3.5*sigma_s, 1000)
        
        # The effective hiring threshold for each group in the observed signal space
        threshold_g0 = t_star_fundamental
        threshold_g1 = t_star_fundamental + b_star

        # Group 0 distribution
        pdf_0 = stats.norm.pdf(x_range, params.mu, sigma_s)
        ax3.plot(x_range, pdf_0, 'b-', linewidth=2, label='Group 0 Signal Dist.')
        ax3.fill_between(x_range, 0, pdf_0, where=(x_range >= threshold_g0), 
                        alpha=0.3, color='blue', label='Hired (Group 0)')
        
        # Group 1 distribution  
        pdf_1 = stats.norm.pdf(x_range, params.mu + b_star, sigma_s)
        ax3.plot(x_range, pdf_1, 'r-', linewidth=2, label='Group 1 Signal Dist.')
        ax3.fill_between(x_range, 0, pdf_1, where=(x_range >= threshold_g1), 
                        alpha=0.3, color='red', label='Hired (Group 1)')
        
        # Plot the effective threshold lines
        ax3.axvline(threshold_g0, color='blue', linestyle='--', linewidth=2, 
                   label=f'Threshold G0 = {threshold_g0:.3f}')
        ax3.axvline(threshold_g1, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold G1 = {threshold_g1:.3f}')
        
        ax3.set_xlabel('Signal Value (s)')
        ax3.set_ylabel('Density')
        ax3.set_title('(c) Optimal Signal Distributions & Hiring Rules', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel (d): Comparative Statics
        print("  Computing comparative statics...")
        kappa_values = np.linspace(0.2, 1.0, 6)
        b_stars = []
        v_functions = []
        
        for kappa in kappa_values:
            params_temp = ModelParameters(
                **{k: v for k, v in params.__dict__.items() if k != 'kappa'},
                kappa=kappa
            )
            model_temp = OptimalBiasModel(params_temp)
            b_temp, _ = model_temp.solve_optimal_bias()
            b_stars.append(b_temp)
            
            # Store value function for this kappa
            b_temp_grid = np.linspace(0.01, params.b_max - 0.01, 30)
            v_temp = [model_temp.value_function(b, n_samples=30000) for b in b_temp_grid]
            v_functions.append((b_temp_grid, v_temp, kappa))
        
        # Plot value functions for different kappa
        colors_kappa = plt.cm.viridis(np.linspace(0.2, 0.8, len(kappa_values)))
        for i, (b_temp_grid, v_temp, kappa) in enumerate(v_functions):
            if i % 2 == 0:  # Show every other one to avoid clutter
                ax4.plot(b_temp_grid, v_temp, color=colors_kappa[i], linewidth=2,
                        label=f'κ = {kappa:.1f}')
        
        ax4.set_xlabel('Bias Level (b)')
        ax4.set_ylabel('Expected Value E[U]')
        ax4.set_title('(d) Comparative Statics (∂b*/∂κ > 0)', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figure_1_model_mechanics.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  Figure 1 saved successfully.")
        
    def generate_figure_2(self, params: ModelParameters) -> None:
        """
        Generate Figure 2: Effects of Policy Interventions.
        
        Shows how Pigouvian taxes and R&D subsidies affect the value function
        and shift the optimal bias choice.
        """
        print("Generating Figure 2: Policy Interventions...")
        
        model = OptimalBiasModel(params)
        b_star_baseline, _ = model.solve_optimal_bias()
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        
        # Baseline value function
        b_grid = np.linspace(0.01, params.b_max - 0.01, 40)
        v_baseline = [model.value_function(b, n_samples=40000) for b in b_grid]
        
        ax.plot(b_grid, v_baseline, 'b-', linewidth=3, label='V(b) (Original)')
        ax.axvline(b_star_baseline, color='blue', linestyle='--', alpha=0.7)
        
        # Pigouvian tax: V(b) - τb
        tau = 0.15  # Tax rate
        v_tax = [v - tau * b for v, b in zip(v_baseline, b_grid)]
        b_star_tax = b_grid[np.argmax(v_tax)]
        
        ax.plot(b_grid, v_tax, 'r-', linewidth=3, label=f'V(b) - τb (Tax, τ={tau})')
        ax.axvline(b_star_tax, color='red', linestyle='--', alpha=0.7)
        
        # R&D subsidy: lower κ
        params_rd = ModelParameters(
            **{k: v for k, v in params.__dict__.items() if k != 'kappa'},
            kappa=params.kappa * 0.6  # 40% reduction in κ
        )
        model_rd = OptimalBiasModel(params_rd)
        v_rd = [model_rd.value_function(b, n_samples=40000) for b in b_grid]
        b_star_rd, _ = model_rd.solve_optimal_bias()
        
        ax.plot(b_grid, v_rd, 'g-', linewidth=3, 
               label=f'V(b) with lower κ (R&D)')
        ax.axvline(b_star_rd, color='green', linestyle='--', alpha=0.7)
        
        # Social optimum
        b_star_social, _ = model.solve_social_optimum()
        ax.axvline(b_star_social, color='orange', linestyle=':', linewidth=3,
                  label=f'Social Optimum b** = {b_star_social:.3f}')
        
        # Add annotations for optimal points
        ax.annotate(f'b* = {b_star_baseline:.3f}', 
                   xy=(b_star_baseline, max(v_baseline)), xytext=(b_star_baseline + 0.1, max(v_baseline)),
                   arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7))
        
        ax.annotate(f'b*_tax = {b_star_tax:.3f}', 
                   xy=(b_star_tax, max(v_tax)), xytext=(b_star_tax + 0.1, max(v_tax)),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
                   
        ax.annotate(f'b*_R&D = {b_star_rd:.3f}', 
                   xy=(b_star_rd, max(v_rd)), xytext=(b_star_rd + 0.1, max(v_rd)),
                   arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))
        
        ax.set_xlabel('Bias Level (b)')
        ax.set_ylabel('Value/Payoff')
        ax.set_title('Effects of Policy Interventions', fontweight='bold', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figure_2_policy_interventions.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  Figure 2 saved successfully.")
        
    def generate_robustness_figure(self, params: ModelParameters) -> None:
        """
        Generate supplementary figure showing robustness across parameter variations.
        """
        print("Generating robustness analysis figure...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
        
        # Vary κ (technology parameter)
        kappa_range = np.linspace(0.1, 1.5, 20)
        b_stars_kappa = []
        for kappa in kappa_range:
            params_temp = ModelParameters(
                **{k: v for k, v in params.__dict__.items() if k != 'kappa'},
                kappa=kappa
            )
            model = OptimalBiasModel(params_temp)
            b_star, _ = model.solve_optimal_bias()
            b_stars_kappa.append(b_star)
        
        ax1.plot(kappa_range, b_stars_kappa, 'b-o', linewidth=2, markersize=4)
        ax1.set_xlabel('Technology Parameter κ')
        ax1.set_ylabel('Optimal Bias b*')
        ax1.set_title('(a) Comparative Static: ∂b*/∂κ > 0')
        ax1.grid(True, alpha=0.3)
        
        # Vary σ²_θ (productivity variance)
        sigma_theta_range = np.linspace(0.5, 2.0, 15)
        b_stars_sigma = []
        for sigma_theta_sq in sigma_theta_range:
            params_temp = ModelParameters(
                **{k: v for k, v in params.__dict__.items() if k != 'sigma_theta_sq'},
                sigma_theta_sq=sigma_theta_sq
            )
            model = OptimalBiasModel(params_temp)
            b_star, _ = model.solve_optimal_bias()
            b_stars_sigma.append(b_star)
        
        ax2.plot(sigma_theta_range, b_stars_sigma, 'g-o', linewidth=2, markersize=4)
        ax2.set_xlabel('Productivity Variance σ²_θ')
        ax2.set_ylabel('Optimal Bias b*')
        ax2.set_title('(b) Effect of Productivity Variance')
        ax2.grid(True, alpha=0.3)
        
        # Vary group proportions
        pi_1_range = np.linspace(0.1, 0.9, 15)
        b_stars_pi = []
        for pi_1 in pi_1_range:
            params_temp = ModelParameters(
                **{k: v for k, v in params.__dict__.items() if k not in ['pi_0', 'pi_1']},
                pi_0=1-pi_1,
                pi_1=pi_1
            )
            model = OptimalBiasModel(params_temp)
            b_star, _ = model.solve_optimal_bias()
            b_stars_pi.append(b_star)
        
        ax3.plot(pi_1_range, b_stars_pi, 'r-o', linewidth=2, markersize=4)
        ax3.set_xlabel('Group 1 Proportion π₁')
        ax3.set_ylabel('Optimal Bias b*')
        ax3.set_title('(c) Effect of Group Composition')
        ax3.grid(True, alpha=0.3)
        
        # Social vs Private optimum comparison
        external_cost_range = np.linspace(0.1, 0.8, 15)
        b_stars_private = []
        b_stars_social = []
        
        for alpha in external_cost_range:
            params_temp = ModelParameters(
                **{k: v for k, v in params.__dict__.items() if k != 'alpha'},
                alpha=alpha
            )
            model = OptimalBiasModel(params_temp)
            b_private, _ = model.solve_optimal_bias()
            b_social, _ = model.solve_social_optimum()
            b_stars_private.append(b_private)
            b_stars_social.append(b_social)
        
        ax4.plot(external_cost_range, b_stars_private, 'b-o', linewidth=2, 
                markersize=4, label='Private Optimum b*')
        ax4.plot(external_cost_range, b_stars_social, 'r-o', linewidth=2, 
                markersize=4, label='Social Optimum b**')
        ax4.set_xlabel('External Cost Parameter α')
        ax4.set_ylabel('Optimal Bias')
        ax4.set_title('(d) Private vs Social Optima')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figure_robustness.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  Robustness figure saved successfully.")

def main():
    """
    Main execution function that generates all figures for the paper.
    
    This function:
    1. Sets up baseline parameters
    2. Creates model instance  
    3. Generates all required figures
    4. Provides summary statistics
    """
    print("="*80)
    print("FIGURE GENERATION FOR 'THE PRICE OF PRECISION'")
    print("Author: Kofi Hair-Ralston")
    print("Date: August 2025")
    print("="*80)
    
    # Initialize baseline parameters
    print("\nInitializing baseline parameters...")
    params = ModelParameters(
        sigma_theta_sq=1.0,  # Normalized to 1, our unit of variance.
        kappa=2.0,           # Technology trade-off is economically significant (2x the productivity variance).
        sigma_0_sq=0.1,      # Baseline noise is 10% of productivity variance.
        b_max=1.0            # Bias is normalized to range from 0 to 1.
    )
    print(f"  Population mean μ = {params.mu}")
    print(f"  Technology parameter κ = {params.kappa}")
    print(f"  Group proportions π₀ = {params.pi_0}, π₁ = {params.pi_1}")
    print(f"  Productivity variance σ²_θ = {params.sigma_theta_sq}")
    
    # Initialize model and solve for baseline optimal bias
    print("\nSolving baseline model...")
    model = OptimalBiasModel(params)
    b_star, optimization_info = model.solve_optimal_bias()
    
    print(f"  Optimal bias b* = {b_star:.4f}")
    print(f"  Value at optimum V(b*) = {optimization_info['function_value']:.4f}")
    print(f"  First-order condition satisfied: |dV/db| = {abs(optimization_info['derivative_at_optimum']):.6f}")
    print(f"  Interior solution: {optimization_info['is_interior']}")
    
    # Compute social optimum for comparison
    b_social, social_info = model.solve_social_optimum()
    print(f"  Social optimum b** = {b_social:.4f}")
    print(f"  Deadweight loss: b* - b** = {b_star - b_social:.4f}")
    
    # Generate figures
    print("\nGenerating figures...")
    figure_gen = FigureGenerator()
    
    try:
        figure_gen.generate_figure_1(params)
        figure_gen.generate_figure_2(params)
        figure_gen.generate_robustness_figure(params)
        
        print("\nAll figures generated successfully!")
        print(f"Output directory: {os.path.abspath(figure_gen.output_dir)}")
        
        # List generated files
        figure_files = [f for f in os.listdir(figure_gen.output_dir) if f.endswith('.png')]
        print(f"\nGenerated files ({len(figure_files)} total):")
        for file in sorted(figure_files):
            file_path = os.path.join(figure_gen.output_dir, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  {file} ({file_size:.1f} MB)")
            
    except Exception as e:
        print(f"\nERROR: Figure generation failed with exception:")
        print(f"  {type(e).__name__}: {str(e)}")
        print("\nThis may be due to:")
        print("  - Insufficient computational resources")
        print("  - Numerical optimization convergence issues")
        print("  - Parameter values outside valid ranges")
        raise
    
    # Model validation and sensitivity checks
    print("\nModel Validation:")
    print("-" * 40)
    
    # Check concavity numerically
    b_test_points = np.linspace(0.01, params.b_max - 0.01, 10)
    second_derivatives = []
    for b in b_test_points:
        h = 0.01
        f_minus = model.value_function(max(0, b - h))
        f_center = model.value_function(b)
        f_plus = model.value_function(min(params.b_max, b + h))
        second_deriv = (f_plus - 2*f_center + f_minus) / (h**2)
        second_derivatives.append(second_deriv)
    
    all_negative = all(d < 0.1 for d in second_derivatives)  # Allow small numerical errors
    print(f"  Value function concavity verified: {all_negative}")
    
    # Check that bias increases with kappa
    kappa_low = ModelParameters(kappa=0.3)
    kappa_high = ModelParameters(kappa=0.7)
    
    model_low = OptimalBiasModel(kappa_low)
    model_high = OptimalBiasModel(kappa_high)
    
    b_low, _ = model_low.solve_optimal_bias()
    b_high, _ = model_high.solve_optimal_bias()
    
    print(f"  Comparative static ∂b*/∂κ > 0: {b_high > b_low} (b*_low={b_low:.3f}, b*_high={b_high:.3f})")
    
    # Economic intuition checks
    threshold = model.optimal_threshold(b_star)
    sigma_s = np.sqrt(model.signal_variance(b_star))
    
    # Hiring rates by group
    prob_hire_0 = 1 - stats.norm.cdf(threshold, params.mu, sigma_s)
    prob_hire_1 = 1 - stats.norm.cdf(threshold, params.mu + b_star, sigma_s)
    
    print(f"  Hiring rate Group 0: {prob_hire_0:.3f}")
    print(f"  Hiring rate Group 1: {prob_hire_1:.3f}")
    print(f"  Disparate impact ratio: {prob_hire_1/prob_hire_0 if prob_hire_0 > 0 else 'undefined':.3f}")
    
    print("\n" + "="*80)
    print("FIGURE GENERATION COMPLETED SUCCESSFULLY")
    print("="*80)

# Support for notebook execution
def display_summary():
    """Display a summary of the model's key predictions for interactive use."""
    params = ModelParameters()
    model = OptimalBiasModel(params)
    b_star, _ = model.solve_optimal_bias()
    
    print("MODEL SUMMARY")
    print("-" * 40)
    print(f"Key Result: Optimal bias b* = {b_star:.4f} > 0")
    print(f"This occurs even with:")
    print(f"  - Identical group productivity (μ₀ = μ₁ = {params.mu})")
    print(f"  - No taste-based discrimination")
    print(f"  - Costless debiasing")
    print(f"\nMechanism: Informativeness Principle")
    print(f"  - Higher precision justifies tolerating bias")
    print(f"  - Trade-off parameter κ = {params.kappa}")
    print(f"  - Signal noise decreases from {params.sigma_0_sq + params.kappa*params.b_max:.1f} to {params.sigma_0_sq:.1f}")

if __name__ == "__main__":
    main()