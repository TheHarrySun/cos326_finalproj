(* Contains the probability distribution utilities *)

open Types

(* Softplus function: σ(ρ) = log(1 + e^ρ)
   This is used to ensure σ > 0 by parameterizing σ = softplus(ρ)
   where ρ can be any real number. *)
let softplus x : float = Float.log (1.0 +. Float.exp x)

let sigma (rho : float) : float = softplus rho

(* Box-Muller transform for sampling from standard normal N(0,1).
   Given two independent uniform random variables U₁, U₂ ~ Uniform(0,1),
   Z = √(-2 ln U₁) · cos(2πU₂) follows N(0,1) distribution.
   Alternative formula using sin would also work. *)
let sample_normal () : float =
  let u = Random.float 1.0 in
  let v = Random.float 1.0 in
  Float.sqrt (-2.0 *. Float.log u) *. Float.cos (2.0 *. Float.pi *. v)


(* Reparameterization trick: w = μ + ε·σ(ρ) where ε ~ N(0,1)
   This allows gradients to flow through the sampling operation.
   Instead of sampling w ~ N(μ, σ²), we sample ε ~ N(0,1) deterministically
   and compute w as a deterministic function of μ, ρ, and ε. *)
let reparam (p : param) : float =
  let eps = sample_normal () in
  p.mu +. eps *. (sigma p.rho)

(* Function used to compute the natural log of the PDF of a Gaussian with mean mu and standard devication sigam for a certain value x. *)
let log_of_gaussianpdf (mu : float) (sigma : float) (x : float) : float =
  let coeff = 1. /. (Float.sqrt (2.0 *. Float.pi) *. sigma) in
  let exponent = -. ((x -. mu) *. (x -. mu)) /. (2.0 *. sigma *. sigma) in
  Float.log coeff +. exponent

(* KL divergence between two univariate Gaussians: KL(q||p)
   where q = N(μ_q, σ_q²) and p = N(μ_p, σ_p²)
   
   Formula: KL(q||p) = log(σ_p/σ_q) + (σ_q² + (μ_q - μ_p)²)/(2σ_p²) - 1/2
   
   Derivation from KL(q||p) = ∫ q(x) log(q(x)/p(x)) dx:
   = E_q[log q(x)] - E_q[log p(x)]
   = -H(q) - E_q[log p(x)]
   where H(q) is the entropy of q. *)
let kl_divergence (mu_q : float) (sigma_q : float) (mu_p : float) (sigma_p : float) : float = 
  Float.log (sigma_p /. sigma_q) +. (sigma_q ** 2. +. (mu_q -. mu_p) ** 2.) /. (2. *. sigma_p ** 2.) -. 0.5