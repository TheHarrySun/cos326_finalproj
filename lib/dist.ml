(* Contains the probability distribution utilities *)

open Types

(* Softplus function: σ(ρ) = log(1 + e^ρ)
   This is used to ensure σ > 0 by parameterizing σ = softplus(ρ)
   where ρ can be any real number. *)
let softplus x : float = Float.log (1.0 +. Float.exp x)

let sigma (rho : float) : float = softplus rho

(* sampling from standard normal N(0,1). *)
let sample_normal () : float =
  let u = Random.float 1.0 in
  let v = Random.float 1.0 in
  Float.sqrt (-2.0 *. Float.log u) *. Float.cos (2.0 *. Float.pi *. v)


(* Reparameterization trick: w = μ + ε·σ(ρ) where ε ~ N(0,1) *)
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
   
   Formula: KL(q||p) = log(σ_p/σ_q) + (σ_q² + (μ_q - μ_p)²)/(2σ_p²) - 1/2 *)
let kl_divergence (mu_q : float) (sigma_q : float) (mu_p : float) (sigma_p : float) : float = 
  Float.log (sigma_p /. sigma_q) +. (sigma_q ** 2. +. (mu_q -. mu_p) ** 2.) /. (2. *. sigma_p ** 2.) -. 0.5