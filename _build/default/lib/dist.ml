(* Contains the probability distribution utilities *)

open Types

let softplus x : float = Float.log (1.0 +. Float.exp x)

let sigma (rho : float) : float = softplus rho

(* one method for randomly sampling a Gaussian is using two uniform variables (we'll call U1 and U2) and computing a Gaussian N(0, 1) with the formulas sqrt(-2 * ln(U1)) * cos(2pi*U2) or sqrt(-2 * ln(U1)) * cos(2pi*U2) *)
let sample_normal () : float =
  let u = Random.float 1.0 in
  let v = Random.float 1.0 in
  Float.sqrt (-2.0 *. Float.log u) *. Float.cos (2.0 *. Float.pi *. v)


let reparam (p : param) : float =
  let eps = sample_normal () in
  p.mu +. eps *. (sigma p.rho)

(* Function used to compute the natural log of the PDF of a Gaussian with mean mu and standard devication sigam for a certain value x. *)
let log_of_gaussianpdf (mu : float) (sigma : float) (x : float) : float =
  let coeff = 1. /. (Float.sqrt (2.0 *. Float.pi) *. sigma) in
  let exponent = -. ((x -. mu) *. (x -. mu)) /. (2.0 *. sigma *. sigma) in
  Float.log coeff +. exponent

(* Function used to compute the KL divergence between two normal distributions, q = N(mu_q, sigma_q^2) and p = N(mu_p, sigma_p^2) *)
let kl_divergence (mu_q : float) (sigma_q : float) (mu_p : float) ~(sigma_p : float) : float = 
  Float.log (sigma_p /. sigma_q) +. (sigma_q ** 2. +. (mu_q -. mu_p) ** 2.) /. (2. *. sigma_p ** 2.) -. 0.5