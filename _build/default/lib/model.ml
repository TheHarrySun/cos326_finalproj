(* code for instantiating a model and running through it *)

open Types
open Layer

let sample_inst_model (m : model) : inst_model =
  List.map sample_inst_layer m.layers

let forward_inst (inst : inst_model) (x : float array) : float array =
  List.fold_left (
    fun input layer ->
      Layer.forward_inst_layer layer input
  )
  x inst

let sample_mlp (input_dim : int) (hidden : int) (output_dim : int) () =
  let l1 =
    Layer.init_layer input_dim hidden Activation.Relu in
  let l2 =
    Layer.init_layer hidden output_dim Activation.Identity in
  { 
    layers = [ l1; l2 ];
    log_noise_precision = { mu = 0.0; rho = -2.0 };
  }

(* Generic MLP builder that accepts a list of layer sizes and activation functions
   Example: create_mlp [1; 20; 10; 1] [Activation.Relu; Activation.Relu; Activation.Identity]
   Creates: input(1) -> hidden1(20,relu) -> hidden2(10,relu) -> output(1,identity) *)
let create_mlp (layer_sizes : int list) (activations : Activation.activation list) : model =
  if List.length layer_sizes < 2 then
    failwith "Need at least 2 layer sizes (input and output)"
  else if List.length activations <> List.length layer_sizes - 1 then
    failwith "Need exactly (num_layers - 1) activation functions"
  else
    let rec build_layers sizes acts =
      match sizes, acts with
      | [], _ | [_], _ -> []
      | in_dim :: out_dim :: rest_sizes, act :: rest_acts ->
          let layer = Layer.init_layer in_dim out_dim act in
          layer :: build_layers (out_dim :: rest_sizes) rest_acts
      | _ -> failwith "Mismatched sizes and activations"
    in
    {
      layers = build_layers layer_sizes activations;
      (* Initialize with higher noise precision: log_tau = 0.5 -> tau = 1.65 *)
      log_noise_precision = { mu = 0.5; rho = -2.0 };
    }

(* computing the KL divergence of a whole model *)
let compute_total_kl (m : model) (prior_mu : float) (prior_sigma : float) : float = 
    (* helper function to compute the kl of a layer *)
    let compute_layer_kl (layer : layer) : float =
      (* helper function to compute the kl of a param *)
      let compute_kl_of_param (param : param) : float = 
        let sigma_q = Dist.sigma param.rho in
        Dist.kl_divergence param.mu sigma_q prior_mu prior_sigma
      in

      let weight_kl = begin
        Array.fold_left (fun (acc_row : float) (row : param array) -> 
          Array.fold_left (fun (acc_param : float) (param : param) -> 
            acc_param +. compute_kl_of_param param
          ) acc_row row
        ) 0.0 layer.weights
      end in
      
      let bias_kl = begin
        Array.fold_left (fun (acc_param : float) (param : param) ->
          acc_param +. compute_kl_of_param param
        ) 0.0 layer.bias
      end in
      
      weight_kl +. bias_kl
    in
  List.fold_left (fun (acc : float) (layer : layer) -> 
    acc +. compute_layer_kl layer
  ) 0.0 m.layers

(* compute the negative log likelihood for a particular batch
   
   For Gaussian likelihood p(y|x,w,τ) = N(y; f(x,w), 1/τ) where τ is precision:
   NLL = Σ_i [(τ/2)·(y_i - f(x_i,w))² - (1/2)·log(τ) + (1/2)·log(2π)]
   
   With learnable log_precision parameter matching Python implementation *)
let compute_nll (m : model) (inst : inst_model) (data : (float array * float array) list) : float =
  let log_noise_precision = Dist.reparam m.log_noise_precision in
  let noise_precision = Float.exp log_noise_precision in
  
  List.fold_left (fun acc (x, y_true) -> begin
    let y_pred = forward_inst inst x in
    (* compute sum of squared errors *)
    let sse = ref 0.0 in
    for i = 0 to Array.length y_true - 1 do
      let error = y_pred.(i) -. y_true.(i) in
      sse := !sse +. (error *. error)
    done;
    (* NLL = (τ/2)·SSE - (N/2)·log(τ) + (N/2)·log(2π) *)
    let n = float_of_int (Array.length y_true) in
    acc +. (0.5 *. noise_precision *. !sse) -. (0.5 *. n *. log_noise_precision) +. (0.5 *. n *. Float.log (2.0 *. Float.pi))
  end
  ) 0.0 data


(* Evidence Lower Bound (ELBO) for Bayesian Neural Network
   
   ELBO = E_q[log p(D|w)] - β·KL(q(w)||p(w))
   
   where:
   - E_q[log p(D|w)] ≈ (1/M) Σ log p(D|w^(m)) with w^(m) ~ q(w)  [MC estimate]
   - log p(D|w) = -NLL(w) for Gaussian likelihood
   - KL(q(w)||p(w)) is sum of KL divergences for all weight parameters
   - β is a scaling factor (β-VAE): β<1 reduces regularization
   
   Both NLL and KL are total (not normalized) values.
   Normalization only happens for display in the training loop.
   
   Maximizing ELBO ≡ Minimizing: NLL + β·KL *)
let compute_elbo (m : model) (data : (float array * float array) list) (num_samples : int) (beta : float) (prior_mu : float) (prior_sigma : float) : float =
  (* beta is now passed as parameter for easy configuration *)

  let kl_term = beta *. compute_total_kl m prior_mu prior_sigma in

  let nll_sum = ref 0.0 in
  for _ = 1 to num_samples do
    let inst = sample_inst_model m in
    nll_sum := !nll_sum +. compute_nll m inst data
  done;
  let avg_nll = !nll_sum /. float_of_int num_samples in

  avg_nll +. kl_term


(* Print model weights for debugging *)
let print_model_weights (m : model) : unit =
  Printf.printf "\n=================================\n";
  Printf.printf "Model Weights (All Parameters)\n";
  Printf.printf "=================================\n\n";
  
  (* Print noise precision *)
  Printf.printf "Noise precision (log_tau):\n";
  Printf.printf "  mu = %.6f, rho = %.6f, sigma = %.6f\n" 
    m.log_noise_precision.mu 
    m.log_noise_precision.rho
    (Dist.sigma m.log_noise_precision.rho);
  Printf.printf "  tau = exp(mu) = %.6f\n\n" (Float.exp m.log_noise_precision.mu);
  
  (* Print layer weights *)
  List.iteri (fun layer_idx (layer : layer) ->
    Printf.printf "Layer %d:\n" layer_idx;
    Printf.printf "  Weights shape: %d x %d\n" 
      (Array.length layer.weights) 
      (Array.length layer.weights.(0));
    
    (* Print all weight values *)
    Printf.printf "  Weights:\n";
    Array.iteri (fun i row ->
      Array.iteri (fun j p ->
        Printf.printf "    W[%d,%d]: mu=%.6f, rho=%.6f, sigma=%.6f\n" 
          i j p.mu p.rho (Dist.sigma p.rho)
      ) row
    ) layer.weights;
    
    (* Print all bias values *)
    Printf.printf "  Biases:\n";
    Array.iteri (fun i p ->
      Printf.printf "    b[%d]: mu=%.6f, rho=%.6f, sigma=%.6f\n" 
        i p.mu p.rho (Dist.sigma p.rho)
    ) layer.bias;
    
    Printf.printf "\n"
  ) m.layers;
  
  Printf.printf "=================================\n\n"