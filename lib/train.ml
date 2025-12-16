open Types
open Layer

(* for computing gradients *)

(* initialize a model_grad with zero gradients all around *)
let init_zero_grads (m : model) : model_grad =
  let init_zero_grads_layer (l : layer) : layer_grad =
    let w_grads = Array.map (fun row -> 
      Array.map (fun _ -> {grad_mu = 0.0; grad_rho = 0.0}
      ) row 
    ) l.weights
    in
    let b_grads = Array.map (fun _ -> 
      {grad_mu = 0.0; grad_rho = 0.0}) l.bias
    in
    {weight_grads = w_grads; bias_grads = b_grads}
  in
  {
    layer_grads = List.map init_zero_grads_layer m.layers;
    noise_grad = {grad_mu = 0.0; grad_rho = 0.0};
  }

(* initialize the inst grads as 0 *)
let init_zero_inst_grads (inst : inst_model) : inst_model_grad =
  let init_zero_inst_grads_layer (l : inst_layer) : inst_layer_grad = 
    let w_grads = Array.map (fun row -> 
      Array.map (fun _ -> 0.0) row) l.weights in
    let b_grads = Array.map (fun _ -> 0.0) l.bias in
    {weight_grads = w_grads; bias_grads = b_grads}
  in
  List.map init_zero_inst_grads_layer inst

(* Backpropagation for one data point using Gaussian NLL *)
let backprop_single_point (noise_precision : float) (inst : inst_model) (cache : Layer.forward_cache) (y : float array) : inst_model_grad = 
  (* initialize zero gradients *)
  let grads = init_zero_inst_grads inst in
  
  (* Gradient of NLL w.r.t. output: ∂NLL/∂ŷ = τ·(ŷ - y) *)
  let initial_delta = Array.map2 (fun pred target ->
    noise_precision *. (pred -. target)) cache.output y in
  
  let layers_rev = List.rev inst in
  let caches_rev = List.rev cache.layer_caches in
  let grads_rev = List.rev grads in

  let rec backprop_layers layers caches grads_list delta_next =
    match layers, caches, grads_list with
    | [], [], [] -> []
    | layer_hd :: layer_tl, cache_hd :: cache_tl, grad_hd :: grad_tl -> 
      let delta_pre = Array.mapi (fun i d ->
        let pre = cache_hd.pre_activation.(i) in
        let post = cache_hd.post_activation.(i) in
        let deriv = Activation.deriv layer_hd.activation pre post in
        d *. deriv
      ) delta_next in
      
      (* Now compute weight and bias gradients using ∂L/∂z *)
      for i = 0 to Array.length layer_hd.weights - 1 do
        for j = 0 to Array.length layer_hd.weights.(0) - 1 do
          grad_hd.weight_grads.(i).(j) <- delta_pre.(i) *. cache_hd.input.(j)
        done
      done;

      for i = 0 to Array.length layer_hd.bias - 1 do
        grad_hd.bias_grads.(i) <- delta_pre.(i)
      done;

      let delta_prev =
        if layer_tl <> [] then begin
          let input_size = Array.length cache_hd.input in
          (* Compute W^T · ∂L/∂z to get ∂L/∂a_{prev} *)
          Array.init input_size (fun j -> 
            let sum = ref 0.0 in
            for i = 0 to Array.length delta_pre - 1 do
              sum := !sum +. layer_hd.weights.(i).(j) *. delta_pre.(i)
            done;
            !sum
          )
          end else
            [||]
          in
          grad_hd :: backprop_layers layer_tl cache_tl grad_tl delta_prev

    | _ -> failwith "Mismatched layers/caches/grads" 
        in
        List.rev (backprop_layers layers_rev caches_rev grads_rev initial_delta)

(* backpropagation function *)
let backprop (noise_precision : float) (inst : inst_model) (data : (float array * float array) list) : inst_model_grad =
  let total_grads = init_zero_inst_grads inst in
  let n_data = float_of_int (List.length data) in

  List.iter (fun (x, y) -> 
    let cache = Layer.forward_with_cache inst x in

    let grads_for_this_point = backprop_single_point noise_precision inst cache y in

    (* Use List.iter2 to ensure proper alignment between total_grads and point gradients *)
    List.iter2 (fun total_layer point_layer ->
      for i = 0 to Array.length total_layer.weight_grads - 1 do
        for j = 0 to Array.length total_layer.weight_grads.(0) - 1 do
          total_layer.weight_grads.(i).(j) <- total_layer.weight_grads.(i).(j) +. point_layer.weight_grads.(i).(j)
        done
      done;

      for i = 0 to Array.length total_layer.bias_grads - 1 do
        total_layer.bias_grads.(i) <- total_layer.bias_grads.(i) +. point_layer.bias_grads.(i)
      done
    ) total_grads grads_for_this_point
    ) data;
  
  (* Average gradients over all data points *)
  List.map (fun layer_grad ->
    {
      weight_grads = Array.map (fun row ->
        Array.map (fun g -> g /. n_data) row
      ) layer_grad.weight_grads;
      bias_grads = Array.map (fun g -> g /. n_data) layer_grad.bias_grads;
    }
  ) total_grads


type sampled_layer = {
  inst : inst_layer;
  eps_w : float array array;
  eps_b : float array;
}

let sample_layer_with_epsilons (l : layer) : sampled_layer =
  let eps_w = Array.map (fun row -> 
    Array.map (fun _ -> Dist.sample_normal ()) row
    ) l.weights
  in

  let eps_b = Array.map (fun _ -> Dist.sample_normal ()) l.bias in
  
  let w = Array.map2 (fun param_row eps_row -> 
    Array.map2 (fun p eps -> 
      p.mu +. eps *. (Dist.sigma p.rho)
      ) param_row eps_row
    ) l.weights eps_w in
  
  let b = Array.map2 (fun p eps ->
    p.mu +. eps *. (Dist.sigma p.rho)
  ) l.bias eps_b in

  {
    inst = {weights = w; bias = b; activation = l.activation};
    eps_w = eps_w;
    eps_b = eps_b;
  }

let sample_model_with_epsilons (m : model) : sampled_layer list =
  List.map sample_layer_with_epsilons m.layers
  
(* Derivative of softplus *)
let softplus_deriv (rho : float) : float =
  let exp_rho = Float.exp rho in
  exp_rho /. (1.0 +. exp_rho)

let rec map3 f l1 l2 l3 =
  match l1, l2, l3 with
  | x1 :: t1, x2 :: t2, x3 :: t3 ->
      f x1 x2 x3 :: map3 f t1 t2 t3
  | [], [], [] ->
      []
  | _ ->
      invalid_arg "map3: lists have different lengths"

(* Reparameterization gradient computation *)
let reparam_gradients (m : model) (sampled_layers : sampled_layer list) (inst_grads : inst_model_grad) : layer_grad list =
  map3 (fun (layer : layer) (sampled_layer : sampled_layer) (inst_grad : inst_layer_grad) : layer_grad ->
    let w_grads = Array.mapi (fun i param_row -> 
      Array.mapi (fun j param ->
        let grad_w = inst_grad.weight_grads.(i).(j) in
        let eps = sampled_layer.eps_w.(i).(j) in
        let sigma_deriv = softplus_deriv param.rho in
        {
          grad_mu = grad_w;
          grad_rho = grad_w *. eps *. sigma_deriv
        }
        ) param_row
      ) layer.weights in
    let b_grads = Array.mapi (fun i param ->
      let grad_b = inst_grad.bias_grads.(i) in
      let eps = sampled_layer.eps_b.(i) in
      let sigma_deriv = softplus_deriv param.rho in
      {
        grad_mu = grad_b;
        grad_rho = grad_b *. eps *. sigma_deriv;
      }
      ) layer.bias in

      {weight_grads = w_grads ; bias_grads = b_grads}
    ) m.layers sampled_layers inst_grads


(* KL divergence gradient computations *)
let compute_kl_gradients (m : model) (prior_mu : float) (prior_sigma : float) : layer_grad list =
  let prior_var = prior_sigma *. prior_sigma in

  List.map (fun (layer : layer) : layer_grad ->
    let w_grads : param_grad array array = Array.map (fun row ->
      Array.map (fun param -> 
        let sigma_q = Dist.sigma param.rho in
        let var_q = sigma_q *. sigma_q in

        (* ∂KL/∂μ = (μ - μ_prior) / σ²_prior *)
        let grad_mu = (param.mu -. prior_mu) /. prior_var in

        (* ∂KL/∂ρ = [(σ²_q - σ²_p)/(σ_q · σ²_p)] · σ'(ρ) *)
        let sigma_deriv = softplus_deriv param.rho in
        let grad_rho = ((var_q -. prior_var) /. (sigma_q *. prior_var) *. sigma_deriv) in

        {grad_mu = grad_mu ; grad_rho = grad_rho}
        ) row
      ) layer.weights in
    
    let b_grads = Array.map (fun param ->
      let sigma_q = Dist.sigma param.rho in
      let var_q = sigma_q *. sigma_q in

      let grad_mu = (param.mu -. prior_mu) /. prior_var in

      let sigma_deriv = softplus_deriv param.rho in
      let grad_rho = (var_q -. prior_var) /. (sigma_q *. prior_var) *. sigma_deriv in

      {grad_mu; grad_rho}
      ) layer.bias in

      {weight_grads = w_grads; bias_grads = b_grads}
    ) m.layers


(* helper function to add two model_grad structures element wise *)
let add_gradients (g1 : model_grad) (g2 : model_grad) : model_grad =
  let combined_layers = List.map2 (fun (layer1 : layer_grad) (layer2 : layer_grad) : layer_grad ->
    let w_grads = Array.map2 (fun row1 row2 ->
      Array.map2 (fun p1 p2 ->
        {
          grad_mu = p1.grad_mu +. p2.grad_mu;
          grad_rho = p1.grad_rho +. p2.grad_rho;
        }
      ) row1 row2
    ) layer1.weight_grads layer2.weight_grads in

    let b_grads = Array.map2 (fun p1 p2 ->
      {
        grad_mu = p1.grad_mu +. p2.grad_mu;
        grad_rho = p1.grad_rho +. p2.grad_rho;
      }
    ) layer1.bias_grads layer2.bias_grads in

    {weight_grads = w_grads; bias_grads = b_grads}
  ) g1.layer_grads g2.layer_grads in
  
  {
    layer_grads = combined_layers;
    noise_grad = {
      grad_mu = g1.noise_grad.grad_mu +. g2.noise_grad.grad_mu;
      grad_rho = g1.noise_grad.grad_rho +. g2.noise_grad.grad_rho;
    };
  }


(* helper function to scale gradients by a constant *)
let scale_gradients (grad : model_grad) (scale : float) : model_grad =
  let scaled_layers = List.map (fun (layer : layer_grad) : layer_grad ->
    let w_grads = Array.map (fun row ->
      Array.map (fun p -> 
      {
        grad_mu = p.grad_mu *. scale;
        grad_rho = p.grad_rho *. scale;
      }
      ) row
    ) layer.weight_grads in

    let b_grads = Array.map (fun p ->
      {
        grad_mu = p.grad_mu *. scale;
        grad_rho = p.grad_rho *. scale;
      }
      ) layer.bias_grads in
    
    {
      weight_grads = w_grads;
      bias_grads = b_grads;
    }
  ) grad.layer_grads in
  
  {
    layer_grads = scaled_layers;
    noise_grad = {
      grad_mu = grad.noise_grad.grad_mu *. scale;
      grad_rho = grad.noise_grad.grad_rho *. scale;
    };
  } 




(* compute gradient of NLL w.r.t. noise precision parameter *)
let compute_noise_gradient (m : model) (sampled_log_precision : float) (epsilon : float) (data : (float array * float array) list) (inst : inst_model) : param_grad =
  let noise_precision = Float.exp sampled_log_precision in
  
  (* Compute total SSE and count *)
  let (sse, n) = List.fold_left (fun (acc_sse, acc_n) (x, y_true) ->
    let y_pred = Model.forward_inst inst x in
    let point_sse = ref 0.0 in
    for i = 0 to Array.length y_true - 1 do
      let error = y_pred.(i) -. y_true.(i) in
      point_sse := !point_sse +. (error *. error)
    done;
    (acc_sse +. !point_sse, acc_n +. float_of_int (Array.length y_true))
  ) (0.0, 0.0) data in
  
  (* ∂NLL/∂log_τ *)
  let grad_log_tau = 0.5 *. (noise_precision *. sse -. n) in
  
  (* Reparameterization: log_τ = μ + ε·σ(ρ), so ∂/∂μ = 1, ∂/∂ρ = ε·σ'(ρ) *)
  let sigma_deriv = softplus_deriv m.log_noise_precision.rho in
  {
    grad_mu = grad_log_tau;
    grad_rho = grad_log_tau *. epsilon *. sigma_deriv;
  }

(* MAIN GRADIENT COMPUTATION FUNCTION *)
let compute_gradient (m : model) (data : (float array * float array) list) (num_samples : int) (beta : float) (prior_mu : float) (prior_sigma : float) : model_grad =
  (* beta is now passed as parameter for easy configuration *)
  let accumulated_grads = ref (init_zero_grads m) in

  (* Monte Carlo estimation: sample M models and average gradients *)
  for _ = 1 to num_samples do
    let sampled_layers = sample_model_with_epsilons m in
    let inst = List.map (fun sl -> sl.inst) sampled_layers in

    (* Sample noise precision with epsilon for reparameterization *)
    let noise_epsilon = Dist.sample_normal () in
    let sampled_log_precision = m.log_noise_precision.mu +. noise_epsilon *. (Dist.sigma m.log_noise_precision.rho) in
    let noise_precision = Float.exp sampled_log_precision in

    (* Compute ∇_w NLL via backpropagation on sampled weights *)
    let inst_grads = backprop noise_precision inst data in

    (* Convert ∇_w to ∇_θ={μ,ρ} using reparameterization trick *)
    let param_grads = reparam_gradients m sampled_layers inst_grads in
    
    (* Compute noise precision gradient using the SAME sampled noise *)
    let noise_grad = compute_noise_gradient m sampled_log_precision noise_epsilon data inst in

    (* Accumulate gradients from this sample *)
    let sample_grad = {
      layer_grads = param_grads;
      noise_grad = noise_grad;
    } in
    accumulated_grads := add_gradients !accumulated_grads sample_grad
  done;

  (* Average over M samples *)
  let avg_grads = scale_gradients !accumulated_grads (1.0 /. float_of_int num_samples) in

  (* Add KL gradient (analytical, no sampling needed) scaled by β *)
  let kl_grads = compute_kl_gradients m prior_mu prior_sigma in
  
  (* Compute KL gradient for noise parameter *)
  let sigma_q = Dist.sigma m.log_noise_precision.rho in
  let var_q = sigma_q *. sigma_q in
  let prior_var = prior_sigma *. prior_sigma in
  let noise_kl_grad_mu = (m.log_noise_precision.mu -. prior_mu) /. prior_var in
  let sigma_deriv = softplus_deriv m.log_noise_precision.rho in
  let noise_kl_grad_rho = ((var_q -. prior_var) /. (sigma_q *. prior_var)) *. sigma_deriv in
  
  let scaled_kl = {
    layer_grads = List.map (fun (layer : layer_grad) : layer_grad ->
      let w_grads = Array.map (fun row ->
        Array.map (fun p -> 
        {
          grad_mu = p.grad_mu *. beta;
          grad_rho = p.grad_rho *. beta;
        }
        ) row
      ) layer.weight_grads in

      let b_grads = Array.map (fun p ->
        {
          grad_mu = p.grad_mu *. beta;
          grad_rho = p.grad_rho *. beta;
        }
        ) layer.bias_grads in
      
      {
        weight_grads = w_grads;
        bias_grads = b_grads;
      }
    ) kl_grads;
    noise_grad = {
      grad_mu = noise_kl_grad_mu *. beta;
      grad_rho = noise_kl_grad_rho *. beta;
    };
  } in

  add_gradients avg_grads scaled_kl




(* Clip gradient to prevent NaN *)
let clip_grad g max_norm =
  let clipped_mu = max (-. max_norm) (min max_norm g.grad_mu) in
  let clipped_rho = max (-. max_norm) (min max_norm g.grad_rho) in
  {grad_mu = clipped_mu; grad_rho = clipped_rho}


(* APPLY GRADIENTS TO UPDATE MODEL PARAMS USING GRADIENT DESCENT *)
let update_params (m : model) (grads : model_grad) (lr : float) : model =
  let max_grad_norm = 5.0 in  (* Very high clip threshold - effectively no clipping *)
  let max_rho = 5.0 in          (* Allow larger sigma for expressive weights *)
  let min_rho = -3.0 in         (* Prevent overly deterministic weights *)
  
  let new_layers = List.map2 (fun (layer : layer) (layer_grad : layer_grad) : layer ->
    let new_weights = Array.map2 (fun param_row grad_row ->
      Array.map2 (fun param grad ->
        let clipped = clip_grad grad max_grad_norm in
        let new_mu = param.mu -. lr *. clipped.grad_mu in
        let new_rho_unclamped = param.rho -. lr *. clipped.grad_rho in
        (* Clamp rho to prevent sigma from becoming too large or too small *)
        let new_rho = max min_rho (min max_rho new_rho_unclamped) in
        (* Check for NaN and keep old values if detected *)
        {
          mu = if classify_float new_mu = FP_nan then param.mu else new_mu;
          rho = if classify_float new_rho = FP_nan then param.rho else new_rho;
        }
      ) param_row grad_row
    ) layer.weights layer_grad.weight_grads in

    let new_bias = Array.map2 (fun param grad -> 
      let clipped = clip_grad grad max_grad_norm in
      let new_mu = param.mu -. lr *. clipped.grad_mu in
      let new_rho_unclamped = param.rho -. lr *. clipped.grad_rho in
      let new_rho = max min_rho (min max_rho new_rho_unclamped) in
      {
        mu = if classify_float new_mu = FP_nan then param.mu else new_mu;
        rho = if classify_float new_rho = FP_nan then param.rho else new_rho;
      }
    ) layer.bias layer_grad.bias_grads in

    {weights = new_weights; bias = new_bias; activation = layer.activation}
  ) m.layers grads.layer_grads in
  
  (* Update noise precision parameter *)
  let clipped_noise = clip_grad grads.noise_grad max_grad_norm in
  let new_noise_mu_unclamped = m.log_noise_precision.mu -. lr *. clipped_noise.grad_mu in
  (* Clamp log_noise_precision.mu to keep noise precision >= 0.1 (log >= -2.3) 
     This allows more flexibility for datasets with large initial errors *)
  let new_noise_mu = max (-2.3) new_noise_mu_unclamped in
  let new_noise_rho_unclamped = m.log_noise_precision.rho -. lr *. clipped_noise.grad_rho in
  let new_noise_rho = max min_rho (min max_rho new_noise_rho_unclamped) in
  let new_log_noise_precision = {
    mu = if classify_float new_noise_mu = FP_nan then m.log_noise_precision.mu else new_noise_mu;
    rho = if classify_float new_noise_rho = FP_nan then m.log_noise_precision.rho else new_noise_rho;
  } in
  
  {
    layers = new_layers;
    log_noise_precision = new_log_noise_precision;
  }


(* FULL TRAINING FUNCTION - returns (trained_model, loss_history) *)
let train (m : model) (data : (float array * float array) list) (num_epochs : int) (lr : float) (num_samples : int) (beta : float) (prior_mu : float) (prior_sigma : float) : model * float list =
  let loss_history = ref [] in
  let n_data = float_of_int (List.length data) in
  
  let rec train_loop model epoch =
    if epoch > num_epochs then model
    else begin
      let loss = Model.compute_elbo model data num_samples beta prior_mu prior_sigma in
      let loss_per_point = loss /. n_data in
      loss_history := loss_per_point :: !loss_history;
      
      if epoch mod 100 == 0 then begin
        (* Compute NLL and KL separately for reporting *)
        let inst = Model.sample_inst_model model in
        let nll = Model.compute_nll model inst data in
        let kl = Model.compute_total_kl model prior_mu prior_sigma in
        let nll_per_point = nll /. n_data in
        let kl_per_point = kl /. n_data in
        let noise_precision = Float.exp (Dist.reparam model.log_noise_precision) in
        
        (* Compute MSE (mean squared error) on predictions *)
        let mse = List.fold_left (fun acc (x, y_true) ->
          let y_pred = Model.forward_inst inst x in
          let point_error = ref 0.0 in
          for i = 0 to Array.length y_true - 1 do
            let diff = y_pred.(i) -. y_true.(i) in
            point_error := !point_error +. (diff *. diff)
          done;
          acc +. !point_error
        ) 0.0 data in
        let mse_per_point = mse /. n_data in
        let rmse = Float.sqrt mse_per_point in
        
        Printf.printf "Epoch %4d/%d | Loss: %.6f | NLL: %.6f | KL: %.6f | MSE: %.6f | RMSE: %.4f | τ: %.4f\n%!" 
          epoch num_epochs loss nll_per_point kl_per_point mse_per_point rmse noise_precision
      end;

      let grads = compute_gradient model data num_samples beta prior_mu prior_sigma in
      
      (* Print gradient magnitudes for each layer *)
      if epoch mod 100 == 0 then begin
        Printf.printf "  Gradient magnitudes:\n";
        List.iteri (fun layer_idx (layer_grad : layer_grad) ->
          (* Compute average absolute gradient for weights *)
          let sum_w = ref 0.0 in
          let count_w = ref 0 in
          Array.iter (fun row ->
            Array.iter (fun g ->
              sum_w := !sum_w +. abs_float g.grad_mu;
              count_w := !count_w + 1
            ) row
          ) layer_grad.weight_grads;
          let avg_w = !sum_w /. float_of_int !count_w in
          
          (* Compute average absolute gradient for biases *)
          let sum_b = ref 0.0 in
          let count_b = ref 0 in
          Array.iter (fun g ->
            sum_b := !sum_b +. abs_float g.grad_mu;
            count_b := !count_b + 1
          ) layer_grad.bias_grads;
          let avg_b = !sum_b /. float_of_int !count_b in
          
          Printf.printf "    Layer %d: |∇W_μ| = %.6f, |∇b_μ| = %.6f\n" 
            layer_idx avg_w avg_b
        ) grads.layer_grads;
        Printf.printf "\n%!"
      end;

      let new_model = update_params model grads lr in

      train_loop new_model (epoch + 1)
    end
  in

  let final_model = train_loop m 1 in
  (final_model, List.rev !loss_history)