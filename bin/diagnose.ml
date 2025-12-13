open Final_proj

let () =
  Printf.printf "=== BNN Diagnostic Tool ===\n\n";
  
  Util.make_rng 42;
  
  (* Small test *)
  let data = Util.make_sin_data 10 in
  let model = Model.sample_mlp 1 10 1 () in
  
  (* Check individual components *)
  let inst = Model.sample_inst_model model in
  let nll = Model.compute_nll model inst data in
  let kl = Model.compute_total_kl model 0.0 1.0 in
  let kl_scaled = kl /. float_of_int (List.length data) in
  
  Printf.printf "NLL (likelihood term): %.6f\n" nll;
  Printf.printf "KL (regularization): %.6f\n" kl;
  Printf.printf "KL scaled by n_data: %.6f\n" kl_scaled;
  Printf.printf "ELBO = NLL + KL_scaled: %.6f\n\n" (nll +. kl_scaled);
  
  (* Check if KL is too strong *)
  if kl_scaled > nll then
    Printf.printf "⚠️  WARNING: KL term dominates! Model heavily regularized.\n\n"
  else
    Printf.printf "✓ NLL and KL are balanced.\n\n";
  
  (* Check parameter magnitudes *)
  let layer1 = List.hd model.layers in
  let sample_param = layer1.weights.(0).(0) in
  let sample_sigma = Dist.sigma sample_param.rho in
  
  Printf.printf "Sample parameter:\n";
  Printf.printf "  mu: %.6f\n" sample_param.mu;
  Printf.printf "  rho: %.6f\n" sample_param.rho;
  Printf.printf "  sigma: %.6f\n\n" sample_sigma;
  
  (* Check prediction scale *)
  Printf.printf "Sample predictions:\n";
  for i = 0 to 4 do
    let x = float_of_int i -. 2.0 in
    let pred = (Model.forward_inst inst [| x |]).(0) in
    Printf.printf "  f(%.1f) = %.6f (true sin: %.4f)\n" 
      x pred (Float.sin x)
  done;
  
  Printf.printf "\n=== Recommendations ===\n";
  Printf.printf "1. Try larger learning rate (0.1)\n";
  Printf.printf "2. Try weaker prior (sigma=10 instead of 1)\n";
  Printf.printf "3. Try better initialization (mu in [-0.5, 0.5])\n"
