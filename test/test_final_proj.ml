open Final_proj

(* Test utilities *)
let assert_float_close ~msg expected actual tolerance =
  let diff = abs_float (expected -. actual) in
  if diff > tolerance then
    Printf.printf "FAIL: %s\n  Expected: %.6f\n  Got: %.6f\n  Diff: %.6f\n" 
      msg expected actual diff
  else
    Printf.printf "PASS: %s\n" msg

let assert_true ~msg condition =
  if condition then
    Printf.printf "PASS: %s\n" msg
  else
    Printf.printf "FAIL: %s\n" msg

(* Test 1: Distribution utilities *)
let test_distributions () =
  Printf.printf "\n=== Testing Distribution Utilities ===\n";
  
  (* Test softplus *)
  let sp1 = Dist.softplus 0.0 in
  assert_float_close ~msg:"softplus(0) ≈ 0.693" 0.693 sp1 0.01;
  
  let sp2 = Dist.softplus (-3.0) in
  assert_float_close ~msg:"softplus(-3) ≈ 0.0486" 0.0486 sp2 0.01;
  
  (* Test sigma *)
  let sig1 = Dist.sigma (-3.0) in
  assert_float_close ~msg:"sigma(-3) ≈ 0.0486" 0.0486 sig1 0.01;
  
  (* Test sampling produces values *)
  let sample = Dist.sample_normal () in
  assert_true ~msg:"sample_normal produces finite value" (Float.is_finite sample);
  
  (* Test reparameterization *)
  let param = { Types.mu = 0.5; rho = -2.0 } in
  let w = Dist.reparam param in
  assert_true ~msg:"reparam produces finite weight" (Float.is_finite w)

(* Test 2: Layer operations *)
let test_layers () =
  Printf.printf "\n=== Testing Layer Operations ===\n";
  
  Util.make_rng 42;
  
  (* Initialize a simple layer *)
  let layer = Layer.init_layer 2 3 Float.tanh in
  
  assert_true ~msg:"Layer has correct weight shape" 
    (Array.length layer.weights = 3 && Array.length layer.weights.(0) = 2);
  
  assert_true ~msg:"Layer has correct bias shape"
    (Array.length layer.bias = 3);
  
  (* Sample an instance *)
  let inst = Layer.sample_inst_layer layer in
  
  assert_true ~msg:"Instantiated layer has correct shape"
    (Array.length inst.weights = 3 && Array.length inst.bias = 3);
  
  (* Forward pass *)
  let x = [| 1.0; 2.0 |] in
  let output = Layer.forward_inst_layer inst x in
  
  assert_true ~msg:"Forward pass produces output of correct size"
    (Array.length output = 3);
  
  assert_true ~msg:"Forward pass produces finite values"
    (Array.for_all Float.is_finite output)

(* Test 3: Forward with cache *)
let test_forward_cache () =
  Printf.printf "\n=== Testing Forward with Cache ===\n";
  
  Util.make_rng 42;
  
  let model = Model.sample_mlp 1 5 1 () in
  let inst = Model.sample_inst_model model in
  
  let x = [| 0.5 |] in
  let cache = Layer.forward_with_cache inst x in
  
  assert_true ~msg:"Cache has correct number of layers"
    (List.length cache.layer_caches = 2);
  
  assert_true ~msg:"Cache output is finite"
    (Array.for_all Float.is_finite cache.output);
  
  assert_true ~msg:"Cache input matches x"
    (cache.input.(0) = x.(0));
  
  Printf.printf "  Output value: %.6f\n" cache.output.(0)

(* Test 4: ELBO computation *)
let test_elbo () =
  Printf.printf "\n=== Testing ELBO Computation ===\n";
  
  Util.make_rng 42;
  
  (* Create simple test data *)
  let data = [
    ([| 0.0 |], [| 0.0 |]);
    ([| 1.0 |], [| 1.0 |]);
    ([| -1.0 |], [| -1.0 |]);
  ] in
  
  let model = Model.sample_mlp 1 5 1 () in
  let elbo = Model.compute_elbo model data 5 0.5 0.0 1.0 in
  
  Printf.printf "  ELBO value: %.6f\n" elbo;
  assert_true ~msg:"ELBO is finite" (Float.is_finite elbo);
  assert_true ~msg:"ELBO is positive (since it's a loss)" (elbo > 0.0)

(* Test 5: Backpropagation *)
let test_backprop () =
  Printf.printf "\n=== Testing Backpropagation ===\n";
  
  Util.make_rng 42;
  
  let data = [([| 1.0 |], [| 0.5 |])] in
  
  let model = Model.sample_mlp 1 3 1 () in
  let inst = Model.sample_inst_model model in
  
  let grads = Train.backprop 1.0 inst data in
  
  assert_true ~msg:"Backprop produces correct number of layer gradients"
    (List.length grads = 2);
  
  let layer1_grad = List.nth grads 0 in
  assert_true ~msg:"Layer 1 gradients are finite"
    (Array.for_all (fun row -> Array.for_all Float.is_finite row) 
      layer1_grad.weight_grads);
  
  Printf.printf "  Sample gradient: %.6f\n" 
    layer1_grad.weight_grads.(0).(0)

(* Test 6: Gradient computation *)
let test_gradients () =
  Printf.printf "\n=== Testing Full Gradient Computation ===\n";
  
  Util.make_rng 42;
  
  let data = [
    ([| 0.0 |], [| 0.0 |]);
    ([| 1.0 |], [| 1.0 |]);
  ] in
  
  let model = Model.sample_mlp 1 3 1 () in
  let grads = Train.compute_gradient model data 3 0.5 0.0 1.0 in
  
  assert_true ~msg:"Gradients have correct number of layers"
    (List.length grads.layer_grads = 2);
  
  let layer1_grad = List.nth grads.layer_grads 0 in
  let sample_mu_grad = layer1_grad.weight_grads.(0).(0).grad_mu in
  let sample_rho_grad = layer1_grad.weight_grads.(0).(0).grad_rho in
  
  Printf.printf "  Sample grad_mu: %.6f\n" sample_mu_grad;
  Printf.printf "  Sample grad_rho: %.6f\n" sample_rho_grad;
  
  assert_true ~msg:"grad_mu is finite" (Float.is_finite sample_mu_grad);
  assert_true ~msg:"grad_rho is finite" (Float.is_finite sample_rho_grad)

(* Test 7: Parameter update *)
let test_update () =
  Printf.printf "\n=== Testing Parameter Update ===\n";
  
  Util.make_rng 42;
  
  let model = Model.sample_mlp 1 3 1 () in
  let data = [([| 0.0 |], [| 0.0 |]); ([| 1.0 |], [| 1.0 |])] in
  
  let grads = Train.compute_gradient model data 3 0.5 0.0 1.0 in
  let updated_model = Train.update_params model grads 0.01 in
  
  let orig_param = (List.hd model.layers).weights.(0).(0) in
  let new_param = (List.hd updated_model.layers).weights.(0).(0) in
  
  Printf.printf "  Original mu: %.6f, rho: %.6f\n" orig_param.mu orig_param.rho;
  Printf.printf "  Updated mu: %.6f, rho: %.6f\n" new_param.mu new_param.rho;
  
  assert_true ~msg:"Parameters changed after update"
    (orig_param.mu <> new_param.mu || orig_param.rho <> new_param.rho)

(* Test 8: Small training run *)
let test_training () =
  Printf.printf "\n=== Testing Training (10 epochs) ===\n";
  
  Util.make_rng 42;
  
  (* Simple linear data *)
  let data = List.init 10 (fun i ->
    let x = float_of_int i /. 5.0 -. 1.0 in
    ([| x |], [| x *. 0.5 |])
  ) in
  
  let model = Model.sample_mlp 1 5 1 () in
  let initial_loss = Model.compute_elbo model data 3 0.5 0.0 1.0 in
  
  Printf.printf "  Initial loss: %.6f\n" initial_loss;
  
  let (trained, _loss_history) = Train.train model data 10 0.01 3 0.5 0.0 1.0 in
  let final_loss = Model.compute_elbo trained data 3 0.5 0.0 1.0 in
  
  Printf.printf "  Final loss: %.6f\n" final_loss;
  Printf.printf "  Improvement: %.6f\n" (initial_loss -. final_loss);
  
  assert_true ~msg:"Loss decreased after training" (final_loss < initial_loss)

(* Test 9: Check data generation *)
let test_data_generation () =
  Printf.printf "\n=== Testing Data Generation ===\n";
  
  Util.make_rng 42;
  
  let data = Util.make_sin_data 10 in
  
  assert_true ~msg:"Generated correct number of points"
    (List.length data = 10);
  
  let (x, y) = List.hd data in
  assert_true ~msg:"Input is 1D" (Array.length x = 1);
  assert_true ~msg:"Output is 1D" (Array.length y = 1);
  
  Printf.printf "  Sample point: x=%.3f, y=%.3f\n" x.(0) y.(0);
  
  assert_true ~msg:"x in range [-5, 5]" (x.(0) >= -5.0 && x.(0) <= 5.0);
  assert_true ~msg:"y is finite" (Float.is_finite y.(0))

(* Main test runner *)
let () =
  Printf.printf "\n";
  Printf.printf "╔════════════════════════════════════╗\n";
  Printf.printf "║  Bayesian Neural Network Tests     ║\n";
  Printf.printf "╚════════════════════════════════════╝\n";
  
  test_distributions ();
  test_layers ();
  test_forward_cache ();
  test_elbo ();
  test_backprop ();
  test_gradients ();
  test_update ();
  test_data_generation ();
  test_training ();
  
  Printf.printf "\n";
  Printf.printf "╔════════════════════════════════════╗\n";
  Printf.printf "║  All Tests Complete!               ║\n";
  Printf.printf "╚════════════════════════════════════╝\n";
  Printf.printf "\n"
