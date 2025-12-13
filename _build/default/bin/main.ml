open Final_proj

(* =============================================================================
   BAYESIAN NEURAL NETWORK - Main Training Script
   =============================================================================
   
   This script trains a Bayesian Neural Network on synthetic sin wave data.
   
   TO CONFIGURE TRAINING:
   Simply edit the hyperparameter values below. All hyperparameters are
   centralized in one place, and the same values are automatically used
   throughout training, evaluation, and display.
   
   Key hyperparameters:
   - epochs: Number of training iterations
   - learning_rate: Step size for gradient descent (0.001-0.01 typical)
   - mc_samples: Monte Carlo samples for gradient estimation (higher = more stable)
   - beta: KL weight/regularization strength (lower = less regularization)
           Typical values: 0.001 (weak) to 1.0 (standard VAE)
   ============================================================================= *)

let () =
  Printf.printf "=================================\n";
  Printf.printf "Bayesian Neural Network Training\n";
  Printf.printf "=================================\n\n";
  
  (* Set random seed for reproducibility *)
  Util.make_rng 42;
  
  (* ===== HYPERPARAMETERS - Edit these to configure training ===== *)
  let n_points = 1000 in
  
  (* Model architecture: list of layer sizes [input_dim; hidden1; hidden2; ...; output_dim]
     Examples:
     - [1; 20; 1]        = single hidden layer with 20 neurons
     - [1; 50; 20; 1]    = two hidden layers with 50 and 20 neurons
     - [1; 64; 32; 16; 1] = three hidden layers with 64, 32, and 16 neurons *)
  let architecture = [1; 20; 1] in
  
  let epochs = 10000 in
  let learning_rate = 0.0001 in  (* Match Python's lr *)
  let mc_samples = 3 in         (* Match Python *)
  let beta = 1.0 in             (* Test with beta=1.0 like Python *)
  
  (* Prior distribution parameters: p(w) = N(prior_mu, prior_sigma²)
     Standard choice: prior_mu = 0.0, prior_sigma = 1.0 (standard normal)
     Tighter prior (e.g., prior_sigma = 0.1) increases regularization
     Wider prior (e.g., prior_sigma = 2.0) decreases regularization *)
  let prior_mu = 0.0 in
  let prior_sigma = 1.0 in
  (* ============================================================== *)
  
  (* Generate training data *)
  Printf.printf "Generating training data...\n";
  let data = Util.make_polynomial_data n_points in
  Printf.printf "Generated %d data points (polynomial curve)\n\n" (List.length data);
  
  (* Initialize model with specified architecture *)
  Printf.printf "Initializing model...\n";
  
  (* Create activation functions: relu for all hidden layers, identity for output *)
  let num_layers = List.length architecture - 1 in
  let activations = List.init num_layers (fun i ->
    if i < num_layers - 1 then Model.relu else (fun x -> x)
  ) in
  
  let model = Model.create_mlp architecture activations in
  
  (* Print initial weights *)
  Printf.printf "\n=== INITIAL ===";
  Model.print_model_weights model;
  
  (* Format architecture string for display *)
  let arch_str = String.concat " -> " (List.map string_of_int architecture) in
  Printf.printf "Model: [%s]\n\n" arch_str;
  
  (* Compute initial loss *)
  let initial_loss = Model.compute_elbo model data mc_samples beta prior_mu prior_sigma in
  let initial_loss_per_point = initial_loss /. float_of_int n_points in
  Printf.printf "Initial loss: %.6f (%.4f per point)\n\n" initial_loss initial_loss_per_point;
  
  (* Train *)
  Printf.printf "Starting training...\n";
  Printf.printf "Hyperparameters:\n";
  Printf.printf "  - Architecture: [%s]\n" arch_str;
  Printf.printf "  - Epochs: %d\n" epochs;
  Printf.printf "  - Learning rate: %.4f\n" learning_rate;
  Printf.printf "  - MC samples: %d\n" mc_samples;
  Printf.printf "  - KL weight (beta): %.3f\n" beta;
  Printf.printf "  - Prior: N(%.2f, %.2f²)\n\n" prior_mu prior_sigma;
  
  let (trained_model, loss_history) = Train.train model data epochs learning_rate mc_samples beta prior_mu prior_sigma in
  
  (* Compute final loss *)
  let final_loss = Model.compute_elbo trained_model data mc_samples beta prior_mu prior_sigma in
  let final_loss_per_point = final_loss /. float_of_int n_points in
  Printf.printf "\n=================================\n";
  Printf.printf "Training complete!\n";
  Printf.printf "Final loss: %.6f (%.4f per point)\n" final_loss final_loss_per_point;
  Printf.printf "Improvement: %.6f\n" (initial_loss -. final_loss);
  Printf.printf "Loss reduction: %.1f%%\n" ((initial_loss -. final_loss) /. initial_loss *. 100.0);
  Printf.printf "=================================\n\n";
  
  (* Print trained model weights *)
  Printf.printf "\n=== FINAL ===";
  Model.print_model_weights trained_model;
  
  (* Save loss history to CSV *)
  Printf.printf "Saving training history...\n";
  let loss_oc = open_out "bnn_loss.csv" in
  Printf.fprintf loss_oc "epoch,loss_per_point\n";
  List.iteri (fun i loss ->
    Printf.fprintf loss_oc "%d,%.6f\n" (i + 1) loss
  ) loss_history;
  close_out loss_oc;
  Printf.printf "  Saved loss history to bnn_loss.csv\n\n";
  
  (* Generate predictions for visualization *)
  Printf.printf "Generating predictions for plotting...\n";
  
  (* Save training data to CSV *)
  let train_oc = open_out "bnn_training.csv" in
  Printf.fprintf train_oc "x,y\n";
  List.iter (fun (x_arr, y_arr) ->
    Printf.fprintf train_oc "%.6f,%.6f\n" x_arr.(0) y_arr.(0)
  ) data;
  close_out train_oc;
  Printf.printf "  Saved training data to bnn_training.csv\n";
  
  (* Generate predictions on a dense grid for smooth plotting *)
  let num_pred_points = 100 in
  let x_min = -5.0 in
  let x_max = 5.0 in
  let pred_oc = open_out "bnn_predictions.csv" in
  Printf.fprintf pred_oc "x,true_y,mean,std,lower,upper\n";
  
  for i = 0 to num_pred_points - 1 do
    let x = x_min +. (float_of_int i /. float_of_int (num_pred_points - 1)) *. (x_max -. x_min) in
    let true_y = 2. *. (x ** 2.) +. x +. 1. in

    
    (* Sample multiple predictions to estimate mean and uncertainty *)
    let num_samples = 50 in
    let predictions = Array.init num_samples (fun _ ->
      let inst = Model.sample_inst_model trained_model in
      (Model.forward_inst inst [| x |]).(0)
    ) in
    
    (* Compute statistics *)
    let mean = Array.fold_left (+.) 0.0 predictions /. float_of_int num_samples in
    let variance = Array.fold_left (fun acc p -> 
      acc +. (p -. mean) ** 2.0
    ) 0.0 predictions /. float_of_int num_samples in
    let std = Float.sqrt variance in
    let lower = mean -. 2.0 *. std in  (* 95% confidence interval *)
    let upper = mean +. 2.0 *. std in
    
    Printf.fprintf pred_oc "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" x true_y mean std lower upper
  done;
  close_out pred_oc;
  Printf.printf "  Saved predictions to bnn_predictions.csv\n\n";
  
  (* Test predictions on a few points *)
  Printf.printf "Sample Predictions:\n";
  Printf.printf "%-8s %-12s %-12s %-12s\n" "x" "True y" "Mean Pred" "Error";
  Printf.printf "%-8s %-12s %-12s %-12s\n" "--------" "------------" "------------" "------------";
  
  let test_points = [-2.0; -1.8; -1.6; -1.4; -1.2; -1.0; -0.8; -0.6; -0.4; -0.2; 0.0; 0.2; 0.4; 0.6; 0.8; 1.0; 2.0] in
  List.iter (fun x ->
    let true_y = Float.sin x in
    (* Average over multiple samples for stable prediction *)
    let num_samples = 50 in
    let predictions = Array.init num_samples (fun _ ->
      let inst = Model.sample_inst_model trained_model in
      (Model.forward_inst inst [| x |]).(0)
    ) in
    let pred_y = Array.fold_left (+.) 0.0 predictions /. float_of_int num_samples in
    let error = abs_float (pred_y -. true_y) in
    Printf.printf "%-8.2f %-12.4f %-12.4f %-12.4f\n" x true_y pred_y error
  ) test_points;
  
  Printf.printf "\n";
  Printf.printf "=================================\n";
  Printf.printf "Run 'python3 plot_bnn.py' to visualize results\n";
  Printf.printf "=================================\n"