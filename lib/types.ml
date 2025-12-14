(* Core types for parameters, layers, and models *)

type param = {
  mu : float;
  rho : float;
}

type layer = {
  weights : param array array; 
  bias : param array;
  activation : Activation.activation;
}

type model = {
  layers : layer list;
  log_noise_precision : param;  (* learnable noise precision *)
}

type inst_layer = {
  weights : float array array;
  bias : float array;
  activation : Activation.activation;
}

type inst_model = inst_layer list

(* types for storing gradients *)
type param_grad = {
  grad_mu : float;
  grad_rho : float;
}

type layer_grad = {
  weight_grads : param_grad array array;
  bias_grads : param_grad array;
}

type model_grad = {
  layer_grads : layer_grad list;
  noise_grad : param_grad;
}

type inst_layer_grad = {
  weight_grads : float array array;
  bias_grads : float array;
}

type inst_model_grad = inst_layer_grad list
