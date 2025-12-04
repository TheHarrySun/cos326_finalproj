(* Core types for paramters, layesr, and models *)

type param = {
  mu : float;
  rho : float;
}

type layer = {
  weights : param array array; 
  bias : param array;
  activation : float -> float;
}

type model = layer list

type inst_layer = {
  weights : float array array;
  bias : float array;
}

type inst_model = inst_layer list