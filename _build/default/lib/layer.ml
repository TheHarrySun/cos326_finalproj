(* Defining the layers *)
open Types
open Dist

(* a function to initialize a parameter *)
let init_param () : param =
  { mu = Random.float 0.1 -. 0.05; rho = -3.0}

(* a function for initializing an entire layer, given an activation function, input dimension, and output dimension *)
let init_layer (input_dim : int) (output_dim : int) (activation : float -> float) : layer =
  let w : param array array = Array.init output_dim (fun _ -> Array.init input_dim (fun _ -> init_param ())) in
  let b : param array = Array.init output_dim (fun _ -> init_param()) in
  { weights = w; bias = b; activation = activation}

(* a function to sample an instantiated layer *)
let sample_inst_layer (l : layer) : inst_layer = 
  let w : float array array = Array.map (fun row -> Array.map (fun p -> reparam p) row) l.weights in
  let b : float array = Array.map (fun p -> reparam p) l.bias in
  { weights = w; bias = b}

(* forward pass through an instantiated layer *)
let forward_inst_layer (inst : inst_layer) (x : float array) : float array = 
  let out_dim = Array.length inst.bias in
  Array.init out_dim (fun i -> 
    let sum = ref inst.bias.(i) in
    for j = 0 to Array.length x - 1 do
      sum := !sum +. inst.weights.(i).(j) *. x.(j)
    done;
    !sum  
  )