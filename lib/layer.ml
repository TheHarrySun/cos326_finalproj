(* Defining the layers *)
open Types
open Dist
open Activation

type forward_cache = {
  input : float array;
  layer_caches : layer_cache list;
  output : float array;
}
and 
layer_cache = {
  input : float array; (* input to this layer *)
  pre_activation : float array; (* the weighted sum *)
  post_activation : float array; (* activation(weighted sum) *)
}

(* a function to initialize a parameter with Xavier/Glorot initialization *)
let init_param (fan_in : int) (fan_out : int) : param =
  (* Xavier initialization: scale = sqrt(2 / (fan_in + fan_out)) *)
  let scale = sqrt (2.0 /. float_of_int (fan_in + fan_out)) in
  { 
    mu = (Random.float (2.0 *. scale)) -. scale;  (* Uniform[-scale, scale] *)
    rho = -1.5  (* softplus(-1.5) ≈ 0.20, closer to Python's 0.22 *)
  }

(* a function for initializing an entire layer, given an activation function, input dimension, and output dimension *)
let init_layer (input_dim : int) (output_dim : int) (activation : Activation.activation) : layer =
  let w : param array array = Array.init output_dim (fun _ -> 
    Array.init input_dim (fun _ -> init_param input_dim output_dim)
  ) in
  let b : param array = Array.init output_dim (fun _ -> 
    init_param input_dim output_dim
  ) in
  { weights = w; bias = b; activation = activation}

(* a function to sample an instantiated layer *)
let sample_inst_layer (l : layer) : inst_layer = 
  let w : float array array = Array.map (fun row -> Array.map (fun p -> reparam p) row) l.weights in
  let b : float array = Array.map (fun p -> reparam p) l.bias in
  { weights = w; bias = b; activation = l.activation}

(* forward pass through an instantiated layer *)
let forward_inst_layer (inst : inst_layer) (x : float array) : float array = 
  let out_dim = Array.length inst.bias in
  Array.init out_dim (fun i -> 
    let sum = ref inst.bias.(i) in
    for j = 0 to Array.length x - 1 do
      sum := !sum +. inst.weights.(i).(j) *. x.(j)
    done;
    Activation.apply inst.activation !sum  
  )

(* forward pass through model and cache the pre-activation and post-activation values *)
let forward_with_cache (inst : inst_model) (x : float array) : forward_cache =
  let layer_caches = ref [] in
  let current_input = ref x in

  List.iter (fun layer -> 
    (* compute weighted sum *)
    let z = Array.init (Array.length layer.bias) (fun i -> 
      let sum = ref layer.bias.(i) in
      for j = 0 to Array.length !current_input - 1 do 
        sum := !sum +. layer.weights.(i).(j) *. !current_input.(j)
      done;
      !sum
    ) in

    let a = Array.map (Activation.apply layer.activation) z in

    layer_caches := {
    input = !current_input;
    pre_activation = z;
    post_activation = a
    } :: !layer_caches;

    current_input := a
  ) inst;

  {
    input = x;
    layer_caches = List.rev !layer_caches;
    output = !current_input
  }
