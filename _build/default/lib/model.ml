(* code for instantiating a model and running through it *)

open Types
open Layer

let sample_inst_model (m : model) : inst_model =
  List.map sample_inst_layer m

let forward_inst (inst : inst_model) (x : float array) (activation : float -> float) : float array =
  List.fold_left (
    fun input layer ->
      let z = Layer.forward_inst_layer layer input in
      Array.map activation z
  )
  x inst

let sample_mlp (input_dim : int) (hidden : int) (output_dim : int) () =
  let l1 =
    Layer.init_layer input_dim hidden tanh in
  let l2 =
    Layer.init_layer hidden output_dim (fun x -> x) in
  [ l1; l2 ]