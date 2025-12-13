(* stores activation functions and their derivatives *)

let tanh x = Float.tanh x
let relu x = max 0.0 x
let identity x = x

(* activation function derivatives *)
let activation_deriv (activation : float -> float) (output : float) : float =
  if activation == tanh then
    1.0 -. output *. output
  else if activation == identity then
    1.0
  else if activation == relu then
    if output > 0.0 then 1.0 else 0.0
  else
    (* defualt assume identity *)
    1.0